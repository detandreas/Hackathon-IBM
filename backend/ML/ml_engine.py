"""
ML Engine — unified interface for training, loading, and predicting
with the XGBoost models used by the EarthRisk AI API.

Models:
  - ERA5 models: predict 5 climate variables from satellite + lag features
  - Combined models (short-term): predict NDVI/NDBI/NDMI/BSI for next obs
  - Combined models (long-term): predict monthly satellite index trends

All predictions are precomputed at startup and cached per-region.
"""

import os
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_DATA = _BACKEND / "data"
_MODELS = _BACKEND / "models"

TARGET_CLIMATE = [
    "temp_2m_C", "precip_mm_day", "soil_water_vol_m3m3",
    "net_solar_rad_Jm2", "skin_temp_C",
]
TARGET_INDICES = ["NDVI", "NDBI", "NDMI", "BSI"]
SPECTRAL_BANDS = [
    "B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir16", "B12_swir22",
]
CLIMATE_FEATURES = [
    "soil_water_vol_m3m3", "net_solar_rad_Jm2", "temp_2m_C",
    "skin_temp_C", "precip_mm_day",
]


class MLEngine:
    """Singleton-ish engine that trains/loads models and serves predictions."""

    def __init__(self):
        self.era5_models: dict | None = None
        self.era5_scaler = None
        self.era5_features: list | None = None

        self.short_term_models: dict | None = None
        self.short_term_scaler = None
        self.short_term_features: list | None = None

        self.long_term_models: dict | None = None
        self.long_term_scaler = None
        self.long_term_features: list | None = None

        self._era5_df: pd.DataFrame | None = None
        self._s2_df: pd.DataFrame | None = None
        self._ready = False

        self._risk_features_cache: dict[str, dict] = {}
        self._trend_data_cache: dict[str, list] = {}
        self._climate_cache: dict[str, dict] = {}
        self._indices_next_cache: dict[str, dict] = {}
        self._indices_lt_cache: dict[str, dict] = {}

    @property
    def ready(self) -> bool:
        return self._ready

    # ── data loading ──────────────────────────────────────────────

    def _load_data(self):
        era5_path = _DATA / "era5_land_features.csv"
        s2_path = _DATA / "sentinel2_features.csv"
        if not era5_path.exists() or not s2_path.exists():
            log.warning("CSV data files not found in %s", _DATA)
            return False

        self._era5_df = pd.read_csv(era5_path)
        self._s2_df = pd.read_csv(s2_path)
        self._era5_df["time"] = pd.to_datetime(
            self._era5_df["time"], format="ISO8601", utc=True
        ).dt.tz_localize(None)
        self._s2_df["datetime"] = pd.to_datetime(
            self._s2_df["datetime"], format="ISO8601", utc=True
        ).dt.tz_localize(None)
        self._era5_df.sort_values("time", inplace=True)
        self._s2_df.sort_values("datetime", inplace=True)
        self._era5_df.reset_index(drop=True, inplace=True)
        self._s2_df.reset_index(drop=True, inplace=True)
        return True

    # ── training ──────────────────────────────────────────────────

    def train_all(self):
        """Train ERA5 + combined (short/long-term) models and save to disk."""
        _MODELS.mkdir(parents=True, exist_ok=True)
        if not self._load_data():
            raise RuntimeError("Cannot train: data files missing")

        self._train_era5_models()
        self._train_combined_models()
        self._ready = True
        log.info("All models trained and saved.")

    def _train_era5_models(self):
        era5 = self._era5_df.copy()
        s2 = self._s2_df.copy()

        for feat in TARGET_CLIMATE:
            for lag in [1, 2, 3, 7]:
                era5[f"{feat}_lag_{lag}"] = era5[feat].shift(lag)

        era5["date"] = era5["time"].dt.normalize()
        s2_merge = s2[["datetime"] + TARGET_INDICES + SPECTRAL_BANDS + ["cloud_cover_pct"]].copy()
        s2_merge["date"] = s2_merge["datetime"].dt.normalize()

        merged = pd.merge_asof(
            era5.sort_values("date"),
            s2_merge.sort_values("date"),
            on="date", direction="nearest",
        )
        for col in TARGET_INDICES + SPECTRAL_BANDS + ["cloud_cover_pct"]:
            merged[col] = merged[col].ffill().bfill()
        merged.dropna(subset=[f"{TARGET_CLIMATE[0]}_lag_7"], inplace=True)

        lag_feats = [f"{f}_lag_{l}" for f in TARGET_CLIMATE for l in [1, 2, 3, 7]]
        input_feats = TARGET_INDICES + SPECTRAL_BANDS + ["cloud_cover_pct"] + lag_feats

        X = merged[input_feats].fillna(merged[input_feats].mean())
        y = merged[TARGET_CLIMATE]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {}
        for feat in TARGET_CLIMATE:
            m = XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            m.fit(X_tr, y_tr[feat])
            models[feat] = m

        joblib.dump(models, _MODELS / "era5_models.pkl")
        joblib.dump(scaler, _MODELS / "scaler_era5.pkl")
        joblib.dump(input_feats, _MODELS / "features_era5.pkl")
        with open(_MODELS / "metadata_era5.pkl", "wb") as f:
            pickle.dump({
                "target_features": TARGET_CLIMATE,
                "input_features": input_feats,
                "satellite_indices": TARGET_INDICES,
                "spectral_bands": SPECTRAL_BANDS,
                "lag_features": lag_feats,
            }, f)

        self.era5_models = models
        self.era5_scaler = scaler
        self.era5_features = input_feats
        log.info("ERA5 models trained (%d features).", len(input_feats))

    def _train_combined_models(self):
        s2 = self._s2_df.copy()
        era5 = self._era5_df.copy()

        # --- short-term ---
        for idx in TARGET_INDICES:
            for lag in [1, 2, 3]:
                s2[f"{idx}_lag_{lag}"] = s2[idx].shift(lag)
        s2_clean = s2.dropna().copy()

        s2_clean["month_str"] = s2_clean["month"]
        era5["month_str"] = era5["time"].dt.strftime("%Y-%m")
        era5_monthly = era5.groupby("month_str")[CLIMATE_FEATURES].mean().reset_index()
        era5_monthly.columns = ["month_str"] + [f"{c}_monthly" for c in CLIMATE_FEATURES]

        short_data = pd.merge(s2_clean, era5_monthly, on="month_str", how="left").dropna()

        lag_st = [f"{idx}_lag_{l}" for idx in TARGET_INDICES for l in [1, 2, 3]]
        clim_mo = [f"{c}_monthly" for c in CLIMATE_FEATURES]
        st_features = SPECTRAL_BANDS + lag_st + clim_mo

        X_s = short_data[st_features].fillna(short_data[st_features].mean())
        y_s = short_data[TARGET_INDICES]
        sc_s = StandardScaler()
        X_ss = sc_s.fit_transform(X_s)
        Xtr, Xte, ytr, yte = train_test_split(X_ss, y_s, test_size=0.2, random_state=42)

        st_models = {}
        for idx in TARGET_INDICES:
            m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
            m.fit(Xtr, ytr[idx])
            st_models[idx] = m

        # --- long-term ---
        s2_lt = self._s2_df.copy()
        s2_lt["year_month"] = s2_lt["datetime"].dt.to_period("M")
        monthly_idx = s2_lt.groupby("year_month")[TARGET_INDICES].mean().reset_index()
        monthly_idx["year_month"] = monthly_idx["year_month"].astype(str)

        lt_data = monthly_idx.copy()
        for idx in TARGET_INDICES:
            for lag in [1, 2, 3, 6, 12]:
                lt_data[f"{idx}_lag_{lag}"] = lt_data[idx].shift(lag)
        lt_data.dropna(inplace=True)

        era5_monthly_all = era5.groupby("month_str")[CLIMATE_FEATURES].mean().reset_index()
        era5_monthly_all.columns = ["year_month"] + CLIMATE_FEATURES
        lt_data = pd.merge(lt_data, era5_monthly_all, on="year_month", how="left").dropna()

        lag_lt = [f"{idx}_lag_{l}" for idx in TARGET_INDICES for l in [1, 2, 3, 6, 12]]
        lt_features = lag_lt + CLIMATE_FEATURES

        X_l = lt_data[lt_features].fillna(lt_data[lt_features].mean())
        y_l = lt_data[TARGET_INDICES]
        sc_l = StandardScaler()
        X_ls = sc_l.fit_transform(X_l)
        Xtr, Xte, ytr, yte = train_test_split(X_ls, y_l, test_size=0.2, random_state=42)

        lt_models = {}
        for idx in TARGET_INDICES:
            m = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
            m.fit(Xtr, ytr[idx])
            lt_models[idx] = m

        # save
        joblib.dump(st_models, _MODELS / "short_term_models.pkl")
        joblib.dump(sc_s, _MODELS / "scaler_short_term.pkl")
        joblib.dump(st_features, _MODELS / "features_short_term.pkl")
        joblib.dump(lt_models, _MODELS / "long_term_models.pkl")
        joblib.dump(sc_l, _MODELS / "scaler_long_term.pkl")
        joblib.dump(lt_features, _MODELS / "features_long_term.pkl")
        with open(_MODELS / "metadata.pkl", "wb") as f:
            pickle.dump({
                "target_indices": TARGET_INDICES,
                "climate_features": CLIMATE_FEATURES,
                "spectral_bands": SPECTRAL_BANDS,
                "short_term_features": st_features,
                "long_term_features": lt_features,
            }, f)

        self.short_term_models = st_models
        self.short_term_scaler = sc_s
        self.short_term_features = st_features
        self.long_term_models = lt_models
        self.long_term_scaler = sc_l
        self.long_term_features = lt_features
        log.info("Combined models trained (short-term %d feats, long-term %d feats).",
                 len(st_features), len(lt_features))

    # ── loading ───────────────────────────────────────────────────

    def load(self) -> bool:
        """Load pre-trained models from disk.  Returns True on success."""
        required = [
            _MODELS / "era5_models.pkl",
            _MODELS / "short_term_models.pkl",
            _MODELS / "long_term_models.pkl",
        ]
        if not all(p.exists() for p in required):
            log.info("Pre-trained models not found; will train from scratch.")
            return False

        self.era5_models = joblib.load(_MODELS / "era5_models.pkl")
        self.era5_scaler = joblib.load(_MODELS / "scaler_era5.pkl")
        self.era5_features = joblib.load(_MODELS / "features_era5.pkl")

        self.short_term_models = joblib.load(_MODELS / "short_term_models.pkl")
        self.short_term_scaler = joblib.load(_MODELS / "scaler_short_term.pkl")
        self.short_term_features = joblib.load(_MODELS / "features_short_term.pkl")

        self.long_term_models = joblib.load(_MODELS / "long_term_models.pkl")
        self.long_term_scaler = joblib.load(_MODELS / "scaler_long_term.pkl")
        self.long_term_features = joblib.load(_MODELS / "features_long_term.pkl")

        self._ready = True
        log.info("Pre-trained models loaded from %s", _MODELS)
        return True

    def ensure_ready(self):
        """Load or train models so the engine is ready to serve."""
        if self._ready:
            return
        if not self._load_data():
            log.warning("No CSV data — ML engine disabled.")
            return
        if not self.load():
            log.info("Training models for the first time…")
            self.train_all()
        self._precompute_all_regions()

    # ── precomputation cache ──────────────────────────────────────

    def _precompute_all_regions(self):
        """Batch-precompute ML predictions for every region in the data."""
        if not self._ready:
            return
        era5 = self._era5_df
        s2 = self._s2_df
        if era5 is None or "region" not in era5.columns:
            log.warning("No region column in ERA5 data; skipping precomputation.")
            return

        region_names = era5["region"].dropna().unique().tolist()
        log.info("Precomputing ML predictions for %d regions…", len(region_names))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ── batch ERA5 lag features (done ONCE for whole dataset) ──
            era5_with_lags = era5.copy()
            for feat in TARGET_CLIMATE:
                for lag in [1, 2, 3, 7]:
                    era5_with_lags[f"{feat}_lag_{lag}"] = era5_with_lags.groupby("region")[feat].shift(lag)

            # ── batch S2 lag features (done ONCE) ──
            s2_with_lags = s2.copy()
            for idx in TARGET_INDICES:
                for lag in [1, 2, 3]:
                    s2_with_lags[f"{idx}_lag_{lag}"] = s2_with_lags.groupby("region")[idx].shift(lag)

            # ── batch monthly climate for short-term merge ──
            era5_month = era5.copy()
            era5_month["month_str"] = era5_month["time"].dt.strftime("%Y-%m")
            era5_monthly = era5_month.groupby("month_str")[CLIMATE_FEATURES].mean().reset_index()
            era5_monthly_st = era5_monthly.copy()
            era5_monthly_st.columns = ["month_str"] + [f"{c}_monthly" for c in CLIMATE_FEATURES]

            # ── batch S2 monthly for long-term ──
            s2_lt = s2.copy()
            s2_lt["year_month"] = s2_lt["datetime"].dt.to_period("M").astype(str)
            s2_monthly_all = s2_lt.groupby(["region", "year_month"])[TARGET_INDICES].mean().reset_index()

            era5_monthly_lt = era5_monthly.copy()
            era5_monthly_lt.columns = ["year_month"] + CLIMATE_FEATURES

            for name in region_names:
                try:
                    self._climate_cache[name] = self._predict_climate_batch(
                        name, era5_with_lags, s2
                    )
                except Exception as exc:
                    log.debug("predict_climate failed for %s: %s", name, exc)
                    self._climate_cache[name] = None

                try:
                    self._indices_next_cache[name] = self._predict_indices_next_batch(
                        name, s2_with_lags, era5_monthly_st
                    )
                except Exception as exc:
                    log.debug("predict_indices_next failed for %s: %s", name, exc)
                    self._indices_next_cache[name] = None

                try:
                    self._indices_lt_cache[name] = self._predict_indices_lt_batch(
                        name, s2_monthly_all, era5_monthly_lt
                    )
                except Exception as exc:
                    log.debug("predict_indices_lt failed for %s: %s", name, exc)
                    self._indices_lt_cache[name] = None

            for name in region_names:
                self._risk_features_cache[name] = self._compute_risk_features_impl(name)

            for name in region_names:
                self._trend_data_cache[name] = self._predict_trend_data_batch(
                    name, era5_month, s2
                )

        log.info("ML precomputation complete for regions: %s",
                 ", ".join(region_names))

    # ── batch prediction helpers (avoid redundant DF operations) ──

    def _predict_climate_batch(self, region_name, era5_with_lags, s2):
        if self.era5_models is None:
            return None
        region_era5 = era5_with_lags[era5_with_lags["region"] == region_name]
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2
        if region_era5.empty or region_s2.empty:
            return None

        latest = region_era5.iloc[-1:].copy()
        latest_s2 = region_s2.iloc[-1:]
        for col in TARGET_INDICES + SPECTRAL_BANDS:
            latest[col] = latest_s2[col].values[0]
        latest["cloud_cover_pct"] = latest_s2["cloud_cover_pct"].values[0]

        X = latest[self.era5_features].fillna(0)
        X_scaled = self.era5_scaler.transform(X)

        preds = {}
        for feat in TARGET_CLIMATE:
            preds[feat] = float(self.era5_models[feat].predict(X_scaled)[0])
        return preds

    def _predict_indices_next_batch(self, region_name, s2_with_lags, era5_monthly_st):
        if self.short_term_models is None:
            return None
        region_s2 = s2_with_lags[s2_with_lags["region"] == region_name] if "region" in s2_with_lags.columns else s2_with_lags
        if region_s2.empty:
            return None

        latest = region_s2.iloc[-1:].copy()
        latest["month_str"] = latest["month"].values[0]
        merged = pd.merge(latest, era5_monthly_st, on="month_str", how="left")
        merged = merged.fillna(merged.select_dtypes(include=[np.number]).mean())

        X = merged[self.short_term_features].fillna(0)
        X_scaled = self.short_term_scaler.transform(X)

        preds = {}
        for idx in TARGET_INDICES:
            preds[idx] = float(self.short_term_models[idx].predict(X_scaled)[0])
        return preds

    def _predict_indices_lt_batch(self, region_name, s2_monthly_all, era5_monthly_lt):
        if self.long_term_models is None:
            return None
        region_monthly = s2_monthly_all[s2_monthly_all["region"] == region_name].copy()
        if region_monthly.empty:
            return None

        for idx in TARGET_INDICES:
            for lag in [1, 2, 3, 6, 12]:
                region_monthly[f"{idx}_lag_{lag}"] = region_monthly[idx].shift(lag)
        latest = region_monthly.dropna().tail(1).copy()
        if latest.empty:
            return None

        latest["month_str"] = latest["year_month"]
        merged = pd.merge(latest.rename(columns={"month_str": "year_month_merge"}),
                          era5_monthly_lt,
                          left_on="year_month_merge", right_on="year_month", how="left",
                          suffixes=("", "_era5"))
        numeric_means = merged.select_dtypes(include=[np.number]).mean()
        merged = merged.fillna(numeric_means)

        available_feats = [f for f in self.long_term_features if f in merged.columns]
        if len(available_feats) < len(self.long_term_features):
            for f in self.long_term_features:
                if f not in merged.columns:
                    merged[f] = 0

        X = merged[self.long_term_features].fillna(0)
        X_scaled = self.long_term_scaler.transform(X)

        preds = {}
        for idx in TARGET_INDICES:
            preds[idx] = float(self.long_term_models[idx].predict(X_scaled)[0])
        return preds

    def _predict_trend_data_batch(self, region_name, era5_with_month, s2):
        region_era5 = era5_with_month[era5_with_month["region"] == region_name]
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2
        if region_era5.empty:
            return []

        monthly_climate = region_era5.groupby("month_str").agg(
            temp=("temp_2m_C", "mean"),
            soil=("soil_water_vol_m3m3", "mean"),
        ).reset_index()

        if not region_s2.empty and "month" in region_s2.columns:
            monthly_ndvi = region_s2.groupby("month")["NDVI"].mean().reset_index()
            monthly_ndvi.columns = ["month_str", "ndvi"]
            monthly_climate = pd.merge(monthly_climate, monthly_ndvi, on="month_str", how="left")

        temps = monthly_climate["temp"].dropna().tolist()
        baseline_temp = np.mean(temps[:12]) if len(temps) >= 12 else (np.mean(temps) if temps else 20)
        ndvi_vals = monthly_climate["ndvi"].dropna().tolist() if "ndvi" in monthly_climate.columns else []
        baseline_ndvi = np.mean(ndvi_vals[:6]) if len(ndvi_vals) >= 6 else (np.mean(ndvi_vals) if ndvi_vals else 0.5)

        trend = []
        for _, row in monthly_climate.iterrows():
            ndvi = row.get("ndvi", baseline_ndvi) if pd.notna(row.get("ndvi")) else baseline_ndvi
            temp = row["temp"] if pd.notna(row["temp"]) else baseline_temp
            soil = row["soil"] if pd.notna(row["soil"]) else 0.25

            ndvi_drop = max(0, (baseline_ndvi - ndvi) / baseline_ndvi * 100) if baseline_ndvi > 0 else 0
            temp_inc = max(0, temp - baseline_temp)
            stress = max(0, min(1, 1 - (soil / 0.4)))

            score = (0.30 * min(ndvi_drop / 100, 1)
                     + 0.25 * min(temp_inc / 4.5, 1)
                     + 0.25 * stress
                     + 0.20 * 0.5) * 100
            score = round(max(1, min(99, score)), 1)
            trend.append({"date": row["month_str"], "score": score, "predicted": False})

        future = self._risk_features_cache.get(region_name)
        if future and trend:
            last_date = trend[-1]["date"]
            try:
                last_dt = pd.Timestamp(last_date + "-01")
            except Exception:
                last_dt = pd.Timestamp.now()

            for offset in range(1, 13):
                future_dt = last_dt + pd.DateOffset(months=offset)
                future_score = (
                    0.30 * min(future["ndvi_drop"] / 100, 1)
                    + 0.25 * min(future["temp_increase"] / 4.5, 1)
                    + 0.25 * future["land_stress"]
                    + 0.20 * (future["asset_proximity"] / 100)
                ) * 100
                noise = np.sin(offset * 0.7) * 3
                future_score = round(max(1, min(99, future_score + noise)), 1)
                trend.append({
                    "date": future_dt.strftime("%Y-%m"),
                    "score": future_score,
                    "predicted": True,
                })

        return trend

    # ── raw prediction implementations (no caching) ──────────────

    def _predict_climate_impl(self, region_name: str) -> dict | None:
        if self.era5_models is None:
            return None
        era5 = self._era5_df
        s2 = self._s2_df

        region_era5 = era5[era5["region"] == region_name] if "region" in era5.columns else era5
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2

        if region_era5.empty or region_s2.empty:
            return None

        era5_copy = region_era5.copy()
        for feat in TARGET_CLIMATE:
            for lag in [1, 2, 3, 7]:
                era5_copy[f"{feat}_lag_{lag}"] = era5_copy[feat].shift(lag)
        latest = era5_copy.iloc[-1:].copy()

        latest_s2 = region_s2.iloc[-1:]
        for col in TARGET_INDICES + SPECTRAL_BANDS:
            latest[col] = latest_s2[col].values[0]
        latest["cloud_cover_pct"] = latest_s2["cloud_cover_pct"].values[0]

        X = latest[self.era5_features].fillna(0)
        X_scaled = self.era5_scaler.transform(X)

        preds = {}
        for feat in TARGET_CLIMATE:
            preds[feat] = float(self.era5_models[feat].predict(X_scaled)[0])
        return preds

    def _predict_indices_next_impl(self, region_name: str) -> dict | None:
        if self.short_term_models is None:
            return None

        s2 = self._s2_df
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2

        if region_s2.empty:
            return None

        temp = region_s2.copy()
        for idx in TARGET_INDICES:
            for lag in [1, 2, 3]:
                temp[f"{idx}_lag_{lag}"] = temp[idx].shift(lag)
        latest = temp.iloc[-1:].copy()

        era5 = self._era5_df.copy()
        era5["month_str"] = era5["time"].dt.strftime("%Y-%m")
        era5_monthly = era5.groupby("month_str")[CLIMATE_FEATURES].mean().reset_index()
        era5_monthly.columns = ["month_str"] + [f"{c}_monthly" for c in CLIMATE_FEATURES]

        latest["month_str"] = latest["month"].values[0]
        merged = pd.merge(latest, era5_monthly, on="month_str", how="left")
        merged = merged.fillna(merged.select_dtypes(include=[np.number]).mean())

        X = merged[self.short_term_features].fillna(0)
        X_scaled = self.short_term_scaler.transform(X)

        preds = {}
        for idx in TARGET_INDICES:
            preds[idx] = float(self.short_term_models[idx].predict(X_scaled)[0])
        return preds

    def _predict_indices_lt_impl(self, region_name: str) -> dict | None:
        if self.long_term_models is None:
            return None

        s2 = self._s2_df
        era5 = self._era5_df
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2

        if region_s2.empty:
            return None

        temp = region_s2.copy()
        temp["year_month"] = temp["datetime"].dt.to_period("M")
        monthly = temp.groupby("year_month")[TARGET_INDICES].mean().reset_index()
        monthly["year_month"] = monthly["year_month"].astype(str)

        for idx in TARGET_INDICES:
            for lag in [1, 2, 3, 6, 12]:
                monthly[f"{idx}_lag_{lag}"] = monthly[idx].shift(lag)
        latest = monthly.dropna().tail(1).copy()

        if latest.empty:
            return None

        era5_copy = era5.copy()
        era5_copy["month_str"] = era5_copy["time"].dt.strftime("%Y-%m")
        era5_monthly = era5_copy.groupby("month_str")[CLIMATE_FEATURES].mean().reset_index()
        era5_monthly.columns = ["month_str"] + CLIMATE_FEATURES

        latest["month_str"] = latest["year_month"].values[0]
        merged = pd.merge(latest, era5_monthly, on="month_str", how="left")
        merged = merged.fillna(merged.select_dtypes(include=[np.number]).mean())

        X = merged[self.long_term_features].fillna(0)
        X_scaled = self.long_term_scaler.transform(X)

        preds = {}
        for idx in TARGET_INDICES:
            preds[idx] = float(self.long_term_models[idx].predict(X_scaled)[0])
        return preds

    def _compute_risk_features_impl(self, region_name: str) -> dict | None:
        climate = self._climate_cache.get(region_name)
        indices_next = self._indices_next_cache.get(region_name)
        indices_lt = self._indices_lt_cache.get(region_name)

        if not climate and not indices_next:
            return None

        ndvi_drop = 30.0
        if indices_next and indices_lt:
            current_ndvi = indices_next.get("NDVI", 0.25)
            baseline_ndvi = indices_lt.get("NDVI", current_ndvi)
            if baseline_ndvi > 0:
                ndvi_drop = max(0, (baseline_ndvi - current_ndvi) / baseline_ndvi * 100)

        temp_increase = 1.5
        if climate:
            era5 = self._era5_df
            region_era5 = era5[era5["region"] == region_name] if "region" in era5.columns else era5
            if not region_era5.empty:
                baseline_temp = region_era5["temp_2m_C"].mean()
                predicted_temp = climate.get("temp_2m_C", baseline_temp)
                temp_increase = max(0, predicted_temp - baseline_temp)

        land_stress = 0.5
        if climate:
            soil = climate.get("soil_water_vol_m3m3", 0.2)
            land_stress = max(0, min(1, 1 - (soil / 0.4)))

        return {
            "ndvi_drop": round(min(95, max(5, ndvi_drop)), 1),
            "temp_increase": round(min(4.5, max(0.3, temp_increase)), 2),
            "land_stress": round(min(0.95, max(0.05, land_stress)), 3),
            "asset_proximity": 50.0,
        }

    def _predict_trend_data_impl(self, region_name: str) -> list:
        era5 = self._era5_df
        s2 = self._s2_df

        region_era5 = era5[era5["region"] == region_name] if "region" in era5.columns else era5
        region_s2 = s2[s2["region"] == region_name] if "region" in s2.columns else s2

        if region_era5.empty:
            return []

        era5_copy = region_era5.copy()
        era5_copy["month_str"] = era5_copy["time"].dt.strftime("%Y-%m")
        monthly_climate = era5_copy.groupby("month_str").agg(
            temp=("temp_2m_C", "mean"),
            soil=("soil_water_vol_m3m3", "mean"),
        ).reset_index()

        s2_copy = region_s2.copy()
        if not s2_copy.empty and "month" in s2_copy.columns:
            monthly_ndvi = s2_copy.groupby("month")["NDVI"].mean().reset_index()
            monthly_ndvi.columns = ["month_str", "ndvi"]
            monthly_climate = pd.merge(monthly_climate, monthly_ndvi, on="month_str", how="left")

        temps = monthly_climate["temp"].dropna().tolist()
        baseline_temp = np.mean(temps[:12]) if len(temps) >= 12 else (np.mean(temps) if temps else 20)
        ndvi_vals = monthly_climate["ndvi"].dropna().tolist() if "ndvi" in monthly_climate.columns else []
        baseline_ndvi = np.mean(ndvi_vals[:6]) if len(ndvi_vals) >= 6 else (np.mean(ndvi_vals) if ndvi_vals else 0.5)

        trend = []
        for _, row in monthly_climate.iterrows():
            ndvi = row.get("ndvi", baseline_ndvi) if pd.notna(row.get("ndvi")) else baseline_ndvi
            temp = row["temp"] if pd.notna(row["temp"]) else baseline_temp
            soil = row["soil"] if pd.notna(row["soil"]) else 0.25

            ndvi_drop = max(0, (baseline_ndvi - ndvi) / baseline_ndvi * 100) if baseline_ndvi > 0 else 0
            temp_inc = max(0, temp - baseline_temp)
            stress = max(0, min(1, 1 - (soil / 0.4)))

            score = (0.30 * min(ndvi_drop / 100, 1)
                     + 0.25 * min(temp_inc / 4.5, 1)
                     + 0.25 * stress
                     + 0.20 * 0.5) * 100
            score = round(max(1, min(99, score)), 1)
            trend.append({"date": row["month_str"], "score": score, "predicted": False})

        future = self._risk_features_cache.get(region_name)
        if future and trend:
            last_date = trend[-1]["date"]
            try:
                last_dt = pd.Timestamp(last_date + "-01")
            except Exception:
                last_dt = pd.Timestamp.now()

            for offset in range(1, 13):
                future_dt = last_dt + pd.DateOffset(months=offset)
                future_score = (
                    0.30 * min(future["ndvi_drop"] / 100, 1)
                    + 0.25 * min(future["temp_increase"] / 4.5, 1)
                    + 0.25 * future["land_stress"]
                    + 0.20 * (future["asset_proximity"] / 100)
                ) * 100
                noise = np.sin(offset * 0.7) * 3
                future_score = round(max(1, min(99, future_score + noise)), 1)
                trend.append({
                    "date": future_dt.strftime("%Y-%m"),
                    "score": future_score,
                    "predicted": True,
                })

        return trend

    # ── public cached prediction API ──────────────────────────────

    def predict_climate(self, region_name: str) -> dict | None:
        """Return cached climate prediction for a region."""
        if not self._ready:
            return None
        if region_name in self._climate_cache:
            return self._climate_cache[region_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._predict_climate_impl(region_name)
        self._climate_cache[region_name] = result
        return result

    def predict_indices_next(self, region_name: str) -> dict | None:
        """Return cached next-observation index prediction for a region."""
        if not self._ready:
            return None
        if region_name in self._indices_next_cache:
            return self._indices_next_cache[region_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._predict_indices_next_impl(region_name)
        self._indices_next_cache[region_name] = result
        return result

    def predict_indices_longterm(self, region_name: str) -> dict | None:
        """Return cached long-term index prediction for a region."""
        if not self._ready:
            return None
        if region_name in self._indices_lt_cache:
            return self._indices_lt_cache[region_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._predict_indices_lt_impl(region_name)
        self._indices_lt_cache[region_name] = result
        return result

    def compute_risk_features(self, region_name: str) -> dict | None:
        """Return cached risk features for a region."""
        if not self._ready:
            return None
        if region_name in self._risk_features_cache:
            return self._risk_features_cache[region_name]
        self.predict_climate(region_name)
        self.predict_indices_next(region_name)
        self.predict_indices_longterm(region_name)
        result = self._compute_risk_features_impl(region_name)
        self._risk_features_cache[region_name] = result
        return result

    def predict_trend_data(self, region_name: str) -> list:
        """Return cached trend data for a region."""
        if not self._ready:
            return []
        if region_name in self._trend_data_cache:
            return self._trend_data_cache[region_name]
        self.compute_risk_features(region_name)
        result = self._predict_trend_data_impl(region_name)
        self._trend_data_cache[region_name] = result
        return result

    def get_all_predictions(self) -> dict:
        """Return all cached predictions as a summary dict."""
        return {
            "regions_computed": list(self._risk_features_cache.keys()),
            "risk_features": {
                k: v for k, v in self._risk_features_cache.items() if v is not None
            },
            "climate_predictions": {
                k: v for k, v in self._climate_cache.items() if v is not None
            },
            "indices_next": {
                k: v for k, v in self._indices_next_cache.items() if v is not None
            },
            "indices_longterm": {
                k: v for k, v in self._indices_lt_cache.items() if v is not None
            },
        }


# Module-level singleton
engine = MLEngine()
