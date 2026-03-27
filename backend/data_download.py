"""
Climate Risk MVP - Data Download Script
Downloads Sentinel-2 (STAC API) and ERA5-Land (CDS API) data.
Sentinel-2: parallel band downloads with overview reads for speed.
Exports DB-ready DataFrames + CSV files.
"""

import os
import time
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import xarray as xr
from pystac_client import Client as STACClient
import cdsapi

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "tif"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

# ── Configuration ───────────────────────────────────────────────
BBOX = [23.5, 37.8, 24.0, 38.1]      # [west, south, east, north]
YEARS = list(range(2015, 2026))       # 2015-2025
MONTHS = list(range(4, 10))           # April-September
MAX_CLOUD_COVER = 30
MAX_WORKERS = 12
SCENES_PER_PERIOD = 1                 # best scene per year-month combo
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


# ── Sentinel-2 helpers ──────────────────────────────────────────

def _safe_ratio(a: float, b: float) -> float:
    return a / b if b != 0 else np.nan


def _read_band_mean(href: str, bbox: list, overview_level: int = 2) -> float:
    """
    Read a Sentinel-2 COG band clipped to bbox using overview level for speed.
    overview_level=2 gives ~40m resolution (enough for mean aggregation).
    """
    with rasterio.open(href, overview_level=overview_level) as src:
        dst_bounds = transform_bounds(
            "EPSG:4326", src.crs, bbox[0], bbox[1], bbox[2], bbox[3],
        )
        window = from_bounds(*dst_bounds, transform=src.transform)
        data = src.read(1, window=window).astype(np.float32)
        data = data / 10_000.0
        valid = data[data > 0]
        return float(np.mean(valid)) if valid.size > 0 else np.nan


def _process_one_scene(item, bbox: list, band_assets: dict) -> dict | None:
    """Download all bands for a single scene and compute indices."""
    means: dict[str, float] = {}

    for asset_key, band_label in band_assets.items():
        asset = item.assets.get(asset_key)
        if asset is None:
            means[band_label] = np.nan
            continue
        try:
            means[band_label] = _read_band_mean(asset.href, bbox)
        except Exception:
            means[band_label] = np.nan

    b02, b04 = means.get("B02", np.nan), means.get("B04", np.nan)
    b08, b11 = means.get("B08", np.nan), means.get("B11", np.nan)

    if np.isnan(b04) or np.isnan(b08):
        return None

    return {
        "item_id": item.id,
        "datetime": item.properties.get("datetime"),
        "cloud_cover_pct": item.properties.get("eo:cloud_cover"),
        "mgrs_tile": item.properties.get("grid:code",
                                          item.properties.get("s2:mgrs_tile", "")),
        "B02_blue": b02,
        "B03_green": means.get("B03", np.nan),
        "B04_red": b04,
        "B08_nir": b08,
        "B11_swir16": b11,
        "B12_swir22": means.get("B12", np.nan),
        "NDVI": _safe_ratio(b08 - b04, b08 + b04),
        "NDBI": _safe_ratio(b11 - b08, b11 + b08),
        "NDMI": _safe_ratio(b08 - b11, b08 + b11),
        "BSI": _safe_ratio((b11 + b04) - (b08 + b02),
                           (b11 + b04) + (b08 + b02)),
    }


# ── Sentinel-2 fetch ────────────────────────────────────────────

def fetch_sentinel2_bands(bbox: list, years: list[int], months: list[int],
                          scenes_per_period: int = 1) -> pd.DataFrame:
    """
    Parallel download of Sentinel-2 L2A raster bands (COGs).
    Searches each year×month combo separately and keeps the best
    (lowest cloud cover) scene(s) per period.
    Uses overview reads + ThreadPoolExecutor for speed.
    """
    stac_url = "https://earth-search.aws.element84.com/v1"
    client = STACClient.open(stac_url)

    items = []
    for year in years:
        year_count = 0
        for month in months:
            date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-30"
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=date_range,
                max_items=10,
                query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
            )
            period_items = sorted(
                search.items(),
                key=lambda x: x.properties.get("eo:cloud_cover", 100),
            )
            picked = period_items[:scenes_per_period]
            items.extend(picked)
            year_count += len(picked)
        print(f"    {year}: {year_count} scenes (months {months[0]}-{months[-1]})")

    print(f"\n  Total: {len(items)} scenes across {len(years)} years × "
          f"{len(months)} months")
    print(f"  Downloading bands in parallel ({MAX_WORKERS} workers)…")

    band_assets = {
        "blue": "B02", "green": "B03", "red": "B04",
        "nir": "B08", "swir16": "B11", "swir22": "B12",
    }

    records: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process_one_scene, item, bbox, band_assets): item.id
            for item in items
        }
        for future in as_completed(futures):
            done += 1
            scene_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    records.append(result)
                    print(f"  [{done}/{len(items)}] {scene_id} ✓")
                else:
                    print(f"  [{done}/{len(items)}] {scene_id} — skipped (no valid data)")
            except Exception as exc:
                print(f"  [{done}/{len(items)}] {scene_id} — error: {exc}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ── ERA5-Land ───────────────────────────────────────────────────

def fetch_era5_land(bbox: list, years: list[int], months: list[int]) -> pd.DataFrame:
    """
    Download ERA5-Land monthly-averaged reanalysis from the Copernicus CDS.
    Supports multiple years × multiple months in a single request.
    Requires ~/.cdsapirc — see https://cds.climate.copernicus.eu/how-to-api
    """
    c = cdsapi.Client()
    tmp_dir = tempfile.mkdtemp(prefix="era5_")
    download_path = os.path.join(tmp_dir, "era5_land_download")

    area = [bbox[3], bbox[0], bbox[1], bbox[2]]  # [N, W, S, E]

    print(f"  Requesting CDS download  years={years[0]}-{years[-1]}  "
          f"months={months[0]:02d}-{months[-1]:02d}")
    print(f"  Area [N,W,S,E]: {area}")

    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "volumetric_soil_water_layer_1",
                "skin_temperature",
                "surface_net_solar_radiation",
            ],
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
            "area": area,
            "format": "netcdf",
        },
        download_path,
    )

    nc_path = download_path
    if zipfile.is_zipfile(download_path):
        print("  Extracting ZIP archive…")
        with zipfile.ZipFile(download_path, "r") as zf:
            nc_files = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_files:
                nc_files = zf.namelist()
            extracted = nc_files[0]
            zf.extract(extracted, tmp_dir)
            nc_path = os.path.join(tmp_dir, extracted)

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    df = ds.to_dataframe().reset_index()
    ds.close()

    rename_map = {
        "t2m": "temp_2m_K",
        "tp": "total_precip_m",
        "swvl1": "soil_water_vol_m3m3",
        "skt": "skin_temp_K",
        "ssr": "net_solar_rad_Jm2",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    if "valid_time" in df.columns:
        df.rename(columns={"valid_time": "time"}, inplace=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


# ── DB preparation ──────────────────────────────────────────────

def prepare_sentinel_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Sentinel-2 DataFrame for database storage.
    Drops rows with all-NaN bands, rounds floats, adds proper types.
    """
    if df.empty:
        return df

    band_cols = [c for c in df.columns if c.startswith("B0") or c.startswith("B1")]
    index_cols = ["NDVI", "NDBI", "NDMI", "BSI"]

    out = df.copy()
    out = out.dropna(subset=["NDVI"]).reset_index(drop=True)

    for col in band_cols:
        out[col] = out[col].round(6)
    for col in index_cols:
        out[col] = out[col].round(4)

    out["cloud_cover_pct"] = out["cloud_cover_pct"].round(2)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out["month"] = out["datetime"].dt.to_period("M").astype(str)

    col_order = ["item_id", "datetime", "month", "mgrs_tile", "cloud_cover_pct",
                 *band_cols, *index_cols]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order]


def prepare_era5_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ERA5-Land DataFrame for database storage.
    Drops NaN land points (sea), converts units, adds month column.
    """
    if df.empty:
        return df

    out = df.copy()

    value_cols = ["temp_2m_K", "total_precip_m", "soil_water_vol_m3m3",
                  "skin_temp_K", "net_solar_rad_Jm2"]
    existing_value_cols = [c for c in value_cols if c in out.columns]
    out = out.dropna(subset=existing_value_cols, how="all").reset_index(drop=True)

    if "temp_2m_K" in out.columns:
        out["temp_2m_C"] = (out["temp_2m_K"] - 273.15).round(2)
    if "skin_temp_K" in out.columns:
        out["skin_temp_C"] = (out["skin_temp_K"] - 273.15).round(2)
    if "total_precip_m" in out.columns:
        out["precip_mm_day"] = (out["total_precip_m"] * 1000).round(3)

    drop_cols = ["number", "expver", "temp_2m_K", "skin_temp_K", "total_precip_m"]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True)
        out["month"] = out["time"].dt.to_period("M").astype(str)

    out["latitude"] = out["latitude"].round(4)
    out["longitude"] = out["longitude"].round(4)

    for col in ["soil_water_vol_m3m3", "net_solar_rad_Jm2"]:
        if col in out.columns:
            out[col] = out[col].round(4)

    return out


# ── Main ────────────────────────────────────────────────────────

def main():
    sep = "=" * 64
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(sep)
    print("  Climate Risk MVP — Data Download (parallel)")
    print(f"  Region bbox : {BBOX}")
    print(f"  Period      : months {MONTHS[0]}-{MONTHS[-1]}, "
          f"years {YEARS[0]}-{YEARS[-1]}  ({len(YEARS)}y × {len(MONTHS)}m)")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(sep)

    # ── 1. Sentinel-2 raster bands (parallel) ───────────────────
    print(f"\n[1/2] Sentinel-2 L2A bands + indices  "
          f"(Apr-Sep {YEARS[0]}-{YEARS[-1]})")
    t0 = time.time()
    df_sentinel_raw = fetch_sentinel2_bands(BBOX, YEARS, MONTHS, SCENES_PER_PERIOD)
    elapsed_s2 = time.time() - t0
    print(f"\n  Done in {elapsed_s2:.1f}s  |  shape: {df_sentinel_raw.shape}")

    df_sentinel = prepare_sentinel_for_db(df_sentinel_raw)
    print(f"  DB-ready shape: {df_sentinel.shape}")
    print(f"  Columns: {list(df_sentinel.columns)}\n")
    print("  ╔══ Sentinel-2 — First 10 (DB-ready) ══╗")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(df_sentinel.head(10).to_string(index=True))

    sentinel_path = os.path.join(OUTPUT_DIR, "sentinel2_features.csv")
    df_sentinel.to_csv(sentinel_path, index=False)
    print(f"\n  Saved → {sentinel_path}")

    # ── 2. ERA5-Land ────────────────────────────────────────────
    print(f"\n{'-' * 64}")
    print(f"[2/2] ERA5-Land monthly means  (Apr-Sep {YEARS[0]}-{YEARS[-1]})")
    t0 = time.time()

    try:
        df_era5_raw = fetch_era5_land(BBOX, YEARS, MONTHS)
        elapsed_era5 = time.time() - t0
        print(f"\n  Done in {elapsed_era5:.1f}s  |  shape: {df_era5_raw.shape}")

        df_era5 = prepare_era5_for_db(df_era5_raw)
        print(f"  DB-ready shape: {df_era5.shape}")
        print(f"  Columns: {list(df_era5.columns)}\n")
        print("  ╔══ ERA5-Land — First 10 (DB-ready) ══╗")
        print(df_era5.head(10).to_string(index=True))

        era5_path = os.path.join(OUTPUT_DIR, "era5_land_features.csv")
        df_era5.to_csv(era5_path, index=False)
        print(f"\n  Saved → {era5_path}")

    except Exception as exc:
        print(f"\n  ⚠  ERA5 download failed: {exc}")
        print("  Visit: https://cds.climate.copernicus.eu/how-to-api")
        df_era5 = pd.DataFrame()

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  DB Schema Reference:")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ sentinel2_features                               │")
    print("  │   item_id TEXT PK, datetime TIMESTAMPTZ,         │")
    print("  │   month TEXT, mgrs_tile TEXT,                     │")
    print("  │   cloud_cover_pct FLOAT,                         │")
    print("  │   B02..B12 FLOAT (reflectance),                  │")
    print("  │   NDVI FLOAT, NDBI FLOAT, NDMI FLOAT, BSI FLOAT │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │ era5_land_features                               │")
    print("  │   time TIMESTAMPTZ, month TEXT,                  │")
    print("  │   latitude FLOAT, longitude FLOAT,               │")
    print("  │   temp_2m_C FLOAT, skin_temp_C FLOAT,            │")
    print("  │   precip_mm_day FLOAT,                           │")
    print("  │   soil_water_vol_m3m3 FLOAT,                     │")
    print("  │   net_solar_rad_Jm2 FLOAT                        │")
    print("  └──────────────────────────────────────────────────┘")
    print(f"\n  CSV files in: {OUTPUT_DIR}/")
    print(sep)

    return df_sentinel, df_era5


if __name__ == "__main__":
    main()
