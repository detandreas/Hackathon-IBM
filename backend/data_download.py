"""
Climate Risk MVP - Data Download Script
Downloads Sentinel-2 (STAC API) and ERA5-Land (CDS API) data
for 10 Greek regions. Exports DB-ready DataFrames + CSV files.

Parallelism strategy:
  - STAC metadata searches run in parallel (thread pool)
  - Sentinel-2 COG band reads run in parallel (thread pool)
  - All regions run in parallel (thread pool)
  - Sentinel-2 and ERA5 run concurrently (thread pool)
"""

import os
import time
import tempfile
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import xarray as xr
from pystac_client import Client as STACClient
import cdsapi

from regions import REGIONS, GREECE_BBOX, nearest_region_name_vectorized

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "tif"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

# ── Configuration ───────────────────────────────────────────────
YEARS = list(range(2015, 2026))
MONTHS = list(range(4, 10))
MAX_CLOUD_COVER = 30
MAX_WORKERS = 12
STAC_SEARCH_WORKERS = 8
SCENES_PER_PERIOD = 1
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    """Thread-safe print."""
    with _print_lock:
        print(*args, **kwargs)


# ── Sentinel-2 helpers ──────────────────────────────────────────

def _safe_ratio(a: float, b: float) -> float:
    return a / b if b != 0 else np.nan


def _read_band_mean(href: str, bbox: list, overview_level: int = 2) -> float:
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

def _search_one_period(client, bbox, year, month, scenes_per_period):
    """Search STAC for a single year-month combo. Thread-safe."""
    date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-30"
    try:
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
        return period_items[:scenes_per_period]
    except Exception:
        return []


def fetch_sentinel2_bands(bbox: list, years: list[int], months: list[int],
                          scenes_per_period: int = 1,
                          region_name: str = "") -> pd.DataFrame:
    """
    Parallel download of Sentinel-2 L2A raster bands (COGs).
    Both STAC searches and band reads are parallelized.
    """
    stac_url = "https://earth-search.aws.element84.com/v1"
    client = STACClient.open(stac_url)
    prefix = f"  [{region_name}]" if region_name else "  "

    # Parallel STAC searches across all year×month combos
    combos = [(year, month) for year in years for month in months]
    items = []

    with ThreadPoolExecutor(max_workers=STAC_SEARCH_WORKERS) as search_pool:
        future_to_combo = {
            search_pool.submit(
                _search_one_period, client, bbox, y, m, scenes_per_period
            ): (y, m)
            for y, m in combos
        }
        for future in as_completed(future_to_combo):
            result = future.result()
            items.extend(result)

    _tprint(f"{prefix} {len(items)} scenes found — downloading bands "
            f"({MAX_WORKERS} workers)...")

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
                    _tprint(f"{prefix} [{done}/{len(items)}] {scene_id} ok")
                else:
                    _tprint(f"{prefix} [{done}/{len(items)}] {scene_id} -- skipped")
            except Exception as exc:
                _tprint(f"{prefix} [{done}/{len(items)}] {scene_id} -- error: {exc}")

    df = pd.DataFrame(records)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
        df.sort_values("datetime", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ── ERA5-Land ───────────────────────────────────────────────────

def fetch_era5_land(bbox: list, years: list[int], months: list[int]) -> pd.DataFrame:
    """
    Download ERA5-Land monthly-averaged reanalysis from the Copernicus CDS.
    Requires ~/.cdsapirc -- see https://cds.climate.copernicus.eu/how-to-api
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
        print("  Extracting ZIP archive...")
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
    if df.empty:
        return df

    band_cols = [c for c in df.columns if c.startswith("B0") or c.startswith("B1")]
    index_cols = ["NDVI", "NDBI", "NDMI", "BSI"]

    out = df.copy()
    out = out.dropna(subset=["NDVI"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)

    for col in band_cols:
        out[col] = out[col].round(6)
    for col in index_cols:
        out[col] = out[col].round(4)

    out["cloud_cover_pct"] = out["cloud_cover_pct"].round(2)
    out["datetime"] = pd.to_datetime(out["datetime"], format="ISO8601", utc=True)
    out["month"] = out["datetime"].dt.to_period("M").astype(str)

    col_order = ["item_id", "datetime", "month", "region", "mgrs_tile",
                 "cloud_cover_pct", *band_cols, *index_cols]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order]


def prepare_era5_for_db(df: pd.DataFrame) -> pd.DataFrame:
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

    col_order = ["time", "month", "region", "latitude", "longitude",
                 "soil_water_vol_m3m3", "net_solar_rad_Jm2",
                 "temp_2m_C", "skin_temp_C", "precip_mm_day"]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order]


# ── Region-level worker ──────────────────────────────────────────

def _fetch_region_sentinel(key: str, info: dict) -> pd.DataFrame:
    """Fetch Sentinel-2 for a single region. Runs in its own thread."""
    _tprint(f"\n  --- {info['name']} (bbox={info['bbox']}) ---")
    df = fetch_sentinel2_bands(
        info["bbox"], YEARS, MONTHS, SCENES_PER_PERIOD,
        region_name=info["name"],
    )
    if not df.empty:
        df["region"] = info["name"]
    _tprint(f"  [{info['name']}] done: {len(df)} scenes")
    return df


# ── Main ────────────────────────────────────────────────────────

def main():
    sep = "=" * 64
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    region_names = [r["name"] for r in REGIONS.values()]
    print(sep)
    print("  Climate Risk MVP -- Multi-Region Data Download (fully parallel)")
    print(f"  Regions     : {len(REGIONS)} ({', '.join(region_names)})")
    print(f"  Period      : months {MONTHS[0]}-{MONTHS[-1]}, "
          f"years {YEARS[0]}-{YEARS[-1]}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(sep)

    t0_total = time.time()

    # ── Launch Sentinel-2 (all regions) and ERA5 concurrently ──
    with ThreadPoolExecutor(max_workers=len(REGIONS) + 1) as master_pool:

        # Sentinel-2: one future per region (each spawns its own sub-pools)
        s2_futures = {
            master_pool.submit(_fetch_region_sentinel, key, info): info["name"]
            for key, info in REGIONS.items()
        }

        # ERA5: single future
        era5_future = master_pool.submit(fetch_era5_land, GREECE_BBOX, YEARS, MONTHS)

        # Collect Sentinel-2 results
        all_sentinel: list[pd.DataFrame] = []
        for future in as_completed(s2_futures):
            region = s2_futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_sentinel.append(df)
            except Exception as exc:
                _tprint(f"  [{region}] error: {exc}")

        df_sentinel_raw = (pd.concat(all_sentinel, ignore_index=True)
                           if all_sentinel else pd.DataFrame())
        _tprint(f"\n  Sentinel-2 all regions done  |  total shape: "
                f"{df_sentinel_raw.shape}")

        df_sentinel = prepare_sentinel_for_db(df_sentinel_raw)
        sentinel_path = os.path.join(OUTPUT_DIR, "sentinel2_features.csv")
        df_sentinel.to_csv(sentinel_path, index=False)
        _tprint(f"  Saved -> {sentinel_path}  ({len(df_sentinel)} rows)")

        # Collect ERA5 result
        try:
            df_era5_raw = era5_future.result()
            _tprint(f"\n  ERA5 done  |  shape: {df_era5_raw.shape}")

            df_era5_raw["region"] = nearest_region_name_vectorized(
                df_era5_raw["latitude"].values,
                df_era5_raw["longitude"].values,
            )

            df_era5 = prepare_era5_for_db(df_era5_raw)
            era5_path = os.path.join(OUTPUT_DIR, "era5_land_features.csv")
            df_era5.to_csv(era5_path, index=False)
            _tprint(f"  Saved -> {era5_path}  ({len(df_era5)} rows)")

        except Exception as exc:
            _tprint(f"\n  ERA5 download failed: {exc}")
            _tprint("  Visit: https://cds.climate.copernicus.eu/how-to-api")
            df_era5 = pd.DataFrame()

    elapsed = time.time() - t0_total

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Download complete in {elapsed:.1f}s")
    if not df_sentinel.empty:
        print(f"  Sentinel-2: {len(df_sentinel)} scenes across "
              f"{df_sentinel['region'].nunique()} regions")
    if not df_era5.empty:
        print(f"  ERA5-Land:  {len(df_era5)} rows across "
              f"{df_era5['region'].nunique()} regions")
    print(f"  Run db_setup.py to build the database.")
    print(sep)

    return df_sentinel, df_era5


if __name__ == "__main__":
    main()
