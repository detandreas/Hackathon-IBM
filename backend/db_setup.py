"""
Climate Risk MVP - Database Setup
Creates an SQLite database from the downloaded CSV data,
with proper schema, indexes, and useful views.
"""

import os
import sqlite3

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "climate_risk.db")

SENTINEL_CSV = os.path.join(DATA_DIR, "sentinel2_features.csv")
ERA5_CSV = os.path.join(DATA_DIR, "era5_land_features.csv")


def create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS sentinel2_features;
        DROP TABLE IF EXISTS era5_land_features;
        DROP VIEW  IF EXISTS monthly_climate_summary;
        DROP VIEW  IF EXISTS sentinel_yearly_trend;

        CREATE TABLE sentinel2_features (
            item_id         TEXT PRIMARY KEY,
            datetime        TEXT NOT NULL,
            month           TEXT NOT NULL,
            mgrs_tile       TEXT,
            cloud_cover_pct REAL,
            B02_blue        REAL,
            B03_green       REAL,
            B04_red         REAL,
            B08_nir         REAL,
            B11_swir16      REAL,
            B12_swir22      REAL,
            NDVI            REAL,
            NDBI            REAL,
            NDMI            REAL,
            BSI             REAL
        );

        CREATE TABLE era5_land_features (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            time                TEXT NOT NULL,
            month               TEXT NOT NULL,
            latitude            REAL NOT NULL,
            longitude           REAL NOT NULL,
            soil_water_vol_m3m3 REAL,
            net_solar_rad_Jm2   REAL,
            temp_2m_C           REAL,
            skin_temp_C         REAL,
            precip_mm_day       REAL
        );

        CREATE INDEX idx_sentinel_month ON sentinel2_features(month);
        CREATE INDEX idx_era5_month     ON era5_land_features(month);
        CREATE INDEX idx_era5_coords    ON era5_land_features(latitude, longitude);
        CREATE INDEX idx_era5_time      ON era5_land_features(time);
    """)


def create_views(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE VIEW monthly_climate_summary AS
        SELECT
            month,
            ROUND(AVG(temp_2m_C), 2)           AS avg_temp_C,
            ROUND(MAX(temp_2m_C), 2)           AS max_temp_C,
            ROUND(AVG(precip_mm_day), 3)       AS avg_precip_mm,
            ROUND(AVG(soil_water_vol_m3m3), 4) AS avg_soil_moisture,
            ROUND(AVG(skin_temp_C), 2)         AS avg_skin_temp_C,
            COUNT(*)                            AS n_grid_points
        FROM era5_land_features
        GROUP BY month
        ORDER BY month;

        CREATE VIEW sentinel_yearly_trend AS
        SELECT
            SUBSTR(month, 1, 4)                AS year,
            SUBSTR(month, 6, 2)                AS mon,
            ROUND(AVG(NDVI), 4)                AS avg_NDVI,
            ROUND(AVG(NDMI), 4)                AS avg_NDMI,
            ROUND(AVG(BSI), 4)                 AS avg_BSI,
            ROUND(AVG(cloud_cover_pct), 2)     AS avg_cloud,
            COUNT(*)                            AS n_scenes
        FROM sentinel2_features
        GROUP BY year, mon
        ORDER BY year, mon;
    """)


def load_csv(conn: sqlite3.Connection, csv_path: str, table: str):
    df = pd.read_csv(csv_path)
    df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def main():
    sep = "=" * 60

    print(sep)
    print("  Climate Risk MVP — Database Setup")
    print(sep)

    if not os.path.exists(SENTINEL_CSV) or not os.path.exists(ERA5_CSV):
        print("\n  CSV files not found. Run data_download.py first.")
        return

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"\n  Removed existing DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    print("\n  [1/4] Creating tables + indexes…")
    create_tables(conn)

    print("  [2/4] Loading sentinel2_features…")
    n_s2 = load_csv(conn, SENTINEL_CSV, "sentinel2_features")
    print(f"         → {n_s2} rows inserted")

    print("  [3/4] Loading era5_land_features…")
    n_era5 = load_csv(conn, ERA5_CSV, "era5_land_features")
    print(f"         → {n_era5} rows inserted")

    print("  [4/4] Creating views…")
    create_views(conn)

    conn.commit()

    # ── Verification queries ────────────────────────────────────
    print(f"\n{'-' * 60}")
    print("  Verification queries:\n")

    print("  ── Sentinel-2: scenes per year ──")
    df = pd.read_sql_query("""
        SELECT SUBSTR(month,1,4) AS year, COUNT(*) AS scenes
        FROM sentinel2_features GROUP BY year ORDER BY year
    """, conn)
    print(df.to_string(index=False))

    print("\n  ── ERA5-Land: monthly climate summary (sample) ──")
    df = pd.read_sql_query("""
        SELECT * FROM monthly_climate_summary LIMIT 12
    """, conn)
    print(df.to_string(index=False))

    print("\n  ── Sentinel-2: yearly NDVI trend ──")
    df = pd.read_sql_query("""
        SELECT * FROM sentinel_yearly_trend
    """, conn)
    print(df.to_string(index=False))

    conn.close()

    db_size = os.path.getsize(DB_PATH) / 1024
    print(f"\n{sep}")
    print(f"  Database: {DB_PATH}")
    print(f"  Size:     {db_size:.1f} KB")
    print(f"  Tables:   sentinel2_features ({n_s2} rows)")
    print(f"            era5_land_features ({n_era5} rows)")
    print(f"  Views:    monthly_climate_summary")
    print(f"            sentinel_yearly_trend")
    print(sep)


if __name__ == "__main__":
    main()
