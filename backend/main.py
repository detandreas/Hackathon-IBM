"""
EarthRisk AI — FastAPI Backend
Climate risk intelligence for insurance underwriters.
Uses XGBoost ML predictions from Sentinel-2 / ERA5-Land data, with deterministic fallback.
"""

import math
import uuid
import json
import sqlite3
import csv
import io
import os
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from regions import REGIONS, nearest_region_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("earthrisk")

# ─── ML Engine ─────────────────────────────────────────────────────────────────
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ML.ml_engine import engine as ml_engine
        ml_engine.ensure_ready()
        ML_AVAILABLE = ml_engine.ready
    log.info("ML engine ready: %s", ML_AVAILABLE)
except Exception as _ml_err:
    log.warning("ML engine not available: %s", _ml_err)
    ml_engine = None
    ML_AVAILABLE = False

# ─── OpenAI ───────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
except Exception:
    OPENAI_AVAILABLE = False
    openai_client = None

# ─── PDF ──────────────────────────────────────────────────────────────────────
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EarthRisk AI",
    description="Climate risk intelligence API for insurance underwriters",
    version="2.0.0",
)

cors_origins_env = os.environ.get("CORS_ORIGINS", "")
if cors_origins_env.strip():
    allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    # Backward-compatible default for local/dev and single-domain demos.
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Vercel serverless runtime is read-only outside /tmp.
if os.environ.get("VERCEL"):
    DB_PATH = os.environ.get("DB_PATH", "/tmp/climate_risk.db")
else:
    DB_PATH = os.environ.get("DB_PATH", os.path.join(_HERE, "data", "climate_risk.db"))


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS risk_snapshots (
            id TEXT PRIMARY KEY, area_name TEXT, lat REAL, lon REAL,
            score REAL, tier TEXT, factors TEXT, summary TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS underwriter_feedback (
            id TEXT PRIMARY KEY, snapshot_id TEXT, action TEXT,
            override_score REAL, reason TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS asset_portfolios (
            id TEXT PRIMARY KEY, insurer_id TEXT, name TEXT,
            lat REAL, lon REAL, value REAL, proximity_risk INTEGER, created_at TEXT
        );
    """)
    conn.commit()
    conn.close()


init_db()


# ─── Models ───────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    lat: float
    lon: float
    area_name: str


class FeedbackRequest(BaseModel):
    snapshot_id: str
    action: str
    override_score: Optional[float] = None
    reason: Optional[str] = None


# ─── Deterministic PRNG (matches frontend JS implementation) ──────────────────

def _seeded_random(seed):
    s = int(seed) & 0xFFFFFFFF
    def gen():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 0xFFFFFFFF
    return gen


# ─── Climate data helpers ─────────────────────────────────────────────────────

def _has_climate_tables() -> bool:
    """Check if the DB has populated climate data tables."""
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT COUNT(*) as n FROM sentinel2_features"
        ).fetchone()
        conn.close()
        return row and row["n"] > 0
    except Exception:
        return False


_CLIMATE_DATA_READY = None


def climate_data_ready() -> bool:
    global _CLIMATE_DATA_READY
    if _CLIMATE_DATA_READY is None:
        _CLIMATE_DATA_READY = _has_climate_tables()
    return _CLIMATE_DATA_READY


def ml_features_for_region(region_name: str) -> dict | None:
    """Compute risk features using ML predictions (XGBoost)."""
    if ML_AVAILABLE and ml_engine is not None:
        try:
            return ml_engine.compute_risk_features(region_name)
        except Exception:
            pass
    return None


def real_features_for_region(conn, region_name: str) -> dict | None:
    """Compute risk features from real satellite/climate data (SQL fallback)."""
    try:
        s2 = conn.execute("""
            SELECT
                AVG(CASE WHEN month >= '2023-01' THEN NDVI END)  AS recent_ndvi,
                AVG(CASE WHEN month <  '2020-01' THEN NDVI END)  AS baseline_ndvi
            FROM sentinel2_features WHERE region = ?
        """, (region_name,)).fetchone()

        era5 = conn.execute("""
            SELECT
                AVG(CASE WHEN month >= '2023-01' THEN temp_2m_C END)           AS recent_temp,
                AVG(CASE WHEN month <  '2020-01' THEN temp_2m_C END)           AS baseline_temp,
                AVG(CASE WHEN month >= '2023-01' THEN soil_water_vol_m3m3 END) AS recent_soil
            FROM era5_land_features WHERE region = ?
        """, (region_name,)).fetchone()

        has_ndvi = (s2 and s2["recent_ndvi"] is not None
                    and s2["baseline_ndvi"] is not None)
        has_era5 = (era5 and era5["recent_temp"] is not None
                    and era5["baseline_temp"] is not None)

        if not has_ndvi and not has_era5:
            return None

        ndvi_drop = 30.0
        if has_ndvi and s2["baseline_ndvi"] > 0:
            ndvi_drop = max(0, (s2["baseline_ndvi"] - s2["recent_ndvi"])
                           / s2["baseline_ndvi"] * 100)

        temp_increase = 1.5
        if has_era5:
            temp_increase = max(0, era5["recent_temp"] - era5["baseline_temp"])

        land_stress = 0.5
        if era5 and era5["recent_soil"] is not None and era5["recent_soil"] > 0:
            land_stress = max(0, min(1, 1 - (era5["recent_soil"] / 0.4)))

        return {
            "ndvi_drop": round(min(95, max(5, ndvi_drop)), 1),
            "temp_increase": round(min(4.5, max(0.3, temp_increase)), 2),
            "land_stress": round(min(0.95, max(0.05, land_stress)), 3),
            "asset_proximity": 50.0,
        }
    except Exception:
        return None


def get_region_trend_data(conn, region_name: str) -> list:
    """Build monthly risk score time series from real data."""
    try:
        rows = conn.execute("""
            SELECT e.month,
                   AVG(e.temp_2m_C)           AS temp,
                   AVG(e.soil_water_vol_m3m3) AS soil,
                   AVG(s.NDVI)                AS ndvi
            FROM era5_land_features e
            LEFT JOIN sentinel2_features s
              ON e.region = s.region AND e.month = s.month
            WHERE e.region = ?
            GROUP BY e.month
            ORDER BY e.month
        """, (region_name,)).fetchall()

        if not rows:
            return []

        temps = [r["temp"] for r in rows if r["temp"] is not None]
        ndvis = [r["ndvi"] for r in rows if r["ndvi"] is not None]
        baseline_temp = sum(temps[:12]) / max(len(temps[:12]), 1) if temps else 20
        baseline_ndvi = sum(ndvis[:6]) / max(len(ndvis[:6]), 1) if ndvis else 0.5

        trend_data = []
        for r in rows:
            ndvi = r["ndvi"] if r["ndvi"] is not None else baseline_ndvi
            temp = r["temp"] if r["temp"] is not None else baseline_temp
            soil = r["soil"] if r["soil"] is not None else 0.25

            ndvi_drop_pct = max(0, (baseline_ndvi - ndvi) / baseline_ndvi * 100) if baseline_ndvi > 0 else 0
            temp_inc = max(0, temp - baseline_temp)
            stress = max(0, min(1, 1 - (soil / 0.4)))

            score = (
                0.30 * min(ndvi_drop_pct / 100, 1)
                + 0.25 * min(temp_inc / 4.5, 1)
                + 0.25 * stress
                + 0.20 * 0.5
            ) * 100
            score = round(max(1, min(99, score)), 1)

            trend_data.append({"date": r["month"], "score": score})

        return trend_data
    except Exception:
        return []


# ─── Risk computation ─────────────────────────────────────────────────────────

def deterministic_features(lat: float, lon: float) -> dict:
    """Synthetic fallback when no real data is available."""
    seed = lat * 137.5 + lon * 239.7
    return {
        "ndvi_drop": min(95.0, max(5.0, round(
            abs(math.sin(seed * 0.31 + 1.1)) * 60
            + abs(math.cos(seed * 0.17)) * 35, 1))),
        "temp_increase": min(4.5, max(0.3, round(
            abs(math.sin(seed * 0.53 + 2.3)) * 3.5 + 0.5, 2))),
        "land_stress": min(0.95, max(0.05, round(
            abs(math.sin(seed * 0.79 + 0.7)) * 0.7 + 0.1, 3))),
        "asset_proximity": min(95.0, max(5.0, round(
            abs(math.cos(seed * 0.43 + 1.9)) * 80 + 10, 1))),
    }


def compute_score(features: dict) -> float:
    score = (
        0.30 * (features["ndvi_drop"] / 100)
        + 0.25 * (features["temp_increase"] / 4.5)
        + 0.25 * features["land_stress"]
        + 0.20 * (features["asset_proximity"] / 100)
    ) * 100
    return round(min(99.0, max(1.0, score)), 1)


def score_to_tier(score: float) -> str:
    if score >= 76:
        return "CRITICAL"
    elif score >= 51:
        return "HIGH"
    elif score >= 26:
        return "MEDIUM"
    return "LOW"


def generate_trend_data(score: float, lat: float, lon: float) -> list:
    """Synthetic 48-month trend fallback."""
    trend_type = "rising" if score > 65 else ("improving" if score < 35 else "stable")
    seed = lat * 71.3 + lon * 43.7
    data = []
    base = score - (12 if trend_type == "rising" else -8 if trend_type == "improving" else 0)
    for month in range(48):
        noise = math.sin(seed * (month + 1) * 0.31) * 5
        drift = (
            month * 0.25 if trend_type == "rising"
            else -month * 0.17 if trend_type == "improving"
            else math.sin(month * 0.4) * 3
        )
        val = round(min(99, max(1, base + drift + noise)), 1)
        year = 2021 + month // 12
        mo = (month % 12) + 1
        data.append({"date": f"{year}-{mo:02d}", "score": val})
    return data


async def generate_ai_summary(area_name, score, features, tier):
    if not OPENAI_AVAILABLE or not os.environ.get("OPENAI_API_KEY"):
        return generate_fallback_summary(area_name, score, features, tier)
    prompt = (
        f"You are a climate risk analyst for an insurance company. "
        f"Area: {area_name}. Risk score: {score}/100 ({tier} tier). "
        f"Vegetation loss: {features['ndvi_drop']}%. "
        f"Temperature increase: {features['temp_increase']}°C above baseline. "
        f"Land stress index: {features['land_stress']:.2f}. "
        f"Asset proximity score: {features['asset_proximity']}%. "
        f"Write a 2-3 sentence professional risk briefing for an underwriter. "
        f"Be specific, actionable, and reference the actual data values."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise, professional climate risk analyst. Write factual, data-driven briefings for insurance underwriters."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150, temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return generate_fallback_summary(area_name, score, features, tier)


def generate_fallback_summary(area_name, score, features, tier):
    tier_text = {
        "CRITICAL": "critical — immediate underwriting action required",
        "HIGH": "high — elevated loading factor recommended",
        "MEDIUM": "moderate — standard risk protocols apply",
        "LOW": "low — within acceptable risk parameters",
    }.get(tier, "moderate")
    return (
        f"{area_name} presents a {tier_text}, with a composite climate risk score of {score}/100. "
        f"Satellite-derived vegetation loss stands at {features['ndvi_drop']}%, indicating "
        f"{'significant' if features['ndvi_drop'] > 50 else 'moderate'} biomass degradation, "
        f"compounded by a {features['temp_increase']}°C temperature anomaly above the 2000-2020 baseline. "
        f"Land stress index of {features['land_stress']:.2f} and asset proximity factor of {features['asset_proximity']:.0f}% "
        f"suggest {'heightened' if score > 60 else 'contained'} exposure for property and casualty portfolios in this zone."
    )


# ─── Patch generation from real data ──────────────────────────────────────────

_patches_cache: list | None = None


def generate_patches_from_db() -> list:
    """Generate 200 patches from pre-computed database predictions (for Vercel when ML unavailable)."""
    patches = []
    conn = get_db()

    try:
        snapshots = conn.execute("SELECT * FROM risk_snapshots").fetchall()
        snapshot_map = {s["area_name"]: s for s in snapshots}
    finally:
        conn.close()

    patch_id = 1

    for key, info in REGIONS.items():
        region_name = info["name"]
        base_snapshot = snapshot_map.get(region_name)

        if not base_snapshot:
            continue

        base_score = base_snapshot["score"]
        base_factors = json.loads(base_snapshot["factors"])
        rng = _seeded_random(info["center_lat"] * 1000 + info["center_lon"] * 100)

        for area in info["areas"]:
            area_name = area["name"]
            lat = area["lat"]
            lon = area["lon"]

            raw_score = base_score + (rng() - 0.5) * 18
            score = round(max(5, min(99, raw_score)))
            tier = score_to_tier(score)
            trend = "rising" if score > 65 else ("improving" if score < 35 else "stable")

            factors = {
                "ndvi_drop": round(min(95, max(5,
                    base_factors["ndvi_drop"] + (rng() - 0.5) * 10)), 1),
                "temp_increase": round(min(4.5, max(0.3,
                    base_factors["temp_increase"] + (rng() - 0.5) * 0.5)), 2),
                "land_stress": round(min(0.95, max(0.05,
                    base_factors["land_stress"] + (rng() - 0.5) * 0.1)), 3),
                "asset_proximity": round(min(95, max(5, score * 0.6 + rng() * 30)), 1),
            }

            trend_data = generate_trend_data(score, lat, lon)

            patches.append({
                "id": f"patch-{patch_id:03d}",
                "name": area_name,
                "region": info["display_region"],
                "cluster": region_name,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "score": score,
                "tier": tier,
                "trend": trend,
                "trendData": trend_data,
                "factors": factors,
                "real_data": True,
                "ml_prediction": False,
            })
            patch_id += 1

    log.info("Generated %d patches from database", len(patches))
    return patches


def generate_all_patches() -> list:
    """Generate 200 risk patches using ML predictions (cached after first call)."""
    global _patches_cache
    if _patches_cache is not None:
        return _patches_cache

    use_real = climate_data_ready()
    conn = get_db() if use_real else None

    patches = []
    patch_id = 1

    for key, info in REGIONS.items():
        base_features = None
        region_trend = None

        base_features = ml_features_for_region(info["name"])

        if not base_features and conn:
            base_features = real_features_for_region(conn, info["name"])

        if ML_AVAILABLE and ml_engine is not None:
            region_trend = ml_engine.predict_trend_data(info["name"])

        if not region_trend and conn:
            region_trend = get_region_trend_data(conn, info["name"])

        if not base_features:
            base_features = deterministic_features(
                info["center_lat"], info["center_lon"])

        base_score = compute_score(base_features)
        rng = _seeded_random(info["center_lat"] * 1000 + info["center_lon"] * 100)

        for area in info["areas"]:
            area_name = area["name"] if isinstance(area, dict) else area
            lat = area["lat"] if isinstance(area, dict) else info["center_lat"] + (rng() - 0.5) * info["spread"] * 2
            lon = area["lon"] if isinstance(area, dict) else info["center_lon"] + (rng() - 0.5) * info["spread"] * 2

            raw_score = base_score + (rng() - 0.5) * 18
            score = round(max(5, min(99, raw_score)))
            tier = score_to_tier(score)
            trend = "rising" if score > 65 else ("improving" if score < 35 else "stable")

            factors = {
                "ndvi_drop": round(min(95, max(5,
                    base_features["ndvi_drop"] + (rng() - 0.5) * 10)), 1),
                "temp_increase": round(min(4.5, max(0.3,
                    base_features["temp_increase"] + (rng() - 0.5) * 0.5)), 2),
                "land_stress": round(min(0.95, max(0.05,
                    base_features["land_stress"] + (rng() - 0.5) * 0.1)), 3),
                "asset_proximity": round(min(95, max(5,
                    score * 0.6 + rng() * 30)), 1),
            }

            if region_trend and len(region_trend) >= 6:
                trend_data = region_trend
            else:
                trend_data = generate_trend_data(score, lat, lon)

            patches.append({
                "id": f"patch-{patch_id:03d}",
                "name": area_name,
                "region": info["display_region"],
                "cluster": info["name"],
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "score": score,
                "tier": tier,
                "trend": trend,
                "trendData": trend_data,
                "factors": factors,
                "real_data": use_real and base_features is not None,
                "ml_prediction": ML_AVAILABLE,
            })
            patch_id += 1

    if conn:
        conn.close()

    _patches_cache = patches
    log.info("Generated %d patches (ML=%s, real_data=%s)", len(patches), ML_AVAILABLE, use_real)
    return patches


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "openai": OPENAI_AVAILABLE,
        "climate_data": climate_data_ready(),
        "ml_models": ML_AVAILABLE,
    }


@app.get("/api/ml/predictions")
def get_ml_predictions():
    """Return all cached ML predictions for debugging and frontend consumption."""
    if not ML_AVAILABLE or ml_engine is None:
        return {"ml_available": False, "predictions": {}}
    return {
        "ml_available": True,
        "predictions": ml_engine.get_all_predictions(),
    }


@app.get("/api/ml/region/{region_name}")
def get_ml_region(region_name: str):
    """Return ML predictions for a specific region."""
    if not ML_AVAILABLE or ml_engine is None:
        raise HTTPException(503, "ML engine not available")
    risk = ml_engine.compute_risk_features(region_name)
    trend = ml_engine.predict_trend_data(region_name)
    climate = ml_engine.predict_climate(region_name)
    indices = ml_engine.predict_indices_next(region_name)
    return {
        "region": region_name,
        "risk_features": risk,
        "trend_data": trend,
        "climate_prediction": climate,
        "indices_prediction": indices,
    }


@app.get("/api/regions")
def get_regions():
    """Return 200 risk patches across 10 Greek regions with real data scores."""
    # Use ML engine if available (local development)
    if ML_AVAILABLE and ml_engine is not None:
        return generate_all_patches()
    # Fall back to pre-computed database predictions (Vercel deployment)
    return generate_patches_from_db()


@app.get("/api/regions/{region_id}/trends")
def get_region_trends(region_id: str):
    """Return detailed time series for a region."""
    info = REGIONS.get(region_id)
    if not info:
        raise HTTPException(404, "Region not found")

    conn = get_db()
    try:
        ndvi_rows = conn.execute("""
            SELECT month, ROUND(AVG(NDVI), 4) AS ndvi,
                   ROUND(AVG(NDMI), 4) AS ndmi,
                   ROUND(AVG(BSI), 4) AS bsi
            FROM sentinel2_features WHERE region = ?
            GROUP BY month ORDER BY month
        """, (info["name"],)).fetchall()

        climate_rows = conn.execute("""
            SELECT month,
                   ROUND(AVG(temp_2m_C), 2) AS temp,
                   ROUND(AVG(precip_mm_day), 3) AS precip,
                   ROUND(AVG(soil_water_vol_m3m3), 4) AS soil_moisture
            FROM era5_land_features WHERE region = ?
            GROUP BY month ORDER BY month
        """, (info["name"],)).fetchall()
    finally:
        conn.close()

    return {
        "region": info["name"],
        "ndvi_trend": [
            {"date": r["month"], "ndvi": r["ndvi"], "ndmi": r["ndmi"], "bsi": r["bsi"]}
            for r in ndvi_rows
        ],
        "climate_trend": [
            {"date": r["month"], "temp": r["temp"], "precip": r["precip"],
             "soil_moisture": r["soil_moisture"]}
            for r in climate_rows
        ],
    }


@app.post("/api/score")
async def score_area(req: ScoreRequest):
    """Compute climate risk score — uses real data when available."""
    region_key = nearest_region_key(req.lat, req.lon)
    region_name = REGIONS[region_key]["name"] if region_key else "Unknown"

    # Priority: ML → SQL → deterministic
    features = ml_features_for_region(region_name)

    if not features and climate_data_ready():
        conn = get_db()
        features = real_features_for_region(conn, region_name)
        conn.close()

    if not features:
        features = deterministic_features(req.lat, req.lon)

    score = compute_score(features)
    tier = score_to_tier(score)

    trend_data_list = []
    if ML_AVAILABLE and ml_engine is not None:
        trend_data_list = ml_engine.predict_trend_data(region_name)
    if not trend_data_list and climate_data_ready():
        conn = get_db()
        trend_data_list = get_region_trend_data(conn, region_name)
        conn.close()
    if not trend_data_list:
        trend_data_list = generate_trend_data(score, req.lat, req.lon)

    summary = await generate_ai_summary(req.area_name, score, features, tier)

    snapshot_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO risk_snapshots (id, area_name, lat, lon, score, tier, factors, summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot_id, req.area_name, req.lat, req.lon, score, tier,
             json.dumps(features), summary, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "id": snapshot_id,
        "area_name": req.area_name,
        "lat": req.lat,
        "lon": req.lon,
        "score": score,
        "tier": tier,
        "factors": features,
        "summary": summary,
        "trend_data": trend_data_list,
        "created_at": now,
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO underwriter_feedback (id, snapshot_id, action, override_score, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (feedback_id, req.snapshot_id, req.action, req.override_score, req.reason, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "feedback_id": feedback_id}


@app.get("/api/feedback/stats")
def feedback_stats():
    conn = get_db()
    try:
        rows = conn.execute("SELECT action, reason FROM underwriter_feedback").fetchall()
    finally:
        conn.close()
    total = len(rows)
    agrees = sum(1 for r in rows if r["action"] == "agree")
    reason_counts: dict[str, int] = {}
    for r in rows:
        if r["reason"]:
            reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
    common = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
    return {
        "total_signals": total,
        "agree_rate": round(agrees / total * 100, 1) if total > 0 else 0,
        "common_reasons": [{"reason": r, "count": c} for r, c in common],
    }


@app.post("/api/assets/upload")
async def upload_assets(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    assets = []
    now = datetime.now(timezone.utc).isoformat()

    HIGH_RISK_CENTERS = [
        (39.6, 22.4, 0.45), (38.6, 23.6, 0.40),
        (36.2, 28.0, 0.30), (37.5, 22.3, 0.45),
    ]

    conn = get_db()
    try:
        for row in reader:
            name = row.get("name", "").strip()
            try:
                lat = float(row.get("lat", 0))
                lon = float(row.get("lon", 0))
                value = float(row.get("value", 0))
            except (ValueError, TypeError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            prox_risk = 0
            for clat, clon, radius in HIGH_RISK_CENTERS:
                if math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2) <= radius:
                    prox_risk = 1
                    break
            asset_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO asset_portfolios (id, insurer_id, name, lat, lon, value, proximity_risk, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, "default", name, lat, lon, value, prox_risk, now),
            )
            assets.append({"id": asset_id, "name": name, "lat": lat, "lon": lon,
                           "value": value, "proximity_risk": prox_risk})
        conn.commit()
    finally:
        conn.close()

    feats = [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
         "properties": {"id": a["id"], "name": a["name"], "value": a["value"],
                        "proximity_risk": a["proximity_risk"]}}
        for a in assets
    ]
    return {
        "type": "FeatureCollection", "features": feats,
        "meta": {"total": len(assets),
                 "high_risk_count": sum(1 for a in assets if a["proximity_risk"])},
    }


@app.get("/api/history")
def get_history():
    conn = get_db()
    try:
        snapshots = conn.execute(
            """SELECT s.*, f.action FROM risk_snapshots s
               LEFT JOIN underwriter_feedback f ON f.snapshot_id = s.id
               ORDER BY s.created_at DESC LIMIT 50"""
        ).fetchall()
    finally:
        conn.close()
    return {
        "snapshots": [
            {"id": row["id"], "area_name": row["area_name"], "lat": row["lat"],
             "lon": row["lon"], "score": row["score"], "tier": row["tier"],
             "factors": json.loads(row["factors"]) if row["factors"] else {},
             "summary": row["summary"], "created_at": row["created_at"],
             "action": row["action"]}
            for row in snapshots
        ]
    }


@app.get("/api/stats")
def get_stats():
    """KPIs match the map: same patch list as GET /api/regions (scores include local variation)."""
    if ML_AVAILABLE and ml_engine is not None:
        patches = generate_all_patches()
    else:
        patches = generate_patches_from_db()

    conn = get_db()
    try:
        fb = conn.execute("SELECT COUNT(*) AS cnt FROM underwriter_feedback").fetchone()
    finally:
        conn.close()

    feedback_count = int(fb["cnt"] or 0) if fb else 0

    scores = [
        p["score"]
        for p in patches
        if isinstance(p.get("score"), (int, float))
    ]
    n = len(scores)
    if n == 0:
        return {
            "total_snapshots": 0,
            "avg_score": 0,
            "critical_count": 0,
            "feedback_count": feedback_count,
        }

    return {
        "total_snapshots": n,
        "avg_score": round(sum(scores) / n),
        "critical_count": sum(1 for s in scores if s >= 76),
        "feedback_count": feedback_count,
    }


@app.get("/api/report/pdf")
def generate_pdf_report(
    area_name: str = Query(...),
    score: float = Query(...),
    summary: str = Query(""),
    snapshot_id: str = Query("N/A"),
):
    if not FPDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF generation not available")

    tier = score_to_tier(score)
    features = deterministic_features(0, 0)

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM risk_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row:
            features = json.loads(row["factors"]) if row["factors"] else features
            if not summary:
                summary = row["summary"] or ""
    finally:
        conn.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(10, 15, 30)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 212, 170)
    pdf.set_xy(15, 12)
    pdf.cell(0, 10, "EarthRisk AI", ln=False)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 190, 210)
    pdf.set_xy(70, 14)
    pdf.cell(0, 8, "Climate Risk Intelligence Report", ln=False)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 130, 150)
    pdf.set_xy(15, 28)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  Snapshot: {snapshot_id[:18]}...")

    pdf.set_xy(0, 42)
    pdf.set_fill_color(20, 30, 50)
    pdf.rect(0, 42, 210, 35, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 48)
    pdf.cell(130, 10, area_name, ln=False)

    score_color = (
        (239, 68, 68) if tier == "CRITICAL"
        else (245, 158, 11) if tier == "HIGH"
        else (234, 179, 8) if tier == "MEDIUM"
        else (0, 212, 170)
    )
    pdf.set_fill_color(*score_color)
    pdf.rect(155, 45, 40, 25, "F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(155, 48)
    pdf.cell(40, 10, str(int(score)), align="C", ln=False)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(155, 59)
    pdf.cell(40, 5, tier, align="C")

    pdf.set_xy(15, 85)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, "RISK FACTOR BREAKDOWN", ln=True)

    headers = ["Factor", "Value", "Weight", "Contribution"]
    col_w = [75, 35, 30, 45]
    data = [
        ["Vegetation Loss (NDVI Drop)", f"{features['ndvi_drop']}%", "30%", f"{round(features['ndvi_drop'] * 0.3, 1)}"],
        ["Temperature Increase", f"{features['temp_increase']}°C", "25%", f"{round(features['temp_increase'] / 4.5 * 25, 1)}"],
        ["Land Stress Index", f"{features['land_stress']:.3f}", "25%", f"{round(features['land_stress'] * 25, 1)}"],
        ["Asset Proximity Score", f"{features['asset_proximity']}%", "20%", f"{round(features['asset_proximity'] * 0.2, 1)}"],
    ]

    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(150, 165, 190)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=0, fill=True, align="L")
    pdf.ln()

    for j, row_data in enumerate(data):
        pdf.set_fill_color(*(18, 26, 44) if j % 2 == 0 else (22, 33, 56))
        pdf.set_text_color(200, 210, 230)
        pdf.set_font("Helvetica", "", 9)
        for i, cell in enumerate(row_data):
            pdf.cell(col_w[i], 7, str(cell), border=0, fill=True, align="L")
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, "AI RISK BRIEFING", ln=True)
    pdf.set_fill_color(10, 20, 40)
    pdf.set_text_color(200, 210, 230)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(15)
    summary_text = summary or generate_fallback_summary(area_name, score, features, tier)
    pdf.multi_cell(180, 6, summary_text, fill=True)

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, "METHODOLOGY", ln=True)
    pdf.set_text_color(140, 155, 180)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        180, 5,
        "Risk score computed via weighted composite formula: Score = 0.30*VegetationLoss + 0.25*TempIncrease + "
        "0.25*LandStress + 0.20*AssetProximity, normalised to 0-100. Vegetation loss derived from Sentinel-2 "
        "NDVI time series (2015-2025). Temperature anomaly from ERA5 reanalysis vs 2015-2019 baseline. "
        "Land stress from soil moisture indices. AI briefing generated by GPT-4o-mini.",
    )

    pdf.set_y(-20)
    pdf.set_fill_color(10, 15, 30)
    pdf.rect(0, pdf.get_y() - 2, 210, 25, "F")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 100, 130)
    pdf.cell(0, 6,
             f"EarthRisk AI v2.0  ·  Snapshot ID: {snapshot_id}  ·  Regulatory Audit Trail  ·  IBM Hackathon 2026",
             align="C")

    pdf_output = pdf.output()
    pdf_bytes = pdf_output.encode('latin-1') if isinstance(pdf_output, str) else pdf_output
    filename = f"earthrisk-{area_name.replace(' ', '-').lower()}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
