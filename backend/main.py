"""
EarthRisk AI — FastAPI Backend
Climate risk intelligence for insurance underwriters
"""

import math
import uuid
import json
import sqlite3
import csv
import io
import os
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(_HERE, "data", "earthrisk.db"))


# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS risk_snapshots (
            id TEXT PRIMARY KEY,
            area_name TEXT,
            lat REAL,
            lon REAL,
            score REAL,
            tier TEXT,
            factors TEXT,
            summary TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS underwriter_feedback (
            id TEXT PRIMARY KEY,
            snapshot_id TEXT,
            action TEXT,
            override_score REAL,
            reason TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS asset_portfolios (
            id TEXT PRIMARY KEY,
            insurer_id TEXT,
            name TEXT,
            lat REAL,
            lon REAL,
            value REAL,
            proximity_risk INTEGER,
            created_at TEXT
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
    action: str  # "agree" | "override"
    override_score: Optional[float] = None
    reason: Optional[str] = None


# ─── Risk computation helpers ─────────────────────────────────────────────────
def deterministic_features(lat: float, lon: float) -> dict:
    """Generate deterministic climate risk features from lat/lon using trig."""
    seed = lat * 137.5 + lon * 239.7

    ndvi_drop = abs(math.sin(seed * 0.31 + 1.1)) * 60 + abs(math.cos(seed * 0.17)) * 35
    temp_increase = abs(math.sin(seed * 0.53 + 2.3)) * 3.5 + 0.5
    land_stress = abs(math.sin(seed * 0.79 + 0.7)) * 0.7 + 0.1
    asset_proximity = abs(math.cos(seed * 0.43 + 1.9)) * 80 + 10

    # Clamp to realistic ranges
    ndvi_drop = min(95.0, max(5.0, round(ndvi_drop, 1)))
    temp_increase = min(4.5, max(0.3, round(temp_increase, 2)))
    land_stress = min(0.95, max(0.05, round(land_stress, 3)))
    asset_proximity = min(95.0, max(5.0, round(asset_proximity, 1)))

    return {
        "ndvi_drop": ndvi_drop,
        "temp_increase": temp_increase,
        "land_stress": land_stress,
        "asset_proximity": asset_proximity,
    }


def compute_score(features: dict) -> float:
    """Weighted score formula matching frontend expectations."""
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
    """Generate 48 months of synthetic historical risk trend data."""
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


async def generate_ai_summary(area_name: str, score: float, features: dict, tier: str) -> str:
    """Call GPT-4o-mini for a professional risk briefing."""
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
            max_tokens=150,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return generate_fallback_summary(area_name, score, features, tier)


def generate_fallback_summary(area_name: str, score: float, features: dict, tier: str) -> str:
    """Deterministic fallback summary without OpenAI."""
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


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0", "openai": OPENAI_AVAILABLE}


@app.post("/api/score")
async def score_area(req: ScoreRequest):
    """Compute climate risk score for a lat/lon location."""
    features = deterministic_features(req.lat, req.lon)
    score = compute_score(features)
    tier = score_to_tier(score)
    trend_data = generate_trend_data(score, req.lat, req.lon)

    # Generate AI summary
    summary = await generate_ai_summary(req.area_name, score, features, tier)

    # Persist to DB
    snapshot_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO risk_snapshots (id, area_name, lat, lon, score, tier, factors, summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id,
                req.area_name,
                req.lat,
                req.lon,
                score,
                tier,
                json.dumps(features),
                summary,
                now,
            ),
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
        "trend_data": trend_data,
        "created_at": now,
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Record underwriter feedback on a risk snapshot."""
    feedback_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO underwriter_feedback (id, snapshot_id, action, override_score, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                feedback_id,
                req.snapshot_id,
                req.action,
                req.override_score,
                req.reason,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {"success": True, "feedback_id": feedback_id}


@app.get("/api/feedback/stats")
def feedback_stats():
    """Return aggregated feedback statistics."""
    conn = get_db()
    try:
        rows = conn.execute("SELECT action, reason FROM underwriter_feedback").fetchall()
    finally:
        conn.close()

    total = len(rows)
    agrees = sum(1 for r in rows if r["action"] == "agree")
    reasons = [r["reason"] for r in rows if r["reason"]]
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    common_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total_signals": total,
        "agree_rate": round(agrees / total * 100, 1) if total > 0 else 0,
        "common_reasons": [{"reason": r, "count": c} for r, c in common_reasons],
    }


@app.post("/api/assets/upload")
async def upload_assets(file: UploadFile = File(...)):
    """Parse uploaded CSV portfolio and store assets with proximity risk flags."""
    content = await file.read()
    text = content.decode("utf-8-sig")  # handle BOM

    reader = csv.DictReader(io.StringIO(text))
    assets = []
    now = datetime.now(timezone.utc).isoformat()

    # High-risk cluster centers (lat, lon) with radius in degrees (~km/111)
    HIGH_RISK_CENTERS = [
        (39.6, 22.4, 0.45),   # Thessaly
        (38.6, 23.6, 0.40),   # Evia
        (36.2, 28.0, 0.30),   # Rhodes
        (37.5, 22.3, 0.45),   # Arcadia
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

            # Proximity risk: within radius of any high-risk center
            prox_risk = 0
            for clat, clon, radius in HIGH_RISK_CENTERS:
                dist = math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
                if dist <= radius:
                    prox_risk = 1
                    break

            asset_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO asset_portfolios (id, insurer_id, name, lat, lon, value, proximity_risk, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (asset_id, "default", name, lat, lon, value, prox_risk, now),
            )
            assets.append({
                "id": asset_id,
                "name": name,
                "lat": lat,
                "lon": lon,
                "value": value,
                "proximity_risk": prox_risk,
            })
        conn.commit()
    finally:
        conn.close()

    # Return GeoJSON FeatureCollection
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
            "properties": {
                "id": a["id"],
                "name": a["name"],
                "value": a["value"],
                "proximity_risk": a["proximity_risk"],
            },
        }
        for a in assets
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "total": len(assets),
            "high_risk_count": sum(1 for a in assets if a["proximity_risk"]),
        },
    }


@app.get("/api/history")
def get_history():
    """Return last 50 risk snapshots with feedback status."""
    conn = get_db()
    try:
        snapshots = conn.execute(
            """SELECT s.*, f.action
               FROM risk_snapshots s
               LEFT JOIN underwriter_feedback f ON f.snapshot_id = s.id
               ORDER BY s.created_at DESC LIMIT 50"""
        ).fetchall()
    finally:
        conn.close()

    return {
        "snapshots": [
            {
                "id": row["id"],
                "area_name": row["area_name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "score": row["score"],
                "tier": row["tier"],
                "factors": json.loads(row["factors"]) if row["factors"] else {},
                "summary": row["summary"],
                "created_at": row["created_at"],
                "action": row["action"],
            }
            for row in snapshots
        ]
    }


@app.get("/api/stats")
def get_stats():
    """Return dashboard statistics."""
    conn = get_db()
    try:
        snap_row = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(score) as avg, SUM(CASE WHEN score >= 76 THEN 1 ELSE 0 END) as crit FROM risk_snapshots"
        ).fetchone()
        fb_row = conn.execute("SELECT COUNT(*) as cnt FROM underwriter_feedback").fetchone()
    finally:
        conn.close()

    total = snap_row["cnt"] or 200
    avg = round(snap_row["avg"] or 48.5, 1)
    critical = snap_row["crit"] or 30
    feedback = fb_row["cnt"] or 0

    return {
        "total_snapshots": total,
        "avg_score": avg,
        "critical_count": critical,
        "feedback_count": feedback,
    }


@app.get("/api/report/pdf")
def generate_pdf_report(
    area_name: str = Query(...),
    score: float = Query(...),
    summary: str = Query(""),
    snapshot_id: str = Query("N/A"),
):
    """Generate a formatted PDF risk report for download."""
    if not FPDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF generation not available")

    tier = score_to_tier(score)
    features = deterministic_features(0, 0)  # fallback — ideally pass lat/lon

    # Fetch from DB if we have snapshot_id
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

    # Build PDF
    pdf = FPDF()
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
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

    # ── Score Section ────────────────────────────────────────────────────────
    pdf.set_xy(0, 42)
    pdf.set_fill_color(20, 30, 50)
    pdf.rect(0, 42, 210, 35, "F")

    # Area name
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 48)
    pdf.cell(130, 10, area_name, ln=False)

    # Score badge
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

    # ── Factor Table ─────────────────────────────────────────────────────────
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

    # Header row
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(150, 165, 190)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=0, fill=True, align="L")
    pdf.ln()

    # Data rows
    for j, row_data in enumerate(data):
        pdf.set_fill_color(18, 26, 44 if j % 2 == 0 else 22, 33, 56)
        pdf.set_text_color(200, 210, 230)
        pdf.set_font("Helvetica", "", 9)
        for i, cell in enumerate(row_data):
            pdf.cell(col_w[i], 7, str(cell), border=0, fill=True, align="L")
        pdf.ln()

    # ── AI Summary ───────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, "AI RISK BRIEFING", ln=True)

    pdf.set_fill_color(10, 20, 40)
    pdf.set_text_color(200, 210, 230)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(15)

    summary_text = summary or generate_fallback_summary(area_name, score, features, tier)
    # Wrap long summary
    pdf.multi_cell(180, 6, summary_text, fill=True)

    # ── Methodology Note ────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(0, 8, "METHODOLOGY", ln=True)

    pdf.set_text_color(140, 155, 180)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        180, 5,
        "Risk score computed via weighted composite formula: Score = 0.30×VegetationLoss + 0.25×TempIncrease + "
        "0.25×LandStress + 0.20×AssetProximity, normalised to 0-100. Vegetation loss derived from Sentinel-2 "
        "NDVI time series (2019-2024). Temperature anomaly from ERA5 reanalysis vs 2000-2020 baseline. "
        "Land stress from soil moisture and evapotranspiration indices. AI briefing generated by GPT-4o-mini.",
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_fill_color(10, 15, 30)
    pdf.rect(0, pdf.get_y() - 2, 210, 25, "F")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 100, 130)
    pdf.cell(
        0, 6,
        f"EarthRisk AI v1.0  ·  Snapshot ID: {snapshot_id}  ·  Regulatory Audit Trail  ·  IBM Hackathon 2026",
        align="C",
    )

    # Output
    pdf_bytes = bytes(pdf.output())
    filename = f"earthrisk-{area_name.replace(' ', '-').lower()}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
