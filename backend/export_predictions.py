#!/usr/bin/env python3
"""
Export pre-computed ML predictions to database.
Run this locally once to populate predictions, then deploy lightweight API to Vercel.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ML.ml_engine import engine as ml_engine
from regions import REGIONS
from main import get_db, score_to_tier, deterministic_features

def export_predictions():
    """Pre-compute all regional predictions and store in database."""

    print("🚀 Initializing ML engine...")
    ml_engine.ensure_ready()

    if not ml_engine.ready:
        print("❌ ML engine failed to initialize")
        return False

    print(f"✓ ML engine ready. Exporting predictions for {len(REGIONS)} regions...\n")

    conn = get_db()
    conn.execute("DELETE FROM risk_snapshots")  # Clear old snapshots

    for region_key, region_data in REGIONS.items():
        try:
            lat = region_data["center_lat"]
            lon = region_data["center_lon"]
            region_name = region_data["name"]

            # Get ML predictions from cache
            features = ml_engine.compute_risk_features(region_name)
            if not features:
                features = deterministic_features(lat, lon)

            # Compute risk score
            from main import compute_score
            risk_score = compute_score(features)
            tier = score_to_tier(risk_score)

            # Generate summary
            summary_prompt = (
                f"Climate risk assessment for {region_name}: "
                f"Risk score {risk_score:.1f}/100 ({tier}). "
                f"Key drivers: vegetation loss {features.get('ndvi_drop', 0):.1f}%, "
                f"temperature increase {features.get('temp_increase', 0):.1f}°C. "
                f"Recommend immediate mitigation assessment."
            )

            snapshot_id = f"export-{region_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"

            conn.execute(
                """INSERT INTO risk_snapshots
                   (id, area_name, lat, lon, score, tier, factors, summary, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    region_name,
                    lat,
                    lon,
                    risk_score,
                    tier,
                    json.dumps(features),
                    summary_prompt,
                    datetime.now(timezone.utc).isoformat(),
                )
            )

            print(f"  ✓ {region_name:20} → Score: {risk_score:6.1f} ({tier})")

        except Exception as e:
            print(f"  ✗ {region_name:20} → ERROR: {e}")

    conn.commit()
    conn.close()

    print(f"\n✅ Successfully exported predictions to database")
    return True

if __name__ == "__main__":
    export_predictions()
