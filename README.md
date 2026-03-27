# EarthRisk AI

**Climate risk intelligence for the insurance industry**

> Real satellite data · XGBoost ML scoring · Explainable AI · Regulatory audit trail · Serverless deployment

Built for the IBM Hackathon 2026 — EarthRisk AI gives insurance underwriters a production-quality tool to assess, override, and export climate risk assessments for any location in Greece, powered by pre-computed XGBoost ML predictions and GPT-4o-mini AI briefings.

---

## Quick Start

### Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python export_predictions.py    # Pre-compute ML predictions (run once)
uvicorn main:app --reload --port 8000
# API at http://localhost:8000/docs
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

**Environment Variables** (backend/.env):
```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### Data Pipeline

```bash
cd backend
python data_download.py   # Download Sentinel-2 + ERA5-Land data (optional)
python db_setup.py        # Build SQLite database from CSVs
python export_predictions.py  # Pre-compute all regional risk scores
```

---

## Architecture

### 10 Greek Regions → 200 Sub-locations

```
Region (1 DB record)
├─ Pre-computed ML risk score (0-100)
├─ Risk factors (NDVI, temp, soil moisture, assets)
└─ 20 Sub-areas (local coordinates)
    └─ API generates 200 patches with deterministic variance

Example: Thessaly (25.3/100 LOW)
├─ Larissa Plains
├─ Karditsa Valley
├─ Trikala Basin
└─ ... (20 areas total)
```

### ML Pipeline

```
Data Sources
├─ Sentinel-2 (satellite vegetation indices: NDVI, NDBI, NDMI, BSI)
├─ ERA5-Land (climate: temp, precip, soil moisture, radiation)
└─ Portfolio (insurers' asset locations)

XGBoost Models
├─ ERA5 Model (daily climate predictions)
├─ Short-term Model (next satellite observation)
└─ Long-term Model (6-month to 5-year trends)

Risk Score = 0.30×VegLoss + 0.25×TempRise + 0.25×LandStress + 0.20×AssetProximity

Storage
├─ Local: ML engine caches predictions in memory
└─ Vercel: Pre-computed scores stored in SQLite database
```

### Vercel Deployment Strategy

**Problem:** XGBoost, rasterio, netcdf4, scikit-learn require C extensions that timeout on Vercel.

**Solution:** Pre-compute locally, deploy predictions.

```
Local Development                 Production (Vercel)
├─ Full ML stack                  ├─ Slim API (6 deps)
├─ ML models load                 ├─ ML unavailable
├─ XGBoost predictions run        ├─ Database used instead
└─ Scores cached                  └─ Same 200 patches served

/api/regions:
├─ If ML available → generate_all_patches()      (local)
└─ If ML unavailable → generate_patches_from_db() (Vercel)
```

**Dependencies (Vercel api/requirements.txt):**
```
fastapi>=0.115
uvicorn[standard]>=0.32
fpdf2>=2.8
pydantic>=2.10
python-dotenv>=1.0
openai>=1.14
```

---

## Project Structure

```
Hackathon-IBM/
├── backend/                          # FastAPI + ML pipeline
│   ├── main.py                       # API endpoints
│   ├── export_predictions.py         # Export ML predictions to DB (run once)
│   ├── regions.py                    # 10 Greek regions + 200 sub-areas
│   ├── requirements.txt              # Full stack (for local dev)
│   ├── data/
│   │   ├── climate_risk.db           # SQLite (10 regional scores)
│   │   ├── era5_land_features.csv    # Climate data (14MB)
│   │   └── sentinel2_features.csv    # Satellite data (86KB)
│   ├── models/
│   │   ├── era5_models.pkl
│   │   ├── short_term_models.pkl
│   │   ├── long_term_models.pkl
│   │   └── [feature scalers & metadata]
│   └── ML/
│       ├── ml_engine.py              # Inference engine (caches predictions)
│       ├── train_era5_model.py       # ERA5 training script
│       └── train_combined_model.py   # Satellite indices training
│
├── frontend/                         # React + Vite + TailwindCSS + deck.gl
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── AppPage.jsx           # Main app with map + priority queue
│   │   │   ├── GreeceMap.jsx         # deck.gl map visualization
│   │   │   ├── ScorePanel.jsx        # Risk details + PDF export
│   │   │   ├── HistoryDrawer.jsx     # Past assessments
│   │   │   ├── StatsBar.jsx          # Dashboard stats
│   │   │   ├── PortfolioUploader.jsx # Asset portfolio import
│   │   │   └── LandingPage.jsx       # Marketing landing page
│   │   └── data/
│   │       └── greeceData.js         # Fallback dummy data (if API offline)
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── api/                              # Vercel serverless handler
│   ├── index.py                      # WSGI wrapper for FastAPI
│   └── requirements.txt              # Slim dependencies only
│
├── vercel.json                       # Vercel deployment config
├── AGENTS.md                         # Vercel best practices
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regions` | Get 200 patches (uses ML cache locally, DB on Vercel) |
| `POST` | `/api/score` | Compute risk score for custom lat/lon |
| `GET` | `/api/health` | Health check (ML + OpenAI status) |
| `GET` | `/api/report/pdf` | Download formatted PDF report |
| `POST` | `/api/feedback` | Log underwriter agree/override |
| `GET` | `/api/feedback/stats` | Feedback aggregation stats |
| `GET` | `/api/history` | Last 50 risk assessments |
| `GET` | `/api/stats` | Dashboard statistics |
| `POST` | `/api/assets/upload` | Upload CSV portfolio (name, lat, lon, value) |
| `GET` | `/api/regions/{region_id}/trends` | Detailed time series for region |

---

## Database Schema

**risk_snapshots** (10 records, one per Greek region)
```sql
CREATE TABLE risk_snapshots (
    id TEXT PRIMARY KEY,              -- UUID
    area_name TEXT,                   -- Region name
    lat REAL, lon REAL,               -- Center coordinates
    score REAL,                       -- 0-100 risk score
    tier TEXT,                        -- CRITICAL|HIGH|MEDIUM|LOW
    factors TEXT,                     -- JSON: ndvi_drop, temp_increase, land_stress, asset_proximity
    summary TEXT,                     -- AI-generated briefing (from export_predictions.py)
    created_at TEXT                   -- ISO timestamp
);
```

Data populated by: `python export_predictions.py`

---

## Risk Tiers

| Tier | Score | Color | Meaning |
|------|-------|-------|---------|
| CRITICAL | 76-100 | 🔴 Red (#EF4444) | Severe stress; manual underwriting required |
| HIGH | 51-75 | 🟠 Amber (#F59E0B) | Elevated risk; apply loading factors |
| MEDIUM | 26-50 | 🟡 Yellow (#EAB308) | Moderate; monitor annually |
| LOW | 0-25 | 🟢 Teal (#00D4AA) | Stable; standard terms apply |

---

## Deployment

### Local to Vercel

1. **Pre-compute predictions locally:**
   ```bash
   python backend/export_predictions.py
   git add backend/data/climate_risk.db
   ```

2. **Push to GitHub:**
   ```bash
   git push origin main
   ```

3. **Deploy via Vercel CLI or Web:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import repository
   - Set environment: `OPENAI_API_KEY=sk-...`
   - Deploy

4. **Verify Vercel:**
   ```bash
   curl https://your-vercel-app.vercel.app/api/health
   # Should return: {"ml_models": false, "openai": true, ...}
   ```

The `/api/regions` endpoint automatically uses the database fallback (no ML needed on Vercel).

---

## Key Features

✅ **200 Hyper-Local Patches** — 20 sub-areas per Greek region
✅ **XGBoost Risk Scoring** — ML models trained on 10 years satellite data
✅ **Explainable Factors** — NDVI, temperature, soil moisture, asset proximity
✅ **Underwriter Override** — Log feedback with timestamp for audit trail
✅ **PDF Export** — Formatted reports with risk breakdown
✅ **Portfolio Integration** — Upload insurer assets; auto-compute proximity risk
✅ **AI Briefings** — GPT-4o-mini summaries for each assessment
✅ **Serverless Ready** — Pre-computed predictions for Vercel deployment
✅ **Regulatory Trail** — Every assessment stored as immutable snapshot

---

## Tech Stack

**Backend:**
- FastAPI (Python web framework)
- XGBoost (ML predictions)
- SQLite (risk snapshots)
- fpdf2 (PDF reports)
- OpenAI API (AI briefings)

**Frontend:**
- React 18 + Vite
- TailwindCSS (styling)
- deck.gl (map visualization)
- Recharts (trend charts)
- MapLibre GL (base map)

**Infrastructure:**
- Vercel (serverless deployment)
- GitHub (git hosting)

---

## Development Notes

### Adding a New Region

Edit `backend/regions.py`:
```python
REGIONS = {
    "new_region": {
        "name": "New Region",
        "center_lat": 39.0,
        "center_lon": 22.0,
        "areas": [
            {"name": "Sub-area 1", "lat": 39.1, "lon": 22.1},
            {"name": "Sub-area 2", "lat": 39.2, "lon": 22.2},
            # ... 20 total
        ]
    },
    ...
}
```

Then re-run: `python export_predictions.py`

### Testing

```bash
# Test backend
curl http://localhost:8000/api/regions
curl http://localhost:8000/api/health

# Test frontend
npm run dev  # at http://localhost:5173
```

---

## License

MIT — Built for IBM Hackathon 2026
