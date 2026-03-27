# EarthRisk AI

**Climate risk intelligence for the insurance industry**

> Real satellite data · XGBoost ML scoring · Explainable AI · Regulatory audit trail

Built for the IBM Hackathon 2026 — EarthRisk AI gives insurance underwriters a production-quality tool to assess, override, and export climate risk assessments for any location in Greece, powered by Sentinel-2 satellite data and GPT-4o-mini AI briefings.

---

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Environment Variables

Create a `.env` file in `/backend/`:

```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### Data Pipeline

```bash
cd backend
python data_download.py   # Download Sentinel-2 + ERA5-Land data
python db_setup.py        # Build SQLite database from CSVs
```

---

## Project Structure

```
Hackathon-IBM/
├── backend/                   # FastAPI + data pipeline
│   ├── main.py                # API endpoints (FastAPI)
│   ├── data_download.py       # Sentinel-2 & ERA5-Land downloader
│   ├── db_setup.py            # SQLite schema + CSV loader
│   ├── requirements.txt       # All Python dependencies
│   └── data/                  # Generated data (gitignored)
│       ├── climate_risk.db
│       ├── sentinel2_features.csv
│       └── era5_land_features.csv
├── frontend/                  # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── LandingPage.jsx
│   │   │   ├── AppPage.jsx
│   │   │   ├── GreeceMap.jsx
│   │   │   ├── ScorePanel.jsx
│   │   │   ├── StatsBar.jsx
│   │   │   ├── PortfolioUploader.jsx
│   │   │   └── HistoryDrawer.jsx
│   │   └── data/
│   │       └── greeceData.js
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── vercel.json                # Vercel deployment config
├── .gitignore
└── README.md
```

---

## The 4 Data Moats

### 1. Satellite-Derived Risk Scores
- Sentinel-2 NDVI time series (2015-2025, April-September)
- ERA5-Land temperature, precipitation, soil moisture reanalysis
- Composite score: `0.30×VegLoss + 0.25×TempRise + 0.25×LandStress + 0.20×AssetProximity`

### 2. Underwriter Feedback Loop
- Every AGREE/OVERRIDE logged to SQLite with timestamp
- Feedback statistics endpoint tracks agreement rates
- Override reasons build ground-truth training corpus

### 3. Regulatory Audit Trail
- Every risk assessment stored as immutable snapshot with UUID
- Full factor breakdown preserved at assessment time
- PDF exports include snapshot ID for compliance traceability

### 4. Portfolio Integration
- Insurers upload CSV portfolios (name, lat, lon, value)
- Proximity risk auto-computed against known HIGH/CRITICAL zones

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/score` | Compute risk score for lat/lon |
| `POST` | `/api/feedback` | Submit underwriter agree/override |
| `GET`  | `/api/feedback/stats` | Feedback aggregation stats |
| `POST` | `/api/assets/upload` | Upload CSV portfolio |
| `GET`  | `/api/history` | Last 50 risk snapshots |
| `GET`  | `/api/stats` | Dashboard statistics |
| `GET`  | `/api/report/pdf` | Download formatted PDF report |
| `GET`  | `/api/health` | Health check |

---

## Deploy to Vercel

1. Push to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import the repo
4. Set environment variable: `OPENAI_API_KEY` = your OpenAI key
5. Click **Deploy**

---

## Design System

- **Background**: Deep Navy `#0A0F1E`
- **Primary**: Electric Teal `#00D4AA`
- **High Risk**: Amber `#F59E0B`
- **Critical**: Coral `#EF4444`
- **Typography**: Inter (Google Fonts)

---

## License

MIT — built for IBM Hackathon 2026
