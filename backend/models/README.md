# XGBoost Models for Climate & Risk Prediction

## Overview

This module provides trained XGBoost models for:
1. **ERA5 Climate Forecasting** - Predict temperature, precipitation, soil moisture, radiation
2. **Satellite Indices Prediction** - Predict NDVI, NDBI, NDMI, BSI
3. **Insurance Risk Scoring** - Calculate 0-100 risk score for areas

## Structure

```
models/
├── __init__.py          # Module exports
├── train.py             # Training pipelines
├── predict.py           # Prediction functions
├── risk_score.py        # Risk calculation
├── artifacts/           # Trained model files
│   ├── era5_models.pkl
│   ├── satellite_models.pkl
│   ├── scaler_era5.pkl
│   └── ...
└── README.md
```

## Usage

### Training Models

```python
from backend.models import train_era5_models, train_satellite_models

# Train ERA5 climate models
models, scaler, features, results = train_era5_models()

# Train satellite indices models
models, scaler, features, results = train_satellite_models()
```

### Making Predictions

```python
from backend.models import predict_era5, predict_satellite_indices

# Predict ERA5 features for next 3 days
predictions = predict_era5(days_ahead=3)

# Predict satellite indices
predictions = predict_satellite_indices()
```

### Risk Scoring

```python
from backend.models import calculate_risk_score, get_current_risk_score

# Calculate risk from custom metrics
metrics = {
    'NDVI': 0.35,
    'temp_2m_C': 24.2,
    'precip_mm_day': 0.13,
    'NDMI': 0.03,
    'BSI': 0.02,
    'NDBI': -0.03
}
overall_score, individual_scores = calculate_risk_score(metrics)

# Get current risk score
risk = get_current_risk_score()
print(f"Current risk: {risk['overall_score']:.1f}/100")
print(f"Tier: {risk['tier']}")
```

## API Endpoints

All predictions are also available via FastAPI endpoints:

```
GET  /api/v1/predict/era5?days_ahead=1      # ERA5 forecast
GET  /api/v1/predict/satellite               # Satellite prediction
GET  /api/v1/risk/current                    # Current risk score
POST /api/v1/risk/calculate                  # Custom risk calculation
GET  /api/v1/health                          # Health check
```

## Model Performance

### ERA5 Climate Models
| Feature | R² | RMSE | MAE |
|---------|----|----- |----|
| Temperature | 0.89 | 1.46°C | 0.97°C |
| Radiation | 0.87 | 1.24M J/m² | 0.91M J/m² |
| Skin Temp | 0.85 | 2.19°C | 1.50°C |
| Precipitation | 0.80 | 0.42 mm/day | 0.23 mm/day |
| Soil Moisture | 0.51 | 0.044 m³/m³ | 0.033 m³/m³ |

### Satellite Indices Models
| Index | R² | Test Samples |
|-------|----|----|
| NDVI | 0.73 | 10 |
| NDBI | 0.51 | 10 |
| NDMI | 0.51 | 10 |
| BSI | 0.48 | 10 |

**Note:** Satellite models use small test sets. Consider collecting more data for improvement.

## Risk Scoring System

**Golden Standards (0 = Safe, 100 = Risky):**

| Metric | Safe Zone | Risky Zone | Weight |
|--------|-----------|-----------|--------|
| NDVI (Vegetation) | >0.5 | <0.3 | 20% |
| NDMI (Moisture) | 0.1-0.3 | <-0.5 or >0.5 | 20% |
| Precipitation | 10-30 mm/day | <0.1 or >100 mm/day | 20% |
| Temperature | 15-25°C | <0°C or >45°C | 20% |
| BSI (Bare Soil) | <0.1 | >0.3 | 10% |
| NDBI (Urban) | <0 | >0.2 | 10% |

## Files

- `train.py` - Model training code (~400 lines)
- `predict.py` - Prediction functions (~200 lines)
- `risk_score.py` - Risk scoring logic (~200 lines)
- `artifacts/` - Trained model pickle files (not in git)

## Requirements

See `requirements.txt` for dependencies. Key packages:
- xgboost >= 2.0
- scikit-learn >= 1.3
- pandas >= 2.2
- numpy >= 1.26
- joblib >= 1.3

## Data Location

Training data is loaded from `backend/data/`:
- `era5_land_features.csv` - 1,056 daily climate records (2015-2025)
- `sentinel2_features.csv` - 53 satellite observations (2017-2025)

## Notes

1. Models are trained on merged ERA5 + Sentinel2 data
2. Lag features created before train/test split (no temporal leakage)
3. StandardScaler used for feature normalization
4. Risk scores are location-independent (threshold-based)
5. Predictions include uncertainty estimates when available

## Future Improvements

- [ ] Add uncertainty quantification (confidence intervals)
- [ ] Collect more satellite data (currently only 53 obs)
- [ ] Add regional variations to risk thresholds
- [ ] Implement cross-validation for satellite models
- [ ] Add extreme event detection
- [ ] Integrate with IPCC climate scenarios
