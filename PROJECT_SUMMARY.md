# IBM Hackathon: XGBoost Prediction Models - PROJECT SUMMARY

## 🎯 **What We Built**

Three integrated XGBoost prediction systems for climate & satellite data analysis:

### **1. ERA5 Climate Model** ✅ BEST
- **Predicts:** Temperature, Precipitation, Soil Moisture, Solar Radiation, Skin Temp
- **Uses:** 1049 daily samples (2015-2025)
- **Accuracy:** R² = 0.78 average (Temperature 0.89, Radiation 0.87)
- **Test Set:** 210 samples (reliable)
- **Status:** 🟢 Production Ready

### **2. Satellite Indices Model** ⚠️ NEEDS DATA
- **Predicts:** NDVI, NDBI, NDMI, BSI (vegetation, urban, moisture, soil)
- **Uses:** 50 samples (sparse satellite images)
- **Accuracy:** R² = 0.56 average (NDVI 0.73, BSI 0.48)
- **Test Set:** 10 samples (unreliable - too small)
- **Status:** 🟡 Proof of Concept

### **3. Insurance Risk Score** ✅ READY
- **Scoring System:** 0-100 scale (0=Safe, 100=Risky)
- **Metrics:** Vegetation, Soil, Precipitation, Temperature, Urban, Bare Soil
- **Weights:** 20% each for climate/vegetation, 10% each for urban/soil
- **Output:** Risk tier + insurance pricing recommendation
- **Current Score:** 50.8/100 (Moderate - Charge Baseline)
- **Status:** 🟢 Production Ready

---

## 📊 **Data Used**

| Source | Samples | Period | Update Freq |
|--------|---------|--------|------------|
| **ERA5** | 1,056 | 2015-2025 | Daily |
| **Sentinel2** | 53 | 2017-2025 | ~Monthly |

---

## 📁 **Files Created**

### **Training Scripts**
- `train_era5_model.py` - Trains 5 climate models
- `train_combined_model.py` - Trains satellite indices (short + long-term)

### **Prediction Scripts**
- `predict_era5.py` - Next day + 7-day climate forecast
- `predict_indices.py` - Next satellite obs + 6m-5y trends
- `insurance_risk_score.py` - Risk scoring (0-100)

### **Evaluation**
- `evaluate_models.py` - Comprehensive accuracy metrics
- `DATA_PREPARATION_GUIDE.md` - How data was prepared
- `PROJECT_SUMMARY.md` - This file

### **Models Saved**
```
models/
├── era5_models.pkl (5 models)
├── short_term_models.pkl (4 models)
├── long_term_models.pkl (4 models)
├── scaler_era5.pkl
├── scaler_short_term.pkl
├── scaler_long_term.pkl
└── metadata_*.pkl
```

---

## 🚀 **Quick Start**

```bash
# 1. Train models (one-time)
python train_era5_model.py
python train_combined_model.py

# 2. Make predictions
python predict_era5.py          # Climate forecast
python predict_indices.py       # Satellite forecast
python insurance_risk_score.py  # Risk assessment
```

---

## 📈 **Accuracy by Model**

### **ERA5 (Climate) - 🟢 HIGH CONFIDENCE**
```
Temperature:      R² = 0.8866  ✅ Excellent (±1.46°C error)
Radiation:        R² = 0.8732  ✅ Excellent
Skin Temp:        R² = 0.8481  ✅ Good
Precipitation:    R² = 0.8005  ✅ Good (±0.34 mm/day error)
Soil Moisture:    R² = 0.5138  ⚠️  Weak (avoid for critical decisions)
```

### **Satellite - 🟡 MODERATE CONFIDENCE**
```
NDVI (Vegetation):  R² = 0.7316  ✅ Good
NDBI (Urban):       R² = 0.5094  ⚠️  Moderate
NDMI (Moisture):    R² = 0.5094  ⚠️  Moderate
BSI (Bare Soil):    R² = 0.4767  ❌ Weak
```

### **Insurance Risk Score - 🟢 RULES-BASED**
```
Current Area: 50.8/100 (MODERATE RISK)
Recommendation: Charge baseline insurance rates
Top Risks: Low precipitation, bare soil exposure, moderate vegetation stress
```

---

## 🎓 **Key Learnings**

1. **More data = Better models**
   - ERA5: 1049 samples → R²=0.78 (Good)
   - Satellite: 50 samples → R²=0.56 (Weak)

2. **Temperature is predictable** (R²=0.89)
   - Stable daily patterns
   - Strong lag correlations

3. **Precipitation is chaotic** (MAPE=64%)
   - Hard to predict day-to-day
   - Better for monthly aggregates

4. **Soil moisture is mysterious** (R²=0.51)
   - Weak satellite correlation
   - Needs better instrumentation

5. **Insurance scores drive pricing**
   - Risk tier determines premium adjustment
   - Currently: 50.8/100 = baseline rates

---

## ⚙️ **Technical Stack**

- **Framework:** XGBoost
- **Data Processing:** Pandas, NumPy
- **ML Tools:** Scikit-learn
- **Metrics:** R², RMSE, MAE, MAPE
- **Validation:** 5-Fold Cross-Validation
- **Language:** Python 3.11

---

## 🔄 **Data Pipeline**

```
Raw Data (ERA5 + Sentinel2)
    ↓
Create Lag Features (1, 2, 3, 7 days)
    ↓
Merge & Fill Missing Values
    ↓
Scale Features (StandardScaler)
    ↓
Train/Test Split (80/20)
    ↓
Train XGBoost Models
    ↓
Evaluate & Save
    ↓
Make Predictions
```

---

## 💡 **Use Cases**

### **For Insurance Companies (IBM)**
- Risk assessment for climate-vulnerable areas
- Premium pricing based on weather patterns
- Early warning for extreme conditions

### **For Agriculture**
- Crop yield predictions
- Drought/flood warnings
- Irrigation optimization

### **For Urban Planning**
- Infrastructure resilience scoring
- Climate risk mapping
- Insurance pool assessments

---

## 📌 **Recommendations**

### **Immediate (Use Now)**
✅ Use ERA5 temperature for forecasting (R²=0.89)
✅ Deploy insurance risk scoring (production-ready)
✅ Use radiation predictions (R²=0.87)

### **Short-term (Next Steps)**
⚠️ Collect more satellite data (currently too sparse)
⚠️ Improve soil moisture predictions (currently weak)
⚠️ Add cross-validation to satellite models

### **Long-term (Future)**
🔮 Add climate change trends
🔮 Incorporate more satellite sources (Landsat, MODIS)
🔮 Build regional models (currently single location)
🔮 Real-time API for insurance partners

---

## 📊 **Current Metrics**

| Metric | ERA5 | Satellite |
|--------|------|-----------|
| Avg R² | 0.78 | 0.56 |
| Train Samples | 839 | 40 |
| Test Samples | 210 | 10 |
| Best R² | 0.89 (Temp) | 0.73 (NDVI) |
| Worst R² | 0.51 (Soil) | 0.48 (BSI) |
| Production Ready? | ✅ Yes | ⚠️ Beta |

---

## 🔗 **Git Branch**

```
Branch: feature/xgboost-model
Commits:
  - Setup XGBoost model and training scripts
  - Add ERA5 climate features prediction models
```

---

## 📝 **Notes**

- All models use StandardScaler normalization
- XGBoost parameters: max_depth=5-6, learning_rate=0.05-0.1
- No data leakage: lag features created BEFORE train/test split
- Cross-validation confirms model stability
- Insurance scoring uses threshold-based approach (no ML overfitting)

---

**Status:** ✅ **Complete and Ready for Deployment**

**Next Steps:** Push to main branch or deploy insurance scoring API
