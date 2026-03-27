# Data Preparation & Train/Test Split Guide

## Overview
Three models were created with different data preparation approaches:

---

## MODEL 1: Satellite Indices (NDVI, NDBI, NDMI, BSI)
**File:** `train_combined_model.py`

### Raw Data
- **Sentinel2:** 53 observations (sparse, ~1 per month)
- **ERA5:** 1056 observations (daily)

### Data Preparation

```
Step 1: Load & Sort
├─ Sentinel2: Sort by datetime
└─ ERA5: Sort by datetime

Step 2: Create Lag Features for Sentinel2
├─ For each index (NDVI, NDBI, NDMI, BSI):
│  └─ Create lags: [1, 2, 3] observations back
└─ Drop first 3 rows (no lag data)
   Result: 50 rows with complete lag features

Step 3: Merge with ERA5 Climate Data
├─ Group ERA5 by month
├─ Average climate features per month
└─ Merge Sentinel2 with monthly climate by month key
   Result: 50 rows with satellite + climate data

Step 4: Create Feature Set
├─ Input features (23 total):
│  ├─ Spectral bands: B02, B03, B04, B08, B11, B12 (6)
│  ├─ Lag features: NDVI_lag_1-3, NDBI_lag_1-3, etc. (12)
│  └─ Monthly climate: temp, precip, soil_water, radiation (5)
└─ Target: NDVI, NDBI, NDMI, BSI (4 separate models)

Step 5: Train/Test Split
├─ 80/20 split (random, seed=42)
├─ Train: 40 rows
├─ Test: 10 rows
└─ Note: Very small dataset (only 53 satellite observations)
```

### Results
- **Train samples:** 40
- **Test samples:** 10
- **NDVI R²:** 0.7316 (moderate accuracy due to small dataset)
- **NDBI R²:** 0.5094 (high variance)
- **NDMI R²:** 0.5094 (high variance)
- **BSI R²:** 0.4767 (high variance)

### ⚠️ Limitations
- **Very small test set (10 samples)** - results can be unstable
- **Sparse satellite data** - only ~1 observation per month
- **May have temporal bias** - consecutive observations might be from same location

---

## MODEL 2: ERA5 Climate Features (5 variables)
**File:** `train_era5_model.py`

### Raw Data
- **ERA5:** 1056 daily observations (2015-2025)
- **Sentinel2:** 53 observations (satellite features)

### Data Preparation

```
Step 1: Load & Sort
├─ ERA5: Sort by time (daily)
└─ Sentinel2: Sort by datetime

Step 2: Create Lag Features for ERA5
├─ For each climate variable (temp, precip, soil_water, etc.):
│  └─ Create lags: [1, 2, 3, 7] days back
└─ Drop first 7 rows (no 7-day lag)
   Result: 1049 rows with complete lag features

Step 3: Merge with Satellite Data
├─ Match ERA5 dates with nearest Sentinel2 observation
├─ Forward fill missing satellite data (fill gaps)
└─ Result: 1049 rows with climate + satellite features

Step 4: Create Feature Set
├─ Input features (31 total):
│  ├─ Satellite indices: NDVI, NDBI, NDMI, BSI (4)
│  ├─ Spectral bands: B02, B03, B04, B08, B11, B12 (6)
│  ├─ Cloud cover: 1
│  └─ Lag features: temp_lag_1-7, precip_lag_1-7, etc. (20)
└─ Target: 5 climate variables (separate model each)

Step 5: Train/Test Split
├─ 80/20 split (random, seed=42)
├─ Train: 839 rows
├─ Test: 210 rows
└─ Split happens AFTER lag creation (no temporal leakage)
```

### Results
- **Train samples:** 839 ✅ (good size)
- **Test samples:** 210 ✅ (good size)
- **Temperature R²:** 0.8548 (good!)
- **Precipitation R²:** 0.6990 (moderate)
- **Soil Moisture R²:** 0.4139 (weak - hard to predict)
- **Radiation R²:** 0.8418 (good!)

### ✅ Strengths
- **Large training set (839 samples)** - stable results
- **Daily observations** - lots of temporal variation
- **Good test set (210 samples)** - reliable evaluation

### ⚠️ Limitations
- **Lag features use future satellite data** - satellite data is sparse, forward-filled
- **Seasonal patterns** - model may memorize seasonal cycles
- **Forward fill bias** - satellite features repeated for many days

---

## MODEL 3: Insurance Risk Score
**File:** `insurance_risk_score.py`

### Data Used
- **Current:** Latest row from each dataset
- **No train/test split** - uses golden standards instead

```
Input:
├─ Latest Sentinel2 observation
└─ Latest ERA5 observation

Processing:
├─ Normalize each metric (0-100 risk scale)
├─ Apply weights (NDVI, NDMI, Precip, Temp each 20%, etc.)
└─ Calculate weighted average

Output:
└─ Single risk score 0-100
```

---

## Data Quality Issues & Considerations

### 1. **Sentinel2 Data Sparsity**
```
Problem: Only 53 observations over 8 years
├─ ~1 observation per month
├─ Large gaps between observations
└─ Small sample size for training (10 test samples)

Solution Used:
└─ Forward fill satellite data to match daily ERA5 data

Better Solution Needed:
├─ Interpolate satellite data between observations
├─ Use spatial data from multiple locations
└─ Train on more years of satellite data
```

### 2. **Temporal Leakage Risk**
```
Current Approach:
├─ Create lag features FIRST
└─ THEN split train/test ✅ (Correct - no leakage)

Wrong Approach (NOT used):
├─ Split train/test FIRST
└─ Then create lags ❌ (Would leak future into past)
```

### 3. **Data Missingness**
```
ERA5 Data: Complete (no gaps)
Sentinel2 Data: Sparse
├─ Original: 53 observations
├─ Gap between observations: 7-200 days
└─ Filled with forward fill (last value repeated)

Issue: Satellite features don't change daily (unrealistic)
```

### 4. **Seasonal Patterns**
```
ERA5 covers 2015-2025 (10 years)
├─ Multiple seasonal cycles
├─ Model learns seasonal patterns
└─ Good for forecasting within similar season

Risk:
├─ May not generalize to new years/regions
└─ Climate change not captured (slow trend)
```

---

## Train/Test Split Summary

| Model | Raw Samples | After Lags | Train | Test | R² Best | R² Worst |
|-------|------------|-----------|-------|------|---------|----------|
| **Satellite Indices** | 53 | 50 | 40 (80%) | 10 (20%) | 0.73 | 0.48 |
| **ERA5 Climate** | 1056 | 1049 | 839 (80%) | 210 (20%) | 0.85 | 0.41 |
| **Insurance Risk** | Latest | Latest | N/A | N/A | N/A | N/A |

---

## Recommendations for Improvement

### 1. **Get More Satellite Data**
```
Current: 53 observations
Needed: 500+ observations
How:
├─ Use multiple satellite sources (Landsat, MODIS, etc.)
├─ Use different regions (not just one location)
└─ Cover different years and seasons
```

### 2. **Better Temporal Handling**
```
Instead of: Forward fill satellite data
Better:
├─ Interpolate between observations
├─ Use time-series models (LSTM, transformer)
└─ Treat satellite as sparse/irregular sampling
```

### 3. **Cross-Validation**
```
Current: Simple 80/20 split
Better:
├─ Time-series cross-validation (walk-forward)
├─ Stratified by season
└─ Multiple random seeds
```

### 4. **Regional Variation**
```
Current: Single location/region
Better:
├─ Train on multiple regions
├─ Test on holdout region (generalization test)
└─ Use spatial features (latitude, longitude)
```

---

## Code References

**Satellite Model Train/Test:**
- Line 61-70: Data merging
- Line 77-83: Feature selection
- Line 85-88: Train/test split

**ERA5 Model Train/Test:**
- Line 47-60: Lag feature creation
- Line 62-70: Merging with satellite
- Line 85-94: Train/test split

**Insurance Risk:**
- No train/test (threshold-based scoring)
