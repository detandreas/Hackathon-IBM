import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
era5 = pd.read_csv('./data/era5_land_features.csv')
sentinel2 = pd.read_csv('./data/sentinel2_features.csv')

print(f"ERA5 shape: {era5.shape}")
print(f"Sentinel2 shape: {sentinel2.shape}")
print("\nERA5 columns:", era5.columns.tolist())
print("Sentinel2 columns:", sentinel2.columns.tolist())

# Convert time columns to datetime
era5['time'] = pd.to_datetime(era5['time'])
sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

# Sort by time
era5 = era5.sort_values('time').reset_index(drop=True)
sentinel2 = sentinel2.sort_values('datetime').reset_index(drop=True)

print("\nERA5 time range:", era5['time'].min(), "to", era5['time'].max())
print("Sentinel2 time range:", sentinel2['datetime'].min(), "to", sentinel2['datetime'].max())

# Merge datasets on month (since location/time overlap is limited)
era5['month_str'] = era5['time'].dt.strftime('%Y-%m')
sentinel2['month_str'] = sentinel2['month']

merged = pd.merge(era5, sentinel2, on='month_str', how='outer', suffixes=('_era5', '_sentinel2'))

print(f"\nMerged data shape: {merged.shape}")
print(f"Merged columns: {merged.columns.tolist()}")
print(f"Missing values:\n{merged.isnull().sum()}")

# Drop rows with too many missing values
#merged = merged.dropna(thresh=merged.shape[1] * 0.9)
merged = merged.dropna(thresh= 0.9)

print(f"\nAfter dropping sparse rows: {merged.shape}")

# Fill remaining NaN values with forward fill or mean
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    merged[col] = merged[col].fillna(merged[col].mean())

# Select features for the model (numeric features)
target_col = 'temp_2m_C'  # Predicting temperature
feature_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude', target_col]]

print(f"\nFeatures ({len(feature_cols)}): {feature_cols[:10]}...")
print(f"Target: {target_col}")

# Create lag features for time series
def create_lag_features(df, target_col, lags=[1, 2, 3, 7]):
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
    df_copy = df_copy.dropna()
    return df_copy

merged_with_lags = create_lag_features(merged, target_col, lags=[1, 2, 3, 7])

print(f"\nData with lag features: {merged_with_lags.shape}")

# Prepare features and target
X = merged_with_lags[feature_cols + [f'{target_col}_lag_{i}' for i in [1, 2, 3, 7]]]
y = merged_with_lags[target_col]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("\nTraining XGBoost model...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Feature importance
print("\n" + "="*50)
print("TOP 10 IMPORTANT FEATURES")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save model
import joblib
import pickle

joblib.dump(model, './xgb_model.pkl')
print("\nModel saved to ./xgb_model.pkl")

# Save scaler and metadata
with open('./scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('./feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Scaler and feature names saved")
print("\nYou can make predictions using:")
print("  model = joblib.load('./xgb_model.pkl')")
print("  scaler = pickle.load(open('./scaler.pkl', 'rb'))")
print("  predictions = model.predict(scaler.transform(X_new))")
