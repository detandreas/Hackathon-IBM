"""
Script to make predictions with the trained XGBoost model
"""
import pandas as pd
import numpy as np
import joblib
import pickle

# Load model and preprocessing objects
print("Loading model...")
model = joblib.load('./xgb_model.pkl')
scaler = pickle.load(open('./scaler.pkl', 'rb'))
feature_names = pickle.load(open('./feature_names.pkl', 'rb'))

print(f"Model loaded with {len(feature_names)} features")
print(f"Features: {feature_names}\n")

def predict_temperature(input_data):
    """
    Make predictions on new data

    Args:
        input_data: pd.DataFrame with the same features as training data

    Returns:
        predictions: array of predicted temperatures
    """
    # Ensure features are in the correct order
    X = input_data[feature_names]

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    return predictions

# Example: Make predictions on the test set
print("Example prediction on recent data:\n")

# Load the original data
era5 = pd.read_csv('./data/era5_land_features.csv')
sentinel2 = pd.read_csv('./data/sentinel2_features.csv')

# Convert time columns to datetime
era5['time'] = pd.to_datetime(era5['time'])
sentinel2['datetime'] = pd.to_datetime(sentinel2['datetime'])

# Get the most recent data
recent_era5 = era5.tail(10).copy()
recent_era5['month_str'] = recent_era5['time'].dt.strftime('%Y-%m')

# Create lag features for the recent data
recent_era5 = recent_era5.sort_values('time').reset_index(drop=True)
for lag in [1, 2, 3, 7]:
    recent_era5[f'temp_2m_C_lag_{lag}'] = recent_era5['temp_2m_C'].shift(lag)

# Remove rows with NaN lag features
recent_era5 = recent_era5.dropna()

if len(recent_era5) > 0:
    # Get the sentinel2 features (most recent)
    recent_sentinel2 = sentinel2.tail(1).copy()
    recent_sentinel2['month_str'] = recent_sentinel2['month']

    # Merge
    merged_recent = pd.merge(recent_era5, recent_sentinel2, on='month_str', how='left', suffixes=('', '_s2'))

    # Select only the features we need
    X_recent = merged_recent[feature_names].fillna(merged_recent[feature_names].mean())

    if len(X_recent) > 0:
        predictions = predict_temperature(X_recent)

        print(f"Actual temperatures: {merged_recent['temp_2m_C'].values}")
        print(f"Predicted temperatures: {predictions}")
        print(f"Differences (MAE): {np.abs(merged_recent['temp_2m_C'].values - predictions).mean():.4f}°C")
else:
    print("Not enough data with lag features for prediction")

print("\n✅ Ready to make predictions on new data!")
