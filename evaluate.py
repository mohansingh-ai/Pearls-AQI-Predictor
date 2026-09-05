import os
import numpy as np
import pandas as pd
import hopsworks
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def evaluate_forecast_horizons():
    print("🔐 Logging into Hopsworks to fetch test data and model...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    
    # 1. Download Model Registry version 2
    mr = project.get_model_registry()
    model_obj = mr.get_model("karachi_aqi_model_v3", version=2) 
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
    
    # 2. Retrieve Feature Store Data
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_weather_features", version=3)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Replicate feature engineering
    print("🛠️ Engineering lags and rolling metrics...")
    df['pm2_5_lag_1h'] = df['pm2_5'].shift(1)
    df['pm2_5_lag_2h'] = df['pm2_5'].shift(2)
    df['pm2_5_lag_3h'] = df['pm2_5'].shift(3)
    df['pm2_5_lag_24h'] = df['pm2_5'].shift(24)
    
    df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6).mean()
    df['temp_rolling_6h'] = df['temperature'].shift(1).rolling(window=6).mean()
    df['wind_rolling_6h'] = df['wind_speed'].shift(1).rolling(window=6).mean()
    
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/23.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
    
    df = df.dropna().reset_index(drop=True)
    
    # 4. Chronological test split (20% test set)
    train_size = int(len(df) * 0.8)
    df_test = df.iloc[train_size:].copy().reset_index(drop=True)
    
    expected_features = model.feature_names_in_
    
    # 5. Fast Bulk Prediction (Vectorized instead of slow row-by-row loops)
    print("⚡ Running fast vectorized model evaluation...")
    X_test = df_test[expected_features]
    predictions = model.predict(X_test)
    
    df_test['predicted_pm25'] = predictions
    df_test['hours_elapsed'] = np.arange(len(df_test)) % 72
    
    # 6. Compute Metrics for 2-Day (48h) vs 3-Day (72h) Horizons
    print("\n" + "="*50)
    print("📊 MODEL ACCURACY REPORT ACROSS FORECAST HORIZONS")
    print("="*50)
    
    for horizon_name, max_hours in [("2-Day Forecast (48 Hours)", 48), ("3-Day Forecast (72 Hours)", 72)]:
        subset = df_test[df_test['hours_elapsed'] <= max_hours]
        
        y_true = subset['pm2_5']
        y_pred = subset['predicted_pm25']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📌 {horizon_name}:")
        print(f"   -> RMSE : {rmse:.2f}")
        print(f"   -> MAE  : {mae:.2f}")
        print(f"   -> R²   : {r2:.2f}")
        
    print("="*50)

if __name__ == "__main__":
    evaluate_forecast_horizons()