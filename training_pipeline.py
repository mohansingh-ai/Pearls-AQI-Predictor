import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def train_and_upload_multistep_models():
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    
    # 1. Fetch Historical Data from Feature Store
    print("📊 Fetching historical data...")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_weather_features", version=3)
    df = fg.read().sort_values("timestamp").reset_index(drop=True)
    
    # 2. Base Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    
    # 3. Create Future Targets (Y) for 72 Hours
    print("🛠️ Engineering 72-hour future targets...")
    for i in range(1, 73):
        df[f'target_pm25_{i}h'] = df['pm2_5'].shift(-i)
        
    df = df.dropna().reset_index(drop=True)
    
    # 4. Define Inputs (X) - We only need current weather and current PM2.5
    features = ['temperature', 'humidity', 'wind_speed', 'pm2_5', 'hour', 'month']
    X = df[features]
    
    # 5. Define Outputs (Y) for each model
    Y_day1 = df[[f'target_pm25_{i}h' for i in range(1, 25)]]  # Hours 1 to 24
    Y_day2 = df[[f'target_pm25_{i}h' for i in range(25, 49)]] # Hours 25 to 48
    Y_day3 = df[[f'target_pm25_{i}h' for i in range(49, 73)]] # Hours 49 to 72
    
    # 6. Train the 3 Models
    print("🧠 Training Model 1 (Day 1: 1-24h)...")
    model_day1 = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    model_day1.fit(X, Y_day1)
    
    print("🧠 Training Model 2 (Day 2: 25-48h)...")
    model_day2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    model_day2.fit(X, Y_day2)
    
    print("🧠 Training Model 3 (Day 3: 49-72h)...")
    model_day3 = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    model_day3.fit(X, Y_day3)
    
    # 7. Save Models Locally
    print("💾 Saving models locally...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_day1, "models/aqi_model_day1.pkl")
    joblib.dump(model_day2, "models/aqi_model_day2.pkl")
    joblib.dump(model_day3, "models/aqi_model_day3.pkl")
    
    # 8. Upload to Hopsworks Model Registry (Creating Version 3)
    print("☁️ Uploading to Hopsworks...")
    mr = project.get_model_registry()
    model_schema = mr.get_model("karachi_aqi_model_v3") # Fetching schema to reuse
    
    new_model = mr.python.create_model(
        name="karachi_aqi_multistep_models", 
        description="Direct Multi-Step Forecasting Models (Day 1, Day 2, Day 3)"
    )
    new_model.save("models") # Uploads the entire 'models' folder containing all 3 files
    
    print("✅ Models successfully trained and uploaded!")

if __name__ == "__main__":
    train_and_upload_multistep_models()