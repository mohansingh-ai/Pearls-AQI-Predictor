import os
import requests
import pandas as pd
import pytz
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LATITUDE", "24.8607")
LON = os.getenv("LONGITUDE", "67.0011")

def fetch_current_environmental_data(lat, lon, api_key):
    # Fetch comprehensive weather data
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    w_resp = requests.get(weather_url, timeout=30).json()

    # Fetch comprehensive air pollution data (gases + AQI)
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    aq_resp = requests.get(aqi_url, timeout=30).json()

    # Get local Karachi time
    karachi_tz = pytz.timezone("Asia/Karachi")
    current_time = pd.Timestamp.now(tz=karachi_tz).tz_localize(None).floor("h")

    # Extract component dictionaries safely
    components = aq_resp.get('list', [{}])[0].get('components', {})
    main_aqi = aq_resp.get('list', [{}])[0].get('main', {}).get('aqi', 1)
    
    w_main = w_resp.get('main', {})
    w_wind = w_resp.get('wind', {})

    # Construct the exact 21-column schema Hopsworks expects
    df_row = pd.DataFrame([{
        "city": "Karachi",
        "timestamp": current_time,
        "unix_time": int(current_time.timestamp() * 1000),
        "aqi_index": int(main_aqi),
        "co": float(components.get("co", 0.0)),
        "no": float(components.get("no", 0.0)),
        "no2": float(components.get("no2", 0.0)),
        "o3": float(components.get("o3", 0.0)),
        "so2": float(components.get("so2", 0.0)),
        "pm2_5": float(components.get("pm2_5", 0.0)),
        "pm10": float(components.get("pm10", 0.0)),
        "nh3": float(components.get("nh3", 0.0)),
        "temperature": float(w_main.get("temp", 0.0)),
        "humidity": int(w_main.get("humidity", 0)),
        "pressure": int(w_main.get("pressure", 0)),
        "wind_speed": float(w_wind.get("speed", 0.0)),
        "wind_deg": int(w_wind.get("deg", 0)),
        "hour": int(current_time.hour),
        "day": int(current_time.day),
        "day_of_week": int(current_time.dayofweek),
        "month": int(current_time.month)
    }])

    return df_row

def main():
    if not HOPSWORKS_API_KEY or not OPENWEATHER_API_KEY:
        raise ValueError("API Keys are not set in environment or repository secrets.")

    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    print("Fetching feature group 'aqi_weather_features'...")
    fg = fs.get_feature_group(name="aqi_weather_features", version=3)

    print("Fetching comprehensive environmental data for Karachi...")
    df_new = fetch_current_environmental_data(LAT, LON, OPENWEATHER_API_KEY)
    
    print("Inserting perfectly mapped row into Feature Group...")
    fg.insert(df_new, write_options={"wait_for_job": False})
    print("✅ Successfully inserted hourly features into Hopsworks!")

if __name__ == "__main__":
    main()