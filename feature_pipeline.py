import os
import requests
import pandas as pd
import pytz
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

# Load local .env variables if running locally
load_dotenv()

# Environment credentials
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
LAT = os.getenv("LATITUDE", "24.8607")
LON = os.getenv("LONGITUDE", "67.0011")

def fetch_current_environmental_data(lat, lon):
    """
    Fetches the latest real-time weather and air quality (PM2.5) metrics for Karachi.
    """
    # 1. Fetch current weather metrics (temperature, humidity, wind speed)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
    )
    w_resp = requests.get(weather_url, timeout=30).json()
    current_weather = w_resp.get("current", {})

    # 2. Fetch current air quality metrics (PM2.5)
    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&current=pm2_5&timezone=auto"
    )
    aq_resp = requests.get(aqi_url, timeout=30).json()
    current_aq = aq_resp.get("current", {})

    # Extract current values
    temp = float(current_weather.get("temperature_2m", 28.0))
    humidity = float(current_weather.get("relative_humidity_2m", 60.0))
    wind_speed = float(current_weather.get("wind_speed_10m", 15.0))
    pm2_5 = float(current_aq.get("pm2_5", 25.0))

    # Timestamp floored to current hour in UTC/local standard
    karachi_tz = pytz.timezone("Asia/Karachi")
    current_time = pd.Timestamp.now(tz=karachi_tz).tz_localize(None).floor("h")

    # Construct the feature row matching the Hopsworks Feature Group schema
    df_row = pd.DataFrame([{
        "city": "Karachi",                      # Primary key
        "timestamp": current_time,              # Primary key / Event time
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pm2_5": pm2_5
    }])

    return df_row

def main():
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is not set in environment or repository secrets.")

    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    print("Fetching feature group 'aqi_weather_features'...")
    fg = fs.get_feature_group(name="aqi_weather_features", version=3)

    print("Fetching current hourly environmental data for Karachi...")
    df_new = fetch_current_environmental_data(LAT, LON)
    print("New Data to Insert:")
    print(df_new)

    print("Inserting new row into Feature Group...")
    fg.insert(df_new, write_options={"wait_for_job": False})
    print("Successfully inserted hourly features into Hopsworks!")

if __name__ == "__main__":
    main()