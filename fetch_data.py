import os

# --- FIX FOR OpenBLAS MEMORY ALLOCATION ERROR ---
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ------------------------------------------------

import time
import requests
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
LAT = os.getenv("LATITUDE", "24.8607")
LON = os.getenv("LONGITUDE", "67.0011")

def fetch_historical_feature_data(days_to_fetch=730, chunk_size_days=30):
    """Fetch 2 years of AQI and Weather data matching Hopsworks V3 schema and data types."""
    
    end_date = datetime.now() - timedelta(days=7)
    start_date = end_date - timedelta(days=days_to_fetch)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...\n")

    # 1. Fetch Open-Meteo Weather
    print("Fetching Open-Meteo Weather History (This might take a moment)...")
    meteo_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,wind_direction_10m"
    
    try:
        meteo_response = requests.get(meteo_url, timeout=30)
        if meteo_response.status_code != 200:
            print(f"Weather API Failed: {meteo_response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Weather API Network Error: {e}")
        return None
        
    meteo_json = meteo_response.json()['hourly']
    df_weather = pd.DataFrame({
        'timestamp': pd.to_datetime(meteo_json['time']),
        'temperature': meteo_json['temperature_2m'],
        'humidity': meteo_json['relative_humidity_2m'],
        'wind_speed': meteo_json['wind_speed_10m'],
        'pressure': meteo_json['surface_pressure'],
        'wind_deg': meteo_json['wind_direction_10m']
    })

    # 2. Fetch OpenWeather AQI in Chunks
    print(f"Fetching OpenWeather AQI in {chunk_size_days}-day chunks...")
    aqi_records = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size_days), end_date)
        
        unix_start = int(current_start.timestamp())
        unix_end = int(current_end.timestamp())
        
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={unix_start}&end={unix_end}&appid={OPENWEATHER_API_KEY}"
        
        success = False
        retries = 0
        max_retries = 3
        
        while not success and retries < max_retries:
            try:
                aqi_response = requests.get(aqi_url, timeout=15)
                
                if aqi_response.status_code == 200:
                    aqi_json = aqi_response.json().get('list', [])
                    for item in aqi_json:
                        aqi_records.append({
                            'timestamp': pd.to_datetime(item['dt'], unit='s'),
                            'aqi_index': item['main']['aqi'],
                            'co': item['components']['co'],
                            'no': item['components']['no'],
                            'no2': item['components']['no2'],
                            'o3': item['components']['o3'],
                            'so2': item['components']['so2'],
                            'pm2_5': item['components']['pm2_5'],
                            'pm10': item['components']['pm10'],
                            'nh3': item['components']['nh3']
                        })
                    print(f" -> Downloaded AQI up to {current_end.strftime('%Y-%m-%d')}")
                    success = True
                else:
                    print(f" -> Bad Status {aqi_response.status_code}. Retrying...")
                    retries += 1
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                print(f" -> Network timeout. Retrying ({retries+1}/{max_retries})...")
                retries += 1
                time.sleep(3)
                
        if not success:
            print(f" ⚠️ Failed chunk starting {current_start.strftime('%Y-%m-%d')}. Skipping.")

        current_start = current_end
        time.sleep(1) 

    df_aqi = pd.DataFrame(aqi_records)
    df_aqi['timestamp'] = df_aqi['timestamp'].dt.floor('h')

    # 3. Merge and Clean
    df_final = pd.merge(df_aqi, df_weather, on='timestamp', how='inner')
    df_final = df_final.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # 4. Generate the missing Primary Keys and Time Columns
    df_final['city'] = 'Karachi'
    df_final['unix_time'] = df_final['timestamp'].apply(lambda x: int(x.timestamp()))
    df_final['hour'] = df_final['timestamp'].dt.hour
    df_final['day'] = df_final['timestamp'].dt.day
    df_final['day_of_week'] = df_final['timestamp'].dt.dayofweek
    df_final['month'] = df_final['timestamp'].dt.month
    
    # --- NEW: TYPE CASTING FIX ---
    # Force these specific columns to be 64-bit integers ('bigint' in Hopsworks)
    cols_to_int = ['aqi_index', 'pressure', 'hour', 'day', 'day_of_week', 'month']
    for col in cols_to_int:
        df_final[col] = df_final[col].fillna(0).astype('int64')
    # -----------------------------
    
    print(f"\n✅ Successfully merged 2-year data!")
    print(f"Total Rows: {len(df_final)}")
    
    return df_final

if __name__ == "__main__":
    historical_df = fetch_historical_feature_data(days_to_fetch=730)
    
    if historical_df is not None:
        print("\n🚀 Logging into Hopsworks to upload data...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        
        print("📊 Retrieving Feature Group (Version 3)...")
        aqi_fg = fs.get_feature_group(name="aqi_weather_features", version=3)
        
        print("⬆️ Inserting 2-year dataset into Hopsworks (This might take a minute)...")
        aqi_fg.insert(historical_df)
        print("🎉 SUCCESS! Your 2-year data is now in Hopsworks.")
        