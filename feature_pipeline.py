import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LATITUDE", "24.8607")
LON = os.getenv("LONGITUDE", "67.0011")

def fetch_current_aqi_weather():
    # Fetch live weather from Open-Meteo
    meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,wind_direction_10m&forecast_days=1"
    meteo_resp = requests.get(meteo_url, timeout=15).json()['hourly']
    
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
    time_list = meteo_resp['time']
    
    try:
        idx = time_list.index(current_time)
    except ValueError:
        idx = 0 

    temp = meteo_resp['temperature_2m'][idx]
    humidity = meteo_resp['relative_humidity_2m'][idx]
    wind_speed = meteo_resp['wind_speed_10m'][idx]
    pressure = meteo_resp['surface_pressure'][idx]
    wind_deg = meteo_resp['wind_direction_10m'][idx]

    # Fetch live pollution from OpenWeatherMap
    ow_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    ow_resp = requests.get(ow_url, timeout=10).json().get('list', [{}])[0]
    components = ow_resp.get('components', {})
    
    pm2_5 = components.get('pm2_5', 0.0)
    co = components.get('co', 0.0)
    no = components.get('no', 0.0)
    no2 = components.get('no2', 0.0)
    o3 = components.get('o3', 0.0)
    so2 = components.get('so2', 0.0)
    pm10 = components.get('pm10', 0.0)
    nh3 = components.get('nh3', 0.0)

    data = {
        'timestamp': [pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:00:00'))],
        'temperature': [temp],
        'humidity': [humidity],
        'wind_speed': [wind_speed],
        'pressure': [pressure],
        'wind_deg': [wind_deg],
        'co': [co],
        'no': [no],
        'no2': [no2],
        'o3': [o3],
        'so2': [so2],
        'pm10': [pm10],
        'nh3': [nh3],
        'pm2_5': [pm2_5]
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    print("Fetching current hourly environmental data...")
    df_new = fetch_current_aqi_weather()
    
    print("Inserting new row into Feature Group...")
    fg = fs.get_feature_group(name="aqi_weather_features", version=3)
    fg.insert(df_new, write_options={"wait_for_job": False})
    print("Successfully updated Hopsworks Feature Store with current live hour data!")