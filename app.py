
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import joblib
import logging
from datetime import datetime
import threading
import time
from prophet import Prophet

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.prediction_service import Prediction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class Config:
    OPENAQ_API_KEY = "YOUR_OPENAQ_API_KEY"  # Get from https://explore.openaq.org
    WEATHERSTACK_API_KEY = "9b482a8fc8114edd53b5a30d85a80efe"  
    VISUAL_CROSSING_API_KEY = "QLZZG8H6JJJQSBEN7XF7BW4AK"  

    # City configurations with coordinates
    CITIES = {
        "Delhi": {"lat": 28.7041, "lon": 77.1025, "openaq_id": 2178},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777, "openaq_id": 1641},
        "Chennai": {"lat": 13.0827, "lon": 80.2707, "openaq_id": 5346},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639, "openaq_id": 7616},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "openaq_id": 2170}
    }

class RealTimeDataCollector:
    def __init__(self):
        self.latest_data = {}
        self.data_history = {city: [] for city in Config.CITIES.keys()}

        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = Prediction()

        # Loading RandomForest model
        try:
            self.rf_model = joblib.load('models/rf_model.pkl')
            logger.info("RandomForest model loaded successfully")
        except:
            logger.warning("RandomForest model not found. Training new model...")
            self.rf_model = None

        # Prophet models
        self.prophet_models = {}

    def fetch_openaq_data(self, city_name):
        try:
            city_info = Config.CITIES.get(city_name)
            if not city_info:
                return None

            url = "https://api.openaq.org/v2/latest"
            headers = {"X-API-Key": Config.OPENAQ_API_KEY}
            params = {
                "city": city_name,
                "country": "IN",
                "limit": 10
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # Parse the response and extract pollutant data
                    measurements = data['results'][0]['measurements']
                    pollutant_data = {}

                    for measurement in measurements:
                        param = measurement.get('parameter')
                        value = measurement.get('value')
                        if param and value is not None:
                            pollutant_data[param] = value

                    return pollutant_data

            return None
        except Exception as e:
            logger.error(f"Error fetching OpenAQ data for {city_name}: {e}")
            return None

    def fetch_weather_data(self, city_name):
        try:
            city_info = Config.CITIES.get(city_name)
            if not city_info:
                return None

            # Visual Crossing API call
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city_info['lat']},{city_info['lon']}"
            params = {
                "unitGroup": "metric",
                "key": Config.VISUAL_CROSSING_API_KEY,
                "include": "current"
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                current_conditions = data.get('currentConditions', {})

                weather_data = {
                    'Max_Temperature_C': current_conditions.get('tempmax', current_conditions.get('temp', 25)),
                    'Min_Temperature_C': current_conditions.get('tempmin', current_conditions.get('temp', 25)),
                    'Avg_Temperature_C': current_conditions.get('temp', 25),
                    'Humidity_Percent': current_conditions.get('humidity', 60),
                    'Rainfall_mm': current_conditions.get('precip', 0),
                    'Wind_Speed_kmh': current_conditions.get('windspeed', 5),
                    'Atmospheric_Pressure_hPa': current_conditions.get('pressure', 1013),
                    'Visibility_km': current_conditions.get('visibility', 10)
                }

                return weather_data

            return None
        except Exception as e:
            logger.error(f"Error fetching weather data for {city_name}: {e}")
            return None

    def combine_and_predict_aqi(self, city_name, pollutant_data, weather_data):
        try:
            if not self.rf_model:
                return None

            # Prepare features for prediction
            current_time = datetime.now()

            combined_data = {
                'PM2.5': pollutant_data.get('pm25', pollutant_data.get('PM2.5', 50)),
                'PM10': pollutant_data.get('pm10', pollutant_data.get('PM10', 75)),
                'NO2': pollutant_data.get('no2', pollutant_data.get('NO2', 40)),
                'SO2': pollutant_data.get('so2', pollutant_data.get('SO2', 15)),
                'CO': pollutant_data.get('co', pollutant_data.get('CO', 1.0)),
                'O3': pollutant_data.get('o3', pollutant_data.get('O3', 30)),
                'NH3': pollutant_data.get('nh3', pollutant_data.get('NH3', 20)),
                **weather_data,
                'year': current_time.year,
                'month': current_time.month,
                'day': current_time.day
            }

            features_df = pd.DataFrame([combined_data])
            features_df = features_df[self.predictor.independent]  


            predicted_aqi = self.rf_model.predict(features_df)[0]
            predicted_aqi = max(0, round(predicted_aqi, 1))  

            # Add AQI to combined data
            combined_data['AQI'] = predicted_aqi
            combined_data['timestamp'] = current_time.isoformat()
            combined_data['city'] = city_name

            return combined_data

        except Exception as e:
            logger.error(f"Error predicting AQI for {city_name}: {e}")
            return None


    #creating prophet model to analyse tremd 
    def create_prophet_trends(self, city_name, parameter, days_future=30):          #parameter refers to name of pollutant or weather feature
        try:
            city_history = self.data_history.get(city_name, [])

            if len(city_history) < 10:  # Need minimum data points
                logger.warning(f"Insufficient historical data for {city_name}")
                return None

            prophet_data = []
            for record in city_history[-100:]:  # Use last 100 records
                if parameter in record:
                    prophet_data.append({
                        'ds': pd.to_datetime(record['timestamp']),
                        'y': record[parameter]
                    })

            if len(prophet_data) < 10:
                return None

            df_prophet = pd.DataFrame(prophet_data)

            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.8
            )

            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=days_future)
            forecast = model.predict(future)

            # Extract trends
            trends = {
                'historical': df_prophet.tail(30).to_dict('records'),
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_future).to_dict('records'),
                'components': {
                    'trend': forecast['trend'].tail(days_future).tolist(),
                    'weekly': forecast['weekly'].tail(days_future).tolist() if 'weekly' in forecast.columns else [],
                    'yearly': forecast['yearly'].tail(days_future).tolist() if 'yearly' in forecast.columns else []
                }
            }

            return trends

        except Exception as e:
            logger.error(f"Error creating Prophet trends for {city_name}, {parameter}: {e}")
            return None

    #real time data for all cities
    def collect_realtime_data(self):
        collected_data = {}

        for city_name in Config.CITIES.keys():
            logger.info(f"Collecting data for {city_name}")

            pollutant_data = self.fetch_openaq_data(city_name)
            if not pollutant_data:
                # Use fallback/mock data if API fails
                pollutant_data = self.generate_mock_pollutant_data(city_name)

            weather_data = self.fetch_weather_data(city_name)
            if not weather_data:
                weather_data = self.generate_mock_weather_data(city_name)

            combined_data = self.combine_and_predict_aqi(city_name, pollutant_data, weather_data)

            if combined_data:
                # Store latest data
                self.latest_data[city_name] = combined_data

                # Add to history
                self.data_history[city_name].append(combined_data)

                # Keep only last 1000 records per city
                if len(self.data_history[city_name]) > 1000:
                    self.data_history[city_name] = self.data_history[city_name][-1000:]

                collected_data[city_name] = combined_data

        return collected_data

    #mock data used in future if API calls fails
    def generate_mock_pollutant_data(self, city_name):
        base_values = {
            "Delhi": {"pm25": 80, "pm10": 120, "no2": 45, "so2": 15, "co": 2.0, "o3": 30},
            "Mumbai": {"pm25": 60, "pm10": 90, "no2": 38, "so2": 12, "co": 1.5, "o3": 35},
            "Chennai": {"pm25": 40, "pm10": 70, "no2": 28, "so2": 8, "co": 1.0, "o3": 40},
            "Kolkata": {"pm25": 70, "pm10": 110, "no2": 42, "so2": 18, "co": 1.8, "o3": 25},
            "Bengaluru": {"pm25": 45, "pm10": 75, "no2": 30, "so2": 10, "co": 1.2, "o3": 38}
        }

        base = base_values.get(city_name, base_values["Delhi"])
        mock_data = {}

        for param, value in base.items():
            variation = np.random.uniform(0.8, 1.2)  # ±20% variation
            mock_data[param.upper()] = round(value * variation, 1)

        mock_data['NH3'] = round(np.random.uniform(15, 30), 1)

        return mock_data

    def generate_mock_weather_data(self, city_name):
        base_values = {
            "Delhi": {"temp": 30, "humidity": 60, "pressure": 1010},
            "Mumbai": {"temp": 32, "humidity": 80, "pressure": 1012},
            "Chennai": {"temp": 34, "humidity": 85, "pressure": 1011},
            "Kolkata": {"temp": 31, "humidity": 85, "pressure": 1009},
            "Bengaluru": {"temp": 26, "humidity": 70, "pressure": 1013}
        }

        base = base_values.get(city_name, base_values["Delhi"])

        return {
            'Max_Temperature_C': base["temp"] + np.random.uniform(2, 5),
            'Min_Temperature_C': base["temp"] - np.random.uniform(3, 8),
            'Avg_Temperature_C': base["temp"] + np.random.uniform(-2, 2),
            'Humidity_Percent': base["humidity"] + np.random.uniform(-10, 10),
            'Rainfall_mm': np.random.uniform(0, 5),
            'Wind_Speed_kmh': np.random.uniform(3, 15),
            'Atmospheric_Pressure_hPa': base["pressure"] + np.random.uniform(-5, 5),
            'Visibility_km': np.random.uniform(3, 15)
        }



# Initialize data collector
data_collector = RealTimeDataCollector()



# background thread to collect data
def background_data_collection():
    while True:
        try:
            data_collector.collect_realtime_data()
            logger.info("Background data collection completed")
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Error in background data collection: {e}")
            time.sleep(60)  # Wait 1 minute on error

# Start background thread
background_thread = threading.Thread(target=background_data_collection, daemon=True)
background_thread.start()

# API Routes
@app.route('/realtime/all', methods=['GET'])
def get_realtime_all_cities():
    try:
        current_data = data_collector.collect_realtime_data()

        if not current_data:
            current_data = data_collector.latest_data

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": current_data
        })
    except Exception as e:
        logger.error(f"Error in get_realtime_all_cities: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/realtime/<city_name>', methods=['GET'])
def get_realtime_city_data(city_name):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        # Get latest data or trigger collection
        city_data = data_collector.latest_data.get(city_name)
        if not city_data:
            current_data = data_collector.collect_realtime_data()
            city_data = current_data.get(city_name)

        if city_data:
            return jsonify({
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": city_data
            })
        else:
            return jsonify({"status": "error", "message": "No data available"}), 404

    except Exception as e:
        logger.error(f"Error in get_realtime_city_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/trends/<city_name>/<parameter>', methods=['GET'])
def get_trends(city_name, parameter):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        days_future = request.args.get('days', 30, type=int)
        days_future = min(days_future, 60)  # Limit to 60 days

        trends = data_collector.create_prophet_trends(city_name, parameter, days_future)

        if trends:
            return jsonify({
                "status": "success",
                "city": city_name,
                "parameter": parameter,
                "trends": trends
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Insufficient data for trend analysis"
            }), 404

    except Exception as e:
        logger.error(f"Error in get_trends: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/historical/<city_name>', methods=['GET'])
def get_historical_data(city_name):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        limit = request.args.get('limit', 100, type=int)
        limit = min(limit, 1000)  # Maximum 1000 records

        city_history = data_collector.data_history.get(city_name, [])
        recent_history = city_history[-limit:] if city_history else []

        return jsonify({
            "status": "success",
            "city": city_name,
            "count": len(recent_history),
            "data": recent_history
        })

    except Exception as e:
        logger.error(f"Error in get_historical_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
#Health check end point
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cities": list(Config.CITIES.keys()),
        "latest_data_count": len(data_collector.latest_data),
        "background_thread_active": background_thread.is_alive()
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "SylphQI Prediction",
        "endpoints": {
            "/realtime/all": "Get real-time data for all cities",
            "/realtime/<city>": "Get real-time data for specific city",
            "/trends/<city>/<parameter>": "Get trend analysis",
            "/historical/<city>": "Get historical data",
            "/health": "Health check"
        }
    })

if __name__ == '__main__':
    logger.info("Starting initial data collection...")
    data_collector.collect_realtime_data()

    logger.info("Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
