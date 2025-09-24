from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import joblib
import logging
from datetime import datetime, timedelta
import threading
import time
import json
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS"])

class Config:
    WEATHERSTACK_API_KEY = "9b482a8fc8114edd53b5a30d85a80efe"
    VISUAL_CROSSING_API_KEY = "QLZZG8H6JJJQSBEN7XF7BW4AK"

    CITIES = {
        "Delhi": {"lat": 28.7041, "lon": 77.1025},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777},
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946}
    }

class RobustDataCollector:
    def __init__(self):
        self.latest_data = {}
        self.data_history = {city: [] for city in Config.CITIES.keys()}
        self.rf_model = None

        # Initialize with some mock historical data to enable trends
        self._initialize_mock_history()

        # Load model if available
        self._load_model()

    def _load_model(self):
        try:
            self.rf_model = joblib.load('models/rf_model.pkl')
            logger.info("RandomForest model loaded successfully")
        except Exception as e:
            logger.warning(f"RandomForest model not found: {e}. Using mock predictions...")
            self.rf_model = None

    def _initialize_mock_history(self):
        """Create mock historical data to enable trend analysis"""
        logger.info("Initializing mock historical data for trend analysis...")

        base_data = {
            "Delhi": {"PM2.5": 80, "PM10": 120, "NO2": 45, "AQI": 150},
            "Mumbai": {"PM2.5": 60, "PM10": 90, "NO2": 38, "AQI": 120},
            "Chennai": {"PM2.5": 40, "PM10": 70, "NO2": 28, "AQI": 85},
            "Kolkata": {"PM2.5": 70, "PM10": 110, "NO2": 42, "AQI": 140},
            "Bengaluru": {"PM2.5": 45, "PM10": 75, "NO2": 30, "AQI": 95}
        }

        # Generate 30 days of mock historical data
        for city_name, base_values in base_data.items():
            city_history = []

            for days_ago in range(30, 0, -1):  # 30 days ago to today
                timestamp = datetime.now() - timedelta(days=days_ago)

                # Add some realistic variation and trends
                day_of_week = timestamp.weekday()
                is_weekend = day_of_week >= 5

                mock_record = {
                    'timestamp': timestamp.isoformat(),
                    'city': city_name
                }

                for param, base_val in base_values.items():
                    # Weekend effect (lower pollution)
                    weekend_factor = 0.8 if is_weekend else 1.0

                    # Daily variation
                    daily_variation = random.uniform(0.7, 1.3)

                    # Seasonal trend (simple)
                    seasonal_factor = 1.0 + 0.2 * np.sin(days_ago * 2 * np.pi / 365)

                    value = base_val * weekend_factor * daily_variation * seasonal_factor
                    mock_record[param] = max(0, round(value, 1))

                # Add weather data
                mock_record.update({
                    'Max_Temperature_C': round(25 + random.uniform(-5, 10), 1),
                    'Min_Temperature_C': round(20 + random.uniform(-5, 8), 1),
                    'Avg_Temperature_C': round(22 + random.uniform(-3, 8), 1),
                    'Humidity_Percent': round(60 + random.uniform(-20, 20), 1),
                    'Rainfall_mm': round(random.uniform(0, 5), 1),
                    'Wind_Speed_kmh': round(random.uniform(3, 15), 1),
                    'Atmospheric_Pressure_hPa': round(1013 + random.uniform(-10, 10), 1),
                    'Visibility_km': round(random.uniform(5, 15), 1),
                    'SO2': round(random.uniform(5, 25), 1),
                    'CO': round(random.uniform(0.5, 2.5), 1),
                    'O3': round(random.uniform(20, 50), 1),
                    'NH3': round(random.uniform(10, 35), 1)
                })

                city_history.append(mock_record)

            self.data_history[city_name] = city_history

        logger.info("Mock historical data initialized for all cities")

    def fetch_weather_data(self, city_name):
        try:
            city_info = Config.CITIES.get(city_name)
            if not city_info:
                return None

            # Try Visual Crossing API first
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city_info['lat']},{city_info['lon']}"
            params = {
                "unitGroup": "metric",
                "key": Config.VISUAL_CROSSING_API_KEY,
                "include": "current"
            }

            response = requests.get(url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                current_conditions = data.get('currentConditions', {})

                return {
                    'Max_Temperature_C': current_conditions.get('tempmax', current_conditions.get('temp', 25)),
                    'Min_Temperature_C': current_conditions.get('tempmin', current_conditions.get('temp', 25)),
                    'Avg_Temperature_C': current_conditions.get('temp', 25),
                    'Humidity_Percent': current_conditions.get('humidity', 60),
                    'Rainfall_mm': current_conditions.get('precip', 0),
                    'Wind_Speed_kmh': current_conditions.get('windspeed', 5),
                    'Atmospheric_Pressure_hPa': current_conditions.get('pressure', 1013),
                    'Visibility_km': current_conditions.get('visibility', 10)
                }
            else:
                logger.warning(f"Weather API failed for {city_name}: {response.status_code}")
                return self.generate_mock_weather_data(city_name)
        except Exception as e:
            logger.error(f"Error fetching weather data for {city_name}: {e}")
            return self.generate_mock_weather_data(city_name)

    def generate_mock_pollutant_data(self, city_name):
        base_values = {
            "Delhi": {"PM2.5": 80, "PM10": 120, "NO2": 45, "SO2": 15, "CO": 2.0, "O3": 30},
            "Mumbai": {"PM2.5": 60, "PM10": 90, "NO2": 38, "SO2": 12, "CO": 1.5, "O3": 35},
            "Chennai": {"PM2.5": 40, "PM10": 70, "NO2": 28, "SO2": 8, "CO": 1.0, "O3": 40},
            "Kolkata": {"PM2.5": 70, "PM10": 110, "NO2": 42, "SO2": 18, "CO": 1.8, "O3": 25},
            "Bengaluru": {"PM2.5": 45, "PM10": 75, "NO2": 30, "SO2": 10, "CO": 1.2, "O3": 38}
        }

        base = base_values.get(city_name, base_values["Delhi"])
        mock_data = {}

        for param, value in base.items():
            variation = random.uniform(0.7, 1.3)
            mock_data[param] = round(value * variation, 1)

        mock_data['NH3'] = round(random.uniform(15, 30), 1)
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
            'Max_Temperature_C': base["temp"] + random.uniform(2, 5),
            'Min_Temperature_C': base["temp"] - random.uniform(3, 8),
            'Avg_Temperature_C': base["temp"] + random.uniform(-2, 2),
            'Humidity_Percent': base["humidity"] + random.uniform(-10, 10),
            'Rainfall_mm': random.uniform(0, 5),
            'Wind_Speed_kmh': random.uniform(3, 15),
            'Atmospheric_Pressure_hPa': base["pressure"] + random.uniform(-5, 5),
            'Visibility_km': random.uniform(3, 15)
        }

    def calculate_aqi(self, pollutant_data):
        """Simple AQI calculation"""
        try:
            pm25 = pollutant_data.get('PM2.5', 50)
            pm10 = pollutant_data.get('PM10', 75)
            no2 = pollutant_data.get('NO2', 40)

            # Simplified AQI calculation
            aqi = (pm25 * 2 + pm10 * 1.5 + no2 * 1.2) / 4.7
            return max(0, round(aqi, 1))
        except:
            return 100  # Default AQI

    def combine_and_predict_aqi(self, city_name, pollutant_data, weather_data):
        try:
            current_time = datetime.now()

            combined_data = {
                **pollutant_data,
                **weather_data,
                'year': current_time.year,
                'month': current_time.month,
                'day': current_time.day,
                'timestamp': current_time.isoformat(),
                'city': city_name
            }

            # Calculate AQI
            if self.rf_model:
                # Use actual model if available
                try:
                    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3',
                              'Max_Temperature_C', 'Min_Temperature_C', 'Avg_Temperature_C', 
                              'Humidity_Percent', 'Rainfall_mm', 'Wind_Speed_kmh',
                              'Atmospheric_Pressure_hPa', 'Visibility_km','year', 'month', 'day']

                    features_df = pd.DataFrame([combined_data])
                    features_df = features_df.reindex(columns=features, fill_value=0)
                    predicted_aqi = self.rf_model.predict(features_df)[0]
                    combined_data['AQI'] = max(0, round(predicted_aqi, 1))
                except Exception as e:
                    logger.error(f"Model prediction failed: {e}")
                    combined_data['AQI'] = self.calculate_aqi(pollutant_data)
            else:
                combined_data['AQI'] = self.calculate_aqi(pollutant_data)

            return combined_data

        except Exception as e:
            logger.error(f"Error combining data for {city_name}: {e}")
            return None

    def collect_realtime_data(self):
        collected_data = {}

        for city_name in Config.CITIES.keys():
            logger.info(f"Collecting data for {city_name}")

            # Always use mock pollutant data for now (since OpenAQ is not working)
            pollutant_data = self.generate_mock_pollutant_data(city_name)

            # Try to get real weather data, fall back to mock
            weather_data = self.fetch_weather_data(city_name)

            combined_data = self.combine_and_predict_aqi(city_name, pollutant_data, weather_data)

            if combined_data:
                self.latest_data[city_name] = combined_data
                self.data_history[city_name].append(combined_data)

                # Keep only last 100 records per city
                if len(self.data_history[city_name]) > 100:
                    self.data_history[city_name] = self.data_history[city_name][-100:]

                collected_data[city_name] = combined_data

        return collected_data

    def create_simple_trends(self, city_name, parameter, days_future=30):
        """Create simple statistical trends without Prophet"""
        try:
            city_history = self.data_history.get(city_name, [])
            if len(city_history) < 5:
                return None

            # Extract parameter values
            values = []
            dates = []
            for record in city_history:
                if parameter in record:
                    values.append(record[parameter])
                    dates.append(record['timestamp'])

            if len(values) < 5:
                return None

            # Simple linear trend
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            trend_line = np.poly1d(z)

            # Generate future predictions
            future_dates = []
            future_values = []

            for i in range(1, days_future + 1):
                future_date = datetime.now() + timedelta(days=i)
                future_dates.append(future_date.isoformat())

                # Simple trend projection with some variation
                trend_value = trend_line(len(values) + i)
                variation = np.random.normal(0, np.std(values) * 0.1)
                future_value = max(0, trend_value + variation)
                future_values.append(round(future_value, 1))

            return {
                'historical': [
                    {'ds': dates[i], 'y': values[i]} 
                    for i in range(max(0, len(dates) - 15), len(dates))
                ],
                'forecast': [
                    {
                        'ds': future_dates[i],
                        'yhat': future_values[i],
                        'yhat_lower': future_values[i] * 0.9,
                        'yhat_upper': future_values[i] * 1.1
                    }
                    for i in range(len(future_dates))
                ],
                'method': 'simple_statistical',
                'message': 'Simple trend analysis (Prophet alternative)'
            }

        except Exception as e:
            logger.error(f"Error creating trends for {city_name}, {parameter}: {e}")
            return None

# Initialize data collector
data_collector = RobustDataCollector()

# Background thread
def background_data_collection():
    while True:
        try:
            data_collector.collect_realtime_data()
            logger.info("Background data collection completed")
            time.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Error in background data collection: {e}")
            time.sleep(60)

background_thread = threading.Thread(target=background_data_collection, daemon=True)
background_thread.start()

# ROBUST API ROUTES
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cities": list(Config.CITIES.keys()),
            "latest_data_count": len(data_collector.latest_data),
            "background_thread_active": background_thread.is_alive(),
            "total_historical_records": sum(len(history) for history in data_collector.data_history.values())
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/realtime/all', methods=['GET'])
def get_realtime_all_cities():
    try:
        # Always try to get fresh data, but don't fail if it's not available
        current_data = data_collector.collect_realtime_data()

        if not current_data:
            current_data = data_collector.latest_data

        if not current_data:
            # Emergency fallback
            current_data = {}
            for city in Config.CITIES.keys():
                pollutant_data = data_collector.generate_mock_pollutant_data(city)
                weather_data = data_collector.generate_mock_weather_data(city)
                current_data[city] = data_collector.combine_and_predict_aqi(city, pollutant_data, weather_data)

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": current_data,
            "data_source": "mixed" if current_data else "mock"
        })

    except Exception as e:
        logger.error(f"Error in get_realtime_all_cities: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/realtime/<city_name>', methods=['GET'])
def get_realtime_city_data(city_name):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        city_data = data_collector.latest_data.get(city_name)
        if not city_data:
            # Try to collect fresh data
            current_data = data_collector.collect_realtime_data()
            city_data = current_data.get(city_name)

        if not city_data:
            # Emergency fallback
            pollutant_data = data_collector.generate_mock_pollutant_data(city_name)
            weather_data = data_collector.generate_mock_weather_data(city_name)
            city_data = data_collector.combine_and_predict_aqi(city_name, pollutant_data, weather_data)

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": city_data
        })

    except Exception as e:
        logger.error(f"Error in get_realtime_city_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/trends/<city_name>/<parameter>', methods=['GET'])
def get_trends(city_name, parameter):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        days_future = request.args.get('days', 30, type=int)
        days_future = min(max(days_future, 1), 60)  # Limit between 1-60 days

        # Use simple trends instead of Prophet
        trends = data_collector.create_simple_trends(city_name, parameter, days_future)

        if trends:
            return jsonify({
                "status": "success",
                "city": city_name,
                "parameter": parameter,
                "trends": trends,
                "days_forecast": days_future
            })
        else:
            return jsonify({
                "status": "error", 
                "message": f"Unable to generate trends for {parameter} in {city_name}",
                "suggestion": "Try a different parameter or wait for more data collection"
            }), 404

    except Exception as e:
        logger.error(f"Error in get_trends: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/historical/<city_name>', methods=['GET'])
def get_historical_data(city_name):
    try:
        if city_name not in Config.CITIES:
            return jsonify({"status": "error", "message": "City not found"}), 404

        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 1000)  # Limit between 1-1000 records

        city_history = data_collector.data_history.get(city_name, [])
        recent_history = city_history[-limit:] if city_history else []

        return jsonify({
            "status": "success",
            "city": city_name,
            "count": len(recent_history),
            "total_available": len(city_history),
            "data": recent_history
        })

    except Exception as e:
        logger.error(f"Error in get_historical_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "AQI Prediction API - Robust Version",
        "status": "online",
        "endpoints": {
            "/api/health": "System health check",
            "/api/realtime/all": "All cities real-time data",
            "/api/realtime/<city>": "Specific city data",
            "/api/trends/<city>/<parameter>?days=N": "Trend analysis",
            "/api/historical/<city>?limit=N": "Historical data"
        },
        "features": [
            "Mock historical data for instant trends",
            "Fallback data generation",
            "Simple statistical forecasting",
            "Robust error handling"
        ]
    })

if __name__ == '__main__':
    logger.info("Initializing robust AQI API server...")
    logger.info("Mock historical data available for immediate trend analysis")

    # Initial data collection
    data_collector.collect_realtime_data()

    logger.info("Starting Flask API server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
