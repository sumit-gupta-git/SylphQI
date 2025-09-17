import pandas as pd
import numpy as np
import logging
import datetime as dt
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add these methods to your existing Prediction class

def load_trained_model(self, model_path):
    try:
        self.model = joblib.load(model_path)
        logger.info(f'Model loaded from {model_path}')
        return True
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        return False

def predict_future_daily(self, current_data, city_name, days_ahead=30):
    """Generate daily predictions for future dates"""
    from datetime import date, timedelta
    
    if not hasattr(self, 'model') or self.model is None:
        logger.error("Model not trained or loaded")
        return None
    
    predictions = []
    today = date.today()
    
    # City encoding (add this as class attribute)
    city_encoding = {
        'delhi': 0, 'kolkata': 1, 'chennai': 2, 
        'bengaluru': 3, 'mumbai': 4
    }
    
    for day in range(1, days_ahead + 1):
        future_date = today + timedelta(days=day)
        
        # Create features for future date
        features = self._create_future_features(current_data, future_date, city_name)
        
        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features])
        features_df = features_df[self.independent]  # Ensure correct column order
        
        try:
            prediction = self.model.predict(features_df)
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'day': day,
                'weekday': future_date.strftime('%A'),
                'predicted_aqi': round(max(0, prediction), 1)
            })
        except Exception as e:
            logger.error(f"Error predicting day {day}: {e}")
            continue
    
    return predictions

def _create_future_features(self, current_data, future_date, city_name):
    """Create features for a future date using current data as baseline"""
    import numpy as np
    
    # Base features from current data
    features = {
        'PM2.5': current_data.get('PM2.5', 50),
        'PM10': current_data.get('PM10', 75),
        'NO2': current_data.get('NO2', 40),
        'SO2': current_data.get('SO2', 15),
        'CO': current_data.get('CO', 1.0),
        'O3': current_data.get('O3', 30),
        'NH3': current_data.get('NH3', 20),
        'Max_Temperature_C': current_data.get('Max_Temperature_C', 35),
        'Min_Temperature_C': current_data.get('Min_Temperature_C', 25),
        'Avg_Temperature_C': current_data.get('Avg_Temperature_C', 30),
        'Humidity_Percent': current_data.get('Humidity_Percent', 65),
        'Rainfall_mm': current_data.get('Rainfall_mm', 0),
        'Wind_Speed_kmh': current_data.get('Wind_Speed_kmh', 10),
        'Atmospheric_Pressure_hPa': current_data.get('Atmospheric_Pressure_hPa', 1013),
        'Visibility_km': current_data.get('Visibility_km', 5),
        'year': future_date.year,
        'month': future_date.month,
        'day': future_date.day
    }
    
    # Add daily patterns
    weekday = future_date.weekday()
    month = future_date.month
    
    # Weekend effects
    if weekday >= 5:  # Saturday, Sunday
        features['PM2.5'] *= np.random.uniform(0.85, 0.95)
        features['PM10'] *= np.random.uniform(0.85, 0.95)
        features['NO2'] *= np.random.uniform(0.7, 0.85)
        features['CO'] *= np.random.uniform(0.8, 0.9)
    
    # Seasonal effects
    if month in [12, 1, 2]:  # Winter
        features['PM2.5'] *= np.random.uniform(1.1, 1.25)
        features['PM10'] *= np.random.uniform(1.05, 1.2)
    elif month in [6, 7, 8]:  # Monsoon
        features['PM2.5'] *= np.random.uniform(0.7, 0.9)
        features['PM10'] *= np.random.uniform(0.65, 0.85)
        features['Rainfall_mm'] = np.random.uniform(5, 25)
    
    return features

def predict_multiple_cities(self, cities_data, days_ahead=30):
    """Predict for multiple cities"""
    results = {}
    
    for city, data in cities_data.items():
        city_predictions = self.predict_future_daily(data, city, days_ahead)
        if city_predictions:
            results[city] = {
                'city': city.title(),
                'predictions': city_predictions,
                'summary': {
                    'tomorrow': city_predictions['predicted_aqi'] if city_predictions else 0,
                    'week_avg': np.mean([p['predicted_aqi'] for p in city_predictions[:7]]) if len(city_predictions) >= 7 else 0,
                    'month_avg': np.mean([p['predicted_aqi'] for p in city_predictions]) if city_predictions else 0
                }
            }
    
    return results
