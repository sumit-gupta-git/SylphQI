import pandas as pd
import numpy as np
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from src.data_preprocessing import  Preprocessor
from src.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Scaler loaded successfully from {SCALER_PATH}")

except Exception as e:
    logger.info(f"ERROR: Could not load model or scaler. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist in the same directory as app.py. Error: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET'])
def predict():
    """
    Predicts the value of AQI for upcoming concentration of pollutants, coming in a JSON format
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded on server. Please check server logs for details."}), 500
    try:
        data = request.get_json(force=True)
        columns = ['Date', 'City', 'State', 'PM2.5', 'PM10', 'NO2',
                            'SO2', 'CO', 'O3', 'NH3', 'Max_Temperature_C', 'Min_Temperature_C',
                            'Avg_Temperature_C', 'Humidity_Percent', 'Rainfall_mm',
                            'Wind_Speed_kmh', 'Wind_Direction', 'Atmospheric_Pressure_hPa',
                            'Visibility_km', 'Month', 'Season']
        
        data = pd.DataFrame(data, columns=columns)
        p = Preprocessor()
        #cleaning the upcoming data
        data = p.clean_data(data)

        #extracting features
        fe = FeatureEngineer()
        data = fe.apply(data)

        #scaling the data
        data = scaler(data)

        #prediction
        predicted_val = model.predict(data)[0]

        response = {
            'predicted_aqi' : predicted_val
        }

        return jsonify(response)
    
    except ValueError as e:
        return jsonify({"error": f"Invalid data type for one or more features. Please ensure all values are numeric. Details: {e}"}), 400
    except Exception as e:
        logger.info(f"An unexpected error occurred: {e}") 
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)