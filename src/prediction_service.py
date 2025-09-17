import pandas as pd
import numpy as np
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Prediction:
    def __init__(self):
        self.independent = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3',
                    'Max_Temperature_C', 'Min_Temperature_C', 'Avg_Temperature_C',
                    'Humidity_Percent', 'Rainfall_mm', 'Wind_Speed_kmh',
                    'Atmospheric_Pressure_hPa', 'Visibility_km', 'year', 'month', 'day']
        
        self.dependent = ['AQI']
        self.model = {}
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}

    
    def model_training(self, df):
        #importing dependencies
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.ensemble import RandomForestRegressor

        X = df[self.independent]
        y = df[self.dependent]

        #train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        #model_training with best parameters found in modeling.ipynb
        model = RandomForestRegressor(n_estimators = 500,
                min_samples_split = 2,
                max_features= 5,
                max_depth=15)
        model.fit(X_train, y_train)
        self.model = model

        logger.info('Model trained')

        return self.model
    
    
    ##Creating Function to Evaluate Model
    def evaluate_model(self, true, predicted):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        logger.info('evaluation completed!')
        return mae, rmse, r2_square
    
#sample run
if __name__ == '__main__':
    df = pd.read_csv('/home/sumit/SylphQI/data/India_Cities_AQI_Weather_2015_2024_Combined.csv')
    
    #feature_engineering
    fe = FeatureEngineer()
    df = fe.apply(df)
    df = fe.numerical_scaler(df)
    

    #model training
    p = Prediction()
    model = p.model_training(df)
    y_pred = model.predict(p.X_test)

    joblib.dump(model,'/home/sumit/SylphQI/models/rf_model.pkl')
    
    mae, rmse, r2_score = p.evaluate_model(p.y_test, y_pred)
    print(mae, rmse, r2_score)