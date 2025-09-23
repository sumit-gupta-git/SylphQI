import pandas as pd
import numpy as np
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self):
        self.scaler = {}

    def extract_date_features(self,df):
        df_engineered = df.copy()
        df_engineered['Date'] = pd.to_datetime(df['Date'])
        df_engineered.set_index('Date')
        for col in df_engineered.columns:
            if col =='Date':
                
                df_engineered['year'] = df_engineered['Date'].dt.year
                df_engineered['month'] = df_engineered['Date'].dt.month
                df_engineered['day'] = df_engineered['Date'].dt.day
                df_engineered['weekday'] = df_engineered['Date'].dt.weekday
                df_engineered['is_weekkend'] = (df_engineered['weekday'] >= 5).astype(int)
                df_engineered['week_of_the_year'] = df_engineered['Date'].dt.isocalendar().week

        logger.info("Date features created")
        return df_engineered
    
    def drop_unnecessary_cols(self, df):
        df_engineered = df.copy()
        for col in ['City', 'State','Date', 'Month', 'Season','Wind_Direction']:
            if col in df.columns:
                df_engineered = df_engineered.drop(col, axis=1)
            else:
                pass
        logger.info("Unnecessary features successfully dropped")
        return df_engineered
    

    
    #master function
    def apply(self, df):
        df = self.extract_date_features(df)
        df = self.drop_unnecessary_cols(df)

        return df
    
#sample run
if __name__ == "__main__":

    f = FeatureEngineer()
    training_data = pd.read_csv('/home/sumit/SylphQI/data/India_Cities_AQI_Weather_2015_2024_Combined.csv')

    df = f.apply(training_data)
    print(df)
