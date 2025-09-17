import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Setting logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    
    def __init__(self):
        self.required_columns = [
            'AQI','Date','PM2.5', 'PM10', 'NO2','SO2', 'CO', 'O3', 'NH3',     #pollutants
            'Max_Temperature_C', 'Min_Temperature_C',                   #weather conditions
            'Avg_Temperature_C', 'Humidity_Percent', 'Rainfall_mm',
            'Wind_Speed_kmh', 'Atmospheric_Pressure_hPa','Visibility_km'
        ]

        self.date_column = 'Date'
    
    def load_data(self, file_path: str):

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Remove unnamed index columns if present
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                df.drop(columns=unnamed_cols, inplace=True)
                logger.info(f"Removed unnamed columns: {unnamed_cols}")
                
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df):
 
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Data validation passed")
        return True
    
    def clean_data(self, df):

        df_cleaned = df.copy()
        
        # Handle missing values
        logger.info("Handling missing values...")

        #fill numerical columns with median value and date column with interpolate
        for col in self.required_columns:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                if col == 'Date':
                    df_cleaned[col] = df_cleaned['Date'].interpolate(method='time')
                    logger.info(f"Filled missing values in date column")

                else:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        return df_cleaned
    
    def preprocess_features(self, df):
        df_processed = df.copy()
        
        for col in self.required_columns:
            # Remove units and convert to float
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
        
        logger.info("Feature preprocessing completed")
        return df_processed
    
    def get_data_summary(self, df):
    
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numerical_summary': df.describe().to_dict()
        }
        
        return summary

#example run
if __name__ == "__main__":

    preprocessor = Preprocessor()

    try:
        df = preprocessor.load_data('/home/sumit/SylphQI/data/India_Cities_AQI_Weather_2015_2024_Combined.csv')
        preprocessor.validate_data(df)
        df_cleaned = preprocessor.clean_data(df)
        df_processed = preprocessor.preprocess_features(df_cleaned)
        
        summary = preprocessor.get_data_summary(df_processed)
        print("Data preprocessing completed successfully!")
        print(f"Final data shape: {summary['shape']}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")