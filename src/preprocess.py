import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION / CONSTANTS ---
# Pro Tip: Keeping these at the top makes the script "Config-driven"
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = [f's_{i}' for i in range(1, 22)] 
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

FEAT_COLS = ['setting_1', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 
             's_11', 's_12', 's_13', 's_14', 's_15', 's_20', 's_21']
TOP_4 = ['s_11', 's_9', 's_4', 's_12']

class DataPipeline:
    """
    A professional-grade pipeline for NASA Turbofan degradation data.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def add_features(self, df, is_training=True):
        """Transformation logic: RUL calculation and rolling statistics."""
        # 1. Target Engineering
        if is_training:
            max_cycle = df.groupby('unit_nr')['time_cycles'].transform('max')
            df['RUL'] = (max_cycle - df['time_cycles']).clip(upper=125)
        
        # 2. Rolling Features
        for col in TOP_4:
            group = df.groupby('unit_nr')[col]
            df[f'{col}_roll_mean'] = group.transform(lambda x: x.rolling(10, min_periods=1).mean())
            df[f'{col}_roll_std'] = group.transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))
        
        return df

    def run(self, input_path, output_dir, is_training=True):
        """Orchestrates the loading, transforming, and saving of data."""
        logging.info(f"Starting pipeline for: {input_path}")
        
        # Load
        df = pd.read_csv(input_path, sep=r'\s+', header=None, names=COL_NAMES)
        
        # Transform
        df = self.add_features(df, is_training)
        
        # Feature Selection
        roll_cols = [f'{c}_roll_mean' for c in TOP_4] + [f'{c}_roll_std' for c in TOP_4]
        all_features = FEAT_COLS + roll_cols
        
        # Scale
        if is_training:
            df[all_features] = self.scaler.fit_transform(df[all_features])
            joblib.dump(self.scaler, 'models/scaler.pkl')
            logging.info("Scaler fitted and saved to models/scaler.pkl")
        else:
            # Load the scaler we fitted during training!
            self.scaler = joblib.load('models/scaler.pkl')
            df[all_features] = self.scaler.transform(df[all_features])
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        X = df[all_features]
        X.to_csv(f"{output_dir}/X_{'train' if is_training else 'test'}.csv", index=False)
        
        if is_training:
            y = df['RUL']
            y.to_csv(f"{output_dir}/y_train.csv", index=False)
            
        logging.info(f"Pipeline complete. Files saved to {output_dir}")

if __name__ == "__main__":
    pipeline = DataPipeline()
    # Path assumes you run from the project root
    pipeline.run(
        input_path='data/train_FD001.txt', 
        output_dir='data/processed', 
        is_training=True
    )