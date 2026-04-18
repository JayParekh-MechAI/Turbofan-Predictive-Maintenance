import pandas as pd
import joblib
import logging
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants (Must match your preprocess.py)
INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = [f's_{i}' for i in range(1, 22)] 
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES
FEAT_COLS = ['setting_1', 's_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
             's_12', 's_13', 's_14', 's_15', 's_20', 's_21']
TOP_4 = ['s_11', 's_9', 's_4', 's_12']

def run_inference(test_path, model_path, scaler_path):
    logging.info("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 1. Load Test Data
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=COL_NAMES)

    # 2. Get the LAST row for each engine
    test_last_shot = df_test.groupby('unit_nr').last().reset_index()

    # 3. Feature Engineering
    for col in TOP_4:
        test_last_shot[f'{col}_roll_mean'] = test_last_shot[col]
        test_last_shot[f'{col}_roll_std'] = 0 

    roll_cols = [f'{c}_roll_mean' for c in TOP_4] + [f'{c}_roll_std' for c in TOP_4]
    all_features = FEAT_COLS + roll_cols

    # 4. Scaling
    test_last_shot[all_features] = scaler.transform(test_last_shot[all_features])

    # 5. Predict
    predictions = model.predict(test_last_shot[all_features])
    
    # 6. Output Results
    results = pd.DataFrame({
        'unit_nr': test_last_shot['unit_nr'],
        'predicted_RUL': predictions.round(2)
    })
    
    return results

def score_predictions(predictions_df, truth_path):
    logging.info("Loading Ground Truth and calculating final scores...")
    
    # Load the actual RUL values from NASA's answer key
    y_true = pd.read_csv(truth_path, header=None, names=['true_RUL']).values.ravel()
    y_pred = predictions_df['predicted_RUL'].values
    
    # Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("         FINAL TEST SCORES")
    print("="*40)
    print(f"Test MAE:   {mae:.2f} cycles")
    print(f"Test RMSE:  {rmse:.2f} cycles")
    print(f"Test R2:    {r2:.2f}")
    print("="*40)

if __name__ == "__main__":
    # Path to the answer key NASA provided
    TRUTH_PATH = 'data/RUL_FD001.txt'

    results_df = run_inference(
        test_path='data/test_FD001.txt',
        model_path='models/random_forest.pkl',
        scaler_path='models/scaler.pkl'
    )
    
    print("\n--- SAMPLE PREDICTIONS (First 10 Engines) ---")
    print(results_df.head(10))

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/predictions.csv', index=False)
    logging.info("Predictions saved to results/predictions.csv")

    # The Final Scoring
    if os.path.exists(TRUTH_PATH):
        score_predictions(results_df, TRUTH_PATH)
    else:
        logging.warning(f"Truth file not found at {TRUTH_PATH}. Skipping scoring.")