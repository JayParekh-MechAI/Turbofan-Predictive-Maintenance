import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_model(x_path, y_path, model_save_path):
    logging.info("Loading processed data...")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).values.ravel() # Convert to 1D array for the model

    # 1. Split into Train and Validation sets
    # This lets us test the model on data it hasn't seen before
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"Training Random Forest on {len(X_train)} samples...")

    # 2. Initialize and Train
    # n_estimators=100 means 100 individual trees voting on the answer
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. Evaluate
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)

    logging.info(f"Validation MAE: {mae:.2f} cycles")
    logging.info(f"Validation RMSE: {rmse:.2f} cycles")
    logging.info(f"R2 Score: {r2:.2f}")

    # 4. Save the Model
    joblib.dump(model, model_save_path)
    logging.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model(
        x_path='data/processed/X_train.csv',
        y_path='data/processed/y_train.csv',
        model_save_path='models/random_forest.pkl'
    )