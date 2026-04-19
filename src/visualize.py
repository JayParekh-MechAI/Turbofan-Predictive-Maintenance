import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
DATASET_ID = 'FD003'
MODEL_TYPE = 'xgboost'

def create_viz():
    # Load the predictions and truth
    preds_path = f'results/predictions_{MODEL_TYPE}_{DATASET_ID}.csv'
    truth_path = f'data/RUL_{DATASET_ID}.txt'
    
    if not os.path.exists(preds_path):
        print("Error: Run src/predict.py first to generate results!")
        return

    preds = pd.read_csv(preds_path)
    truth = pd.read_csv(truth_path, header=None, names=['true_RUL'])
    
    # Combine and sort for a clean "trend" visual
    results = pd.DataFrame({
        'Actual': truth['true_RUL'],
        'Predicted': preds['predicted_RUL']
    }).sort_values(by='Actual').reset_index(drop=True)

    # Calculate metrics for the text box
    mae = mean_absolute_error(results['Actual'], results['Predicted'])
    rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
    r2 = r2_score(results['Actual'], results['Predicted'])

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(results['Actual'], label='Actual RUL (Ground Truth)', color='#1f77b4', linewidth=2)
    plt.plot(results['Predicted'], 'ro', label='Predicted RUL', markersize=4, alpha=0.6)
    
    # Add a trend line for the predictions to show the general path
    plt.plot(results['Predicted'].rolling(window=5).mean(), color='red', alpha=0.3, label='Prediction Trend')

    # Create the text box string
    stats_text = f'Performance Metrics:\nR² Score: {r2:.2f}\nMAE: {mae:.2f} cycles\nRMSE: {rmse:.2f} cycles'
    
    # Position the text box in the upper left
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'NASA Turbofan RUL Prediction: Actual vs Predicted ({MODEL_TYPE} {DATASET_ID})', fontsize=14)
    plt.xlabel('Engine Samples (Sorted by Remaining Life)', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/prediction_plot_{MODEL_TYPE}_{DATASET_ID}.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved to results/prediction_plot_{MODEL_TYPE}_{DATASET_ID}.png")
    plt.show()

if __name__ == "__main__":
    create_viz()