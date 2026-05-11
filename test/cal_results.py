import os 
import sys 
import pandas as pd
import json
# Fix ImportError by ensuring project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import load_dataset
from src.logger import logging as log

DATASET_PATH = os.path.join(project_root, "outputs", "predictions.csv")

# load dataset 
df = load_dataset(file_path=DATASET_PATH)

if df is not None:
    # Use shape[0] for number of rows
    total_records = df.shape[0]
    
    # Calculate correct predictions by comparing columns and summing the boolean mask
    # This avoids the TypeError by not summing non-numeric columns
    correct_predicted = (df['actual'] == df['prediction']).sum()
    
    wrong_predicted = total_records - correct_predicted
    
    if total_records > 0:
        accuracy = (correct_predicted / total_records) * 100
    else:
        accuracy = 0

    print("total records = ", total_records)
    print("correct predictions =", correct_predicted)
    print("wrong predictions =", wrong_predicted)
    print(f"accuracy = {accuracy:.4f} %")   
    
    # saving the results 
    prediction_results = {
        "test_size": float(0.22),
        "total_records": int(total_records),
        "wrong_predictions": int(wrong_predicted),
        "accuracy": float(accuracy)
    }
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    with open(os.path.join(project_root, "outputs", "prediction_results.json"), 'w') as f:
        json.dump(prediction_results, f, indent=4)
    
    log.info(f"Results calculated: Total={total_records}, Correct={correct_predicted}, Accuracy={accuracy:.2f}%. Results saved succesfully")
else:
    print("Error: Could not load prediction data.")
    log.error("Failed to load prediction data. Check file path.")
