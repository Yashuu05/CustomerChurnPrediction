"""
predict the churn using test dataset and saved model
"""
import os 
import sys 
import pandas as pd
import numpy as np

# Fix ImportError by ensuring project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logger import logging as log
from utils.data_utils import load_dataset, save_dataset, split_dataset
from utils.model_utils import load_model

MODEL_PATH = os.path.join(project_root, "models", "best_model_random_forest.joblib")
X_PATH = os.path.join(project_root, "data", "processed", "X_res.csv")
Y_PATH = os.path.join(project_root, "data", "processed", "y_res.csv")
OUTPUT_DIR = os.path.join(project_root, "outputs")


def plot_graph(y_test, y_pred):
    """
    constructs the Scatter graph representing the comparison between actual target and predicted target
    X-axis : y_predict
    y-axis : y actual
    ---
    Inputs: 
    1. y_test : actual target feature
    2. y_pred : predicted values by model
    ---
    Output: scatter plot between y_test and y_pred
    ---
    Note: the graph image is saved in outputs/ folder
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    # y_test = actual labels, y_pred = model predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue') # alpha adds transparency
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # diagonal line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    plt.savefig(os.path.join(project_root, "outputs", f"actual_Vs_prediction.png"))
    plt.close()


def prediction(X_test, y_test, model):
    """
    Use saved model to predict the target values and probabilities on the given test dataset
    ---
    Input:
    1. X_test : test dataset having all features
    2. y_test : actual target label
    3. model : saved weights in .joblib or .pkl
    --
    output: prediction dataframe
    """

    predictions = model.predict(X_test)
        
    # Get churn probability (class 1)
    # Check if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = np.nan
        log.warning("Model does not support predict_proba")

    # Create a results dataframe
    # Flatten y_test if it's a 1-column dataframe
    y_actual = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
        
    predict_data = pd.DataFrame({
        "actual": y_actual,
        "prediction": predictions,
        "churn_probability": probabilities
    })
        
    print(f"Prediction done. Total samples: {len(predict_data)}")

    return predict_data, predictions


if __name__ == "__main__":
    log.info("Initialized prediction script")

    # 1. load required labels and features
    log.info(f"Loading labels and features")
    X= load_dataset(file_path=X_PATH)
    y = load_dataset(file_path=Y_PATH)

    # 2. load saved model
    log.info(f"Loading saved model from {MODEL_PATH}")
    model = load_model(file_path=MODEL_PATH)

    # 3. split into X-test
    log.info("Splitting into X-test")
    _,X_test,_,y_test = split_dataset(randomState=42, testSize=0.22, X=X, y=y)
    # implement the prediction 
    if X is not None and y is not None and model is not None:
        print("Model and test datasets loaded successfully")
        
        log.info("Performing churn prediction")
        # Perform prediction on X_test
        predict_data, y_pred = prediction(X_test=X_test, y_test=y_test, model=model)
        # Save the results in outputs/
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "predictions.csv")
        save_dataset(data=predict_data, file_path=output_file)
        
        log.info(f"Prediction results saved to {output_file}")
        print(f"Prediction results saved to {output_file}")

        log.info("plotting scatter plot")
        plot_graph(y_test=y_test, y_pred=y_pred)
        print("plot contsructed.")
        log.info("Prediction process completed successfully")

    else:
        error_msg = "Error! Critical components (X_test, y_test, or model) failed to load."
        print(error_msg)
        log.error(error_msg)