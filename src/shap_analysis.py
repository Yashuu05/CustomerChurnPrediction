import os
import sys 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import shap
from utils.data_utils import load_dataset, split_dataset
from utils.model_utils import load_model
from src.logger import logging as log

import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def load_X_test():
    # load x
    X = load_dataset(file_path=os.path.join(project_root, "data", "processed", "X_res.csv"))
    # load y
    y = load_dataset(file_path=os.path.join(project_root, "data", "processed", "y_res.csv"))
    # split into train and test
    if X is not None and y is not None:
        _, X_test, _, _ = split_dataset(randomState=42, testSize=0.20, X=X, y=y)
        return X_test
    return None

def implement_shap(Xtest, model):
    try:
        # Extract individual steps to avoid fitted state warnings when slicing
        prep = model.named_steps['prep']
        selector = model.named_steps['selector']
        rf_model = model.named_steps['model']
        
        print("Preprocessing data...")
        # Step-by-step transformation
        X_test_prep = prep.transform(Xtest)
        X_test_transformed = selector.transform(X_test_prep)
        
        # We'll explain a few samples
        test_samples = X_test_transformed.head(5)
        
        print(f"Calculating SHAP values for {len(test_samples)} samples using TreeExplainer...")
        # TreeExplainer is much faster and more accurate for Random Forest
        explainer = shap.TreeExplainer(rf_model)
        
        # Get Explanation object (modern SHAP API)
        shap_explanation = explainer(test_samples)
        
        # For RF classification, shap_explanation.values is typically (samples, features, classes)
        # We select the positive class (index 1) for the waterfall plot
        if len(shap_explanation.values.shape) == 3:
             exp = shap.Explanation(
                 values=shap_explanation.values[:, :, 1],
                 base_values=shap_explanation.base_values[:, 1],
                 data=shap_explanation.data,
                 feature_names=test_samples.columns.tolist()
             )
        else:
             exp = shap_explanation

        print("Generating Waterfall Plot for the first sample...")
        # waterfall_plot needs a single sample's explanation
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(exp[0], show=False)
        
        # Ensure output directory exists
        output_dir = os.path.join(project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, "shap_waterfall.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Waterfall plot successfully saved to: {plot_path}")
        return exp.values
    except Exception as e:
        print(f"Error in implement_shap: {e}")
        import traceback
        traceback.print_exc()
        log.error(f"Error in implement_shap: {e}")
        return None

if __name__ == "__main__":
    print("====== initiated =====")
    log.info("SHAP analysis initiated")
    # load dataset
    print("loading X_test...")
    X_test = load_X_test()
    if X_test is not None:
        log.info("X_test load successful")
        print("loading saved model...")
        model_path = os.path.join(project_root, "models", "best_model_random_forest.joblib")
        if os.path.exists(model_path):
            model = load_model(file_path=model_path)
            
            if model is not None:
                log.info("model load successful.")
                print("calculating SHAP...")
                shap_values = implement_shap(Xtest=X_test, model=model)
                if shap_values is not None:
                    log.info("SHAP analysis successful")
                    print("SHAP analysis completed.")
                
            else:
                print("failed to load model content")
                log.error("Failed to load saved model. Check if file is corrupted.")
        else:
            print(f"Model file not found at {model_path}")
            log.error(f"Model file not found at {model_path}")
        
    else:
        print("Failed to load X_test")
        log.error("Error occurred due to X_test failed to load.")
