"""
train the model on the training and processed dataset and evaluate model's performance

1. load full pipeline from pipelines/
2. load labels and features from data/processed/
3. encode the labels and features using OneHotEncoder
4. fit models
5. save best_params_ of each model in output/ 
6. evaluate each model on roc_auc score, recall score, precision score, f1 score, accuracy score and confusion matrix
7. plot confusion matrix using heatmap, and save in output folder/
    example: [model_name][plot_name].png
8. record each metric of each model by creating a dataframe and save in output/ folder.
    columns: "model_name", "roc_auc", "recall", "precision", "f1_score", "accuracy"
9. create a bar graph to compare each model on each metric using dataframe created aand save in output/ folder
10. save the best model having highest roc_auc score in .joblib or .pkl format.

Artifacts:
- best_params_
- confusion matrix
- performance record dataframe
- bar graph comparisionw

"""
import os 
import sys 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.logger import logging
from utils import data_utils, model_utils
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pipelines import full_pipeline

def load_resources() -> tuple:
    """
    loads the required labels, features and pipeline all at once.
    Input: file path
    Output: X_train, y_train, X_test, y_test, model_pipeline
    """
    try:
        # load X
        X = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "X_res.csv"))
        # load y
        y = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "y_res.csv"))
        # split into train and test
        X_train, X_test, y_train, y_test = data_utils.split_dataset(randomState=42, testSize=0.20, X=X, y=y)
        # pipeline
        cat_cols, num_cols = data_utils.separate_cols_type(data=X)
        grid_pipelines = full_pipeline.build_full_pipeline(categorical_cols=cat_cols, numerical_cols=num_cols)
    
        return X_train, X_test, y_train, y_test, grid_pipelines
    
    except Exception as e:
        print(f"{str(e)}")
        logging.error(msg=f"{str(e)}")
        return None

if __name__ == "__main__" : 
    print("=============== Initialized Training =====================")
    print("\nLoading resources...")
    X_train, X_test, y_train, y_test, grid_pipelines = load_resources()
    if X_train is not None and grid_pipelines is not None: 
        print(f"\nX train: {X_train.shape}\ny train: {y_train.shape}\nX test: {X_test.shape}\ny test: {y_test.shape}")
        
        performance_metrics = []
        best_params_dict = {}
        best_roc_auc = 0
        best_model = None
        best_model_name = ""

        # Ensure output directories exist
        os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

        for model_name, grid_search in grid_pipelines.items():
            print(f"\nTraining {model_name}...")
            logging.info(f"Fitting {model_name}")
            
            # Fit model
            grid_search.fit(X_train, y_train)
            
            # Get best estimator and params
            model = grid_search.best_estimator_
            best_params_dict[model_name] = grid_search.best_params_
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate using model_utils helper
            metrics = model_utils.evalulate_model(y_test=y_test.values.ravel(), y_pred=y_pred)
            # metrics returns [recall, precision, f1, acc, roc]
            recall, precision, f1, acc, roc = metrics
            
            performance_metrics.append({
                "model_name": model_name,
                "roc_auc": roc,
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "accuracy": acc
            })
            
            print(f"{model_name} - ROC-AUC: {roc:.4f}, F1: {f1:.4f}")

            # Plot Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(project_root, "outputs", f"{model_name}_confusion_matrix.png"))
            plt.close()

            # Track best model based on ROC-AUC
            if roc > best_roc_auc:
                best_roc_auc = roc
                best_model = model
                best_model_name = model_name

        # Save best hyperparameters
        with open(os.path.join(project_root, "outputs", "best_hyperparameters.json"), 'w') as f:
            json.dump(best_params_dict, f, indent=4)
        
        # Save performance metrics to DataFrame
        performance_df = pd.DataFrame(performance_metrics)
        performance_df.to_csv(os.path.join(project_root, "outputs", "performance_record.csv"), index=False)
        print("\nPerformance record saved to outputs/performance_record.csv")

        # Create comparison bar graph
        plt.figure(figsize=(12, 8))
        performance_df_melted = performance_df.melt(id_vars="model_name", var_name="Metric", value_name="Score")
        sns.barplot(data=performance_df_melted, x="Metric", y="Score", hue="model_name")
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, "outputs", "model_comparison.png"))
        plt.close()
        print("Comparison bar graph saved to outputs/model_comparison.png")

        # Save the best model
        if best_model is not None:
            model_save_path = os.path.join(project_root, "models", f"best_model_{best_model_name}.joblib")
            model_utils.save_model(file_path=model_save_path, model=best_model)
            print(f"\nBest model ({best_model_name}) saved to {model_save_path}")
            logging.info(f"Best model {best_model_name} saved with ROC-AUC {best_roc_auc}")

    else: 
        print("Issue in load_resources() function while loading datasets.")