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
from pipelines import full_pipeline
from sklearn.preprocessing import OneHotEncoder

def load_resources() -> tuple:
    """
    loads the required labels, features and pipeline all at once.
    Input: file path
    Output: X_train, y_train, X_test, y_test, model_pipeline
    """
    try:
        # load y_train_res (Balanced)
        y_train = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "y_train_res.csv"))
        # load y_test_processed
        y_test = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "y_test_processed.csv"))
        # X_train
        X_train = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "X_train_res.csv"))
        # X_test
        df = data_utils.load_dataset(file_path=os.path.join(project_root, "data", "processed", "cleaned_data.csv"))
        X, y = data_utils.prepare_X_y(data=df, cols_to_drop=["customerID"], target="Churn")
        _,X_test, _, _ = data_utils.split_dataset(randomState=42, testSize=0.20, X=X, y=y)
        # pipeline
        cat_cols, num_cols = data_utils.separate_cols_type(data=X_train)
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
    if X_train is not None: 
        print(f"\nX train: {X_train.shape}\ny train: {y_train.shape}\nX test: {X_test.shape}\ny test: {y_test.shape}")
        print(f"grid pipeline type: {type(grid_pipelines)}")
    else: 
        print("Issue in load_resources() function while loading datasets.")