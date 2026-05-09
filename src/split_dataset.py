"""
split the clean dataset into train and test
"""

import os 
import sys 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.data_utils import load_dataset, save_dataset, split_dataset, prepare_X_y
from src.logger import logging


# 1. load clean dataset
print("loading dataset...")
df = load_dataset(file_path=os.path.join(project_root, "data", "processed", "cleaned_data.csv"))

if df is not None:
    logging.info("Initiated the Splitting of datast.")
    print("\nsplitting dataset....")
    # 2. separate labels and features
    X, y = prepare_X_y(data=df, cols_to_drop=['customerID', 'Churn'], target="Churn")
    # 3. split into train and test labels, features
    X_train, X_test, y_train, y_test = split_dataset(
        randomState=42,
        testSize=0.20,
        X=X,
        y=y   
    )
    print(f"Xtrain: {X_train.shape}\nytrain: {y_train.shape}\nXtest: {X_test.shape}\nytest: {y_test.shape}")
    print("\nsaving dataset...")
    # 4. save to data/processed
    
    # X-train
    save_dataset(data=X_train, file_path=os.path.join(project_root, "data", "processed", "Xtrain.csv"))
    # X-test
    save_dataset(data=X_train, file_path=os.path.join(project_root, "data", "processed", "Xtest.csv"))
    # y-train
    save_dataset(data=X_train, file_path=os.path.join(project_root, "data", "processed", "ytrain.csv"))
    # y-test
    save_dataset(data=X_train, file_path=os.path.join(project_root, "data", "processed", "ytest.csv"))
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        print("all labels and features saved successfully.")
        logging.info("Split Successful")
else:
    logging.error("Failed to split dataset")
    print("Couldn't load the cleaned datset")