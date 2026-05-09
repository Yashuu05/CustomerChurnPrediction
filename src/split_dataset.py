"""
split the clean dataset into labels and features
and encode the labels (y)
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
    logging.info("Initiated the Splitting of dataset")
    print("\n1. encoding labels...")
    df["Churn"] = df["Churn"].map({
        "Yes":1,
        "No":0
    })
    print(f"\n{df['Churn'].head()}")
    logging.info("Encoded labels")
    print("\n2. splitting dataset....")
    # 2. separate labels and features
    X, y = prepare_X_y(data=df, cols_to_drop=['customerID', 'Churn'], target="Churn")
    print(f"Xtrain: {X.shape}\nytrain: {y.shape}")
    print("\n3. saving dataset...")
    # 3. save to data/processed
    # X -> features
    save_dataset(data=X, file_path=os.path.join(project_root, "data", "processed", "X.csv"))
    # y -> labels
    save_dataset(data=y, file_path=os.path.join(project_root, "data", "processed", "y.csv"))
    if X is not None and y is not None:
        print("all labels and features saved successfully.")
        logging.info("Split Successful")
    else:
        print("X and y returns None")
        logging.error("X and Y is a NoneType object. Failed to split dataset.")
else:
    logging.error("Failed to split dataset due to failed load of dataset.")
    print("Couldn't load the cleaned datset")