import sys
import os
import numpy as np
# Fix ImportError by ensuring project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from configs.paths import DATA_PATH
from utils import data_utils
from src.logger import logging

SAVE_FILE_PATH = os.path.join(project_root, "data", "processed", "cleaned_data.csv")
# load dataset using raw string for path
df = data_utils.load_dataset(file_path=DATA_PATH)

if df is None:
    logging.error("Error loading the dataset.")
else:
    # convert the `TotalCharges` into float
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df['InternetService'] = df['InternetService'].replace("No", "No internet service")  
    print(f"TotalCharges dtype: {df['TotalCharges'].dtypes}")
    print(f"\nunique values of 'InternetService': {df['InternetService'].unique()}")

    cat_cols, num_cols = data_utils.separate_cols_type(data=df)

    print(f"\nCategorical cols: {cat_cols}\nNumerical cols: {num_cols}")
    
    # Process numerical columns
    for col in num_cols:
        if df[col].isnull().sum() > 0: 
            # fill with median
            df[col] = df[col].fillna(df[col].median())
            
    # Process categorical columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0: 
            # fill with mode (first value of the series)
            df[col] = df[col].fillna(df[col].mode()[0])
        
    # Check for remaining null values
    null_count = df.isnull().sum().sum()
    print(f"\nRemaining null values:\n{df.isnull().sum()}")

    if null_count == 0:
        logging.info("Dataset successfully processed and cleaned.")
        print("\nSuccess: Dataset is clean.")
        os.makedirs(os.path.dirname(SAVE_FILE_PATH), exist_ok=True)
        data_utils.save_dataset(data=df, file_path=SAVE_FILE_PATH)
        print("saving cleaned data...")
        logging.info(f"Saved Cleaned data in {SAVE_FILE_PATH}")

    else: 
        logging.warning(f"Data cleaning incomplete. Remaining nulls: {null_count}")
        print(f"\nWarning: {null_count} null values remaining.")