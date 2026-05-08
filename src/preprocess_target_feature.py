# preprocess the target feature
import os 
import sys 
import pandas as pd

# Fix ImportError by ensuring project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.paths import CLEANED_DATA_PATH, Y_TEST_SAVE_PATH, Y_TRAIN_SAVE_PATH
from utils.data_utils import load_dataset, save_dataset, prepare_X_y, split_dataset
from sklearn.preprocessing import LabelEncoder
from src.logger import logging


if __name__ == "__main__":
    # 1: load resampled target and test target
    logging.info("loading resampled and test labels")
    print("=========== Preprocessing Initialized =============")
    
    Y_TRAIN_RES_PATH = os.path.join(project_root, "data", "processed", "y_train_res.csv")
    Y_TEST_PATH = os.path.join(project_root, "data", "processed", "y_test.csv")
    
    y_train_res = load_dataset(file_path=Y_TRAIN_RES_PATH)
    y_test = load_dataset(file_path=Y_TEST_PATH)
    
    if y_train_res is not None and y_test is not None:
        logging.info("preprocessing target features using LabelEncoder")
        le = LabelEncoder()
        
        # Flatten to 1D
        y_train_arr = le.fit_transform(y_train_res.values.ravel())
        y_test_arr = le.transform(y_test.values.ravel())
        
        # Convert back to pandas Series for saving
        y_train_preprocessed = pd.Series(y_train_arr, name="Churn")
        y_test_preprocessed = pd.Series(y_test_arr, name="Churn")
        
        logging.info("saving preprocessed target features")
        save_dataset(data=y_train_preprocessed, file_path=Y_TRAIN_SAVE_PATH)
        save_dataset(data=y_test_preprocessed, file_path=Y_TEST_SAVE_PATH)
        
        print(f"y_train_processed shape: {y_train_preprocessed.shape}")
        print(f"y_test_processed shape: {y_test_preprocessed.shape}")
        print("=========== Preprocessing finished ===============")
    else:
        logging.error("Could not load resampled datasets")
        print("Dataset couldn't be loaded")