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


# 1: load dataset 
logging.info("loading dataset")
print("=========== Preprocessing Initialized =============")
df = load_dataset(file_path=CLEANED_DATA_PATH)

if df is not None:
    logging.info("preparing labels and target feature")
    X, y = prepare_X_y(data=df, cols_to_drop=['customerID'], target="Churn")
    
    logging.info("splitting dataset...")
    _, _, y_train, y_test = split_dataset(randomState=42, testSize=0.20, X=X, y=y)
    
    logging.info("preprocessing target features")
    le = LabelEncoder()
    # LabelEncoder returns numpy arrays
    y_train_arr = le.fit_transform(y=y_train)
    y_test_arr = le.transform(y=y_test)

    # Convert back to pandas Series/DataFrame for save_dataset compatibility
    y_train_preprocessed = pd.Series(y_train_arr, name="Churn")
    y_test_preprocessed = pd.Series(y_test_arr, name="Churn")

    logging.info("saving preprocessed target features")
    save_dataset(data=y_train_preprocessed, file_path=Y_TRAIN_SAVE_PATH)
    save_dataset(data=y_test_preprocessed, file_path=Y_TEST_SAVE_PATH)

    print("=========== Preprocessing finished ===============")
else:
    logging.error("Could not load dataset")
    print("Dataset couldn't be loaded")