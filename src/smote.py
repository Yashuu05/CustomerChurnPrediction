"""
implementing SMOTE on imbalanced cleaned dataset
"""

import os
import sys
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from imblearn.over_sampling import SMOTE
import pandas as pd
from src.logger import logging
from utils.data_utils import load_dataset, save_dataset, split_dataset, prepare_X_y

from configs.paths import CLEANED_DATA_PATH, Y_TRAIN_SAVE_PATH
SAVE_X = os.path.join(project_root, "data", "processed", "X_train_res.csv")
SAVE_Y = os.path.join(project_root, "data", "processed", "y_train_res.csv")

def perform_smote(Xtrain, Ytrain):
    """
    This function performs SMOTE to increase the number of minority class
    from dataset to avoid overfitting of model
    Input: imbalanced training dataset
    output: balaned training dataset
    """

    print(f"before oversampling, counts of label '1': {sum(Ytrain == 1)}")
    print(f"before oversampling, counts of label '0' : {sum(Ytrain == 0)}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(Xtrain, Ytrain)

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

    return X_train_res, y_train_res


if __name__ == "__main__":
    print("Intialized ")
    logging.info("Initialised SMOTE")
    logging.info("loading dataset")
    df = load_dataset(file_path=CLEANED_DATA_PATH)
    
    if df is not None:
        logging.info(f"{CLEANED_DATA_PATH} Dataset Splitting")
        X, y = prepare_X_y(data=df, cols_to_drop=['customerID'], target="Churn")
        
        logging.info("Encoding categorical variables for the entire dataset")
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Split encoded data
        X_train_encoded, X_test_encoded, y_train, y_test = split_dataset(randomState=42, testSize=0.20, X=X_encoded, y=y)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        logging.info("Performing SMOTE on training set")
        X_train_res, y_train_res_encoded = perform_smote(Xtrain=X_train_encoded, Ytrain=y_train_encoded)
        
        # Decode target back to strings
        y_train_res = le.inverse_transform(y_train_res_encoded)

        logging.info("SMOTE successfull. Saving balanced data.")
        
        # Paths for new files
        X_TRAIN_RES_PATH = os.path.join(project_root, "data", "processed", "X_train_res.csv")
        Y_TRAIN_RES_PATH = os.path.join(project_root, "data", "processed", "y_train_res.csv")
        X_TEST_PATH = os.path.join(project_root, "data", "processed", "X_test.csv")
        Y_TEST_PATH = os.path.join(project_root, "data", "processed", "y_test.csv")
        
        os.makedirs(os.path.dirname(X_TRAIN_RES_PATH), exist_ok=True)
        save_dataset(data=X_train_res, file_path=X_TRAIN_RES_PATH)
        save_dataset(data=pd.Series(y_train_res, name="Churn"), file_path=Y_TRAIN_RES_PATH)
        
        # Save X_test as encoded as well to match X_train columns
        save_dataset(data=X_test_encoded, file_path=X_TEST_PATH)
        save_dataset(data=pd.Series(y_test, name="Churn"), file_path=Y_TEST_PATH)
        
        logging.info("save successfull.")
        print("Terminated")