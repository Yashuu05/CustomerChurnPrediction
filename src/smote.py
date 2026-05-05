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

DATA_PATH = os.path.join(project_root, "data", "processed", "cleaned_data.csv")
SAVE_X = os.path.join(project_root, "data", "processed", "X_train_res.csv")
SAVE_Y = os.path.join(project_root, "data", "processed", "y_train_res.csv")
Y_PROCESSED_PATH = os.path.join(project_root, "data", "processed", "y_train_processed.csv")

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
    df = load_dataset(file_path=DATA_PATH)
    y_train = load_dataset(file_path=Y_PROCESSED_PATH)
    
    if df is not None:
        logging.info(f"{DATA_PATH} Dataset Splitting")
        X, y = prepare_X_y(data=df, cols_to_drop=['Churn', 'customerID'], target="Churn")
        
        logging.info("Encoding categorical variables for X_train")
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, _, _, _ = split_dataset(randomState=42, testSize=0.20, X=X, y=y)

        logging.info("Performing SMOTE")
        # Flatten y_train to 1D for SMOTE
        y_train_flat = y_train.values.ravel()
        X_train_res, y_train_res = perform_smote(Xtrain=X_train, Ytrain=y_train_flat)

        logging.info("SMOTE successfull. Saving balanced data.")
        os.makedirs(os.path.dirname(SAVE_X), exist_ok=True)
        save_dataset(data=X_train_res, file_path=SAVE_X)
        os.makedirs(os.path.dirname(SAVE_Y), exist_ok=True)
        save_dataset(data=pd.Series(y_train_res, name="Churn"), file_path=SAVE_Y)
        
        logging.info("save successfull.")
        print("Terminated")