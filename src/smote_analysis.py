import os 
import sys 
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.data_utils import load_dataset, save_dataset, separate_cols_type
from src.logger import logging
from imblearn.over_sampling import SMOTENC
import pandas as pd

X_FILE_PATH = os.path.join(project_root, "data", "processed", "X.csv")
Y_FILE_PATH = os.path.join(project_root, "data", "processed", "y.csv")

# 1. load the labels and features
print("1. loading features and labels...")
X = load_dataset(file_path=X_FILE_PATH)
y = load_dataset(file_path=Y_FILE_PATH)

if X is not None and y is not None:
    # Ensure y is 1D Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # print value counts
    print("\nBefore Resampling :")
    print(f"\nX shape: {X.shape}\ny shape: {y.shape}")
    print("value count of yes(1) and no(0):\n")
    print(y.value_counts().reset_index())

    # SMOTE
    print("\nImplementing SNOTE...")
    cat_cols, _ = separate_cols_type(data=X)
    sm = SMOTENC(random_state=42, categorical_features=cat_cols, sampling_strategy="auto")
    X_res, y_res = sm.fit_resample(X=X, y=y)
    print("\nAfter Resampling: ")
    print(f"X shape: {X_res.shape}\ny shape: {y_res.shape}")
    print("Value counts of yes(1) and no(0): \n", y_res.value_counts().reset_index())
    logging.info("implemented SMOTE successfully")
    
    # verify if no row mistmatch
    if X_res.shape[0] != y_res.shape[0]:
        print("Error! Rows mismatched")
        logging.error("Rows Mismatched between X and y")
    else:
        print("No row mismatch")
        try:
            # save labels and features
            print("saving resampled X and y ....")
            os.makedirs(os.path.dirname(os.path.join(project_root, "data", "processed", "X_res.csv")), exist_ok=True)
            save_dataset(data=X_res, file_path=os.path.join(project_root, "data", "processed", "X_res.csv"))
            print("saved X")
            os.makedirs(os.path.dirname(os.path.join(project_root, "data", "processed", "y_res.csv")), exist_ok=True)
            
            # Ensure y_res is 1D before saving
            y_res_final = y_res.values.ravel() if hasattr(y_res, "values") else y_res
            save_dataset(data=pd.Series(y_res_final, name="Churn"), file_path=os.path.join(project_root, "data", "processed", "y_res.csv"))
            print("saved y")
            logging.info("X and Y saved. SMOTE succesful.")
        except Exception as e:
            print(f"{e}")
            logging.error(f"{str(e)}")

else:
    print("Error! failed to load X and Y")
    logging.info("X and Y failed to load")