import os 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
Y_TRAIN_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "y_train_processed.csv")
Y_TEST_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "y_test_processed.csv")