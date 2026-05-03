from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm

models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(),
    "lightgbm": lightgbm.LGBMClassifier()
}