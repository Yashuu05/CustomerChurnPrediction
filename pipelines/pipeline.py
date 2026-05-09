from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified features from a DataFrame.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'drop'):
            return X.drop(columns=self.feature_names, errors='ignore')
        return X

def build_pipeline(categorical_cols, numerical_cols):

    """
    input: categorical and numerical columns of dataset
    output: full pipeline with standardization, imputation and encoding
    """

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))
    ])

    full_pipeline = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numerical_cols)
    ])

    # Set output to pandas to preserve column names for the next steps
    full_pipeline.set_output(transform="pandas")

    return full_pipeline