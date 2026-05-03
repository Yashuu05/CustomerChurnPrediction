from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numerical_cols)
    ])

    return full_pipeline