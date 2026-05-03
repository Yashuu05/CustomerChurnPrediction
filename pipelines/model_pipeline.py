from configs.models_config import models
from pipelines.pipeline import build_pipeline
from sklearn.pipeline import Pipeline

def build_model_pipelines(cat_cols: list, num_cols: list):
    """
    Creates a dictionary of pipelines for each model in models_config.
    Each pipeline includes preprocessing followed by the estimator.
    """
    # Get the shared preprocessing pipeline
    preprocessing_pipeline = build_pipeline(categorical_cols=cat_cols, numerical_cols=num_cols)
    
    pipelines = {}
    
    # Dynamically build pipelines for all models
    for model_name, model_instance in models.items():
        pipelines[model_name] = Pipeline(steps=[
            ('prep', preprocessing_pipeline),
            ('model', model_instance)
        ])

    return pipelines