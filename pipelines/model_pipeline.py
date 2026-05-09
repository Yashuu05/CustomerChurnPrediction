import os
import json
from configs.models_config import models
from pipelines.pipeline import build_pipeline, FeatureSelector
from sklearn.pipeline import Pipeline

def build_model_pipelines(cat_cols: list, num_cols: list):
    """
    Creates a dictionary of pipelines for each model in models_config.
    Each pipeline includes preprocessing followed by the estimator.
    """
    # Use absolute path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drops_path = os.path.join(project_root, 'configs', 'suggested_drops.json')
    
    # Load features to drop
    if os.path.exists(drops_path):
        with open(drops_path, 'r') as f:
            features_to_drop = json.load(f)
    else:
        features_to_drop = []

    # Get the shared preprocessing pipeline
    preprocessing_pipeline = build_pipeline(categorical_cols=cat_cols, numerical_cols=num_cols)
    
    pipelines = {}
    
    # Dynamically build pipelines for all models
    for model_name, model_instance in models.items():
        pipelines[model_name] = Pipeline(steps=[
            ('prep', preprocessing_pipeline),
            ('selector', FeatureSelector(features_to_drop)),
            ('model', model_instance)
        ])

    return pipelines