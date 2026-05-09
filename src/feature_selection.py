import pandas as pd
import os
import sys 
import json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.logger import logging
from utils.data_utils import save_dataset, load_dataset
SAVE_FILE_PATH = os.path.join(project_root, "outputs", "perm_imp.csv")

def get_low_importance_features(importance_df, threshold=0.0):
    """
    Identifies features with importance scores at or below the threshold.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        threshold (float): Importance threshold. Features <= threshold will be returned.
        
    Returns:
        pd.DataFrame: Features to consider dropping.
    """
    low_importance = importance_df[importance_df['Importance'] <= threshold]
    logging.info(f"Identified {len(low_importance)} features with importance <= {threshold}")
    return low_importance

def suggest_original_feature_drops(importance_df):
    """
    Groups one-hot encoded features and suggests original features to drop 
    if all their categories have low importance.
    """
    # Extract original feature names (assuming 'cat__Feature_Category' format)
    importance_df['OriginalFeature'] = importance_df['Feature'].apply(
        lambda x: x.split('__')[1].split('_')[0] if '__' in x else x.split('__')[-1]
    )
    
    feature_stats = importance_df.groupby('OriginalFeature')['Importance'].agg(['max', 'mean']).reset_index()
    
    # Suggest dropping if the maximum importance across all categories is very low
    suggested_drops = feature_stats[feature_stats['max'] <= 0.001]
    
    return suggested_drops

if __name__ == "__main__":
    df = load_dataset(file_path=os.path.join(project_root, "outputs", "feature_importances.csv"))
    print("features with low importance:")
    result = get_low_importance_features(importance_df=df, threshold=0.005)
    print(result)
    
    print("\nSuggested Drops")
    suggested_drops = suggest_original_feature_drops(importance_df=df)
    print(suggested_drops)
    
    # Save features with low importance in congig/
    CONFIG_PATH = os.path.join(project_root, "configs", "suggested_drops.json")
    drops_list = result['Feature'].tolist()
    with open(CONFIG_PATH, 'w') as f:
        json.dump(drops_list, f, indent=4)
    logging.info(f"Saved suggested drops to {CONFIG_PATH}")