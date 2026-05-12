import pandas as pd
from sklearn.metrics import silhouette_score
import json
import os
from src.logger import logging

def evaluate_segmentation(X, labels, output_path, df=None):
    """
    Evaluate DBSCAN segmentation using silhouette score and save results.
    If df is provided, compare with target_segment.
    """
    try:
        unique_labels = set(labels)
        if len(unique_labels) > 1 and not (len(unique_labels) == 1 and -1 in unique_labels):
            score = silhouette_score(X, labels)
            logging.info(f"Silhouette Score: {score}")
        else:
            score = -1
            logging.warning("Silhouette score could not be calculated.")

        results = {
            "silhouette_score": score,
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "num_noise_points": list(labels).count(-1)
        }

        if df is not None and 'target_segment' in df.columns and 'customer_segment' in df.columns:
            # Simple overlap check
            comparison = pd.crosstab(df['target_segment'], df['customer_segment'])
            results['crosstab'] = comparison.to_dict()
            logging.info("Crosstab comparison with target logic completed.")

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Evaluation results saved to {output_path}")
        return results

    except Exception as e:
        logging.error(f"Error in evaluation: {str(e)}")
        print(f"Error in evaluation: {str(e)}")
        return None
