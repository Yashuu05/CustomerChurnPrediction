import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.logger import logging

def plot_customer_segments(df, x_col, y_col, segment_col, output_path):
    """
    Visualize customer segments using a scatter plot.
    """
    try:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=segment_col, palette='viridis', style=segment_col)
        plt.title(f'Customer Segmentation: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"Segmentation plot saved to {output_path}")
        print(f"Success! Plot saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        print(f"Error in visualization: {str(e)}")
