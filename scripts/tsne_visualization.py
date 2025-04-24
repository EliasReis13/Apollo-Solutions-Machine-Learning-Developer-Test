"""
t-SNE visualization script for genetic syndrome embeddings analysis
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define base directory path for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def perform_tsne(embeddings):
    """
    Perform t-SNE dimensionality reduction on embeddings
    Args:
        embeddings (np.ndarray): Array of embeddings with shape (n_samples, n_features)
    Returns:
        np.ndarray: 2D t-SNE transformed coordinates
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    return tsne.fit_transform(embeddings)

def plot_tsne(tsne_results, labels, save_path):
    """
    Create and save t-SNE visualization plot
    Args:
        tsne_results (np.ndarray): 2D t-SNE coordinates
        labels (pd.Series): Category labels for coloring points
        save_path (str): Full path for saving the visualization
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Syndromes")
    plt.title('t-SNE Visualization of Genetic Syndrome Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    """Main visualization pipeline"""
    
    # Load processed data
    input_path = os.path.join(BASE_DIR, 'results', 'flattened_data.pkl')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    df = pd.read_pickle(input_path)
    embeddings = np.stack(df['embedding'].values)
    
    # Generate t-SNE visualization
    tsne_results = perform_tsne(embeddings)
    
    # Save visualization
    plots_dir = os.path.join(BASE_DIR, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'tsne_plot.png')
    
    plot_tsne(tsne_results,
             df['syndrome_id'].astype('category').cat.codes,
             output_path)
    
    print(f"t-SNE visualization saved to {output_path}")