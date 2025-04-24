"""
Data visualization script for analyzing genetic syndrome dataset distribution
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define base directory path for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_statistics(df):
    """
    Generate summary statistics from processed data
    Args:
        df (pd.DataFrame): Input dataframe with syndrome data
    Returns:
        dict: Dictionary containing:
            - num_syndromes: Number of unique syndromes
            - num_subjects: Number of unique subjects
            - images_per_syndrome: Dictionary of image counts per syndrome
    """
    return {
        'num_syndromes': df['syndrome_id'].nunique(),
        'num_subjects': df['subject_id'].nunique(),
        'images_per_syndrome': df.groupby('syndrome_id').size().to_dict()
    }

def plot_class_distribution(stats, save_path):
    """
    Create and save bar plot of image distribution across syndromes
    Args:
        stats (dict): Statistics dictionary from generate_statistics
        save_path (str): Full path for saving the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.bar(stats['images_per_syndrome'].keys(), stats['images_per_syndrome'].values())
    plt.title('Image Distribution by Syndrome')
    plt.xlabel('Syndrome ID')
    plt.ylabel('Number of Images')
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    """Main visualization pipeline"""
    
    # Load processed data
    input_path = os.path.join(BASE_DIR, 'results', 'flattened_data.pkl')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_pickle(input_path)
    stats = generate_statistics(df)
    
    # Create visualization directory and save plot
    plots_dir = os.path.join(BASE_DIR, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_class_distribution(stats, os.path.join(plots_dir, 'class_distribution.png'))
    print(f"Visualization saved to {os.path.join(plots_dir, 'class_distribution.png')}")