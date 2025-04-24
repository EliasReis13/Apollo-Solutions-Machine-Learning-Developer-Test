"""
Data processing script for flattening genetic syndrome embeddings data
"""
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import os

# Define base directory path for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_and_flatten_data():
    """
    Load nested pickle data and flatten into pandas DataFrame
    Returns:
        pd.DataFrame: Contains syndrome IDs, subject IDs, image IDs, and 320D embeddings
    """
    input_path = os.path.join(BASE_DIR, 'data', 'mini_gm_public_v0.1.p')
    
    # Validate input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Load nested dictionary structure from pickle
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Flatten nested structure into dictionary of lists
    flattened_data = defaultdict(list)
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                flattened_data['syndrome_id'].append(syndrome_id)
                flattened_data['subject_id'].append(subject_id)
                flattened_data['image_id'].append(image_id)
                flattened_data['embedding'].append(np.array(embedding))  # Convert to numpy array for consistency
    
    return pd.DataFrame(flattened_data)

def check_data_integrity(df):
    """
    Validate embedding dimensions in the DataFrame
    Args:
        df (pd.DataFrame): Input dataframe with embeddings column
    """
    assert all(len(emb) == 320 for emb in df['embedding']), "Incorrect embedding dimensions"
    print("Data integrity check completed.")

if __name__ == "__main__":
    """Main data processing pipeline"""
    
    output_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute data processing pipeline
    print("Processing raw data...")
    df = load_and_flatten_data()
    check_data_integrity(df)
    
    # Save processed data
    output_path = os.path.join(output_dir, 'flattened_data.pkl')
    df.to_pickle(output_path)
    print(f"Processed data saved to {output_path}")