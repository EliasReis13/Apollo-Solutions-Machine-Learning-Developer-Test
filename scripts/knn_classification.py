"""
KNN evaluation script for genetic syndrome classification using embeddings
"""

import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
import numpy as np
import pandas as pd
import os

# Configure base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
import numpy as np
import pandas as pd
import json
import os

def evaluate_knn(X, y, metric='cosine', k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    metrics = {
        'AUC': [],
        'F1': [],
        'Top3_Accuracy': [],
        'True_Labels': [],
        'Probabilities': []
    }
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        knn.fit(X_train, y_train)
        y_proba = knn.predict_proba(X_test)
        
        # Store data for ROC curves
        metrics['True_Labels'].append(y_test.tolist())  # Convert numpy array to list
        metrics['Probabilities'].append(y_proba.tolist())  # Convert numpy array to list
        
        # Calculate metrics
        metrics['AUC'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        metrics['F1'].append(f1_score(y_test, knn.predict(X_test), average='weighted'))
        metrics['Top3_Accuracy'].append(top_k_accuracy_score(y_test, y_proba, k=3))
    
    return {
        'AUC': np.mean(metrics['AUC']),
        'F1': np.mean(metrics['F1']),
        'Top3_Accuracy': np.mean(metrics['Top3_Accuracy']),
        'True_Labels': metrics['True_Labels'],
        'Probabilities': metrics['Probabilities']
    }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, 'results', 'flattened_data.pkl')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data not found: {input_path}")
    
    df = pd.read_pickle(input_path)
    X = np.stack(df['embedding'].values)
    y = df['syndrome_id'].values
    
    results = {}
    for metric in ['cosine', 'euclidean']:
        for k in range(1, 16):
            key = f'{metric}_k{k}'
            results[key] = evaluate_knn(X, y, metric=metric, k=k)
            print(f"Completed: {key}")
    
    output_path = os.path.join(BASE_DIR, 'results', 'knn_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")