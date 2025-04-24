"""
Visualization script for KNN performance analysis
Generates metrics tables, AUC comparisons, and multiclass ROC curves
"""

import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def generate_metrics_table(results_data, output_dir):
    """
    Generates markdown table with complete performance metrics
    Args:
        results_data (dict): Loaded experiment results
        output_dir (str): Directory path for saving outputs
    """
    metrics = []
    
    # Process each configuration
    for config in results_data:
        if '_k' not in config:
            continue
            
        try:
            metric, k = config.split('_k')
            entry = {
                'Metric': metric.capitalize(),
                'k': int(k),
                'AUC': round(results_data[config]['AUC'], 2),
                'F1-Score': round(results_data[config]['F1'], 2)
            }
            metrics.append(entry)
        except (KeyError, ValueError):
            continue
    
    # Create and save table
    df = pd.DataFrame(metrics)
    table_path = os.path.join(output_dir, 'performance_metrics.md')
    df.to_markdown(table_path, index=False)
    print(f"Metrics table saved to: {table_path}")

def plot_multiclass_roc(experiment_data, plots_dir):
    """
    Generates multiclass ROC curves using One-vs-Rest approach
    for the best performing configuration
    """
    # Find best configuration by AUC
    best_config = None
    max_auc = 0
    for config in experiment_data:
        try:
            if '_k' in config and experiment_data[config]['AUC'] > max_auc:
                best_config = config
                max_auc = experiment_data[config]['AUC']
        except KeyError:
            continue

    if not best_config:
        print("No valid configurations found for ROC curves")
        return

    # Load data for best configuration
    try:
        all_y_true = np.concatenate([np.array(fold) for fold in experiment_data[best_config]['True_Labels']])
        all_y_proba = np.concatenate([np.array(fold) for fold in experiment_data[best_config]['Probabilities']])
    except KeyError as e:
        print(f"Missing data in best configuration: {str(e)}")
        return

    # Get class information
    classes = np.unique(all_y_true)
    n_classes = len(classes)
    
    # Plot settings
    plt.figure(figsize=(10, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Plot ROC for each class
    for i, color in zip(range(n_classes), colors):
        # Binarize labels for current class
        y_true_class = (all_y_true == classes[i]).astype(int)
        fpr, tpr, _ = roc_curve(y_true_class, all_y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Format plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curves ({best_config})\nTotal AUC OvR: {max_auc:.2f}')
    plt.legend(loc="lower right")
    
    # Save output
    output_path = os.path.join(plots_dir, 'multiclass_roc.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multiclass ROC curves saved to: {output_path}")

def plot_metric_comparison_roc(experiment_data, plots_dir):
    """
    Generates comparative ROC curves for both metrics using One-vs-Rest approach
    """
    plt.figure(figsize=(10, 8))
    
    # Configuration parameters
    metrics = ['cosine', 'euclidean']
    colors = ['#1f77b4', '#ff7f0e']
    linestyles = ['-', '--']
    
    for metric, color, ls in zip(metrics, colors, linestyles):
        # Find best k for current metric
        best_k = 0
        best_auc = 0
        for k in range(1, 16):
            config = f'{metric}_k{k}'
            try:
                current_auc = experiment_data[config]['AUC']
                if current_auc > best_auc:
                    best_k = k
                    best_auc = current_auc
            except KeyError:
                continue
        
        if best_k == 0:
            continue
            
        # Aggregate data across folds
        config = f'{metric}_k{best_k}'
        try:
            y_true = np.concatenate([np.array(fold) for fold in experiment_data[config]['True_Labels']])
            y_proba = np.concatenate([np.array(fold) for fold in experiment_data[config]['Probabilities']])
        except KeyError:
            continue
        
        # Binarize labels for OvR
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Compute micro-average ROC
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, linestyle=ls, lw=2,
                 label=f'{metric.capitalize()} (k={best_k}, AUC={roc_auc:.2f})')

    # Plot reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Format plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve Comparison (One-vs-Rest)')
    plt.legend(loc="lower right")
    
    # Save output
    output_path = os.path.join(plots_dir, 'metric_comparison_roc.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metric comparison ROC saved to: {output_path}")

def plot_auc_comparison():
    """
    Main visualization pipeline
    Generates all analysis plots and tables
    """
    # Configure paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Load experiment results
    try:
        with open(os.path.join(results_dir, 'knn_results.json'), 'r') as f:
            experiment_data = json.load(f)
    except FileNotFoundError:
        print("Error: Results file not found. Run knn_classification.py first.")
        return

    # Generate outputs
    generate_metrics_table(experiment_data, results_dir)
    plot_multiclass_roc(experiment_data, plots_dir)
    plot_metric_comparison_roc(experiment_data, plots_dir)

    # Create AUC comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot settings
    metrics = ['cosine', 'euclidean']
    markers = ['o', 's']
    
    for metric, marker in zip(metrics, markers):
        auc_values = []
        k_values = []
        
        # Collect data for metric
        for k in range(1, 16):
            config = f'{metric}_k{k}'
            try:
                auc_values.append(experiment_data[config]['AUC'])
                k_values.append(k)
            except KeyError:
                continue
        
        if auc_values:
            plt.plot(k_values, auc_values, marker+'-', 
                    label=f'{metric.capitalize()} Metric')

    # Format plot
    plt.title('AUC Performance Comparison Across Metrics')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average AUC (One-vs-Rest)')
    plt.xticks(range(1, 16))
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(plots_dir, 'auc_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_auc_comparison()