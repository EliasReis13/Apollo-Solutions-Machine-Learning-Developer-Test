# Apollo Solutions Machine Learning Developer Test

A machine learning pipeline for analyzing genetic syndrome embeddings from image data. Developed for Apollo Solutions' ML Developer Practical Test.

## 📋 Table of Contents
- [Installation](#installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Script Documentation](#-script-documentation)
- [Output Examples](#-output-examples)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## Installation 📥

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apollo-genetic-analysis.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure 🏗️
```bash
genetic-syndrome-analysis/
├── data/
│   └── mini_gm_public_v0.1.p         # Raw dataset (embeddings)
├── results/
│   ├── plots/                        # Generated visualizations
│   │   ├── auc_comparison.png
│   │   ├── class_distribution.png
│   │   └── tsne_visualization.png
│   ├── flattened_data.pkl            # Processed dataset
│   └── knn_results.json              # Classification metrics
├── scripts/
│   ├── data_processing.py            # Data loading & preprocessing
│   ├── eda.py                        # Exploratory data analysis
│   ├── tsne_visualization.py         # Dimensionality reduction
│   ├── knn_classification.py         # KNN implementation
│   └── generate_plots.py             # Metric visualizations
├── main.py                           # Main pipeline controller
├── requirements.txt                  # Dependency list
└── README.md                         # This document
```

## Usage 🚦
### Full Pipeline Execution
```bash
python main.py
```
### Individual Components
```bash
# Data preprocessing
python scripts/data_processing.py

# Generate EDA visualizations
python scripts/eda.py

# Create t-SNE plot
python scripts/tsne_visualization.py

# Run KNN classification
python scripts/knn_classification.py

# Generate performance plots
python scripts/generate_plots.py
```
