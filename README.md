# Apollo Solutions Machine Learning Developer Test

A comprehensive pipeline for analyzing genetic syndrome embeddings using machine learning techniques.

## 📋 Table of Contents
- [Installation](#installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Script Documentation](#-script-documentation)
- [Output Examples](#-output-examples)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apollo-genetic-analysis.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. 📁 Project Structure
apollo_test/
├── data/
│   └── mini_gm_public_v0.1.p         # Raw data file
├── results/
│   ├── plots/                        # Generated visualizations
│   ├── flattened_data.pkl            # Processed data
│   └── knn_results.json              # Classification metrics
├── scripts/
│   ├── data_processing.py            # Data loading/preprocessing
│   ├── eda.py                        # Exploratory analysis
│   ├── tsne_visualization.py         # Dimensionality reduction
│   ├── knn_classification.py         # KNN implementation
│   ├── generate_plots.py             # Metric visualizations
│   └── generate_report.py            # PDF report generation
├── main.py                           # Main pipeline controller
├── requirements.txt                  # Dependency list
└── README.md                         # This file
