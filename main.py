"""
Machine Learning pipeline orchestration script
Executes data processing, analysis, and modeling steps in sequence
"""

import os
import subprocess
import sys

def run_script(script_name: str) -> None:
    """
    Execute a Python script with error handling and output capture
    
    Args:
        script_name: Name of the script to execute (must be in scripts/ directory)
    
    Raises:
        SystemExit: If subprocess returns non-zero exit code
    """
    script_path = os.path.join('scripts', script_name)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            capture_output=True
        )
        print(f"\n✅ {script_name} executed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing {script_name}:")
        print(e.stderr)
        sys.exit(1)

def main() -> None:
    """Main pipeline execution workflow"""
    # Create output directories
    os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
    
    # Define execution order of pipeline components
    pipeline_scripts = [
        "data_processing.py",
        "exploratory_analysis.py",
        "tsne_visualization.py", 
        "knn_classification.py",
        "generate_plots.py"
    ]

    print("🚀 Starting Machine Learning pipeline...")
    
    # Execute each stage sequentially
    for script in pipeline_scripts:
        print(f"\n{'='*50}")
        print(f"⚙️  Running: {script}")
        run_script(script)

    # Final output message
    print("\n" + "="*50)
    print("🎉 Pipeline executed successfully!")
    print("Results available in:")
    print(f"- Visualizations:   {os.path.join('results', 'plots')}")
    print(f"- Processed data:   {os.path.join('results', 'flattened_data.pkl')}")
    print(f"- KNN metrics:      {os.path.join('results', 'knn_results.json')}")

if __name__ == "__main__":
    main()