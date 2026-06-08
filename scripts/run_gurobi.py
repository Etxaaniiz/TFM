import os
import sys
import glob
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config, load_instance, append_results_to_csv
from src.solvers.classic_solvers import solve_gurobi, GUROBI_AVAILABLE

def parse_filename(filepath):
    # e.g., "data/instances/instance_validation_6_0.pkl" -> ("validation", 6, 0)
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    dataset = parts[1]
    N = int(parts[2])
    instance_id = int(parts[3].split(".")[0])
    return (dataset, N, instance_id, filepath)

def main():
    if not GUROBI_AVAILABLE:
        print("Error: Gurobi is not installed or not available in the current environment.")
        sys.exit(1)
        
    config = load_config()
    lambda_val = config['portfolio']['lambda_val']
    
    # Locate all instance files
    files = glob.glob("data/instances/instance_*.pkl")
    if not files:
        print("No instance files found. Please run generate_instances.py first.")
        sys.exit(1)
        
    # Sort files logically: dataset, N, instance_id
    sorted_instances = sorted([parse_filename(f) for f in files], key=lambda x: (x[0], x[1], x[2]))
    
    print(f"Running Gurobi Solver on {len(sorted_instances)} instances...")
    
    # Overwrite results.csv if it exists (since Gurobi is the reference and runs first)
    csv_path = "results/results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    for dataset, N, instance_id, filepath in sorted_instances:
        print(f"Solving instance {dataset} (N={N}, ID={instance_id})...")
        instance = load_instance(filepath)
        
        # Run Gurobi solver
        result = solve_gurobi(instance, lambda_val=lambda_val)
        
        # Append result to CSV
        append_results_to_csv(result, csv_path)
        
    print(f"Gurobi experiments completed. Results saved to {csv_path}.")

if __name__ == "__main__":
    main()
