import os
import sys
import glob
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config, load_instance, append_results_to_csv
from src.solvers.classic_solvers import solve_sa
from src.metrics.metrics import compute_gap

def parse_filename(filepath):
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    dataset = parts[1]
    N = int(parts[2])
    instance_id = int(parts[3].split(".")[0])
    return (dataset, N, instance_id, filepath)

def main():
    config = load_config()
    num_reads = config['solvers']['sa']['num_reads']
    num_sweeps = config['solvers']['sa']['num_sweeps']
    
    files = glob.glob("data/instances/instance_*.pkl")
    if not files:
        print("No instance files found. Please run generate_instances.py first.")
        sys.exit(1)
        
    sorted_instances = sorted([parse_filename(f) for f in files], key=lambda x: (x[0], x[1], x[2]))
    
    csv_path = "results/results.csv"
    
    # Load Gurobi objectives for GAP calculations
    gurobi_objs = {}
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            gurobi_df = df[df['solver'] == 'gurobi']
            for _, row in gurobi_df.iterrows():
                key = (row['dataset'], int(row['N']), int(row['instance_id']))
                gurobi_objs[key] = float(row['objective'])
        except Exception as e:
            print(f"Warning: Could not read Gurobi objectives from results.csv: {e}")
            
    print(f"Running Simulated Annealing (reads={num_reads}, sweeps={num_sweeps}) on all instances...")
    
    for dataset, N, instance_id, filepath in sorted_instances:
        print(f"Solving instance {dataset} (N={N}, ID={instance_id})...")
        instance = load_instance(filepath)
        
        # Run Simulated Annealing
        result = solve_sa(instance, num_reads=num_reads, num_sweeps=num_sweeps)
        
        # Calculate relative GAP compared to Gurobi
        g_obj = gurobi_objs.get((dataset, N, instance_id), None)
        result['gap'] = compute_gap(result['objective'], g_obj)
        
        # Save results
        append_results_to_csv(result, csv_path)
        
    print(f"Simulated Annealing experiments completed. Results saved to {csv_path}.")

if __name__ == "__main__":
    main()
