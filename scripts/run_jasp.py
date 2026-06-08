import os
import sys
import glob
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config, load_instance, append_results_to_csv
from src.quantum.quantum_solvers import solve_jasp
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
    p_list = config['solvers']['jasp_qaoa']['p'] # [2, 3, 4]
    maxiter = config['solvers']['jasp_qaoa']['maxiter']
    shots = config['solvers']['jasp_qaoa']['shots']
    
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
            
    print(f"Running JaspQAOA (JAX-compiled, maxiter={maxiter}, shots={shots}) on N in [10, 12, 14, 16]...")
    
    count = 0
    for dataset, N, instance_id, filepath in sorted_instances:
        # JaspQAOA runs on N in [10, 12, 14, 16] as specified in parameters
        if N not in [10, 12, 14, 16]:
            continue
            
        print(f"Solving instance {dataset} (N={N}, ID={instance_id}) with JaspQAOA...")
        instance = load_instance(filepath)
        
        # Run JaspQAOA for each depth p in configuration
        for p in p_list:
            print(f"  -> Depth p = {p}")
            result = solve_jasp(instance, p=p, maxiter=maxiter, shots=shots)
            
            # Compute relative GAP compared to Gurobi
            g_obj = gurobi_objs.get((dataset, N, instance_id), None)
            result['gap'] = compute_gap(result['objective'], g_obj)
            
            # Save results
            append_results_to_csv(result, csv_path)
            count += 1
            
    print(f"JaspQAOA experiments completed for {count} executions. Results appended to {csv_path}.")

if __name__ == "__main__":
    main()
