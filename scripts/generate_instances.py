import os
import sys
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config, save_instance
from src.portfolio.portfolio_model import build_qubo

def main():
    # Load configuration
    config = load_config()
    
    processed_dir = config['data']['processed_path']
    mu_path = os.path.join(processed_dir, "mu.csv")
    Sigma_path = os.path.join(processed_dir, "Sigma.csv")
    
    # Check if processed data exists
    if not os.path.exists(mu_path) or not os.path.exists(Sigma_path):
        print("Error: Processed return statistics not found. Please run prepare_data.py first.")
        sys.exit(1)
        
    # Load expected returns and covariance matrix
    mu_all = pd.read_csv(mu_path, index_col=0).squeeze("columns")
    Sigma_all = pd.read_csv(Sigma_path, index_col=0)
    
    all_tickers = list(mu_all.index)
    print(f"Loaded {len(all_tickers)} valid tickers from statistics files.")
    
    lambda_val = config['portfolio']['lambda_val']
    seeds = config['experiments']['seeds'] # [42, 43, 44, 45, 46]
    
    # Instance configurations from default_config.yaml
    inst_config = config['experiments']['instances']
    
    datasets = {
        "validation": inst_config['validation'],
        "principal": inst_config['principal'],
        "scalability": inst_config['scalability']
    }
    
    os.makedirs("data/instances", exist_ok=True)
    
    total_instances = 0
    # Generate instances for each dataset category
    for dataset_name, dataset_params in datasets.items():
        Ns = dataset_params['N']
        Ks = dataset_params['K']
        
        for N, K in zip(Ns, Ks):
            print(f"Generating instances for dataset '{dataset_name}' with N={N}, K={K}...")
            
            for i, seed in enumerate(seeds):
                # Set seed for reproducible random selection
                np.random.seed(seed)
                
                # Randomly select N tickers without replacement
                selected_tickers = sorted(list(np.random.choice(all_tickers, size=N, replace=False)))
                
                # Subset returns and covariance
                mu_sub = mu_all.loc[selected_tickers]
                Sigma_sub = Sigma_all.loc[selected_tickers, selected_tickers]
                
                # Build QUBO matrix (P multiplier calculated inside build_qubo)
                # We want the unpenalized max Q0_ij to calculate penalty P
                N_assets = len(selected_tickers)
                Q0 = np.zeros((N_assets, N_assets))
                for idx_i in range(N_assets):
                    Q0[idx_i, idx_i] = lambda_val * Sigma_sub.iloc[idx_i, idx_i] / (K ** 2) - (1.0 - lambda_val) * mu_sub.iloc[idx_i] / K
                    for idx_j in range(idx_i + 1, N_assets):
                        val = lambda_val * Sigma_sub.iloc[idx_i, idx_j] / (K ** 2)
                        Q0[idx_i, idx_j] = val / 2.0
                        Q0[idx_j, idx_i] = val / 2.0
                
                # Heuristic penalty P = 10 * max(|Q0_ij|)
                P = 10.0 * np.max(np.abs(Q0))
                
                # Build the full penalized symmetric QUBO matrix
                Q = build_qubo(mu_sub, Sigma_sub, K, lambda_val, penalty=P)
                
                # Save instance dict
                instance = {
                    "dataset": dataset_name,
                    "N": N,
                    "K": K,
                    "instance_id": i,
                    "seed": seed,
                    "tickers": selected_tickers,
                    "mu": mu_sub.to_numpy(),
                    "Sigma": Sigma_sub.to_numpy(),
                    "Q": Q,
                    "penalty": P,
                    "offset": P * (K ** 2),
                    "lambda_val": lambda_val
                }
                
                filename = f"data/instances/instance_{dataset_name}_{N}_{i}.pkl"
                save_instance(instance, filename)
                total_instances += 1
                
    print(f"Finished generating {total_instances} instances in data/instances/.")

if __name__ == "__main__":
    main()
