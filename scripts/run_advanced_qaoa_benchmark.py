import os
import sys

# Resolve project root relative to script directory and change working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
sys.path.append(project_root)

# Force JAX to use CPU only to bypass incompatible PJRT CUDA plugins in Colab
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.utils.utils import load_config
from src.portfolio.portfolio_model import build_qubo
from src.solvers.classic_solvers import solve_gurobi, solve_sa, GUROBI_AVAILABLE
from src.quantum.quantum_solvers import create_QUBO_cl_cost_function, create_QUBO_cost_operator
from src.quantum.regularized_qaoa import RegularizedQAOAProblem
from src.metrics.metrics import calculate_portfolio_metrics, calculate_qubo_energy, compute_gap
from qrisp import QuantumVariable, dicke_state
from qrisp.qaoa import XY_mixer

def run_solver_instance_advanced(
    instance: Dict[str, Any],
    solver_type: str, # "gurobi", "sa", "xy_normal", "xy_regularized"
    p: int = 2,
    maxiter: int = 100,
    shots: int = 1024,
    alpha: float = 0.1,
    init_type: str = "tqa"
) -> Tuple[Dict[str, Any], int]:
    """
    Runs a specific solver configuration and returns metrics and the number of classical iterations/evaluations.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    start_time = time.perf_counter()
    
    if solver_type == "gurobi":
        res = solve_gurobi(instance, lambda_val=lambda_val)
        return res, 1
        
    elif solver_type == "sa":
        res = solve_sa(instance)
        # SA sweeps * reads represents total Monte Carlo evaluations
        return res, int(instance.get("sa_reads", 1000))
        
    elif solver_type in ["xy_normal", "xy_regularized"]:
        # Setup QuantumVariable
        qv = QuantumVariable(N)
        
        # Dicke state initialization function
        def init_dicke(q_var):
            from qrisp import x
            for i in range(K):
                x(q_var[i])
            dicke_state(q_var, K)
            
        prob_alpha = alpha if solver_type == "xy_regularized" else 0.0
        prob_init_type = "tqa" if solver_type == "xy_regularized" else "random"
        
        xy_qaoa_prob = RegularizedQAOAProblem(
            cost_operator=create_QUBO_cost_operator(Q),
            mixer=XY_mixer,
            cl_cost_function=create_QUBO_cl_cost_function(Q),
            init_function=init_dicke,
            alpha=prob_alpha
        )
        
        # Enable callback to track evaluations
        xy_qaoa_prob.callback = True
        xy_qaoa_prob.optimization_costs = []
        
        # Run optimization
        results_dict = xy_qaoa_prob.run(
            qarg=qv,
            depth=p,
            max_iter=maxiter,
            init_type=prob_init_type,
            mes_kwargs={"shots": shots}
        )
        
        end_time = time.perf_counter()
        runtime = end_time - start_time
        
        # Extract best solution
        best_bitstring = max(results_dict, key=results_dict.get)
        x_sol = np.array([int(bit) for bit in best_bitstring])
        
        # Compute metrics
        metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
        energy = calculate_qubo_energy(x_sol, Q) + offset
        
        tickers_list = instance.get('tickers', [])
        selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
        num_selected = int(np.sum(x_sol))
        
        # Number of classical iterations (function evaluations) is the length of recorded costs
        num_iterations = len(xy_qaoa_prob.optimization_costs)
        
        results = {
            "solver": solver_type,
            "N": N,
            "K": K,
            "p": p,
            "alpha": prob_alpha,
            "init_type": prob_init_type,
            "objective": metrics['objective'],
            "energy": energy,
            "sharpe": metrics['sharpe'],
            "expected_return": metrics['expected_return'],
            "volatility": metrics['volatility'],
            "feasible": metrics['feasible'],
            "runtime_seconds": runtime,
            "selected_tickers": selected_tickers,
            "num_selected": num_selected,
            "solution": x_sol,
            "counts": results_dict
        }
        
        return results, num_iterations
        
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

def main():
    print("======================================================================")
    print("INICIANDO BENCHMARK AVANZADO: ESCALABILIDAD, ESTRÉS Y EFICIENCIA TEMPORAL")
    print("======================================================================")
    
    test_mode = "--test" in sys.argv
    if test_mode:
        print("MODO DE PRUEBA ACTIVO (Parametros mínimos)")
        
    # Load configuration
    config = load_config("configs/phase3_stress_config.yaml")
    stress_config = config['stress_test']
    regimes = stress_config['regimes']
    tickers_pool = stress_config['tickers']
    
    # Configure variables based on test_mode
    if test_mode:
        problem_sizes = [(6, 2)] # N=6, K=2
        regimes_to_run = ["stable"]
        seeds = [42]
        solvers_to_run = [
            ("gurobi", 0, 0.0),
            ("xy_normal", 1, 0.0),
            ("xy_regularized", 1, 0.1)
        ]
        maxiter = 2
        shots = 100
        sa_reads = 100
    else:
        # N=10 (K=3), N=15 (K=4), N=20 (K=5)
        problem_sizes = [(10, 3), (15, 4), (20, 5)]
        regimes_to_run = list(regimes.keys()) # stable, volatile, inflationary
        seeds = [42, 43, 44]
        solvers_to_run = [
            ("gurobi", 0, 0.0),
            ("sa", 0, 0.0),
            ("xy_normal", 2, 0.0),
            ("xy_regularized", 2, 0.05),
            ("xy_regularized", 2, 0.1)
        ]
        maxiter = 100
        shots = 1024
        sa_reads = 1000
        
    # Load raw prices
    raw_path = config['data']['raw_path']
    if not os.path.exists(raw_path):
        print(f"Error: Raw prices file not found at {raw_path}. Run prepare_data.py first.")
        sys.exit(1)
        
    prices_all = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    daily_returns = np.log(prices_all / prices_all.shift(1)).dropna()
    
    os.makedirs("results/qaoa_advanced_analysis", exist_ok=True)
    sweep_path = "results/qaoa_advanced_analysis/advanced_hyperparameters_sweep.csv"
    
    sweep_results = []
    total_runs = 0
    
    # Run loop
    for regime_key in regimes_to_run:
        regime_info = regimes[regime_key]
        regime_name = regime_info['name']
        print(f"\n==================================================")
        print(f"REGIMEN DE MERCADO: {regime_name.upper()}")
        print(f"==================================================")
        
        train_returns_reg = daily_returns.loc[regime_info['train_start']:regime_info['train_end']]
        test_returns_reg = daily_returns.loc[regime_info['test_start']:regime_info['test_end']]
        
        for N, K in problem_sizes:
            print(f"\n--- Dimension del problema: N={N}, K={K} ---")
            
            for seed in seeds:
                print(f"  -> Semilla: {seed}")
                np.random.seed(seed)
                selected_tickers = sorted(list(np.random.choice(tickers_pool, size=N, replace=False)))
                
                # Statistics
                mu = train_returns_reg[selected_tickers].mean() * 252
                Sigma = train_returns_reg[selected_tickers].cov() * 252
                
                # QUBO
                Q0 = np.zeros((N, N))
                for idx_i in range(N):
                    Q0[idx_i, idx_i] = 0.5 * Sigma.iloc[idx_i, idx_i] / (K ** 2) - 0.5 * mu.iloc[idx_i] / K
                    for idx_j in range(idx_i + 1, N):
                        val = 0.5 * Sigma.iloc[idx_i, idx_j] / (K ** 2)
                        Q0[idx_i, idx_j] = val / 2.0
                        Q0[idx_j, idx_i] = val / 2.0
                        
                P = 10.0 * np.max(np.abs(Q0))
                Q = build_qubo(mu.to_numpy(), Sigma.to_numpy(), K, 0.5, penalty=P)
                
                instance = {
                    "dataset": f"stress_{regime_key}",
                    "N": N,
                    "K": K,
                    "instance_id": total_runs,
                    "seed": seed,
                    "tickers": selected_tickers,
                    "mu": mu.to_numpy(),
                    "Sigma": Sigma.to_numpy(),
                    "Q": Q,
                    "penalty": P,
                    "offset": P * (K ** 2),
                    "lambda_val": 0.5,
                    "sa_reads": sa_reads
                }
                
                # Solve Gurobi baseline
                gurobi_res = None
                if GUROBI_AVAILABLE:
                    gurobi_res, _ = run_solver_instance_advanced(instance, "gurobi")
                    gurobi_obj = gurobi_res["objective"]
                    gurobi_bitstring = "".join([str(int(bit)) for bit in gurobi_res["solution"]])
                else:
                    sa_base, _ = run_solver_instance_advanced(instance, "sa")
                    gurobi_obj = sa_base["objective"]
                    gurobi_bitstring = "".join([str(int(bit)) for bit in sa_base["solution"]])
                    
                # Out-of-sample details for Gurobi
                w_g = (gurobi_res["solution"] if GUROBI_AVAILABLE else sa_base["solution"]) / K
                port_returns_g = test_returns_reg[selected_tickers].dot(w_g)
                ann_ret_g = np.mean(port_returns_g) * 252
                ann_vol_g = np.std(port_returns_g) * np.sqrt(252)
                sharpe_g = ann_ret_g / ann_vol_g if ann_vol_g > 1e-9 else 0.0
                
                # Loop through solvers
                for solver_name, p_val, alpha_val in solvers_to_run:
                    if solver_name == "gurobi":
                        sweep_results.append({
                            "N": N, "K": K, "regime": regime_key, "seed": seed, "solver": "gurobi",
                            "p": None, "alpha": None, "in_sample_obj": gurobi_obj, "in_sample_gap": 0.0,
                            "iterations": 1, "out_of_sample_sharpe": sharpe_g,
                            "feasible": True, "runtime": gurobi_res["runtime_seconds"] if GUROBI_AVAILABLE else sa_base["runtime_seconds"]
                        })
                        continue
                        
                    print(f"    Running {solver_name} (p={p_val}, alpha={alpha_val})...")
                    try:
                        res, num_iters = run_solver_instance_advanced(
                            instance, solver_name, p=p_val, maxiter=maxiter, shots=shots, alpha=alpha_val
                        )
                        
                        # Gap calculation
                        gap = compute_gap(res["objective"], gurobi_obj)
                        
                        # Out of sample sharpe
                        w_sol = res["solution"] / K
                        port_returns = test_returns_reg[selected_tickers].dot(w_sol)
                        ann_ret = np.mean(port_returns) * 252
                        ann_vol = np.std(port_returns) * np.sqrt(252)
                        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
                        
                        sweep_results.append({
                            "N": N, "K": K, "regime": regime_key, "seed": seed, "solver": solver_name,
                            "p": p_val, "alpha": alpha_val, "in_sample_obj": res["objective"], "in_sample_gap": gap,
                            "iterations": num_iters, "out_of_sample_sharpe": sharpe,
                            "feasible": res["feasible"], "runtime": res["runtime_seconds"]
                        })
                        print(f"      Gap: {gap:.4f} | Iters: {num_iters} | Sharpe: {sharpe:.4f} | Time: {res['runtime_seconds']:.2f}s")
                        
                    except Exception as e:
                        print(f"      Error running {solver_name}: {e}")
                        sweep_results.append({
                            "N": N, "K": K, "regime": regime_key, "seed": seed, "solver": solver_name,
                            "p": p_val, "alpha": alpha_val, "in_sample_obj": np.nan, "in_sample_gap": np.nan,
                            "iterations": 0, "out_of_sample_sharpe": np.nan,
                            "feasible": False, "runtime": 0.0
                        })
                        
                total_runs += 1
                
    # Save sweep results
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"\n[OK] Barrido avanzado de hiperparámetros guardado en {sweep_path}")
    print("======================================================================")
    print("BENCHMARK AVANZADO COMPLETO")
    print("======================================================================")

if __name__ == "__main__":
    main()
