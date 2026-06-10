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
from src.metrics.metrics import calculate_portfolio_metrics, calculate_qubo_energy, compute_gap
from src.quantum.regularized_qaoa import RegularizedQAOAProblem
from qrisp import QuantumVariable, dicke_state
from qrisp.qaoa import XY_mixer

def run_solver_instance(
    instance: Dict[str, Any],
    solver_type: str, # "gurobi", "sa", "xy_normal", "xy_regularized"
    p: int = 2,
    maxiter: int = 100,
    shots: int = 1024,
    alpha: float = 0.1,
    init_type: str = "tqa", # "tqa" or "random"
    track_trajectory: bool = False
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Runs a specific solver configuration and optionaly tracks optimization trajectory.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    start_time = time.perf_counter()
    trajectory = []
    
    if solver_type == "gurobi":
        res = solve_gurobi(instance, lambda_val=lambda_val)
        return res, []
        
    elif solver_type == "sa":
        res = solve_sa(instance)
        return res, []
        
    elif solver_type in ["xy_normal", "xy_regularized"]:
        # Setup QuantumVariable
        qv = QuantumVariable(N)
        
        # Dicke state initialization function
        def init_dicke(q_var):
            from qrisp import x
            for i in range(K):
                x(q_var[i])
            dicke_state(q_var, K)
            
        # Instantiate Regularized XY-QAOA Problem
        prob_alpha = alpha if solver_type == "xy_regularized" else 0.0
        prob_init_type = "tqa" if solver_type == "xy_regularized" else "random"
        if solver_type == "xy_regularized" and init_type == "random":
            prob_init_type = "random"
            
        xy_qaoa_prob = RegularizedQAOAProblem(
            cost_operator=create_QUBO_cost_operator(Q),
            mixer=XY_mixer,
            cl_cost_function=create_QUBO_cl_cost_function(Q),
            init_function=init_dicke,
            alpha=prob_alpha
        )
        
        if track_trajectory:
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
        
        if track_trajectory:
            trajectory = list(xy_qaoa_prob.optimization_costs)
            
        return results, trajectory
        
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

def main():
    print("======================================================================")
    print("INICIANDO BENCHMARK RIGUROSO DE COMPARACION DE QAOA XY REGULARIZADO")
    print("======================================================================")
    
    test_mode = "--test" in sys.argv
    if test_mode:
        print("MODO DE PRUEBA ACTIVO (Parametros minimos)")
        
    # Configure parameters
    if test_mode:
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'V'] # N=6
        K = 2
        seeds = [42]
        p_list = [1]
        alpha_list = [0.0, 0.1]
        maxiter = 2
        shots = 100
    else:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'PG', 'KO', 'SAN.MC', 'BBVA.MC'] # N=10
        K = 3
        seeds = [42, 43, 44] # Multiple seeds for statistical stability
        p_list = [1, 2, 3]
        alpha_list = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]
        maxiter = 100
        shots = 1024
        
    # Define training and testing windows (Stable Regime style)
    train_start, train_end = "2019-01-01", "2020-12-31"
    test_start, test_end = "2021-01-01", "2021-06-30"
    
    # Load prices
    raw_path = "data/raw/prices.csv"
    if not os.path.exists(raw_path):
        print(f"Error: Raw prices file not found at {raw_path}. Run prepare_data.py first.")
        sys.exit(1)
        
    prices_all = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    prices = prices_all[tickers].ffill().bfill()
    daily_returns = np.log(prices / prices.shift(1)).dropna()
    
    train_returns = daily_returns.loc[train_start:train_end]
    test_returns = daily_returns.loc[test_start:test_end]
    
    os.makedirs("results/qaoa_analysis", exist_ok=True)
    sweep_path = "results/qaoa_analysis/hyperparameters_sweep.csv"
    trajectory_path = "results/qaoa_analysis/convergence_trajectories.csv"
    
    sweep_results = []
    trajectory_results = []
    
    # Run over seeds
    for seed in seeds:
        print(f"\n---> Corriendo experimentos con Semilla: {seed} <---")
        np.random.seed(seed)
        
        # Annualized mu & Sigma in-sample
        mu = train_returns.mean() * 252
        Sigma = train_returns.cov() * 252
        
        # Build QUBO
        N_assets = len(tickers)
        Q0 = np.zeros((N_assets, N_assets))
        for idx_i in range(N_assets):
            Q0[idx_i, idx_i] = 0.5 * Sigma.iloc[idx_i, idx_i] / (K ** 2) - 0.5 * mu.iloc[idx_i] / K
            for idx_j in range(idx_i + 1, N_assets):
                val = 0.5 * Sigma.iloc[idx_i, idx_j] / (K ** 2)
                Q0[idx_i, idx_j] = val / 2.0
                Q0[idx_j, idx_i] = val / 2.0
                
        P = 10.0 * np.max(np.abs(Q0))
        Q = build_qubo(mu.to_numpy(), Sigma.to_numpy(), K, 0.5, penalty=P)
        
        instance = {
            "dataset": "benchmark_analysis",
            "N": N_assets,
            "K": K,
            "instance_id": seed,
            "seed": seed,
            "tickers": tickers,
            "mu": mu.to_numpy(),
            "Sigma": Sigma.to_numpy(),
            "Q": Q,
            "penalty": P,
            "offset": P * (K ** 2),
            "lambda_val": 0.5
        }
        
        # Solve with Gurobi (Baseline exact)
        gurobi_res = None
        if GUROBI_AVAILABLE:
            print("  [Gurobi] Corriendo solucionador exacto...")
            gurobi_res, _ = run_solver_instance(instance, "gurobi")
            gurobi_obj = gurobi_res["objective"]
            gurobi_bitstring = "".join([str(int(bit)) for bit in gurobi_res["solution"]])
        else:
            print("  [Gurobi] No disponible, usando Simulated Annealing para aproximar base...")
            sa_base, _ = run_solver_instance(instance, "sa")
            gurobi_obj = sa_base["objective"]
            gurobi_bitstring = "".join([str(int(bit)) for bit in sa_base["solution"]])
            
        # Log Gurobi / Base Out-of-Sample metrics
        # Compute out-of-sample portfolio Sharpe
        test_returns_sel = test_returns
        g_sol = gurobi_res["solution"] if GUROBI_AVAILABLE else sa_base["solution"]
        w_g = g_sol / K
        port_returns_g = test_returns_sel.dot(w_g)
        ann_ret_g = np.mean(port_returns_g) * 252
        ann_vol_g = np.std(port_returns_g) * np.sqrt(252)
        sharpe_g = ann_ret_g / ann_vol_g if ann_vol_g > 1e-9 else 0.0
        
        sweep_results.append({
            "seed": seed, "solver": "gurobi" if GUROBI_AVAILABLE else "sa_base",
            "p": None, "alpha": None, "init_type": None,
            "in_sample_obj": gurobi_obj, "in_sample_gap": 0.0,
            "success_prob": 1.0, "acc_success_prob": 1.0,
            "out_of_sample_return": ann_ret_g, "out_of_sample_vol": ann_vol_g, "out_of_sample_sharpe": sharpe_g,
            "feasible": True, "runtime": gurobi_res["runtime_seconds"] if GUROBI_AVAILABLE else sa_base["runtime_seconds"]
        })
        
        # 1. Sweep over Circuit Depth (p) for Normal XY-QAOA vs Regularized XY-QAOA (alpha = 0.1)
        for p in p_list:
            # Normal XY-QAOA
            print(f"  [XY-QAOA Normal] Corriendo p={p}...")
            norm_res, norm_traj = run_solver_instance(
                instance, "xy_normal", p=p, maxiter=maxiter, shots=shots, track_trajectory=(p == 2)
            )
            
            # Out-of-sample metrics
            w_norm = norm_res["solution"] / K
            port_returns_norm = test_returns_sel.dot(w_norm)
            ann_ret_norm = np.mean(port_returns_norm) * 252
            ann_vol_norm = np.std(port_returns_norm) * np.sqrt(252)
            sharpe_norm = ann_ret_norm / ann_vol_norm if ann_vol_norm > 1e-9 else 0.0
            
            # Gap and probabilities
            norm_gap = compute_gap(norm_res["objective"], gurobi_obj)
            norm_counts = norm_res.get("counts", {})
            norm_succ = norm_counts.get(gurobi_bitstring, 0.0)
            
            norm_acc = 0.0
            for b_str, pr in norm_counts.items():
                x_b = np.array([int(bit) for bit in b_str])
                metrics_b = calculate_portfolio_metrics(x_b, mu.to_numpy(), Sigma.to_numpy(), K, 0.5)
                if (metrics_b["objective"] - gurobi_obj) <= 0.10 * abs(gurobi_obj):
                    norm_acc += pr
                    
            sweep_results.append({
                "seed": seed, "solver": "xy_qaoa_normal",
                "p": p, "alpha": 0.0, "init_type": "random",
                "in_sample_obj": norm_res["objective"], "in_sample_gap": norm_gap,
                "success_prob": norm_succ, "acc_success_prob": norm_acc,
                "out_of_sample_return": ann_ret_norm, "out_of_sample_vol": ann_vol_norm, "out_of_sample_sharpe": sharpe_norm,
                "feasible": norm_res["feasible"], "runtime": norm_res["runtime_seconds"]
            })
            
            # Store normal p=2 trajectory
            if p == 2 and len(norm_traj) > 0:
                for idx_t, cost in enumerate(norm_traj):
                    trajectory_results.append({
                        "seed": seed, "solver": "xy_qaoa_normal", "iteration": idx_t, "cost": cost
                    })
            
            # Regularized XY-QAOA with alpha = 0.1 (baseline)
            print(f"  [XY-QAOA Regularized] Corriendo p={p}, alpha=0.1...")
            reg_res, reg_traj = run_solver_instance(
                instance, "xy_regularized", p=p, maxiter=maxiter, shots=shots, alpha=0.1, init_type="tqa", track_trajectory=(p == 2)
            )
            
            w_reg = reg_res["solution"] / K
            port_returns_reg = test_returns_sel.dot(w_reg)
            ann_ret_reg = np.mean(port_returns_reg) * 252
            ann_vol_reg = np.std(port_returns_reg) * np.sqrt(252)
            sharpe_reg = ann_ret_reg / ann_vol_reg if ann_vol_reg > 1e-9 else 0.0
            
            reg_gap = compute_gap(reg_res["objective"], gurobi_obj)
            reg_counts = reg_res.get("counts", {})
            reg_succ = reg_counts.get(gurobi_bitstring, 0.0)
            
            reg_acc = 0.0
            for b_str, pr in reg_counts.items():
                x_b = np.array([int(bit) for bit in b_str])
                metrics_b = calculate_portfolio_metrics(x_b, mu.to_numpy(), Sigma.to_numpy(), K, 0.5)
                if (metrics_b["objective"] - gurobi_obj) <= 0.10 * abs(gurobi_obj):
                    reg_acc += pr
                    
            sweep_results.append({
                "seed": seed, "solver": "xy_qaoa_regularized",
                "p": p, "alpha": 0.1, "init_type": "tqa",
                "in_sample_obj": reg_res["objective"], "in_sample_gap": reg_gap,
                "success_prob": reg_succ, "acc_success_prob": reg_acc,
                "out_of_sample_return": ann_ret_reg, "out_of_sample_vol": ann_vol_reg, "out_of_sample_sharpe": sharpe_reg,
                "feasible": reg_res["feasible"], "runtime": reg_res["runtime_seconds"]
            })
            
            # Store regularized p=2 trajectory for alpha = 0.1
            if p == 2 and len(reg_traj) > 0:
                for idx_t, cost in enumerate(reg_traj):
                    trajectory_results.append({
                        "seed": seed, "solver": "xy_qaoa_regularized_alpha_0.1", "iteration": idx_t, "cost": cost
                    })
                    
        # 2. Sweep over Ridge Regularization Penalty (alpha) for fixed p = 2
        p_fixed = 2 if not test_mode else 1
        for alpha in alpha_list:
            # Skip alpha=0.1 as it is already run above (except in test mode)
            if not test_mode and np.isclose(alpha, 0.1):
                continue
                
            print(f"  [XY-QAOA Regularized Alpha Sweep] Corriendo p={p_fixed}, alpha={alpha}...")
            
            # We also run alpha=0.0 with init_type="tqa" to isolate TQA initialization impact
            # vs Ridge penalty.
            res_alpha, traj_alpha = run_solver_instance(
                instance, "xy_regularized", p=p_fixed, maxiter=maxiter, shots=shots, alpha=alpha, init_type="tqa", track_trajectory=(alpha in [0.0, 1.0])
            )
            
            w_alpha = res_alpha["solution"] / K
            port_returns_alpha = test_returns_sel.dot(w_alpha)
            ann_ret_alpha = np.mean(port_returns_alpha) * 252
            ann_vol_alpha = np.std(port_returns_alpha) * np.sqrt(252)
            sharpe_alpha = ann_ret_alpha / ann_vol_alpha if ann_vol_alpha > 1e-9 else 0.0
            
            gap_alpha = compute_gap(res_alpha["objective"], gurobi_obj)
            counts_alpha = res_alpha.get("counts", {})
            succ_alpha = counts_alpha.get(gurobi_bitstring, 0.0)
            
            acc_alpha = 0.0
            for b_str, pr in counts_alpha.items():
                x_b = np.array([int(bit) for bit in b_str])
                metrics_b = calculate_portfolio_metrics(x_b, mu.to_numpy(), Sigma.to_numpy(), K, 0.5)
                if (metrics_b["objective"] - gurobi_obj) <= 0.10 * abs(gurobi_obj):
                    acc_alpha += pr
                    
            sweep_results.append({
                "seed": seed, "solver": "xy_qaoa_regularized",
                "p": p_fixed, "alpha": alpha, "init_type": "tqa",
                "in_sample_obj": res_alpha["objective"], "in_sample_gap": gap_alpha,
                "success_prob": succ_alpha, "acc_success_prob": acc_alpha,
                "out_of_sample_return": ann_ret_alpha, "out_of_sample_vol": ann_vol_alpha, "out_of_sample_sharpe": sharpe_alpha,
                "feasible": res_alpha["feasible"], "runtime": res_alpha["runtime_seconds"]
            })
            
            # Store trajectory details for alpha = 0.0 or alpha = 1.0
            if len(traj_alpha) > 0:
                solver_label = f"xy_qaoa_regularized_alpha_{alpha}"
                for idx_t, cost in enumerate(traj_alpha):
                    trajectory_results.append({
                        "seed": seed, "solver": solver_label, "iteration": idx_t, "cost": cost
                    })
                    
    # Save datasets
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"\n[OK] Barrido de hiperparámetros guardado en {sweep_path}")
    
    if len(trajectory_results) > 0:
        df_traj = pd.DataFrame(trajectory_results)
        df_traj.to_csv(trajectory_path, index=False)
        print(f"[OK] Trayectorias de convergencia guardadas en {trajectory_path}")
        
    print("======================================================================")
    print("BENCHMARK COMPLETO")
    print("======================================================================")

if __name__ == "__main__":
    main()
