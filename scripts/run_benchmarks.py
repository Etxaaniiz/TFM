import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.portfolio.portfolio_model import build_qubo
from src.solvers.classic_solvers import solve_gurobi, solve_sa
from src.quantum.classical_emulators import QuantumStatevectorSimulator, solve_qaoa_pure_numpy
from src.metrics.metrics import calculate_portfolio_metrics, compute_gap

def load_regime_data(processed_dir="data/processed"):
    """Loads In-Sample (Estable) and Out-of-Sample (COVID19 / Inflacionario) data."""
    mu_is_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Estable.csv"))
    cov_is_df = pd.read_csv(os.path.join(processed_dir, "covariance_Estable.csv"), index_col=0)
    
    mu_oos_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Volatil_COVID19.csv"))
    cov_oos_df = pd.read_csv(os.path.join(processed_dir, "covariance_Volatil_COVID19.csv"), index_col=0)
    
    # Common tickers
    tickers = [t for t in mu_is_df['Ticker'] if t in cov_is_df.columns and t in mu_oos_df['Ticker'].values and t in cov_oos_df.columns]
    
    mu_is = mu_is_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_is = cov_is_df.loc[tickers, tickers]
    
    mu_oos = mu_oos_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_oos = cov_oos_df.loc[tickers, tickers]
    
    return tickers, mu_is, cov_is, mu_oos, cov_oos

def evaluate_oos_sharpe(x_sol, mu_oos, cov_oos, K):
    """Calculates Out-of-Sample Sharpe Ratio given binary solution vector."""
    actual_k = np.sum(x_sol)
    if actual_k == 0:
        return 0.0
    w = x_sol / actual_k
    mu_val = np.dot(mu_oos, w)
    var_val = np.dot(w, np.dot(cov_oos, w))
    vol_val = np.sqrt(var_val) if var_val > 0 else 0.0
    return float(mu_val / vol_val) if vol_val > 1e-9 else 0.0

def run_benchmarks(quick_mode=False):
    print("=" * 70)
    print("INICIANDO CAMPAÑA EXPERIMENTAL REAL (TFM QUANTUM PORTFOLIO)")
    print("=" * 70)
    
    processed_dir = os.path.join(project_root, "data", "processed")
    if not os.path.exists(processed_dir):
        print(f"Error: Directorio {processed_dir} no encontrado. Ejecuta prepare_data.py primero.")
        return
        
    all_tickers, mu_is_all, cov_is_all, mu_oos_all, cov_oos_all = load_regime_data(processed_dir)
    print(f"Activos disponibles: {len(all_tickers)}")
    
    if quick_mode:
        Ns = [6, 8, 10]
        seeds = [42, 43]
        ps = [1, 2, 3, 4]
        cobyla_iter = 30
        sa_reads = 200
        sa_sweeps = 200
        print("[MODO RÁPIDO ACTIVADO PARA VALIDACIÓN]", flush=True)
    else:
        Ns = [6, 8, 10, 12, 14, 16, 18, 20]
        seeds = [42, 43, 44, 45, 46]
        ps = [1, 2, 3, 4, 5, 6, 7, 8]
        cobyla_iter = 50
        sa_reads = 500
        sa_sweeps = 500
        sa_sweeps = 500
        
    lambda_val = 0.5
    rows = []
    
    # --------------------------------------------------------------------------
    # 1. MAIN N-SCALING STUDY (Fixed p=3 for QAOA solvers)
    # --------------------------------------------------------------------------
    p_fixed = 3
    print("\n>>> FASE 1.1: ESTUDIO DE ESCALABILIDAD EN N (N in", Ns, ")")
    
    for N in Ns:
        K = N // 2
        print(f"\n--- Dimension N={N}, Cardinalidad K={K} ---")
        
        for seed in seeds:
            # Deterministic selection of N tickers for this seed
            rng = np.random.RandomState(seed)
            selected_idx = rng.choice(len(all_tickers), size=N, replace=False)
            selected_tickers = [all_tickers[i] for i in selected_idx]
            
            mu_is = mu_is_all.loc[selected_tickers].values
            cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values
            mu_oos = mu_oos_all.loc[selected_tickers].values
            cov_oos = cov_oos_all.loc[selected_tickers, selected_tickers].values
            
            Q = build_qubo(mu_is, cov_is, K, lambda_val=lambda_val)
            
            inst = {
                'dataset': 'real_finance_SP500_IBEX',
                'instance_id': f"N{N}_K{K}_s{seed}",
                'N': N,
                'K': K,
                'mu': mu_is,
                'Sigma': cov_is,
                'Q': Q,
                'lambda_val': lambda_val,
                'seed': seed,
                'tickers': selected_tickers
            }
            
            # 1. GUROBI (Exact reference)
            res_gurobi = solve_gurobi(inst, lambda_val=lambda_val)
            gurobi_obj = res_gurobi['objective']
            gurobi_sol = res_gurobi['solution']
            gurobi_oos_sharpe = evaluate_oos_sharpe(gurobi_sol, mu_oos, cov_oos, K)
            
            rows.append({
                "N": N,
                "K": K,
                "Solver": "Gurobi",
                "p": None,
                "seed": seed,
                "Feasibility Ratio (%)": 100.0,
                "Optimization GAP (%)": 0.0,
                "Sharpe Ratio In-Sample": res_gurobi['sharpe'],
                "Sharpe Ratio Out-of-Sample": gurobi_oos_sharpe,
                "Execution Time (s)": res_gurobi['runtime_seconds']
            })
            
            # 2. SIMULATED ANNEALING
            res_sa = solve_sa(inst, num_reads=sa_reads, num_sweeps=sa_sweeps)
            sa_sol = res_sa['solution']
            sa_obj = res_sa['objective']
            sa_gap = compute_gap(sa_obj, gurobi_obj) * 100.0
            sa_feas = 100.0 if res_sa['feasible'] else 0.0
            sa_oos_sharpe = evaluate_oos_sharpe(sa_sol, mu_oos, cov_oos, K)
            
            rows.append({
                "N": N,
                "K": K,
                "Solver": "Simulated Annealing",
                "p": None,
                "seed": seed,
                "Feasibility Ratio (%)": sa_feas,
                "Optimization GAP (%)": sa_gap,
                "Sharpe Ratio In-Sample": res_sa['sharpe'],
                "Sharpe Ratio Out-of-Sample": sa_oos_sharpe,
                "Execution Time (s)": res_sa['runtime_seconds']
            })
            
            # 3. STANDARD QAOA (RX mixer, QUBO cost)
            res_qaoa_rx = solve_qaoa_pure_numpy(inst, p=p_fixed, mixer="rx", init_type="random", alpha=0.0, maxiter=cobyla_iter)
            rx_sol = res_qaoa_rx['solution']
            rx_obj = res_qaoa_rx['objective']
            rx_gap = compute_gap(rx_obj, gurobi_obj) * 100.0
            
            # Calculate feasibility probability mass in statevector
            sim_rx = QuantumStatevectorSimulator(N, K, Q, lambda_val)
            _, rx_probs = sim_rx.simulate_qaoa(p_fixed, res_qaoa_rx['optimal_angles'], mixer="rx")
            feasible_indices = [idx for idx in range(2**N) if bin(idx).count('1') == K]
            rx_feas_ratio = float(np.sum(rx_probs[feasible_indices]) * 100.0)
            rx_oos_sharpe = evaluate_oos_sharpe(rx_sol, mu_oos, cov_oos, K)
            
            rows.append({
                "N": N,
                "K": K,
                "Solver": "Standard QAOA",
                "p": p_fixed,
                "seed": seed,
                "Feasibility Ratio (%)": rx_feas_ratio,
                "Optimization GAP (%)": rx_gap,
                "Sharpe Ratio In-Sample": res_qaoa_rx['sharpe'],
                "Sharpe Ratio Out-of-Sample": rx_oos_sharpe,
                "Execution Time (s)": res_qaoa_rx['runtime_seconds']
            })
            
            # 4. XY-QAOA (XY mixer, Dicke state, unregularized)
            res_xy = solve_qaoa_pure_numpy(inst, p=p_fixed, mixer="xy", init_type="random", alpha=0.0, maxiter=cobyla_iter)
            xy_sol = res_xy['solution']
            xy_obj = res_xy['objective']
            xy_gap = compute_gap(xy_obj, gurobi_obj) * 100.0
            xy_oos_sharpe = evaluate_oos_sharpe(xy_sol, mu_oos, cov_oos, K)
            
            rows.append({
                "N": N,
                "K": K,
                "Solver": "XY-QAOA",
                "p": p_fixed,
                "seed": seed,
                "Feasibility Ratio (%)": 100.0, # Conserved by XY mixer
                "Optimization GAP (%)": xy_gap,
                "Sharpe Ratio In-Sample": res_xy['sharpe'],
                "Sharpe Ratio Out-of-Sample": xy_oos_sharpe,
                "Execution Time (s)": res_xy['runtime_seconds']
            })
            
            # 5. XY-QAOA REGULARIZED (XY mixer, Dicke state, TQA init, Ridge L2)
            res_reg = solve_qaoa_pure_numpy(inst, p=p_fixed, mixer="xy", init_type="tqa", alpha=0.015, maxiter=cobyla_iter)
            reg_sol = res_reg['solution']
            reg_obj = res_reg['objective']
            reg_gap = compute_gap(reg_obj, gurobi_obj) * 100.0
            reg_oos_sharpe = evaluate_oos_sharpe(reg_sol, mu_oos, cov_oos, K)
            
            rows.append({
                "N": N,
                "K": K,
                "Solver": "XY-QAOA Regularized",
                "p": p_fixed,
                "seed": seed,
                "Feasibility Ratio (%)": 100.0,
                "Optimization GAP (%)": reg_gap,
                "Sharpe Ratio In-Sample": res_reg['sharpe'],
                "Sharpe Ratio Out-of-Sample": reg_oos_sharpe,
                "Execution Time (s)": res_reg['runtime_seconds']
            })
            
            print(f"  Seed {seed} | Gurobi: {gurobi_obj:.4f} | SA GAP: {sa_gap:.2f}% | RX GAP: {rx_gap:.2f}% | XY GAP: {xy_gap:.2f}% | Reg GAP: {reg_gap:.2f}%", flush=True)
            
    # --------------------------------------------------------------------------
    # 2. DEPTH p SCALING STUDY (Fixed N=14, K=7)
    # --------------------------------------------------------------------------
    N_fixed = 14 if 14 in Ns else Ns[-1]
    K_fixed = N_fixed // 2
    print(f"\n>>> FASE 1.2: ESTUDIO DE PROFUNDIDAD DEL ANSATZ p in {ps} (N={N_fixed}, K={K_fixed})")
    
    for p in ps:
        print(f"\n--- Profundidad p={p} ---")
        for seed in seeds:
            rng = np.random.RandomState(seed)
            selected_idx = rng.choice(len(all_tickers), size=N_fixed, replace=False)
            selected_tickers = [all_tickers[i] for i in selected_idx]
            
            mu_is = mu_is_all.loc[selected_tickers].values
            cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values
            mu_oos = mu_oos_all.loc[selected_tickers].values
            cov_oos = cov_oos_all.loc[selected_tickers, selected_tickers].values
            
            Q = build_qubo(mu_is, cov_is, K_fixed, lambda_val=lambda_val)
            
            inst = {
                'dataset': 'real_finance_SP500_IBEX',
                'instance_id': f"N{N_fixed}_K{K_fixed}_p{p}_s{seed}",
                'N': N_fixed,
                'K': K_fixed,
                'mu': mu_is,
                'Sigma': cov_is,
                'Q': Q,
                'lambda_val': lambda_val,
                'seed': seed,
                'tickers': selected_tickers
            }
            
            # Gurobi benchmark reference for this instance
            res_gurobi = solve_gurobi(inst, lambda_val=lambda_val)
            gurobi_obj = res_gurobi['objective']
            
            # 1. Standard QAOA at depth p
            res_rx_p = solve_qaoa_pure_numpy(inst, p=p, mixer="rx", init_type="random", alpha=0.0, maxiter=cobyla_iter)
            rx_sol_p = res_rx_p['solution']
            rx_obj_p = res_rx_p['objective']
            rx_gap_p = compute_gap(rx_obj_p, gurobi_obj) * 100.0
            
            sim_rx_p = QuantumStatevectorSimulator(N_fixed, K_fixed, Q, lambda_val)
            _, rx_probs_p = sim_rx_p.simulate_qaoa(p, res_rx_p['optimal_angles'], mixer="rx")
            feasible_indices_p = [idx for idx in range(2**N_fixed) if bin(idx).count('1') == K_fixed]
            rx_feas_p = float(np.sum(rx_probs_p[feasible_indices_p]) * 100.0)
            rx_oos_sharpe_p = evaluate_oos_sharpe(rx_sol_p, mu_oos, cov_oos, K_fixed)
            
            rows.append({
                "N": N_fixed,
                "K": K_fixed,
                "Solver": "Standard QAOA",
                "p": p,
                "seed": seed,
                "Feasibility Ratio (%)": rx_feas_p,
                "Optimization GAP (%)": rx_gap_p,
                "Sharpe Ratio In-Sample": res_rx_p['sharpe'],
                "Sharpe Ratio Out-of-Sample": rx_oos_sharpe_p,
                "Execution Time (s)": res_rx_p['runtime_seconds']
            })
            
            # 2. XY-QAOA Regularized at depth p
            res_reg_p = solve_qaoa_pure_numpy(inst, p=p, mixer="xy", init_type="tqa", alpha=0.015, maxiter=cobyla_iter)
            reg_sol_p = res_reg_p['solution']
            reg_obj_p = res_reg_p['objective']
            reg_gap_p = compute_gap(reg_obj_p, gurobi_obj) * 100.0
            reg_oos_sharpe_p = evaluate_oos_sharpe(reg_sol_p, mu_oos, cov_oos, K_fixed)
            
            rows.append({
                "N": N_fixed,
                "K": K_fixed,
                "Solver": "XY-QAOA Regularized",
                "p": p,
                "seed": seed,
                "Feasibility Ratio (%)": 100.0,
                "Optimization GAP (%)": reg_gap_p,
                "Sharpe Ratio In-Sample": res_reg_p['sharpe'],
                "Sharpe Ratio Out-of-Sample": reg_oos_sharpe_p,
                "Execution Time (s)": res_reg_p['runtime_seconds']
            })
            
            print(f"  Seed {seed} | RX(p={p}) GAP: {rx_gap_p:.2f}% | Reg(p={p}) GAP: {reg_gap_p:.2f}%", flush=True)
            
    # Convert to DataFrame
    df_results = pd.DataFrame(rows)
    
    out_dir = os.path.join(project_root, "output", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    df_results.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print(f"CAMPAÑA EXPERIMENTAL COMPLETADA CON ÉXITO.")
    print(f"Archivo generado: {out_csv}")
    print(f"Total de registros experimentales: {len(df_results)}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar benchmark experimental real de optimización de carteras.")
    parser.add_argument("--quick", action="store_true", help="Ejecutar en modo rápido para pruebas")
    args = parser.parse_args()
    
    run_benchmarks(quick_mode=args.quick)
