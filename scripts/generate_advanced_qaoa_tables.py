import os
import sys

# Resolve project root relative to script directory and change working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
sys.path.append(project_root)

import pandas as pd
import numpy as np

def main():
    print("======================================================================")
    print("GENERANDO TABLAS LATEX DEL BENCHMARK AVANZADO")
    print("======================================================================")
    
    sweep_path = "results/qaoa_advanced_analysis/advanced_hyperparameters_sweep.csv"
    if not os.path.exists(sweep_path):
        sweep_path = "Version2/output/results/advanced_hyperparameters_sweep.csv"
        
    if not os.path.exists(sweep_path):
        print(f"Error: No se encontró el archivo de resultados en {sweep_path}. Ejecuta run_advanced_qaoa_benchmark.py primero.")
        return
        
    df = pd.read_csv(sweep_path)
    os.makedirs("tables", exist_ok=True)
    os.makedirs("Version2/output/tables", exist_ok=True)
    
    # Standardize names
    df["solver_display"] = df.apply(
        lambda r: f"{r['solver']} (α={r['alpha']})" if r['solver'] == "xy_regularized" else r['solver'],
        axis=1
    )
    df["solver_display"] = df["solver_display"].replace({
        "gurobi": "Gurobi (Exacto)",
        "sa": "Simulated Annealing",
        "xy_normal": "XY-QAOA Normal",
        "xy_regularized (α=0.05)": "XY-QAOA Reg (α=0.05)",
        "xy_regularized (α=0.1)": "XY-QAOA Reg (α=0.1)"
    })
    
    regime_names = {
        "stable": "Régimen Estable",
        "volatile": "Régimen Volátil",
        "inflationary": "Régimen Inflacionario"
    }
    
    # We want to create a table for each regime
    for regime_key, regime_name in regime_names.items():
        df_r = df[df["regime"] == regime_key]
        if df_r.empty:
            continue
            
        # Group and calculate mean and std
        grouped = df_r.groupby(["N", "solver_display"]).agg({
            "in_sample_gap": ["mean", "std"],
            "out_of_sample_sharpe": ["mean", "std"],
            "iterations": ["mean"],
            "runtime": ["mean"]
        }).reset_index()
        
        # Flatten columns
        grouped.columns = [
            "N", "Solver", 
            "gap_mean", "gap_std", 
            "sharpe_mean", "sharpe_std", 
            "iterations", "runtime"
        ]
        
        # Sort
        grouped = grouped.sort_values(by=["N", "Solver"])
        
        # LaTeX Table Construction
        latex = r"""\begin{table}[h!]
\centering
\caption{Comparación de Desempeño Cuántico y Clásico en el """ + regime_name + r""" ($N \in \{10, 15, 20\}$)}
\label{tab:advanced_qaoa_""" + regime_key + r"""}
\begin{tabular}{llcccc}
\hline
\textbf{N} & \textbf{Solucionador} & \textbf{In-Sample Gap} & \textbf{Sharpe Neto (OOS)} & \textbf{Iteraciones} & \textbf{Tiempo (s)} \\
\hline
"""
        current_n = None
        for _, row in grouped.iterrows():
            n_val = row["N"]
            # To make it look clean, only print N on the first row of that group
            n_str = str(n_val) if n_val != current_n else ""
            current_n = n_val
            
            # Format numbers
            gap_val = f"{row['gap_mean']*100:.2f}\\% \\pm {row['gap_std']*100:.2f}\\%" if not pd.isna(row['gap_std']) else f"{row['gap_mean']*100:.2f}\\%"
            if row["Solver"] in ["Gurobi (Exacto)", "Simulated Annealing"] and "gap_mean" in row and row["gap_mean"] == 0:
                gap_val = "0.00\\%"
                
            sharpe_val = f"{row['sharpe_mean']:.4f} \\pm {row['sharpe_std']:.4f}" if not pd.isna(row['sharpe_std']) else f"{row['sharpe_mean']:.4f}"
            
            iters_val = f"{int(row['iterations'])}"
            runtime_val = f"{row['runtime']:.3f}"
            
            latex += f"{n_str} & {row['Solver']} & {gap_val} & {sharpe_val} & {iters_val} & {runtime_val} \\\\\n"
            
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        latex_path1 = f"tables/advanced_qaoa_summary_{regime_key}.tex"
        latex_path2 = f"Version2/output/tables/advanced_qaoa_summary_{regime_key}.tex"
        with open(latex_path1, "w", encoding="utf-8") as f:
            f.write(latex)
        with open(latex_path2, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"[OK] Tabla LaTeX para {regime_key} guardada en {latex_path1} y {latex_path2}")
        
    print("\nGeneración de tablas LaTeX completada con éxito.")

if __name__ == "__main__":
    main()
