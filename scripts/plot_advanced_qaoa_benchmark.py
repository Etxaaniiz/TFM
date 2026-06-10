import os
import sys

# Resolve project root relative to script directory and change working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def main():
    print("======================================================================")
    print("GENERANDO GRAFICOS ANALITICOS DEL BENCHMARK AVANZADO")
    print("======================================================================")
    
    sweep_path = "results/qaoa_advanced_analysis/advanced_hyperparameters_sweep.csv"
    if not os.path.exists(sweep_path):
        sweep_path = "Version2/output/results/advanced_hyperparameters_sweep.csv"
        
    if not os.path.exists(sweep_path):
        print(f"Error: No se encontró el archivo de resultados en {sweep_path}. Ejecuta run_advanced_qaoa_benchmark.py primero.")
        return
        
    df = pd.read_csv(sweep_path)
    
    def save_fig(filename):
        os.makedirs("figures", exist_ok=True)
        plt.savefig(os.path.join("figures", filename), dpi=300, bbox_inches='tight')
        os.makedirs("Version2/output/graficos", exist_ok=True)
        plt.savefig(os.path.join("Version2/output/graficos", filename), dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico guardado en figures/{filename} y Version2/output/graficos/{filename}")
    
    # Set styling
    sns.set_theme(style="whitegrid")
    
    # Format labels
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
    
    # ----------------------------------------------------------------
    # GRAFICO 1: OPTIMIZATION GAP VS N POR REGIMEN
    # ----------------------------------------------------------------
    # Filter for quantum solvers to compare Gap
    df_quantum = df[df["solver"].isin(["xy_normal", "xy_regularized"])].copy()
    
    if not df_quantum.empty:
        # Group by regime, N, and solver_display and average
        df_gap = df_quantum.groupby(["regime", "N", "solver_display"])["in_sample_gap"].mean().reset_index()
        
        regimes_present = df_gap["regime"].unique()
        fig, axes = plt.subplots(1, len(regimes_present), figsize=(6 * len(regimes_present), 5.5), sharey=True)
        if len(regimes_present) == 1:
            axes = [axes]
            
        for i, regime in enumerate(regimes_present):
            df_r = df_gap[df_gap["regime"] == regime]
            ax = axes[i]
            sns.lineplot(
                data=df_r,
                x="N",
                y="in_sample_gap",
                hue="solver_display",
                marker="o",
                linewidth=2.2,
                markersize=8,
                ax=ax
            )
            ax.set_title(regime_names.get(regime, regime), fontsize=12, fontweight='bold')
            ax.set_xlabel("Tamaño del Universo (N activos)", fontsize=11)
            ax.set_ylabel("Optimization Gap (In-Sample)", fontsize=11)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xticks(sorted(df_r["N"].unique()))
            if i > 0:
                ax.get_legend().remove()
            else:
                ax.legend(title="Solucionador")
                
        plt.suptitle("Optimization Gap In-Sample vs. Escalabilidad del Universo (N activos)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig("qaoa_advanced_gap_scaling.png")
        plt.close()
        
    # ----------------------------------------------------------------
    # GRAFICO 2: EFICIENCIA DE OPTIMIZACION CLASICA (ITERACIONES VS N)
    # ----------------------------------------------------------------
    # Group by N and solver_display across all regimes/seeds to show general optimization complexity
    df_iters = df[df["solver"].isin(["xy_normal", "xy_regularized"])].copy()
    if not df_iters.empty:
        df_iters_grouped = df_iters.groupby(["N", "solver_display"])["iterations"].mean().reset_index()
        
        plt.figure(figsize=(9, 5.5))
        sns.barplot(
            data=df_iters_grouped,
            x="N",
            y="iterations",
            hue="solver_display",
            palette="muted",
            edgecolor="#94A3B8"
        )
        
        plt.title("Evaluaciones de la Función de Coste Clásico (Iteraciones COBYLA)", fontsize=13, fontweight='bold')
        plt.xlabel("Tamaño del Universo (N activos)", fontsize=11)
        plt.ylabel("Número Promedio de Iteraciones", fontsize=11)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Solucionador", loc="upper left")
        sns.despine()
        
        save_fig("qaoa_advanced_iterations.png")
        plt.close()
        
    # ----------------------------------------------------------------
    # GRAFICO 3: SHARPE RATIO NETO VS N POR REGIMEN
    # ----------------------------------------------------------------
    df_sharpe = df.groupby(["regime", "N", "solver_display"])["out_of_sample_sharpe"].mean().reset_index()
    regimes_present_s = df_sharpe["regime"].unique()
    
    if not df_sharpe.empty:
        fig, axes = plt.subplots(1, len(regimes_present_s), figsize=(6 * len(regimes_present_s), 5.5), sharey=True)
        if len(regimes_present_s) == 1:
            axes = [axes]
            
        for i, regime in enumerate(regimes_present_s):
            df_r = df_sharpe[df_sharpe["regime"] == regime]
            ax = axes[i]
            
            # Draw lines for all solvers
            sns.lineplot(
                data=df_r,
                x="N",
                y="out_of_sample_sharpe",
                hue="solver_display",
                marker="s",
                linewidth=2.2,
                markersize=8,
                ax=ax
            )
            ax.set_title(regime_names.get(regime, regime), fontsize=12, fontweight='bold')
            ax.set_xlabel("Tamaño del Universo (N activos)", fontsize=11)
            ax.set_ylabel("Sharpe Ratio Neto (Out-of-Sample)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xticks(sorted(df_r["N"].unique()))
            if i > 0:
                ax.get_legend().remove()
            else:
                ax.legend(title="Solucionador")
                
        plt.suptitle("Sharpe Ratio Neto Fuera de Muestra vs. Escalabilidad del Universo (N activos)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig("qaoa_advanced_sharpe_scaling.png")
        plt.close()
        
    print("\nGeneración de gráficos analíticos avanzados completada con éxito.")

if __name__ == "__main__":
    main()
