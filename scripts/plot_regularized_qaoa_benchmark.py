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
    print("GENERANDO GRAFICOS ANALITICOS DEL QAOA XY REGULARIZADO")
    print("======================================================================")
    
    sweep_path = "results/qaoa_analysis/hyperparameters_sweep.csv"
    trajectory_path = "results/qaoa_analysis/convergence_trajectories.csv"
    
    if not os.path.exists(sweep_path):
        sweep_path = "Version2/output/results/hyperparameters_sweep.csv"
    if not os.path.exists(trajectory_path):
        trajectory_path = "Version2/output/results/convergence_trajectories.csv"
        
    if not os.path.exists(sweep_path):
        print(f"Error: No se encontró el barrido de hiperparámetros en {sweep_path}. Ejecuta run_regularized_qaoa_benchmark.py primero.")
        return
        
    df_sweep = pd.read_csv(sweep_path)
    
    def save_fig(filename):
        os.makedirs("figures", exist_ok=True)
        plt.savefig(os.path.join("figures", filename), dpi=300, bbox_inches='tight')
        os.makedirs("Version2/output/graficos", exist_ok=True)
        plt.savefig(os.path.join("Version2/output/graficos", filename), dpi=300, bbox_inches='tight')
        print(f"[OK] Gráfico guardado en figures/{filename} y Version2/output/graficos/{filename}")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # ----------------------------------------------------------------
    # GRAFICO 1: IMPACTO DEL DIAL DE REGULARIZACION (ALPHA)
    # ----------------------------------------------------------------
    # Filter for regularized solver with fixed depth p (usually 2, or whatever is max in the sweep)
    # Exclude Gurobi/SA
    p_fixed = df_sweep[df_sweep["solver"] == "xy_qaoa_regularized"]["p"].max()
    df_alpha = df_sweep[(df_sweep["solver"] == "xy_qaoa_regularized") & (df_sweep["p"] == p_fixed)].copy()
    
    if not df_alpha.empty:
        # Group by alpha and average over seeds
        df_alpha_grouped = df_alpha.groupby("alpha").agg({
            "in_sample_gap": ["mean", "std"],
            "out_of_sample_sharpe": ["mean", "std"]
        }).reset_index()
        df_alpha_grouped.columns = ["alpha", "gap_mean", "gap_std", "sharpe_mean", "sharpe_std"]
        
        # We also need Gurobi/SA baseline to draw a reference line
        df_gurobi = df_sweep[df_sweep["solver"].isin(["gurobi", "sa_base"])]
        g_sharpe_mean = df_gurobi["out_of_sample_sharpe"].mean() if not df_gurobi.empty else 0.89
        
        # Create dual plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot A: In-sample Optimization Gap vs Alpha
        # Use semi-log x scale
        axes[0].errorbar(
            df_alpha_grouped["alpha"],
            df_alpha_grouped["gap_mean"] * 100, # convert to %
            yerr=df_alpha_grouped["gap_std"] * 100,
            marker="o",
            color="darkblue",
            linewidth=2.0,
            capsize=4,
            label="XY-QAOA Regularizado"
        )
        axes[0].set_xscale('symlog', linthresh=0.01)
        axes[0].set_title("Optimization Gap In-Sample vs Coeficiente Alpha (α)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Alpha (α) - Escala Logarítmica", fontsize=11)
        axes[0].set_ylabel("Optimization Gap (%)", fontsize=11)
        axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
        
        # Plot B: Out-of-sample Sharpe Ratio vs Alpha
        axes[1].errorbar(
            df_alpha_grouped["alpha"],
            df_alpha_grouped["sharpe_mean"],
            yerr=df_alpha_grouped["sharpe_std"],
            marker="s",
            color="darkgreen",
            linewidth=2.0,
            capsize=4,
            label="XY-QAOA Regularizado"
        )
        # Add Gurobi baseline reference line
        axes[1].axhline(
            y=g_sharpe_mean,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Línea base Gurobi/SA (Sharpe: {g_sharpe_mean:.2f})"
        )
        axes[1].set_xscale('symlog', linthresh=0.01)
        axes[1].set_title("Sharpe Ratio Neto Out-of-Sample vs Coeficiente Alpha (α)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Alpha (α) - Escala Logarítmica", fontsize=11)
        axes[1].set_ylabel("Sharpe Ratio Neto (Out-of-Sample)", fontsize=11)
        axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper right")
        
        plt.suptitle("Impacto de la Regularización Ridge (α) en XY-QAOA (Fase de Entrenamiento vs Prueba)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_fig("qaoa_analysis_alpha_impact.png")
        plt.close()
    else:
        print("Advertencia: No hay datos suficientes de XY-QAOA Regularizado para generar gráfico de Alpha.")
        
    # ----------------------------------------------------------------
    # GRAFICO 2: TRAYECTORIA DE CONVERGENCIA (BARREN PLATEAUS)
    # ----------------------------------------------------------------
    if os.path.exists(trajectory_path):
        df_traj = pd.read_csv(trajectory_path)
        
        # Group by solver and iteration, averaging over seeds
        df_traj_grouped = df_traj.groupby(["solver", "iteration"])["cost"].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        
        solvers_display = {
            "xy_qaoa_normal": "Normal XY-QAOA (Random Init, α=0.0)",
            "xy_qaoa_regularized_alpha_0.0": "Regularizado (TQA Init, α=0.0 - Sin Penalización)",
            "xy_qaoa_regularized_alpha_0.1": "Regularizado (TQA Init, α=0.1 - Penalización Leve)",
            "xy_qaoa_regularized_alpha_1.0": "Regularizado (TQA Init, α=1.0 - Penalización Fuerte)"
        }
        
        colors = {
            "xy_qaoa_normal": "red",
            "xy_qaoa_regularized_alpha_0.0": "orange",
            "xy_qaoa_regularized_alpha_0.1": "green",
            "xy_qaoa_regularized_alpha_1.0": "purple"
        }
        
        for solver_key, solver_label in solvers_display.items():
            df_s = df_traj_grouped[df_traj_grouped["solver"] == solver_key]
            if not df_s.empty:
                plt.plot(
                    df_s["iteration"],
                    df_s["cost"],
                    label=solver_label,
                    color=colors[solver_key],
                    linewidth=2.0
                )
                
        plt.title("Evolución del Coste en el Optimizador Clásico (COBYLA)", fontsize=13, fontweight='bold')
        plt.xlabel("Iteración de Optimización Clásica", fontsize=11)
        plt.ylabel("Valor Esperado de la Función de Coste (E)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Configuración", loc="upper right")
        
        save_fig("qaoa_analysis_convergence.png")
        plt.close()
    else:
        print("Advertencia: No hay datos de trayectoria de convergencia disponibles.")
        
    # ----------------------------------------------------------------
    # GRAFICO 3: IMPACTO DE LA PROFUNDIDAD (P)
    # ----------------------------------------------------------------
    # Filter for Normal vs Regularized with alpha=0.1
    df_depth = df_sweep[df_sweep["solver"].isin(["xy_qaoa_normal", "xy_qaoa_regularized"])].copy()
    # For regularized, only keep alpha = 0.1 to compare p
    df_depth = df_depth[
        (df_depth["solver"] == "xy_qaoa_normal") | 
        ((df_depth["solver"] == "xy_qaoa_regularized") & (np.isclose(df_depth["alpha"], 0.1)))
    ]
    
    if not df_depth.empty and len(df_depth["p"].unique()) > 1:
        plt.figure(figsize=(10, 6))
        
        df_depth["solver_display"] = df_depth["solver"].replace({
            "xy_qaoa_normal": "XY-QAOA Normal (Random Init)",
            "xy_qaoa_regularized": "XY-QAOA Regularizado (TQA+Ridge, α=0.1)"
        })
        
        sns.lineplot(
            data=df_depth,
            x="p",
            y="in_sample_gap",
            hue="solver_display",
            marker="o",
            linewidth=2.0,
            palette=["red", "green"],
            errorbar="se"
        )
        
        plt.title("Optimization Gap vs Profundidad del Circuito (p)", fontsize=13, fontweight='bold')
        plt.xlabel("Profundidad del Circuito (p)", fontsize=11)
        plt.ylabel("Optimization Gap (In-Sample)", fontsize=11)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(df_depth["p"].unique())
        plt.legend(title="Solucionador")
        
        save_fig("qaoa_analysis_depth_gap.png")
        plt.close()
    else:
        print("Advertencia: No hay suficientes niveles de profundidad (p) para graficar el impacto.")
        
    print("\nGeneración de gráficos analíticos completada con éxito.")

if __name__ == "__main__":
    main()
