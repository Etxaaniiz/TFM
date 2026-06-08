import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def setup_style():
    """
    Sets up high-quality plotting styles for the master's thesis.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })
    # Harmonious color palette
    # Blue: Gurobi, Green: SA/Exact, Orange: QAOA, Red: XY-QAOA, Purple: Jasp
    return {
        'gurobi': '#1f77b4',
        'exact': '#2ca02c',
        'simulated_annealing': '#bcbd22',
        'qaoa': '#ff7f0e',
        'xy_qaoa': '#d62728',
        'jasp_qaoa': '#9467bd'
    }

def main():
    csv_path = "results/results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}. Please run experiments first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    os.makedirs("figures", exist_ok=True)
    
    colors = setup_style()
    
    # Create a column for composite solver name to distinguish p values
    # e.g., "qaoa (p=2)"
    df['solver_display'] = df.apply(
        lambda r: f"{r['solver']} (p={int(r['p'])})" if not pd.isna(r['p']) else r['solver'],
        axis=1
    )
    
    # Define order for display
    solvers_to_plot = ['gurobi', 'exact', 'simulated_annealing', 'qaoa (p=1)', 'qaoa (p=2)', 'xy_qaoa (p=1)', 'xy_qaoa (p=2)', 'jasp_qaoa (p=2)']
    # Filter to only show solvers present in results
    df_filtered = df[df['solver_display'].isin(solvers_to_plot) | df['solver'].isin(['gurobi', 'exact', 'simulated_annealing'])].copy()
    
    # ----------------------------------------------------
    # Figure 1: Tiempo vs N
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    # Group and plot lines
    sns.lineplot(data=df_filtered, x='N', y='runtime_seconds', hue='solver_display', marker='o', errorbar=None)
    plt.yscale('log')
    plt.title("Tiempo de Ejecución vs. Número de Activos (N)")
    plt.xlabel("Número de Activos (N)")
    plt.ylabel("Tiempo de Ejecución (segundos) - Escala Log")
    plt.legend(title="Solver", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("figures/tiempo_vs_N.png")
    plt.close()
    
    # ----------------------------------------------------
    # Figure 2: Gap vs N
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    # Exclude Gurobi as it's the 0 reference
    df_no_gurobi = df_filtered[df_filtered['solver'] != 'gurobi'].copy()
    # Convert gap to percentage
    df_no_gurobi['gap_pct'] = df_no_gurobi['gap'] * 100
    sns.lineplot(data=df_no_gurobi, x='N', y='gap_pct', hue='solver_display', marker='o', errorbar=None)
    plt.title("Optimality GAP vs. Número de Activos (N)")
    plt.xlabel("Número de Activos (N)")
    plt.ylabel("GAP relativo (%)")
    plt.legend(title="Solver", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("figures/gap_vs_N.png")
    plt.close()
    
    # ----------------------------------------------------
    # Figure 3: Sharpe vs N
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_filtered, x='N', y='sharpe', hue='solver_display', marker='o', errorbar=None)
    plt.title("Ratio de Sharpe vs. Número de Activos (N)")
    plt.xlabel("Número de Activos (N)")
    plt.ylabel("Ratio de Sharpe (Anualizado)")
    plt.legend(title="Solver", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("figures/sharpe_vs_N.png")
    plt.close()
    
    # ----------------------------------------------------
    # Figure 4: Factibilidad vs N
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    # Convert feasible (boolean) to float percentage
    df_filtered['feasible_pct'] = df_filtered['feasible'].astype(float) * 100
    sns.lineplot(data=df_filtered, x='N', y='feasible_pct', hue='solver_display', marker='o', errorbar=None)
    plt.title("Factibilidad de Soluciones vs. Número de Activos (N)")
    plt.xlabel("Número de Activos (N)")
    plt.ylabel("Ratio de Factibilidad (%)")
    plt.ylim(-5, 105)
    plt.legend(title="Solver", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("figures/factibilidad_vs_N.png")
    plt.close()
    
    # ----------------------------------------------------
    # Figure 5: Gap vs p (for QAOA versions)
    # ----------------------------------------------------
    df_quantum = df[df['solver'].isin(['qaoa', 'xy_qaoa', 'jasp_qaoa'])].copy()
    if not df_quantum.empty:
        df_quantum['gap_pct'] = df_quantum['gap'] * 100
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_quantum, x='p', y='gap_pct', hue='solver', style='solver', marker='s', errorbar=None, markersize=8)
        plt.title("Optimality GAP vs. Profundidad del Circuito (p)")
        plt.xlabel("Profundidad (p)")
        plt.ylabel("GAP relativo (%)")
        plt.xticks(df_quantum['p'].unique())
        plt.legend(title="Algoritmo Cuántico")
        plt.savefig("figures/gap_vs_p.png")
        plt.close()
        
        # ----------------------------------------------------
        # Figure 6: Tiempo vs p (for QAOA versions)
        # ----------------------------------------------------
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_quantum, x='p', y='runtime_seconds', hue='solver', style='solver', marker='s', errorbar=None, markersize=8)
        plt.yscale('log')
        plt.title("Tiempo de Ejecución vs. Profundidad del Circuito (p)")
        plt.xlabel("Profundidad (p)")
        plt.ylabel("Tiempo de Ejecución (segundos) - Escala Log")
        plt.xticks(df_quantum['p'].unique())
        plt.legend(title="Algoritmo Cuántico")
        plt.savefig("figures/tiempo_vs_p.png")
        plt.close()
        
    # ----------------------------------------------------
    # Figure 7: XY-QAOA vs. Standard QAOA (Feasibility Comparison)
    # ----------------------------------------------------
    df_qaoa_xy = df[df['solver'].isin(['qaoa', 'xy_qaoa'])].copy()
    if not df_qaoa_xy.empty:
        df_qaoa_xy['feasible_pct'] = df_qaoa_xy['feasible'].astype(float) * 100
        # Average feasibility over all instances for each N and solver
        df_feas = df_qaoa_xy.groupby(['solver', 'N'])['feasible_pct'].mean().reset_index()
        
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_feas, x='N', y='feasible_pct', hue='solver', palette=[colors['qaoa'], colors['xy_qaoa']])
        plt.title("Comparativa de Factibilidad: QAOA Estándar vs. XY-QAOA")
        plt.xlabel("Número de Activos (N)")
        plt.ylabel("Tasa de Factibilidad Promedio (%)")
        plt.ylim(0, 110)
        plt.legend(title="Algoritmo")
        plt.savefig("figures/xy_vs_qaoa_feasibility.png")
        plt.close()
        
    print("Figures generation completed. Visualizations saved in figures/.")

if __name__ == "__main__":
    main()
