import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def setup_style():
    """
    Sets up high-quality plotting styles for the master's thesis.
    """
    sns.set_theme(style="whitegrid", rc={
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
        'grid.color': '#E2E8F0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#94A3B8',
        'axes.linewidth': 0.8,
        'xtick.color': '#64748B',
        'ytick.color': '#64748B'
    })
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 15,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

def generate_energy_distribution_table(output_path):
    """
    Simulates a small validation instance (N=6, K=2) using Qrisp and generates a
    formatted PDF/PNG table comparing state probabilities and QUBO energies.
    """
    import pickle
    from qrisp import QuantumVariable, dicke_state
    from qrisp.qaoa import QAOAProblem, RX_mixer, XY_mixer, create_QUBO_cost_operator, create_QUBO_cl_cost_function
    from src.metrics.metrics import calculate_qubo_energy
    
    # 1. Load a small validation instance (N=6, K=2)
    instance_path = "data/instances/instance_validation_6_0.pkl"
    if not os.path.exists(instance_path):
        print(f"Warning: Instance file {instance_path} not found. Skipping energy distribution table.")
        return
        
    try:
        with open(instance_path, "rb") as f:
            instance = pickle.load(f)
            
        N = instance['N']
        K = instance['K']
        Q = instance['Q']
        offset = instance.get('offset', 0.0)
        
        print(f"Simulating N={N}, K={K} validation instance for energy distribution table...")
        
        # 2. Run a small QAOA (p=1, shots=2048)
        qv_qaoa = QuantumVariable(N)
        qaoa_prob = QAOAProblem(
            cost_operator=create_QUBO_cost_operator(Q),
            mixer=RX_mixer,
            cl_cost_function=create_QUBO_cl_cost_function(Q)
        )
        res_qaoa = qaoa_prob.run(
            qarg=qv_qaoa,
            depth=1,
            max_iter=10,
            mes_kwargs={"shots": 2048}
        )
        
        # 3. Run a small XY-QAOA (p=1, shots=2048)
        qv_xy = QuantumVariable(N)
        def init_dicke(q_var):
            from qrisp import x
            for i in range(K):
                x(q_var[i])
            dicke_state(q_var, K)
            
        xy_prob = QAOAProblem(
            cost_operator=create_QUBO_cost_operator(Q),
            mixer=XY_mixer,
            cl_cost_function=create_QUBO_cl_cost_function(Q),
            init_function=init_dicke
        )
        res_xy = xy_prob.run(
            qarg=qv_xy,
            depth=1,
            max_iter=10,
            mes_kwargs={"shots": 2048}
        )
        
        # Convert counts to probabilities
        total_qaoa = sum(res_qaoa.values())
        total_xy = sum(res_xy.values())
        
        prob_qaoa = {k: v / total_qaoa for k, v in res_qaoa.items()}
        prob_xy = {k: v / total_xy for k, v in res_xy.items()}
        
        # Combine all unique states from both runs
        all_states = list(set(list(prob_qaoa.keys()) + list(prob_xy.keys())))
        
        # Calculate energy and build rows
        rows = []
        for state in all_states:
            x_vec = np.array([int(b) for b in state])
            energy = calculate_qubo_energy(x_vec, Q) + offset
            is_feasible = sum(x_vec) == K
            
            rows.append({
                "State": state,
                "Energy": energy,
                "Probability (QAOA)": prob_qaoa.get(state, 0.0),
                "Probability (XY Mixer)": prob_xy.get(state, 0.0),
                "Feasible": "Sí" if is_feasible else "No"
            })
            
        df_states = pd.DataFrame(rows)
        df_states.sort_values(by="Energy", inplace=True)
        
        # Select top 10 states by energy for display
        df_display = df_states.head(10).copy()
        
        # Format values
        df_display['Energy'] = df_display['Energy'].apply(lambda e: f"{e:.4f}")
        df_display.reset_index(drop=True, inplace=True)
        
        # Plotting the table using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.axis('off')
        ax.axis('tight')
        
        table_data = []
        headers = ["Estado (Bitstring)", "Energía QUBO", "Probabilidad (QAOA)", "Probabilidad (XY Mixer)", "Factible (\u03a3x_i = K)"]
        table_data.append(headers)
        for idx, row in df_display.iterrows():
            table_data.append([
                row["State"],
                row["Energy"],
                f"{row['Probability (QAOA)'] * 100:.2f}%",
                f"{row['Probability (XY Mixer)'] * 100:.2f}%",
                row["Feasible"]
            ])
            
        table = ax.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.24, 0.18, 0.20, 0.20, 0.18]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        for i, cell in table.get_celld().items():
            cell.set_height(0.08)
            if i[0] == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#0F4C81')  # Navy header
            else:
                if i[0] % 2 == 0:
                    cell.set_facecolor('#F8FAFC')
                else:
                    cell.set_facecolor('white')
                    
                # Soft color gradients for probabilities
                if i[1] == 2:
                    prob_val = df_display.loc[i[0]-1, "Probability (QAOA)"]
                    alpha = min(prob_val * 2.0, 0.8)
                    cell.set_facecolor(plt.cm.Oranges(alpha))
                elif i[1] == 3:
                    prob_val = df_display.loc[i[0]-1, "Probability (XY Mixer)"]
                    alpha = min(prob_val * 2.0, 0.8)
                    cell.set_facecolor(plt.cm.Reds(alpha))
                    
                if i[0] == 1 and i[1] == 0:  # row 1 is the best energy
                    cell.set_text_props(weight='bold')
                    
        plt.title("Distribución de Probabilidades y Energías (Instancia N=6, K=2)", fontsize=13, weight='bold', pad=15)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Energy distribution table saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating energy distribution table: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 1. Setup paths with fallbacks
    csv_path = "output/results/results.csv"
    if not os.path.exists(csv_path):
        csv_path = "results/results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}. Please run experiments first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    output_dir = "output/figures" if "output/results" in csv_path else "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup global styling
    setup_style()
    
    # 2. Preprocess data
    df['p_int'] = df['p'].apply(lambda x: int(x) if not pd.isna(x) else None)
    df['solver_display'] = df.apply(
        lambda r: f"{r['solver']} (p={int(r['p_int'])})" if not pd.isna(r['p_int']) else r['solver'],
        axis=1
    )
    
    # Define solver display settings (with vibrant colors for p=4 representative lines)
    display_colors = {
        'gurobi': '#0F4C81',            # Navy Blue
        'exact': '#0D9488',             # Teal
        'simulated_annealing': '#64748B', # Slate Grey
        'qaoa (p=1)': '#FDBA74',
        'qaoa (p=2)': '#F97316',
        'qaoa (p=3)': '#C2410C',
        'qaoa (p=4)': '#EA580C',         # Vibrant Rust Orange (Representative)
        'xy_qaoa (p=1)': '#FCA5A5',
        'xy_qaoa (p=2)': '#EF4444',
        'xy_qaoa (p=3)': '#B91C1C',
        'xy_qaoa (p=4)': '#E11D48',         # Vibrant Crimson Red (Representative)
        'jasp_qaoa (p=2)': '#C084FC',
        'jasp_qaoa (p=3)': '#8B5CF6',
        'jasp_qaoa (p=4)': '#7C3AED'          # Vibrant Royal Purple (Representative)
    }
    
    display_markers = {
        'gurobi': 'D',
        'exact': '^',
        'simulated_annealing': 'p',
        'qaoa (p=1)': 'o',
        'qaoa (p=2)': 'o',
        'qaoa (p=3)': 'o',
        'qaoa (p=4)': 'o',
        'xy_qaoa (p=1)': 's',
        'xy_qaoa (p=2)': 's',
        'xy_qaoa (p=3)': 's',
        'xy_qaoa (p=4)': 's',
        'jasp_qaoa (p=2)': 'v',
        'jasp_qaoa (p=3)': 'v',
        'jasp_qaoa (p=4)': 'v'
    }
    
    # Clean solver labels for legend
    rename_dict = {
        'gurobi': 'Gurobi (Exacto)',
        'exact': 'ExactSolver (Bruta)',
        'simulated_annealing': 'Simulated Annealing',
        'qaoa (p=1)': 'QAOA (p=1)',
        'qaoa (p=2)': 'QAOA (p=2)',
        'qaoa (p=3)': 'QAOA (p=3)',
        'qaoa (p=4)': 'QAOA (p=4)',
        'xy_qaoa (p=1)': 'XY-QAOA (p=1)',
        'xy_qaoa (p=2)': 'XY-QAOA (p=2)',
        'xy_qaoa (p=3)': 'XY-QAOA (p=3)',
        'xy_qaoa (p=4)': 'XY-QAOA (p=4)',
        'jasp_qaoa (p=2)': 'JaspQAOA (p=2)',
        'jasp_qaoa (p=3)': 'JaspQAOA (p=3)',
        'jasp_qaoa (p=4)': 'JaspQAOA (p=4)'
    }
    
    df['solver_display_clean'] = df['solver_display'].map(rename_dict)
    
    # Harmonized colors and markers mapped to clean names
    clean_colors = {rename_dict.get(k, k): v for k, v in display_colors.items()}
    clean_markers = {rename_dict.get(k, k): v for k, v in display_markers.items()}
    
    # Filter datasets
    solvers_of_interest = list(rename_dict.values())
    df_filtered = df[df['solver_display_clean'].isin(solvers_of_interest)].copy()
    
    # Precompute gaps & approximation ratios
    df_filtered['gap_pct'] = df_filtered['gap'] * 100
    df_filtered['approximation_ratio'] = 1 - df_filtered['gap']
    df_filtered['feasible_pct'] = df_filtered['feasible'].astype(float) * 100

    # Define representative p for comparison plots
    REPRESENTATIVE_P = 4
    
    # Create the comparison subset that only contains the representative p for quantum solvers,
    # and all classical solvers (which have p_int as None/NaN)
    df_comparison = df_filtered[
        (df_filtered['p_int'] == REPRESENTATIVE_P) | (df_filtered['p_int'].isna())
    ].copy()

    # Helper function to apply custom markers & styling in lineplot
    def make_lineplot(data, x, y, title, ylabel, filepath, scale='linear', limits=None, show_gurobi_baseline=False):
        plt.figure(figsize=(7.5, 4.5))
        
        # Calculate mean across seeds/instances
        df_mean = data.groupby(['N', 'solver_display_clean'])[y].mean().reset_index()
        
        # Determine the order based on rename_dict keys that are actually present in df_mean
        present_solvers = df_mean['solver_display_clean'].unique()
        local_ordered_solvers = [rename_dict[s] for s in rename_dict.keys() if rename_dict[s] in present_solvers]
        
        sns.lineplot(
            data=df_mean, x=x, y=y, hue='solver_display_clean', style='solver_display_clean',
            markers=clean_markers, palette=clean_colors, hue_order=local_ordered_solvers,
            style_order=local_ordered_solvers, linewidth=1.75, markersize=7, dashes=False
        )
        
        if scale == 'log':
            plt.yscale('log')
        if limits:
            plt.ylim(limits)
        if show_gurobi_baseline:
            plt.axhline(0 if '%' in ylabel else 1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Gurobi (Óptimo)')
            
        plt.title(title, weight='bold')
        plt.xlabel("Número de Activos ($N$)")
        plt.ylabel(ylabel)
        plt.xticks(sorted(data['N'].unique()))
        
        # Style legend
        plt.legend(title="Solver", bbox_to_anchor=(1.03, 1), loc='upper left', frameon=True, facecolor='white', edgecolor='#E2E8F0')
        sns.despine()
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------
    # Plot 1: Tiempo de Ejecución vs N (Escala Lineal) -> scalability_times.png
    # ----------------------------------------------------
    make_lineplot(
        df_comparison, 'N', 'runtime_seconds',
        "Tiempo de Ejecución de los Solvers vs. Número de Activos ($N$)",
        "Tiempo de Ejecución (segundos)",
        os.path.join(output_dir, "scalability_times.png")
    )
    
    # ----------------------------------------------------
    # Plot 2: Tiempo de Ejecución vs N (Escala Logarítmica) -> final_scalability_times.png
    # ----------------------------------------------------
    make_lineplot(
        df_comparison, 'N', 'runtime_seconds',
        "Tiempo de Ejecución vs. Número de Activos ($N$)",
        "Tiempo de Ejecución (segundos) - Escala Log",
        os.path.join(output_dir, "final_scalability_times.png"),
        scale='log'
    )
    
    # ----------------------------------------------------
    # Plot 3: Tiempos de Simulación con Qrisp -> scalability_times_qrisp_combined.png
    # ----------------------------------------------------
    df_qrisp = df_comparison[df_comparison['solver'].isin(['qaoa', 'xy_qaoa', 'jasp_qaoa'])].copy()
    if not df_qrisp.empty:
        make_lineplot(
            df_qrisp, 'N', 'runtime_seconds',
            "Tiempo de Simulación Cuántica (Qrisp) vs. Número de Activos ($N$)",
            "Tiempo de Simulación (segundos) - Escala Log",
            os.path.join(output_dir, "scalability_times_qrisp_combined.png"),
            scale='log'
        )
        
    # ----------------------------------------------------
    # Plot 4: Optimality GAP vs N -> quality_gap.png
    # ----------------------------------------------------
    df_no_gurobi = df_comparison[df_comparison['solver'] != 'gurobi'].copy()
    make_lineplot(
        df_no_gurobi, 'N', 'gap_pct',
        "Optimality GAP de los Solvers vs. Número de Activos ($N$)",
        "GAP relativo respecto a Gurobi (%)",
        os.path.join(output_dir, "quality_gap.png"),
        show_gurobi_baseline=True
    )
    # Also save as the original name gap_vs_N.png
    make_lineplot(
        df_no_gurobi, 'N', 'gap_pct',
        "Optimality GAP de los Solvers vs. Número de Activos ($N$)",
        "GAP relativo respecto a Gurobi (%)",
        os.path.join(output_dir, "gap_vs_N.png"),
        show_gurobi_baseline=True
    )
    
    # ----------------------------------------------------
    # Plot 5: Ratio de Aproximación vs N -> quality_approximation_ratio.png
    # ----------------------------------------------------
    make_lineplot(
        df_comparison, 'N', 'approximation_ratio',
        "Ratio de Aproximación vs. Número de Activos ($N$)",
        "Ratio de Aproximación ($1 - \\text{GAP}$)",
        os.path.join(output_dir, "quality_approximation_ratio.png"),
        limits=(0.0, 1.05),
        show_gurobi_baseline=True
    )
    
    # ----------------------------------------------------
    # Plot 6: Ratio de Sharpe vs N -> sharpe_vs_N.png
    # ----------------------------------------------------
    make_lineplot(
        df_comparison, 'N', 'sharpe',
        "Ratio de Sharpe Anualizado Promedio vs. Número de Activos ($N$)",
        "Ratio de Sharpe (Anualizado)",
        os.path.join(output_dir, "sharpe_vs_N.png")
    )
    
    # ----------------------------------------------------
    # Plot 7: Tasa de Factibilidad vs N -> feasibility_vs_N.png
    # ----------------------------------------------------
    make_lineplot(
        df_comparison, 'N', 'feasible_pct',
        "Factibilidad de las Soluciones vs. Número de Activos ($N$)",
        "Tasa de Factibilidad (%)",
        os.path.join(output_dir, "feasibility_vs_N.png"),
        limits=(-5, 105)
    )
    # Also save as the original name factibilidad_vs_N.png
    make_lineplot(
        df_comparison, 'N', 'feasible_pct',
        "Factibilidad de las Soluciones vs. Número de Activos ($N$)",
        "Tasa de Factibilidad (%)",
        os.path.join(output_dir, "factibilidad_vs_N.png"),
        limits=(-5, 105)
    )
    
    # ----------------------------------------------------
    # Plot 8: Factibilidad Standalone vs p (for QAOA) -> feasibility_rate_standalone.png
    # ----------------------------------------------------
    df_qaoa_only = df_filtered[df_filtered['solver'] == 'qaoa'].copy()
    if not df_qaoa_only.empty:
        plt.figure(figsize=(7.5, 4.5))
        # Group by p and N
        df_qaoa_feas = df_qaoa_only.groupby(['p_int', 'N'])['feasible_pct'].mean().reset_index()
        # Sort values
        df_qaoa_feas.sort_values(by='p_int', inplace=True)
        
        sns.lineplot(
            data=df_qaoa_feas, x='p_int', y='feasible_pct', hue='N', marker='o',
            palette='crest_r', linewidth=2, markersize=8, dashes=False
        )
        plt.title("Factibilidad de QAOA Estándar vs. Profundidad del Circuito ($p$)", weight='bold')
        plt.xlabel("Profundidad del Circuito ($p$)")
        plt.ylabel("Tasa de Factibilidad (%)")
        plt.xticks(sorted(df_qaoa_feas['p_int'].unique()))
        plt.ylim(-5, 105)
        plt.legend(title="Tamaño ($N$)", frameon=True, facecolor='white', edgecolor='#E2E8F0')
        sns.despine()
        plt.savefig(os.path.join(output_dir, "feasibility_rate_standalone.png"), bbox_inches='tight')
        plt.close()
        
    # ----------------------------------------------------
    # Plot 9: Comparativa Factibilidad QAOA vs XY-QAOA -> xy_vs_qaoa_feasibility.png
    # ----------------------------------------------------
    df_qaoa_xy = df_filtered[df_filtered['solver'].isin(['qaoa', 'xy_qaoa'])].copy()
    if not df_qaoa_xy.empty:
        # Average feasibility over all instances/seeds/p for each N and solver
        df_feas_mean = df_qaoa_xy.groupby(['solver', 'N'])['feasible_pct'].mean().reset_index()
        
        plt.figure(figsize=(7.5, 4.5))
        # Custom palette: Coral for QAOA, Crimson for XY-QAOA
        palette_feas = {'qaoa': '#F97316', 'xy_qaoa': '#B22222'}
        
        ax = sns.barplot(
            data=df_feas_mean, x='N', y='feasible_pct', hue='solver',
            palette=palette_feas, edgecolor='#94A3B8', linewidth=0.5
        )
        
        # Add labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3, fontsize=9)
            
        plt.title("Comparativa de Factibilidad: QAOA Estándar vs. XY-QAOA", weight='bold')
        plt.xlabel("Número de Activos ($N$)")
        plt.ylabel("Tasa de Factibilidad Promedio (%)")
        plt.ylim(0, 115)
        
        # Rename legend labels
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['QAOA Estándar', 'XY-QAOA (Restringido)'], title="Algoritmo", frameon=True)
        sns.despine()
        plt.savefig(os.path.join(output_dir, "xy_vs_qaoa_feasibility.png"), bbox_inches='tight')
        plt.close()
    df_quantum = df_filtered[df_filtered['solver'].isin(['qaoa', 'xy_qaoa'])].copy()
    if not df_quantum.empty:
        # Map solver values to clean names for accurate, automatic legends
        df_quantum['solver'] = df_quantum['solver'].map({'qaoa': 'QAOA Estándar', 'xy_qaoa': 'XY-QAOA'})
        
        plt.figure(figsize=(7.5, 4.5))
        df_quantum_p = df_quantum.groupby(['p_int', 'solver'])['approximation_ratio'].mean().reset_index()
        df_quantum_p.sort_values(by='p_int', inplace=True)
        
        palette_quant = {'QAOA Estándar': '#F97316', 'XY-QAOA': '#B22222'}
        sns.lineplot(
            data=df_quantum_p, x='p_int', y='approximation_ratio', hue='solver', style='solver',
            markers={'QAOA Estándar': 'o', 'XY-QAOA': 's'}, palette=palette_quant, linewidth=2, markersize=8, dashes=False
        )
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Gurobi (Óptimo)')
        plt.title("Ratio de Aproximación vs. Profundidad del Circuito ($p$)", weight='bold')
        plt.xlabel("Profundidad del Circuito ($p$)")
        plt.ylabel("Ratio de Aproximación ($1 - \\text{GAP}$)")
        plt.xticks(sorted(df_quantum_p['p_int'].unique()))
        plt.ylim(0.0, 1.05)
        
        plt.legend(title="Algoritmo", frameon=True)
        sns.despine()
        plt.savefig(os.path.join(output_dir, "comparison_qaoa_vs_xy.png"), bbox_inches='tight')
        plt.close()
        
        # ----------------------------------------------------
        # Plot 11 & 12: Circuit depth impact plots -> gap_vs_p.png & tiempo_vs_p.png
        # ----------------------------------------------------
        # GAP vs p
        plt.figure(figsize=(7.5, 4.5))
        df_gap_p = df_quantum.groupby(['p_int', 'solver'])['gap_pct'].mean().reset_index()
        df_gap_p.sort_values(by='p_int', inplace=True)
        sns.lineplot(
            data=df_gap_p, x='p_int', y='gap_pct', hue='solver', style='solver',
            markers={'QAOA Estándar': 'o', 'XY-QAOA': 's'}, palette=palette_quant, linewidth=2, markersize=8, dashes=False
        )
        plt.axhline(0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='Gurobi (Óptimo)')
        plt.title("Optimality GAP vs. Profundidad del Circuito ($p$)", weight='bold')
        plt.xlabel("Profundidad del Circuito ($p$)")
        plt.ylabel("GAP relativo respecto a Gurobi (%)")
        plt.xticks(sorted(df_gap_p['p_int'].unique()))
        plt.legend(title="Algoritmo", frameon=True)
        sns.despine()
        plt.savefig(os.path.join(output_dir, "gap_vs_p.png"), bbox_inches='tight')
        plt.close()
 
        # Tiempo vs p
        plt.figure(figsize=(7.5, 4.5))
        df_time_p = df_quantum.groupby(['p_int', 'solver'])['runtime_seconds'].mean().reset_index()
        df_time_p.sort_values(by='p_int', inplace=True)
        sns.lineplot(
            data=df_time_p, x='p_int', y='runtime_seconds', hue='solver', style='solver',
            markers={'QAOA Estándar': 'o', 'XY-QAOA': 's'}, palette=palette_quant, linewidth=2, markersize=8, dashes=False
        )
        plt.yscale('log')
        plt.title("Tiempo de Simulación vs. Profundidad del Circuito ($p$)", weight='bold')
        plt.xlabel("Profundidad del Circuito ($p$)")
        plt.ylabel("Tiempo de Simulación (segundos) - Escala Log")
        plt.xticks(sorted(df_time_p['p_int'].unique()))
        plt.legend(title="Algoritmo", frameon=True)
        sns.despine()
        plt.savefig(os.path.join(output_dir, "tiempo_vs_p.png"), bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------
    # Plot 13: Energy Distribution Table -> energy_distribution_table.png
    # ----------------------------------------------------
    generate_energy_distribution_table(os.path.join(output_dir, "energy_distribution_table.png"))
    
    print(f"Figures generation completed successfully. Visualizations saved in: {output_dir}")

if __name__ == "__main__":
    main()
