"""
generate_variance_analysis.py
==============================
Análisis de Varianza Inter-semilla (OE3) — Prioridad 5.

Genera la figura de boxplot del GAP de optimización por semilla,
con y sin regularización Ridge+TQA, calculando la reducción de
desviación estándar.

Usa los datos reales de output/results/results.csv.

Genera:
  - output/figures_tfm/6.Resultados/boxplot_varianza_semillas.png
  - Imprime la tabla de reducción de varianza (std sin vs con regularización).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    print("=" * 70)
    print("ANÁLISIS DE VARIANZA INTER-SEMILLA (OE3)")
    print("=" * 70)

    csv_path = os.path.join(project_root, "output", "results", "results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado. Ejecuta run_benchmarks.py primero.")
        return

    df = pd.read_csv(csv_path)

    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)

    # ── Filtrar datos del estudio de escalado en N (p=3 fijo) ────────────────
    # Incluir Standard QAOA y XY-QAOA Regularized para comparación directa
    solvers_of_interest = ["Standard QAOA", "XY-QAOA Regularized"]
    df_scale = df[
        (df['p'].isna() | (df['p'] == 3)) &
        (df['Solver'].isin(solvers_of_interest))
    ].copy()

    # Etiquetas descriptivas para el gráfico
    label_map = {
        "Standard QAOA": "QAOA estándar\n(RX Mixer)",
        "XY-QAOA Regularized": "XY-QAOA Reg.\n(Ridge + TQA)"
    }
    df_scale['Solver_Label'] = df_scale['Solver'].map(label_map)

    # ── Tabla de varianza por N ───────────────────────────────────────────────
    print("\n  Reducción de std del GAP (%) — QAOA estándar vs. XY-QAOA Reg.:")
    print(f"  {'N':>4}  {'Std QAOA':>12}  {'Std XY-Reg':>12}  {'Reducción':>12}")
    print("  " + "-" * 48)

    Ns = sorted(df_scale['N'].unique())
    reduction_data = []

    for N in Ns:
        df_N = df_scale[df_scale['N'] == N]
        std_qaoa = df_N[df_N['Solver'] == "Standard QAOA"]['Optimization GAP (%)'].std()
        std_xyreg = df_N[df_N['Solver'] == "XY-QAOA Regularized"]['Optimization GAP (%)'].std()
        red_pct = (1.0 - std_xyreg / std_qaoa) * 100.0 if std_qaoa > 1e-9 else 0.0
        reduction_data.append({'N': N, 'std_qaoa': std_qaoa, 'std_xy': std_xyreg, 'reduction_pct': red_pct})
        print(f"  {N:>4}  {std_qaoa:>12.4f}  {std_xyreg:>12.4f}  {red_pct:>11.1f}%")

    global_std_qaoa = df_scale[df_scale['Solver'] == "Standard QAOA"]['Optimization GAP (%)'].std()
    global_std_xy = df_scale[df_scale['Solver'] == "XY-QAOA Regularized"]['Optimization GAP (%)'].std()
    global_red = (1.0 - global_std_xy / global_std_qaoa) * 100.0 if global_std_qaoa > 1e-9 else 0.0
    print(f"  {'GLOBAL':>4}  {global_std_qaoa:>12.4f}  {global_std_xy:>12.4f}  {global_red:>11.1f}%")
    print()
    print(f"  → Reducción global de std del GAP: {global_red:.1f}%")

    # ── Figura 1: Boxplot del GAP por Solver (todos los N agregados) ──────────
    print("\n  Generando figura boxplot_varianza_semillas.png...")

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        sns.set_style('whitegrid')

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif'
    })

    palette = {
        "Standard QAOA": "#EA580C",
        "XY-QAOA Regularized": "#10B981"
    }

    # ── Figura principal: Boxplot del GAP por semilla para cada N ─────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()

    for idx, N in enumerate(Ns[:8]):  # Máx 8 subplots (N=6,8,...,20)
        ax = axes[idx]
        df_N = df_scale[df_scale['N'] == N].copy()

        box_data = []
        box_labels = []
        box_colors = []
        for solver in ["Standard QAOA", "XY-QAOA Regularized"]:
            vals = df_N[df_N['Solver'] == solver]['Optimization GAP (%)'].values
            box_data.append(vals)
            short = "QAOA (RX)" if "Standard" in solver else "XY-Reg."
            box_labels.append(short)
            box_colors.append(palette[solver])

        bp = ax.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            widths=0.5, medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(linewidth=1.2, color='#64748B'),
            capprops=dict(linewidth=1.2, color='#64748B'),
            flierprops=dict(marker='o', markersize=4, alpha=0.5)
        )
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Scatter de puntos individuales (semillas)
        for j, (data, color) in enumerate(zip(box_data, box_colors)):
            x_jitter = np.random.RandomState(42).normal(j + 1, 0.05, size=len(data))
            ax.scatter(x_jitter, data, color=color, alpha=0.8, s=30, zorder=5)

        ax.set_title(f"N = {N}", weight='bold', fontsize=11)
        ax.set_ylabel("GAP (%)" if idx % 4 == 0 else "")
        ax.set_xlabel("")
        sns.despine(ax=ax)

    # Ocultar subplots vacíos si Ns tiene menos de 8 elementos
    for idx in range(len(Ns), 8):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Varianza Inter-Semilla del GAP de Optimización:\n"
        "QAOA Estándar vs. XY-QAOA Regularizado (Ridge + TQA)",
        weight='bold', fontsize=13, y=1.01
    )

    # Leyenda global
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#EA580C", alpha=0.75, label="QAOA estándar (RX)"),
        Patch(facecolor="#10B981", alpha=0.75, label="XY-QAOA Reg. (Ridge + TQA)"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               frameon=True, facecolor='white', edgecolor='#E2E8F0',
               bbox_to_anchor=(0.5, -0.04), fontsize=10)

    plt.tight_layout()

    out_path = os.path.join(output_dir, "boxplot_varianza_semillas.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado en: {out_path}")

    # ── Figura 2: Evolución de std del GAP vs N ───────────────────────────────
    print("  Generando figura std_gap_vs_N.png...")

    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))

    Ns_arr = [r['N'] for r in reduction_data]
    std_qaoa_arr = [r['std_qaoa'] for r in reduction_data]
    std_xy_arr = [r['std_xy'] for r in reduction_data]

    ax2.plot(Ns_arr, std_qaoa_arr, 'o-', color="#EA580C", linewidth=2.0,
             label="QAOA estándar (RX)", markersize=7)
    ax2.plot(Ns_arr, std_xy_arr, 's-', color="#10B981", linewidth=2.0,
             label="XY-QAOA Reg. (Ridge + TQA)", markersize=7)

    # Área entre curvas = varianza reducida
    ax2.fill_between(Ns_arr, std_qaoa_arr, std_xy_arr,
                     alpha=0.12, color='#7C3AED',
                     label=f"Varianza reducida (Δstd global={global_red:.1f}%)")

    ax2.set_xlabel("Número de Activos ($N$)")
    ax2.set_ylabel("Desviación Estándar del GAP (%)")
    ax2.set_title(
        "Reducción de Varianza Inter-Semilla del GAP de Optimización\n"
        "mediante Regularización Ridge + Inicialización TQA",
        weight='bold', pad=12
    )
    ax2.set_xticks(Ns_arr)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine(ax=ax2)
    plt.tight_layout()

    out_path2 = os.path.join(output_dir, "std_gap_vs_N.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path2}")

    print("\n" + "=" * 70)
    print("ANÁLISIS DE VARIANZA INTER-SEMILLA — COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
