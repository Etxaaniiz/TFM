"""
generate_scaling_analysis.py
=============================
Analisis de Escalado Computacional y Punto de Cruce (OE4) — Prioridad 3.

Usa los datos reales de output/results/results.csv para:
  1. Ajustar una regresion exponencial al tiempo de Gurobi vs N.
  2. Comparar con el tiempo del simulador cuantico (XY-QAOA Reg.) vs N.
  3. Estimar el punto de cruce teorico entre ambas curvas.
  4. Generar la figura con escala logaritmica en el eje Y.

Genera:
  - output/figures_tfm/6.Resultados/tiempo_escalado_cruce.png
  - Imprime la tabla de tiempos y los parametros del ajuste.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ── Modelos de ajuste ─────────────────────────────────────────────────────────

def exp_model(N, a, b):
    """Modelo exponencial: T(N) = a * exp(b * N)"""
    return a * np.exp(b * N)


def poly_model(N, a, b):
    """Modelo polinomico cuadratico: T(N) = a * N^b"""
    return a * np.power(N, b)


def statevector_theoretical(N, t_ref, N_ref=6):
    """
    Tiempo teorico del simulador de statevector: T(N) ∝ 2^N.
    Anclado en t_ref a N_ref para comparacion directa con datos reales.
    """
    return t_ref * np.power(2.0, N - N_ref)


def main():
    print("=" * 70)
    print("ANÁLISIS DE ESCALADO COMPUTACIONAL Y PUNTO DE CRUCE (OE4)")
    print("=" * 70)

    csv_path = os.path.join(project_root, "output", "results", "results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado. Ejecuta run_benchmarks.py primero.")
        return

    df = pd.read_csv(csv_path)

    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)

    # ── Filtrar solo el estudio de escalado en N (p fijo = 3 o nulo) ─────────
    df_scale = df[df['p'].isna() | (df['p'] == 3)].copy()

    # ── Tabla de tiempos medios por solver y N ────────────────────────────────
    agg = df_scale.groupby(['Solver', 'N'])['Execution Time (s)'].agg(
        mean='mean', std='std'
    ).reset_index()

    Ns_all = sorted(agg['N'].unique())

    solvers_of_interest = {
        "Gurobi": {"color": "#1E293B", "label": "Gurobi (MIQP exacto)", "marker": "s"},
        "XY-QAOA Regularized": {"color": "#10B981", "label": "XY-QAOA Reg. (emulador)", "marker": "o"},
    }

    print("\n  Tiempos medios de ejecucion por solver y N:")
    print(f"  {'Solver':<25} {'N':>4}  {'Mean(s)':>10}  {'Std(s)':>8}")
    print("  " + "-" * 52)
    for s_name in solvers_of_interest:
        sub = agg[agg['Solver'] == s_name]
        for _, row in sub.iterrows():
            print(f"  {s_name:<25} {int(row['N']):>4}  {row['mean']:>10.6f}  {row['std']:>8.6f}")
        print()

    # ── Ajuste de modelos ─────────────────────────────────────────────────────
    print("  Ajuste de modelos de escalado:")

    fit_results = {}
    N_dense = np.linspace(min(Ns_all), 32, 300)

    # Gurobi: ajuste exponencial
    gurobi_data = agg[agg['Solver'] == "Gurobi"].sort_values('N')
    Ns_g = gurobi_data['N'].values.astype(float)
    Ts_g = gurobi_data['mean'].values

    try:
        popt_g, pcov_g = curve_fit(exp_model, Ns_g, Ts_g, p0=[1e-4, 0.2], maxfev=5000)
        a_g, b_g = popt_g
        perr_g = np.sqrt(np.diag(pcov_g))
        T_fit_g = exp_model(N_dense, a_g, b_g)
        r2_g = 1.0 - np.sum((Ts_g - exp_model(Ns_g, a_g, b_g))**2) / np.sum((Ts_g - Ts_g.mean())**2)
        fit_results["Gurobi"] = {"a": a_g, "b": b_g, "R2": r2_g, "T_fit": T_fit_g}
        print(f"  Gurobi   -> exp(a={a_g:.4e}, b={b_g:.4f})  R²={r2_g:.4f}")
    except RuntimeError:
        T_fit_g = None
        print("  Gurobi   -> ajuste exponencial no convergio (pocos puntos)")

    # XY-QAOA: los datos reales del emulador (polinomico ~ 2^N teorico)
    xy_data = agg[agg['Solver'] == "XY-QAOA Regularized"].sort_values('N')
    Ns_xy = xy_data['N'].values.astype(float)
    Ts_xy = xy_data['mean'].values

    # Ajuste polinomico (potencia)
    try:
        popt_xy, _ = curve_fit(poly_model, Ns_xy, Ts_xy, p0=[1e-5, 2.0], maxfev=5000)
        a_xy, b_xy = popt_xy
        T_fit_xy = poly_model(N_dense, a_xy, b_xy)
        r2_xy = 1.0 - np.sum((Ts_xy - poly_model(Ns_xy, a_xy, b_xy))**2) / np.sum((Ts_xy - Ts_xy.mean())**2)
        fit_results["XY-QAOA Reg."] = {"a": a_xy, "b": b_xy, "R2": r2_xy, "T_fit": T_fit_xy}
        print(f"  XY-QAOA  -> poly(a={a_xy:.4e}, b={b_xy:.4f})  R²={r2_xy:.4f}")
    except RuntimeError:
        T_fit_xy = None
        print("  XY-QAOA  -> ajuste polinomico no convergio")

    # Curva teorica 2^N, anclada en tiempo real a N=min(Ns_xy)
    t_ref = float(Ts_xy[0])
    N_ref = float(Ns_xy[0])
    T_theoretical_2N = statevector_theoretical(N_dense, t_ref, N_ref)

    # ── Punto de cruce ────────────────────────────────────────────────────────
    crossover_N = None
    if T_fit_g is not None:
        diff = T_theoretical_2N - T_fit_g
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            crossover_N = N_dense[sign_changes[0]]
            print(f"\n  Punto de cruce teorico (Gurobi ↔ Emulador 2^N): N ~ {crossover_N:.1f}")
        else:
            print("\n  Punto de cruce: no hay cruce dentro del rango N in [6, 32]")
            if T_fit_g[-1] > T_theoretical_2N[-1]:
                print("  -> El emulador sigue siendo MÁS LENTO que Gurobi en todo el rango observado")
            else:
                print("  -> El emulador ya es MÁS RÁPIDO que Gurobi en N=32 (extrapolado)")

    # ── Figura ────────────────────────────────────────────────────────────────
    print("\n  Generando figura tiempo_escalado_cruce.png...")

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
        'legend.fontsize': 9.5,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif'
    })

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Datos empiricos
    for s_name, style in solvers_of_interest.items():
        sub = agg[agg['Solver'] == s_name].sort_values('N')
        ax.errorbar(
            sub['N'], sub['mean'], yerr=sub['std'],
            fmt=style['marker'] + '-',
            color=style['color'],
            label=style['label'],
            linewidth=1.75,
            capsize=4, capthick=1.2,
            markersize=6,
            zorder=4
        )

    # Curvas de ajuste
    if T_fit_g is not None:
        ax.plot(N_dense, T_fit_g, linestyle='--', color="#1E293B",
                alpha=0.6, linewidth=1.5, label=f"Ajuste exp. Gurobi (b={b_g:.3f})")

    ax.plot(N_dense, T_theoretical_2N, linestyle=':', color="#10B981",
            alpha=0.7, linewidth=1.5, label="Complejidad teorica $\\propto 2^N$")

    # Linea del punto de cruce
    if crossover_N is not None:
        ax.axvline(x=crossover_N, color='#DC2626', linestyle='-.', linewidth=1.2, alpha=0.8,
                   label=f"Cruce teorico N~{crossover_N:.0f}")

    ax.set_yscale('log')
    ax.set_xlabel("Numero de Activos ($N$)")
    ax.set_ylabel("Tiempo de Ejecucion (s, escala log.)")
    ax.set_title(
        "Escalado Computacional: Gurobi vs. Emulador XY-QAOA\n"
        "(Tiempo real + ajuste de complejidad + cruce teorico)",
        weight='bold', pad=12
    )
    ax.set_xticks(Ns_all)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', loc='upper left',
              fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "tiempo_escalado_cruce.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path}")
    print("\n" + "=" * 70)
    print("ANÁLISIS DE ESCALADO — COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
