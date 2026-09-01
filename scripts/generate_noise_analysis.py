"""
generate_noise_analysis.py
===========================
Análisis de Robustez al Ruido (OE2 extensión) — Prioridad 4.

Simula el efecto de ruido cuántico (bit-flip / depolarizing) sobre el
muestreo final de XY-QAOA vs QAOA estándar.

El modelo de ruido se aplica SOLO al muestreo posterior al circuito
(las probabilidades del statevector): por cada estado muestreado,
cada bit se invierte con probabilidad p_err (bit-flip).

Se barre p_err ∈ {0%, 1%, 5%, 10%} con n_shots muestras por configuración.
La tasa de factibilidad se mide como fracción de muestras con Hamming(x) = K.

Genera: output/figures_tfm/6.Resultados/factibilidad_vs_ruido.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.portfolio.portfolio_model import build_qubo
from src.quantum.classical_emulators import QuantumStatevectorSimulator


# ── Configuración del experimento ─────────────────────────────────────────────
N = 12          # Tamaño de instancia fijo para el análisis de ruido
K = 6
p = 3           # Profundidad QAOA fija
lambda_val = 0.5
n_shots = 5000  # Muestras por evaluación de tasa de factibilidad
seeds = [42, 43, 44, 45, 46]  # 5 sub-instancias independientes
p_err_vals = [0.00, 0.01, 0.05, 0.10]  # Intensidades de ruido bit-flip
maxiter = 50    # Iteraciones COBYLA


def load_data(processed_dir):
    """Carga datos reales del régimen Estable."""
    import pandas as pd
    mu_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Estable.csv"))
    cov_df = pd.read_csv(os.path.join(processed_dir, "covariance_Estable.csv"), index_col=0)
    tickers = [t for t in mu_df['Ticker'] if t in cov_df.columns]
    mu_all = mu_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_all = cov_df.loc[tickers, tickers]
    return tickers, mu_all, cov_all


def sample_with_bitflip_noise(probs, N, K, p_err, n_shots, rng):
    """
    Muestrea `n_shots` estados desde la distribución de probabilidad `probs`
    y aplica ruido bit-flip a cada bit con probabilidad `p_err`.

    Devuelve la tasa de factibilidad (fracción de muestras con Hamming(x) == K).
    """
    # Muestrear índices de estado según probs
    indices = rng.choice(len(probs), size=n_shots, p=probs)
    feasible_count = 0

    for idx in indices:
        # Decodificar bit-string
        bits = np.array([int(b) for b in bin(idx)[2:].zfill(N)], dtype=int)
        # Aplicar bit-flip con probabilidad p_err a cada bit
        flip_mask = rng.random(N) < p_err
        bits_noisy = np.bitwise_xor(bits, flip_mask.astype(int))
        if np.sum(bits_noisy) == K:
            feasible_count += 1

    return feasible_count / n_shots


def optimize_qaoa(sim, p, mixer, seed_opt, maxiter_opt):
    """Optimiza ángulos QAOA con COBYLA partiendo de punto aleatorio."""
    rng = np.random.RandomState(seed_opt)
    init_point = rng.rand(2 * p) * np.pi / 2.0

    def objective(params):
        energy, _ = sim.simulate_qaoa(p, params, mixer=mixer)
        return energy

    res = minimize(objective, init_point, method='COBYLA',
                   options={'maxiter': maxiter_opt})
    return res.x


def main():
    print("=" * 70)
    print("ANÁLISIS DE ROBUSTEZ AL RUIDO (OE2 extensión)")
    print(f"  N={N}, K={K}, p={p}, n_shots={n_shots}")
    print(f"  Intensidades de ruido: {[f'{pe*100:.0f}%' for pe in p_err_vals]}")
    print("=" * 70)

    processed_dir = os.path.join(project_root, "data", "processed")
    all_tickers, mu_all, cov_all = load_data(processed_dir)

    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)

    # Estructura: {mixer: {p_err: [feas_rate_per_seed]}}
    results = {
        "Standard QAOA (RX)": {pe: [] for pe in p_err_vals},
        "XY-QAOA Regularizado": {pe: [] for pe in p_err_vals},
    }

    for seed in seeds:
        rng_sel = np.random.RandomState(seed)
        idx_sel = rng_sel.choice(len(all_tickers), size=N, replace=False)
        selected = [all_tickers[i] for i in idx_sel]

        mu_arr = mu_all.loc[selected].values
        cov_arr = cov_all.loc[selected, selected].values
        Q = build_qubo(mu_arr, cov_arr, K, lambda_val=lambda_val)

        print(f"\n  Seed {seed} — Optimizando ángulos QAOA...")

        sim_rx = QuantumStatevectorSimulator(N, K, Q, lambda_val)
        sim_xy = QuantumStatevectorSimulator(N, K, Q, lambda_val)

        # Optimizar ángulos para cada solver
        opt_rx = optimize_qaoa(sim_rx, p, "rx", seed_opt=seed, maxiter_opt=maxiter)
        opt_xy = optimize_qaoa(sim_xy, p, "xy", seed_opt=seed, maxiter_opt=maxiter)

        # Obtener distribución de probabilidad óptima
        _, probs_rx = sim_rx.simulate_qaoa(p, opt_rx, mixer="rx")
        _, probs_xy = sim_xy.simulate_qaoa(p, opt_xy, mixer="xy")

        rng_noise = np.random.RandomState(seed + 100)

        for p_err in p_err_vals:
            feas_rx = sample_with_bitflip_noise(probs_rx, N, K, p_err, n_shots, rng_noise)
            feas_xy = sample_with_bitflip_noise(probs_xy, N, K, p_err, n_shots, rng_noise)

            results["Standard QAOA (RX)"][p_err].append(feas_rx * 100.0)
            results["XY-QAOA Regularizado"][p_err].append(feas_xy * 100.0)

            print(f"    p_err={p_err*100:4.0f}%  |  RX feas={feas_rx*100:.1f}%  |  XY feas={feas_xy*100:.1f}%")

    # ── Figura: Tasa de factibilidad vs Intensidad de ruido ──────────────────
    print("\n  Generando figura factibilidad_vs_ruido.png...")

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
        "Standard QAOA (RX)": "#EA580C",   # Orange-red
        "XY-QAOA Regularizado": "#10B981",  # Emerald green
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    x_ticks = [pe * 100 for pe in p_err_vals]

    for solver_name, color in palette.items():
        means = [np.mean(results[solver_name][pe]) for pe in p_err_vals]
        stds = [np.std(results[solver_name][pe]) for pe in p_err_vals]

        ax.plot(x_ticks, means, marker='o', linewidth=2.0,
                label=solver_name, color=color, zorder=3)
        ax.fill_between(x_ticks,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("Intensidad de ruido ($p_{err}$, %)")
    ax.set_ylabel("Tasa de Factibilidad (%)")
    ax.set_title(
        f"Degradación de Factibilidad bajo Ruido Bit-Flip\n"
        f"(XY-QAOA Reg. vs. QAOA estándar, N={N}, K={K}, p={p})",
        weight='bold', pad=12
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{pe:.0f}%" for pe in x_ticks])
    ax.set_ylim(-5, 108)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', loc='lower left')
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "factibilidad_vs_ruido.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path}")

    # ── Tabla de resultados numéricos ─────────────────────────────────────────
    print("\n  Tabla de resultados (media ± std sobre 5 semillas):")
    print(f"  {'p_err':>6}  {'RX mean':>10}  {'RX std':>8}  {'XY mean':>10}  {'XY std':>8}")
    for pe in p_err_vals:
        rx_m = np.mean(results["Standard QAOA (RX)"][pe])
        rx_s = np.std(results["Standard QAOA (RX)"][pe])
        xy_m = np.mean(results["XY-QAOA Regularizado"][pe])
        xy_s = np.std(results["XY-QAOA Regularizado"][pe])
        print(f"  {pe*100:6.0f}%  {rx_m:10.2f}%  {rx_s:8.2f}  {xy_m:10.2f}%  {xy_s:8.2f}")

    print("\n" + "=" * 70)
    print("ANÁLISIS DE ROBUSTEZ AL RUIDO — COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
