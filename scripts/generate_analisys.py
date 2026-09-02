import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.portfolio.portfolio_model import build_qubo
from src.quantum.classical_emulators import QuantumStatevectorSimulator


# ============================================================================
# NOISE ANALYSIS
# ============================================================================

N_NOISE = 12
K_NOISE = 6
P_NOISE = 3
LAMBDA_VAL = 0.5
N_SHOTS = 5000
SEEDS_NOISE = [42, 43, 44, 45, 46]
P_ERR_VALS = [0.00, 0.01, 0.05, 0.10]
MAXITER_NOISE = 50


def load_noise_data(processed_dir):
    mu_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Estable.csv"))
    cov_df = pd.read_csv(os.path.join(processed_dir, "covariance_Estable.csv"), index_col=0)
    tickers = [t for t in mu_df['Ticker'] if t in cov_df.columns]
    mu_all = mu_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_all = cov_df.loc[tickers, tickers]
    return tickers, mu_all, cov_all


def sample_with_bitflip_noise(probs, n_assets, k_cardinality, p_err, n_shots, rng):
    indices = rng.choice(len(probs), size=n_shots, p=probs)
    feasible_count = 0
    for idx in indices:
        bits = np.array([int(b) for b in bin(idx)[2:].zfill(n_assets)], dtype=int)
        flip_mask = rng.random(n_assets) < p_err
        bits_noisy = np.bitwise_xor(bits, flip_mask.astype(int))
        if np.sum(bits_noisy) == k_cardinality:
            feasible_count += 1
    return feasible_count / n_shots


def optimize_qaoa(sim, p, mixer, seed_opt, maxiter_opt):
    rng = np.random.RandomState(seed_opt)
    init_point = rng.rand(2 * p) * np.pi / 2.0

    def objective(params):
        energy, _ = sim.simulate_qaoa(p, params, mixer=mixer)
        return energy

    res = minimize(objective, init_point, method='COBYLA', options={'maxiter': maxiter_opt})
    return res.x


def run_noise_analysis():
    print("=" * 70)
    print("ANÁLISIS DE ROBUSTEZ AL RUIDO (OE2 extensión)")
    print(f"  N={N_NOISE}, K={K_NOISE}, p={P_NOISE}, n_shots={N_SHOTS}")
    print(f"  Intensidades de ruido: {[f'{pe*100:.0f}%' for pe in P_ERR_VALS]}")
    print("=" * 70)

    processed_dir = os.path.join(project_root, "data", "processed")
    all_tickers, mu_all, cov_all = load_noise_data(processed_dir)

    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "Standard QAOA (RX)": {pe: [] for pe in P_ERR_VALS},
        "XY-QAOA (TQA)": {pe: [] for pe in P_ERR_VALS},
    }

    for seed in SEEDS_NOISE:
        rng_sel = np.random.RandomState(seed)
        idx_sel = rng_sel.choice(len(all_tickers), size=N_NOISE, replace=False)
        selected = [all_tickers[i] for i in idx_sel]

        mu_arr = mu_all.loc[selected].values
        cov_arr = cov_all.loc[selected, selected].values
        Q = build_qubo(mu_arr, cov_arr, K_NOISE, lambda_val=LAMBDA_VAL)

        print(f"\n  Seed {seed} — Optimizando ángulos QAOA...")
        sim_rx = QuantumStatevectorSimulator(N_NOISE, K_NOISE, Q, LAMBDA_VAL)
        sim_xy = QuantumStatevectorSimulator(N_NOISE, K_NOISE, Q, LAMBDA_VAL)

        opt_rx = optimize_qaoa(sim_rx, P_NOISE, "rx", seed_opt=seed, maxiter_opt=MAXITER_NOISE)
        opt_xy = optimize_qaoa(sim_xy, P_NOISE, "xy", seed_opt=seed, maxiter_opt=MAXITER_NOISE)

        _, probs_rx = sim_rx.simulate_qaoa(P_NOISE, opt_rx, mixer="rx")
        _, probs_xy = sim_xy.simulate_qaoa(P_NOISE, opt_xy, mixer="xy")

        rng_noise = np.random.RandomState(seed + 100)

        for p_err in P_ERR_VALS:
            feas_rx = sample_with_bitflip_noise(probs_rx, N_NOISE, K_NOISE, p_err, N_SHOTS, rng_noise)
            feas_xy = sample_with_bitflip_noise(probs_xy, N_NOISE, K_NOISE, p_err, N_SHOTS, rng_noise)
            results["Standard QAOA (RX)"][p_err].append(feas_rx * 100.0)
            results["XY-QAOA (TQA)"][p_err].append(feas_xy * 100.0)
            print(f"    p_err={p_err*100:4.0f}%  |  RX feas={feas_rx*100:.1f}%  |  XY feas={feas_xy*100:.1f}%")

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
        "Standard QAOA (RX)": "#EA580C",
        "XY-QAOA (TQA)": "#10B981",
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    x_ticks = [pe * 100 for pe in P_ERR_VALS]

    for solver_name, color in palette.items():
        means = [np.mean(results[solver_name][pe]) for pe in P_ERR_VALS]
        stds = [np.std(results[solver_name][pe]) for pe in P_ERR_VALS]
        ax.plot(x_ticks, means, marker='o', linewidth=2.0, label=solver_name, color=color, zorder=3)
        ax.fill_between(x_ticks,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("Intensidad de ruido ($p_{err}$, %)")
    ax.set_ylabel("Tasa de Factibilidad (%)")
    ax.set_title(
        f"Degradación de Factibilidad bajo Ruido Bit-Flip\n"
        f"(XY-QAOA TQA vs. QAOA estándar, N={N_NOISE}, K={K_NOISE}, p={P_NOISE})",
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

    print("\n  Tabla de resultados (media ± std sobre 5 semillas):")
    print(f"  {'p_err':>6}  {'RX mean':>10}  {'RX std':>8}  {'XY mean':>10}  {'XY std':>8}")
    for pe in P_ERR_VALS:
        rx_m = np.mean(results["Standard QAOA (RX)"][pe])
        rx_s = np.std(results["Standard QAOA (RX)"][pe])
        xy_m = np.mean(results["XY-QAOA (TQA)"][pe])
        xy_s = np.std(results["XY-QAOA (TQA)"][pe])
        print(f"  {pe*100:6.0f}%  {rx_m:10.2f}%  {rx_s:8.2f}  {xy_m:10.2f}%  {xy_s:8.2f}")


# ============================================================================
# VARIANCE ANALYSIS
# ============================================================================


def run_variance_analysis():
    print("=" * 70)
    print("ANÁLISIS DE VARIANZA INTER-SEMILLA (OE3)")
    print("=" * 70)

    csv_path = os.path.join(project_root, "output", "results", "results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado. Ejecuta run_benchmarks.py primero.")
        return

    df = pd.read_csv(csv_path)
    if 'experiment_phase' in df.columns:
        df = df[df['experiment_phase'] == 'N_scaling']
    if 'is_best_restart' in df.columns:
        df = df[df['is_best_restart'].fillna(True)]

    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)

    solvers_of_interest = ["Standard QAOA", "XY-QAOA Regularized"]
    df_scale = df[
        (df['p'].isna() | (df['p'] == 3)) &
        (df['Solver'].isin(solvers_of_interest))
    ].copy()

    label_map = {
        "Standard QAOA": "QAOA estándar\n(RX Mixer)",
        "XY-QAOA Regularized": "XY-QAOA Reg.\n(Ridge + TQA)"
    }
    df_scale['Solver_Label'] = df_scale['Solver'].map(label_map)

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


    print("  Generando figura std_gap_vs_N.png...")
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))

    Ns_arr = [r['N'] for r in reduction_data]
    std_qaoa_arr = [r['std_qaoa'] for r in reduction_data]
    std_xy_arr = [r['std_xy'] for r in reduction_data]

    ax2.plot(Ns_arr, std_qaoa_arr, 'o-', color="#EA580C", linewidth=2.0,
             label="QAOA estándar (RX)", markersize=7)
    ax2.plot(Ns_arr, std_xy_arr, 's-', color="#10B981", linewidth=2.0,
             label="XY-QAOA Reg. (Ridge + TQA)", markersize=7)
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


# ============================================================================
# SCALING ANALYSIS
# ============================================================================


def exp_model(N, a, b):
    return a * np.exp(b * N)


def poly_model(N, a, b):
    return a * np.power(N, b)


def statevector_theoretical(N, t_ref, N_ref=6):
    return t_ref * np.power(2.0, N - N_ref)


def run_scaling_analysis():
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

    if 'experiment_phase' in df.columns:
        df_scale = df[df['experiment_phase'] == 'N_scaling'].copy()
    else:
        df_scale = df[df['p'].isna() | (df['p'] == 3)].copy()

    if 'is_best_restart' in df_scale.columns:
        df_scale = df_scale[df_scale['is_best_restart'].fillna(True)]

    agg = df_scale.groupby(['Solver', 'N'])['Execution Time (s)'].agg(mean='mean', std='std').reset_index()
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

    print("  Ajuste de modelos de escalado:")
    N_dense = np.linspace(min(Ns_all), 32, 300)

    gurobi_data = agg[agg['Solver'] == "Gurobi"].sort_values('N')
    Ns_g = gurobi_data['N'].values.astype(float)
    Ts_g = gurobi_data['mean'].values

    T_fit_g = None
    try:
        popt_g, _ = curve_fit(exp_model, Ns_g, Ts_g, p0=[1e-4, 0.2], maxfev=5000)
        a_g, b_g = popt_g
        T_fit_g = exp_model(N_dense, a_g, b_g)
        r2_g = 1.0 - np.sum((Ts_g - exp_model(Ns_g, a_g, b_g))**2) / np.sum((Ts_g - Ts_g.mean())**2)
        print(f"  Gurobi   -> exp(a={a_g:.4e}, b={b_g:.4f})  R²={r2_g:.4f}")
    except RuntimeError:
        print("  Gurobi   -> ajuste exponencial no convergio (pocos puntos)")

    xy_data = agg[agg['Solver'] == "XY-QAOA Regularized"].sort_values('N')
    Ns_xy = xy_data['N'].values.astype(float)
    Ts_xy = xy_data['mean'].values

    T_fit_xy = None
    try:
        popt_xy, _ = curve_fit(poly_model, Ns_xy, Ts_xy, p0=[1e-5, 2.0], maxfev=5000)
        a_xy, b_xy = popt_xy
        T_fit_xy = poly_model(N_dense, a_xy, b_xy)
        r2_xy = 1.0 - np.sum((Ts_xy - poly_model(Ns_xy, a_xy, b_xy))**2) / np.sum((Ts_xy - Ts_xy.mean())**2)
        print(f"  XY-QAOA  -> poly(a={a_xy:.4e}, b={b_xy:.4f})  R²={r2_xy:.4f}")
    except RuntimeError:
        print("  XY-QAOA  -> ajuste polinomico no convergio")

    t_ref = float(Ts_xy[0])
    N_ref = float(Ns_xy[0])
    T_theoretical_2N = statevector_theoretical(N_dense, t_ref, N_ref)

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

    if T_fit_g is not None:
        ax.plot(N_dense, T_fit_g, linestyle='--', color="#1E293B",
                alpha=0.6, linewidth=1.5, label=f"Ajuste exp. Gurobi")

    ax.plot(N_dense, T_theoretical_2N, linestyle=':', color="#10B981",
            alpha=0.7, linewidth=1.5, label="Complejidad teorica $\\propto 2^N$")

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
    ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', loc='upper left', fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "tiempo_escalado_cruce.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path}")


# ============================================================================
# RIDGE ALPHA ABLATION (alpha=0 vs alpha>0) - ESTILO EDITORIAL
# ============================================================================


def run_alpha_ablation_analysis():
    print("=" * 70)
    print("ABLACION DE ALPHA (Ridge L2): alpha=0 vs alpha>0")
    print("=" * 70)

    csv_path = os.path.join(
        project_root, "output", "results", "alpha_ablation.csv"
    )
    if not os.path.exists(csv_path):
        print(
            f"Error: {csv_path} no encontrado. Ejecuta 'python scripts/run_benchmarks.py --alpha-ablation' primero."
        )
        return

    df = pd.read_csv(csv_path)
    output_dir = os.path.join(
        project_root, "output", "figures_tfm", "6.Resultados"
    )
    os.makedirs(output_dir, exist_ok=True)

    labels = list(df.sort_values("alpha")["alpha_label"].unique())
    label_off, label_on = labels[0], labels[1]
    alpha_on_val = df[df["alpha_label"] == label_on]["alpha"].iloc[0]

    agg = (
        df.groupby(["alpha_label", "N"])[
            ["Optimization GAP (%)", "Execution Time (s)"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "alpha_label",
        "N",
        "gap_mean",
        "gap_std",
        "time_mean",
        "time_std",
    ]

    Ns = sorted(df["N"].unique())

    # --- Configuración Estética Global ---
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
            "legend.title_fontsize": 11,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 1.0,
        }
    )

    # Paleta balanceada: Slate Grey (Base/Off) vs. Vivid Indigo (Regularizado/On)
    style_cfg = {
        label_off: {
            "color": "#475569",
            "marker": "o",
            "linestyle": "--",
            "alpha_fill": 0.12,
        },
        label_on: {
            "color": "#4F46E5",
            "marker": "s",
            "linestyle": "-",
            "alpha_fill": 0.18,
        },
    }

    # ── 1. GapRegularizado.png ──────────────────────────────────────────
    print("\n  Generando figura GapRegularizado.png...")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in [label_off, label_on]:
        sub = agg[agg["alpha_label"] == label].sort_values("N")
        cfg = style_cfg[label]

        # Banda de dispersión (media ± std)
        ax.fill_between(
            sub["N"],
            np.maximum(0, sub["gap_mean"] - sub["gap_std"]),
            sub["gap_mean"] + sub["gap_std"],
            color=cfg["color"],
            alpha=cfg["alpha_fill"],
            zorder=2,
        )
        # Curva media con marcadores destacados
        ax.plot(
            sub["N"],
            sub["gap_mean"],
            label=label,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.2,
            marker=cfg["marker"],
            markersize=7,
            markeredgewidth=1.8,
            markeredgecolor="white",
            zorder=4,
        )

    ax.set_xlabel("Número de Activos ($N$)", labelpad=8, weight="medium")
    ax.set_ylabel("Optimization GAP (%)", labelpad=8, weight="medium")
    ax.set_title(
        "Impacto de la Regularización Ridge en el GAP de Optimización\n"
        rf"XY-QAOA con TQA ($\alpha = 0$ vs. $\alpha = {alpha_on_val}$)",
        weight="bold",
        pad=14,
    )

    ax.set_xticks(Ns)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#E2E8F0", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.95,
        loc="best",
    )
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    out_path_gap = os.path.join(output_dir, "GapRegularizado.png")
    plt.savefig(out_path_gap, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path_gap}")

    # ── 2. TiempoRegularizado.png ───────────────────────────────────────
    print("  Generando figura TiempoRegularizado.png...")
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))

    for label in [label_off, label_on]:
        sub = agg[agg["alpha_label"] == label].sort_values("N")
        cfg = style_cfg[label]

        # Banda de dispersión
        ax2.fill_between(
            sub["N"],
            np.maximum(0, sub["time_mean"] - sub["time_std"]),
            sub["time_mean"] + sub["time_std"],
            color=cfg["color"],
            alpha=cfg["alpha_fill"],
            zorder=2,
        )
        # Curva media
        ax2.plot(
            sub["N"],
            sub["time_mean"],
            label=label,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.2,
            marker=cfg["marker"],
            markersize=7,
            markeredgewidth=1.8,
            markeredgecolor="white",
            zorder=4,
        )

    ax2.set_xlabel("Número de Activos ($N$)", labelpad=8, weight="medium")
    ax2.set_ylabel("Tiempo de Ejecución (s)", labelpad=8, weight="medium")
    ax2.set_title(
        "Sobrecoste Temporal de la Regularización Ridge\n"
        rf"XY-QAOA con TQA ($\alpha = 0$ vs. $\alpha = {alpha_on_val}$)",
        weight="bold",
        pad=14,
    )

    ax2.set_xticks(Ns)
    ax2.set_ylim(bottom=0)
    ax2.grid(axis="y", linestyle="--", linewidth=0.7, color="#E2E8F0", zorder=0)
    ax2.grid(axis="x", visible=False)
    ax2.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.95,
        loc="best",
    )
    sns.despine(ax=ax2, top=True, right=True)

    plt.tight_layout()
    out_path_time = os.path.join(output_dir, "TiempoRegularizado.png")
    plt.savefig(out_path_time, dpi=300)
    plt.close()
    print(f"  [OK] Guardado en: {out_path_time}")


def main():
    run_noise_analysis()
    run_variance_analysis()
    run_scaling_analysis()
    run_alpha_ablation_analysis()


if __name__ == "__main__":
    main()
