# src/metrics/stats_tests.py
import numpy as np
from scipy.stats import wilcoxon

def paired_wilcoxon(df, solver_a, solver_b, group_col, value_col):
    """Test de Wilcoxon pareado por semilla, para cada valor de group_col (p.ej. N)."""
    results = {}
    for g, sub in df.groupby(group_col):
        a = sub[sub.Solver == solver_a].sort_values('seed')[value_col].values
        b = sub[sub.Solver == solver_b].sort_values('seed')[value_col].values
        stat, p = wilcoxon(a, b)
        results[g] = {"mean_a": a.mean(), "mean_b": b.mean(), "p_value": p}
    return results

def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi