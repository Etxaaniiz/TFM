import os
import sys
import numpy as np
import pytest

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.solvers.classic_solvers import solve_gurobi, GUROBI_AVAILABLE
from src.portfolio.portfolio_model import build_qubo

@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available in the current environment")
def test_solve_gurobi_correctness():
    # Simple portfolio problem
    # 4 assets, select 2
    mu = np.array([0.05, 0.20, 0.10, 0.15])
    Sigma = np.array([
        [0.05, 0.00, 0.00, 0.00],
        [0.00, 0.10, 0.00, 0.00],
        [0.00, 0.00, 0.08, 0.00],
        [0.00, 0.00, 0.00, 0.12]
    ])
    K = 2
    lambda_val = 0.5
    
    # Build QUBO just to have it in the instance
    Q = build_qubo(mu, Sigma, K, lambda_val)
    
    instance = {
        "dataset": "test",
        "N": 4,
        "K": K,
        "instance_id": 0,
        "seed": 42,
        "mu": mu,
        "Sigma": Sigma,
        "Q": Q,
        "offset": 0.0
    }
    
    res = solve_gurobi(instance, lambda_val=lambda_val)
    
    # Assert result keys
    assert res["dataset"] == "test"
    assert res["solver"] == "gurobi"
    assert res["N"] == 4
    assert res["K"] == K
    assert "objective" in res
    assert "sharpe" in res
    assert "expected_return" in res
    assert "volatility" in res
    assert res["feasible"] is True
    
    # Verify that exactly K variables are chosen
    sol = res["solution"]
    assert np.isclose(np.sum(sol), K)
