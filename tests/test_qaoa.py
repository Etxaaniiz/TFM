import os
import sys
import numpy as np
import pytest

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.portfolio.portfolio_model import build_qubo
from src.quantum.quantum_solvers import solve_qaoa, solve_xy, solve_jasp

def test_quantum_solvers_execution():
    # Tiny portfolio problem: N=3, K=1
    mu = np.array([0.1, 0.2, 0.15])
    Sigma = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.015],
        [0.02, 0.015, 0.06]
    ])
    K = 1
    lambda_val = 0.5
    
    Q = build_qubo(mu, Sigma, K, lambda_val, penalty=10.0)
    
    instance = {
        "dataset": "test_quantum",
        "N": 3,
        "K": K,
        "instance_id": 0,
        "seed": 42,
        "mu": mu,
        "Sigma": Sigma,
        "Q": Q,
        "offset": 10.0 * (K ** 2),
        "lambda_val": lambda_val
    }
    
    # 1. Test standard QAOA (p=1, maxiter=2, shots=100)
    # Using small parameters to keep the tests very fast
    res_qaoa = solve_qaoa(instance, p=1, maxiter=2, shots=100)
    assert res_qaoa["solver"] == "qaoa"
    assert res_qaoa["N"] == 3
    assert len(res_qaoa["solution"]) == 3
    assert "objective" in res_qaoa
    
    # 2. Test XY-QAOA (p=1, maxiter=2, shots=100)
    res_xy = solve_xy(instance, p=1, maxiter=2, shots=100)
    assert res_xy["solver"] == "xy_qaoa"
    assert res_xy["N"] == 3
    # XY-QAOA is initialized to Dicke state, so it must always yield a feasible solution (sum(x) == K)
    assert np.isclose(np.sum(res_xy["solution"]), K)
    
    # 3. Test JaspQAOA (p=2, maxiter=2, shots=100)
    res_jasp = solve_jasp(instance, p=2, maxiter=2, shots=100)
    assert res_jasp["solver"] == "jasp_qaoa"
    assert res_jasp["N"] == 3
    assert len(res_jasp["solution"]) == 3
