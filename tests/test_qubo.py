import os
import sys
import numpy as np
import pytest

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.portfolio.portfolio_model import build_qubo

def test_build_qubo_dimensions_and_symmetry():
    # Simple parameters for N=3, K=1
    mu = np.array([0.1, 0.2, 0.15])
    Sigma = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.015],
        [0.02, 0.015, 0.06]
    ])
    K = 1
    lambda_val = 0.5
    
    Q = build_qubo(mu, Sigma, K, lambda_val, penalty=1.0)
    
    # Assert dimensions
    assert Q.shape == (3, 3)
    # Assert symmetry
    assert np.allclose(Q, Q.T)

def test_build_qubo_penalty_impact():
    # Verify that changing penalty changes the QUBO values
    mu = np.array([0.1, 0.2, 0.15])
    Sigma = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.015],
        [0.02, 0.015, 0.06]
    ])
    K = 2
    lambda_val = 0.5
    
    Q1 = build_qubo(mu, Sigma, K, lambda_val, penalty=1.0)
    Q2 = build_qubo(mu, Sigma, K, lambda_val, penalty=10.0)
    
    # Diagonals should differ
    # Q_ii = Q0_ii + P * (1 - 2*K)
    # Difference in P is 9.0. Diff on diagonal should be 9.0 * (1 - 4) = -27.0
    assert np.isclose(Q2[0, 0] - Q1[0, 0], -27.0)
    
    # Off-diagonals should differ
    # Q_ij = Q0_ij + P / 2.0
    # Diff on off-diagonal should be 9.0 / 2.0 = 4.5
    assert np.isclose(Q2[0, 1] - Q1[0, 1], 4.5)
