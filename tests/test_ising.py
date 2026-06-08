import os
import sys
import numpy as np
import pytest

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.portfolio.portfolio_model import qubo_to_ising

def test_qubo_to_ising_equivalence():
    # Define a simple 2-variable QUBO matrix Q
    # E_qubo(x) = x^T * Q * x = Q_00 * x0 + Q_11 * x1 + (Q_01 + Q_10) * x0 * x1
    Q = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ])
    # Total Q_01 + Q_10 is 1.0
    
    h, J, offset = qubo_to_ising(Q)
    
    # Test all 4 combinations of binary values and their corresponding spins
    # x_i in {0, 1} matches s_i in {-1, 1} via x_i = (s_i + 1) / 2
    # E_qubo = Q_00 * x0 + Q_11 * x1 + 2 * Q_01 * x0 * x1
    # E_ising = sum(h_i * s_i) + sum(J_ij * s_i * s_j) + offset
    for x0 in [0, 1]:
        for x1 in [0, 1]:
            s0 = 2 * x0 - 1
            s1 = 2 * x1 - 1
            
            # Calculate QUBO energy
            x = np.array([x0, x1])
            e_qubo = x0 * Q[0, 0] + x1 * Q[1, 1] + (Q[0, 1] + Q[1, 0]) * x0 * x1
            
            # Calculate Ising energy
            e_ising = h[0] * s0 + h[1] * s1 + J.get((0, 1), 0.0) * s0 * s1 + offset
            
            assert np.isclose(e_qubo, e_ising)
