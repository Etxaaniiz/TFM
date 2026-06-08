import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_manager import compute_returns, compute_statistics

def test_compute_returns():
    # Create simple dummy prices
    dates = pd.date_range(start="2026-01-01", periods=5)
    prices = pd.DataFrame({
        "AAPL": [100.0, 101.0, 102.01, 99.9698, 100.9695],
        "MSFT": [200.0, 202.0, 204.02, 199.9396, 201.9390]
    }, index=dates)
    
    returns = compute_returns(prices)
    
    # log(101/100) = log(1.01) ~= 0.0099503
    assert returns.shape == (4, 2)
    assert np.allclose(returns.iloc[0]["AAPL"], np.log(101.0 / 100.0))
    assert np.allclose(returns.iloc[1]["AAPL"], np.log(102.01 / 101.0))

def test_compute_statistics():
    # Create dummy returns
    dates = pd.date_range(start="2026-01-01", periods=3)
    returns = pd.DataFrame({
        "AAPL": [0.01, -0.01],
        "MSFT": [0.02, -0.02]
    }, index=dates[1:])
    
    mu, Sigma = compute_statistics(returns, trading_days=252)
    
    # Expected daily return of AAPL is 0.0. Annualized expected return should be 0.0.
    assert np.isclose(mu["AAPL"], 0.0)
    assert np.isclose(mu["MSFT"], 0.0)
    
    # Variance of AAPL daily return is var([0.01, -0.01]) = 0.0002
    # Annualized variance = 0.0002 * 252 = 0.0504
    assert np.isclose(Sigma.loc["AAPL", "AAPL"], 0.0002 * 252)
    assert np.isclose(Sigma.loc["MSFT", "MSFT"], 0.0008 * 252)
