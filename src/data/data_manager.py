import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads Adjusted Close prices from Yahoo Finance for a list of tickers.
    Removes columns (tickers) that contain any NaN values.
    
    Parameters:
    tickers (list of str): List of ticker symbols.
    start (str): Start date in YYYY-MM-DD format.
    end (str): End date in YYYY-MM-DD format.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame of prices.
    """
    if not tickers:
        raise ValueError("The tickers list cannot be empty.")
        
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    # yfinance download
    data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
    
    # Extract Adjusted Close
    prices = pd.DataFrame()
    for ticker in tickers:
        if ticker in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [ticker]:
            try:
                # If MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    prices[ticker] = data[ticker]['Adj Close']
                else:
                    prices[ticker] = data['Adj Close']
            except KeyError:
                print(f"Warning: Adj Close data not found for ticker {ticker}.")
                
    # Remove assets containing NaN values
    initial_count = len(prices.columns)
    prices = prices.dropna(axis=1, how='any')
    final_count = len(prices.columns)
    
    if final_count < initial_count:
        print(f"Dropped {initial_count - final_count} assets due to missing data (NaN).")
        
    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes logarithmic returns for a given DataFrame of prices.
    
    Parameters:
    prices (pd.DataFrame): DataFrame of historical prices.
    
    Returns:
    pd.DataFrame: DataFrame of daily log returns.
    """
    # R_t = log(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def compute_statistics(returns: pd.DataFrame, trading_days: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Computes annualized expected returns (mu) and annualized covariance matrix (Sigma).
    
    Parameters:
    returns (pd.DataFrame): DataFrame of daily log returns.
    trading_days (int): Number of trading days in a year (default: 252).
    
    Returns:
    Tuple[pd.Series, pd.DataFrame]: Annualized expected returns (mu) and covariance matrix (Sigma).
    """
    # Expected daily returns and daily covariance
    daily_mu = returns.mean()
    daily_sigma = returns.cov()
    
    # Annualize
    mu = daily_mu * trading_days
    Sigma = daily_sigma * trading_days
    
    return mu, Sigma
