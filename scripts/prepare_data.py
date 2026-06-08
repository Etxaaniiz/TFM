import os
import sys
import pandas as pd

# Add src to Python path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.utils import load_config, create_directory_structure
from src.data.data_manager import download_data, compute_returns, compute_statistics

POOL_TICKERS = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'CSCO', 'ADBE',
    # US Blue Chips & Financials
    'JPM', 'V', 'MA', 'PG', 'KO', 'PEP', 'JNJ', 'WMT', 'DIS', 'PFE',
    # IBEX 35 (Spain)
    'SAN.MC', 'BBVA.MC', 'TEF.MC', 'ITX.MC', 'REP.MC', 'IBE.MC', 'CABK.MC', 'SAB.MC', 'ACS.MC', 'FER.MC',
    # US Others
    'NFLX', 'INTC', 'AMD', 'QCOM', 'TXN', 'HON', 'AMGN', 'SBUX', 'MDLZ', 'GILD',
    'NKE', 'ORCL', 'IBM', 'GE', 'CAT', 'GS', 'MS', 'AXP', 'BAC', 'C'
]

def main():
    # Load configuration
    config = load_config()
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    raw_path = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    
    # Create required directory structure
    create_directory_structure()
    
    # Download raw data
    prices = download_data(POOL_TICKERS, start=start_date, end=end_date)
    
    # Save raw data
    prices.to_csv(raw_path)
    print(f"Saved raw prices to {raw_path} ({prices.shape[0]} rows, {prices.shape[1]} columns)")
    
    # Compute returns
    returns = compute_returns(prices)
    returns_path = os.path.join(processed_dir, "returns.csv")
    returns.to_csv(returns_path)
    print(f"Saved log returns to {returns_path}")
    
    # Compute statistics
    mu, Sigma = compute_statistics(returns)
    mu_path = os.path.join(processed_dir, "mu.csv")
    Sigma_path = os.path.join(processed_dir, "Sigma.csv")
    
    mu.to_csv(mu_path)
    Sigma.to_csv(Sigma_path)
    print(f"Saved annualized expected returns (mu) to {mu_path}")
    print(f"Saved annualized covariance matrix (Sigma) to {Sigma_path}")
    print("Data preparation complete successfully!")

if __name__ == "__main__":
    main()
