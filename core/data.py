# core/data.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from datetime import datetime

def load_stock_data(file_path: str) -> pd.DataFrame:
    """Load stock price data from CSV."""
    data = pd.read_csv(file_path)
    
    # Ensure date column exists and is properly formatted
    if 'date' not in data.columns:
        raise ValueError("CSV file must have a 'date' column")
    
    # Convert date to datetime and set as index
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    return data

def filter_date_range(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter data by date range."""
    return data.loc[start_date:end_date].copy()

def calculate_returns_and_covariance(price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Calculate expected returns and covariance matrix."""
    from pypfopt import expected_returns, risk_models
    
def calculate_returns_and_covariance(price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Calculate expected returns and covariance matrix."""
    from pypfopt import expected_returns, risk_models
    
    # Calculate expected returns (annualized)
    mu = expected_returns.mean_historical_return(price_data)
    
    # Replace NaN values if any
    mu = mu.fillna(0.02)  # Default to 2% return
    
    # Calculate covariance matrix
    S = risk_models.sample_cov(price_data)
    
    # Ensure covariance matrix is positive definite
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # If not positive definite, add small adjustment to the diagonal
        min_eig = np.min(np.linalg.eigvals(S))
        if min_eig < 0:
            S = S + (-min_eig + 1e-5) * np.eye(len(S))
    
    return mu, S

def prepare_allocation_weights(tickers: List[str], allocations: Dict[str, float]) -> Dict[str, float]:
    """Prepare allocation weights, ensuring they sum to 1."""
    # If no allocations provided, use equal weights
    if not allocations:
        return {ticker: 1.0/len(tickers) for ticker in tickers}
    
    # Make sure all tickers have an allocation
    complete_allocations = {ticker: allocations.get(ticker, 0) for ticker in tickers}
    
    # Normalize to ensure sum is 1.0
    total = sum(complete_allocations.values())
    if total > 0:
        return {ticker: weight/total for ticker, weight in complete_allocations.items()}
    else:
        # Fallback to equal weights if all weights are zero
        return {ticker: 1.0/len(tickers) for ticker in tickers}

def prepare_constraints(tickers: List[str], allocations: Dict[str, float], 
                        constraints: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Prepare constraints for all tickers."""
    # If no explicit constraints, create default constraints
    if not constraints:
        return {ticker: (0, 1) for ticker in tickers}
    
    # Ensure all tickers have constraints
    complete_constraints = {}
    
    for ticker in tickers:
        if ticker in constraints:
            complete_constraints[ticker] = constraints[ticker]
        elif ticker in allocations:
            # Create constraint based on allocation value
            alloc = allocations[ticker]
            # Set min as max(0, allocation-0.05) and max as min(1, allocation+0.05)
            min_val = max(0, alloc - 0.05)
            max_val = min(1, alloc + 0.05)
            complete_constraints[ticker] = (min_val, max_val)
        else:
            # Default constraint for tickers without allocation or constraint
            complete_constraints[ticker] = (0, 1)
    
    return complete_constraints