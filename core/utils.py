# core/utils.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union

def weights_to_array(tickers: List[str], weights: Dict[str, float]) -> np.ndarray:
    """Convert weights dictionary to numpy array."""
    return np.array([weights.get(ticker, 0) for ticker in tickers])

def calculate_portfolio_metrics(
    weights: Union[np.ndarray, Dict[str, float]], 
    mu: pd.Series, 
    S: pd.DataFrame,
    price_data: pd.DataFrame = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate comprehensive portfolio performance metrics."""
    # Convert weights dict to array if needed
    if isinstance(weights, dict):
        tickers = list(mu.index)
        weights_array = weights_to_array(tickers, weights)
    else:
        weights_array = weights
    
    # Calculate basic portfolio metrics
    portfolio_return = weights_array @ mu
    portfolio_volatility = np.sqrt(weights_array @ S @ weights_array)
    
    # Calculate Sharpe ratio
    sharpe_ratio = 0.0
    if portfolio_volatility > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    metrics = {
        "expected_return": float(portfolio_return),
        "standard_deviation": float(portfolio_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "risk_free_rate": float(risk_free_rate)
    }
    
    # Calculate Sortino ratio if price data is provided
    if price_data is not None:
        sortino_ratio = calculate_sortino_ratio(weights_array, price_data, risk_free_rate)
        metrics["sortino_ratio"] = sortino_ratio
    
    return metrics

def calculate_sortino_ratio(
    weights: np.ndarray,
    price_data: pd.DataFrame,
    risk_free_rate: float = 0.02,
    target_return: float = 0.0
) -> float:
    """Calculate Sortino ratio (differentiating between upside and downside volatility)."""
    # Calculate daily returns
    daily_returns = price_data.pct_change().dropna()
    
    # Convert weights to dict if needed
    if isinstance(weights, np.ndarray):
        tickers = price_data.columns
        weights_dict = {tickers[i]: weights[i] for i in range(len(weights))}
    else:
        weights_dict = weights
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0, index=daily_returns.index)
    for ticker, weight in weights_dict.items():
        if ticker in daily_returns.columns:
            portfolio_returns += daily_returns[ticker] * weight
    
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    
    # Calculate downside returns (returns below target)
    downside_returns = portfolio_returns[portfolio_returns < target_return]
    
    # Calculate downside deviation (annualized)
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2) * 252)
    else:
        # If no downside returns, use small value to avoid division by zero
        downside_deviation = 1e-6
    
    # Calculate expected annual return
    expected_annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    
    # Calculate Sortino ratio
    sortino_ratio = (expected_annual_return - risk_free_rate) / downside_deviation
    
    return float(sortino_ratio)

def ensure_total_is_one(allocations: Dict[str, float]) -> Dict[str, float]:
    """Ensure allocations sum to exactly 1.0."""
    total = sum(allocations.values())
    
    if abs(total - 1.0) > 0.0001:
        # Sort tickers by allocation value
        sorted_tickers = sorted(allocations.keys(), key=lambda t: allocations[t])
        
        # Adjust the largest allocation to make sum = 1.0
        if sorted_tickers:
            allocations[sorted_tickers[-1]] += (1.0 - total)
    
    # Round to 4 decimal places
    return {ticker: round(alloc, 4) for ticker, alloc in allocations.items()}