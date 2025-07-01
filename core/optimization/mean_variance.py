# core/optimization/mean_variance.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pypfopt import efficient_frontier
import warnings

from core.utils import weights_to_array, calculate_portfolio_metrics, ensure_total_is_one

def add_constraints_to_ef(
    ef: efficient_frontier.EfficientFrontier,
    tickers: List[str],
    constraints: Dict[str, Tuple[float, float]]
) -> efficient_frontier.EfficientFrontier:
    """Add constraints to efficient frontier object."""
    for ticker in tickers:
        lower, upper = constraints.get(ticker, (0, 1))
        
        # Add lower and upper bound constraints
        ef.add_constraint(
            lambda w, ticker=ticker, idx=tickers.index(ticker): w[idx] >= lower
        )
        ef.add_constraint(
            lambda w, ticker=ticker, idx=tickers.index(ticker): w[idx] <= upper
        )
    
    return ef

def optimize_max_sharpe(
    mu: pd.Series, 
    S: pd.DataFrame,
    tickers: List[str],
    constraints: Dict[str, Tuple[float, float]],
    risk_free_rate: float = 0.02,
    price_data: pd.DataFrame = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Optimize portfolio for maximum Sharpe ratio."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Create efficient frontier object
        ef = efficient_frontier.EfficientFrontier(mu, S)
        
        # Add constraints
        ef = add_constraints_to_ef(ef, tickers, constraints)
        
        # Optimize for max Sharpe ratio
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            weights = ef.clean_weights()
            
            # Calculate portfolio metrics
            weights_array = weights_to_array(tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, mu, S, price_data, risk_free_rate
            )
            
            return weights, metrics
            
        except Exception as e:
            print(f"Max Sharpe optimization failed: {e}")
            # Fallback to min volatility
            return optimize_min_volatility(mu, S, tickers, constraints, risk_free_rate, price_data)

def optimize_min_volatility(
    mu: pd.Series, 
    S: pd.DataFrame,
    tickers: List[str],
    constraints: Dict[str, Tuple[float, float]],
    risk_free_rate: float = 0.02,
    price_data: pd.DataFrame = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Optimize portfolio for minimum volatility."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Create efficient frontier object
        ef = efficient_frontier.EfficientFrontier(mu, S)
        
        # Add constraints
        ef = add_constraints_to_ef(ef, tickers, constraints)
        
        # Optimize for min volatility
        try:
            ef.min_volatility()
            weights = ef.clean_weights()
            
            # Calculate portfolio metrics
            weights_array = weights_to_array(tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, mu, S, price_data, risk_free_rate
            )
            
            return weights, metrics
            
        except Exception as e:
            print(f"Min volatility optimization failed: {e}")
            # Fallback to equal weights
            equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
            equal_weights_array = weights_to_array(tickers, equal_weights)
            equal_metrics = calculate_portfolio_metrics(
                equal_weights_array, mu, S, price_data, risk_free_rate
            )
            
            return equal_weights, equal_metrics

def generate_efficient_frontier(
    mu: pd.Series, 
    S: pd.DataFrame, 
    tickers: List[str], 
    constraints: Dict[str, Tuple[float, float]],
    optimal_metrics: Dict[str, float],
    optimal_weights: Dict[str, float],
    num_portfolios: int = 20,
    price_data: pd.DataFrame = None,
    risk_free_rate: float = 0.02
) -> List[Dict[str, Any]]:
    """Generate portfolios along the efficient frontier."""
    frontier_portfolios = []
    
    # Get optimal parameters
    optimal_return = optimal_metrics["expected_return"]
    optimal_sharpe = optimal_metrics["sharpe_ratio"]
    
    # Determine return range for frontier
    mean_return = mu.mean()
    min_ret = max(-0.1, mean_return - 0.15)
    max_ret = min(0.3, mean_return + 0.15)
    
    # Create return targets
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    # Create efficient frontier object
    ef = efficient_frontier.EfficientFrontier(mu, S)
    ef = add_constraints_to_ef(ef, tickers, constraints)
    
    # Add optimal portfolio
    optimal_portfolio = {
        "weights": optimal_weights,
        "expected_return": optimal_metrics["expected_return"],
        "standard_deviation": optimal_metrics["standard_deviation"],
        "sharpe_ratio": optimal_metrics["sharpe_ratio"],
        "is_max_sharpe": True
    }
    
    if "sortino_ratio" in optimal_metrics:
        optimal_portfolio["sortino_ratio"] = optimal_metrics["sortino_ratio"]
        
    frontier_portfolios.append(optimal_portfolio)
    
    # Generate portfolios for each target return
    for target_return in target_returns:
        # Skip if too close to optimal return
        if abs(target_return - optimal_return) < 0.0001:
            continue
            
        try:
            # Reset efficient frontier for each target
            ef = efficient_frontier.EfficientFrontier(mu, S)
            ef = add_constraints_to_ef(ef, tickers, constraints)
            
            # Optimize for target return
            ef.efficient_return(target_return)
            weights = ef.clean_weights()
            
            # Calculate metrics
            weights_array = weights_to_array(tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, mu, S, price_data, risk_free_rate
            )
            
            # Add to frontier
            portfolio = {
                "weights": weights,
                "expected_return": metrics["expected_return"],
                "standard_deviation": metrics["standard_deviation"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "is_max_sharpe": False
            }
            
            if "sortino_ratio" in metrics:
                portfolio["sortino_ratio"] = metrics["sortino_ratio"]
                
            frontier_portfolios.append(portfolio)
            
        except Exception as e:
            print(f"Could not generate portfolio with return {target_return}: {e}")
            continue
    
    # Sort by expected return
    frontier_portfolios.sort(key=lambda x: x["expected_return"])
    
    return frontier_portfolios

def run_mean_variance_optimization(
    price_data: pd.DataFrame,
    tickers: List[str],
    allocations: Dict[str, float],
    constraints: Dict[str, Tuple[float, float]],
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """Run Mean-Variance optimization analysis."""
    # Calculate expected returns and covariance
    from core.data import calculate_returns_and_covariance
    mu, S = calculate_returns_and_covariance(price_data)
    
    # Optimize for max Sharpe ratio
    optimal_weights, optimal_metrics = optimize_max_sharpe(
        mu, S, tickers, constraints, risk_free_rate, price_data
    )
    
    # Calculate metrics for provided weights
    provided_weights_array = weights_to_array(tickers, allocations)
    provided_metrics = calculate_portfolio_metrics(
        provided_weights_array, mu, S, price_data, risk_free_rate
    )
    
    # Generate efficient frontier
    frontier_portfolios = generate_efficient_frontier(
        mu, S, tickers, constraints, optimal_metrics, 
        optimal_weights, 20, price_data, risk_free_rate
    )
    
    # Create correlation matrix
    correlation_matrix = price_data.corr().to_dict()
    
    return {
        "provided_portfolio": {
            "weights": allocations,
            "metrics": provided_metrics
        },
        "optimal_portfolio": {
            "weights": optimal_weights,
            "metrics": optimal_metrics
        },
        "efficient_frontier": frontier_portfolios,
        "correlation_matrix": correlation_matrix,
        "method": "MEAN_VARIANCE"
    }