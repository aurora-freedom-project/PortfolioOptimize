# core/optimization/black_litterman.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pypfopt import black_litterman, risk_models
import warnings

from core.utils import weights_to_array, calculate_portfolio_metrics, ensure_total_is_one
from core.optimization.mean_variance import add_constraints_to_ef, generate_efficient_frontier, optimize_max_sharpe

def prepare_black_litterman_model(
    price_data: pd.DataFrame,
    benchmark_weights: Dict[str, float],
    risk_free_rate: float = 0.02,
    risk_aversion: float = 2.5
) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """Prepare data for Black-Litterman model."""
    # Get tickers from price data
    tickers = price_data.columns.tolist()
    
    # Calculate covariance matrix
    cov_matrix = risk_models.sample_cov(price_data)
    
    # Ensure benchmark weights for all tickers
    benchmark_weights_series = pd.Series(
        {ticker: benchmark_weights.get(ticker, 0) for ticker in tickers},
        index=tickers
    )
    
    # Normalize weights
    if abs(benchmark_weights_series.sum() - 1.0) > 0.001:
        benchmark_weights_series = benchmark_weights_series / benchmark_weights_series.sum()
    
    try:
        # Calculate market-implied returns
        market_implied_returns = black_litterman.market_implied_prior_returns(
            benchmark_weights_series,
            risk_aversion,
            cov_matrix,
            risk_free_rate
        )
        
        # Ensure same index as covariance matrix
        market_implied_returns = pd.Series(
            market_implied_returns, 
            index=cov_matrix.index
        )
    except Exception as e:
        # Fallback to historical returns
        from pypfopt import expected_returns
        market_implied_returns = expected_returns.mean_historical_return(price_data)
        print(f"Could not calculate market-implied returns: {e}. Using historical returns.")
    
    # Calculate historical returns for comparison
    from pypfopt import expected_returns
    historical_returns = expected_returns.mean_historical_return(price_data)
    
    # Store additional info
    additional_info = {
        "market_risk_aversion": risk_aversion,
        "risk_free_rate": risk_free_rate,
        "historical_returns": historical_returns.to_dict()
    }
    
    return market_implied_returns, cov_matrix, additional_info

def apply_investor_views(
    market_implied_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    view_dict: Dict[str, Dict],
    confidence_dict: Dict[str, float] = None,
    tau: float = 0.05
) -> Tuple[pd.Series, Dict]:
    """Apply investor views to Black-Litterman model."""
    # Initialize view dictionary
    viewdict = {}
    
    # Use default confidence if not provided
    if confidence_dict is None:
        confidence_dict = {}
    
    # Convert views to Black-Litterman format
    for ticker, view_info in view_dict.items():
        if ticker in market_implied_returns.index:
            if view_info.get("view_type") == "will return":
                value = view_info.get("value", 0) / 100  # Convert percentage to decimal
                viewdict[ticker] = value
    
    # If no valid views, return market implied returns
    if not viewdict:
        return market_implied_returns, {"adjusted": False, "reason": "No valid views provided"}
    
    try:
        # Apply Black-Litterman model
        bl = black_litterman.BlackLittermanModel(
            cov_matrix,
            pi=market_implied_returns,
            absolute_views=viewdict,
            tau=tau
        )
        
        # Get posterior returns
        posterior_returns = bl.bl_returns()
        
        # Return results with metadata
        additional_info = {
            "adjusted": True,
            "prior_returns": market_implied_returns.to_dict(),
            "posterior_returns": posterior_returns.to_dict(),
            "views": viewdict,
            "confidence": confidence_dict
        }
        
        return posterior_returns, additional_info
        
    except Exception as e:
        print(f"Error in Black-Litterman model: {e}")
        return market_implied_returns, {
            "adjusted": False, 
            "reason": f"Error in Black-Litterman calculation: {str(e)}"
        }

def run_black_litterman_optimization(
    price_data: pd.DataFrame,
    tickers: List[str],
    allocations: Dict[str, float],
    constraints: Dict[str, Tuple[float, float]],
    risk_free_rate: float = 0.02,
    investor_views: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """Run Black-Litterman optimization analysis."""
    # Prepare Black-Litterman model
    market_implied_returns, cov_matrix, bl_info = prepare_black_litterman_model(
        price_data, allocations, risk_free_rate
    )
    
    # Apply investor views if provided
    if investor_views:
        posterior_returns, view_info = apply_investor_views(
            market_implied_returns, cov_matrix, investor_views
        )
    else:
        posterior_returns = market_implied_returns
        view_info = {"adjusted": False, "reason": "No views provided"}
    
    # Optimize using posterior returns
    optimal_weights, optimal_metrics = optimize_max_sharpe(
        posterior_returns, cov_matrix, tickers, constraints, risk_free_rate, price_data
    )
    
    # Calculate metrics for provided weights
    provided_weights_array = weights_to_array(tickers, allocations)
    provided_metrics = calculate_portfolio_metrics(
        provided_weights_array, posterior_returns, cov_matrix, price_data, risk_free_rate
    )
    
    # Generate efficient frontier
    frontier_portfolios = generate_efficient_frontier(
        posterior_returns, cov_matrix, tickers, constraints, 
        optimal_metrics, optimal_weights, 20, price_data, risk_free_rate
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
        "black_litterman_info": {
            **bl_info,
            **view_info
        },
        "efficient_frontier": frontier_portfolios,
        "correlation_matrix": correlation_matrix,
        "method": "BLACK_LITTERMAN"
    }