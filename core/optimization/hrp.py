# core/optimization/hrp.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from pypfopt import hierarchical_portfolio
import warnings

from core.utils import weights_to_array, calculate_portfolio_metrics

def run_hierarchical_risk_parity(
    price_data: pd.DataFrame,
    tickers: List[str],
    allocations: Dict[str, float],
    constraints: Dict[str, Tuple[float, float]] = None,  # Not used in HRP but kept for API consistency
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """Run Hierarchical Risk Parity optimization."""
    # Calculate returns
    returns = price_data.pct_change().dropna()
    
    # Initialize HRP
    hrp = hierarchical_portfolio.HRPOpt(returns)
    
    # Optimize weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hrp.optimize()
    
    # Get optimized weights
    optimal_weights = hrp.clean_weights()
    
    # Calculate returns and covariance for metrics
    from core.data import calculate_returns_and_covariance
    mu, S = calculate_returns_and_covariance(price_data)
    
    # Calculate metrics for optimal portfolio
    optimal_weights_array = weights_to_array(tickers, optimal_weights)
    optimal_metrics = calculate_portfolio_metrics(
        optimal_weights_array, mu, S, price_data, risk_free_rate
    )
    
    # Calculate metrics for provided weights
    provided_weights_array = weights_to_array(tickers, allocations)
    provided_metrics = calculate_portfolio_metrics(
        provided_weights_array, mu, S, price_data, risk_free_rate
    )
    
    # Create correlation matrix
    correlation_matrix = price_data.corr().to_dict()
    
    # Get cluster information
    clusters = hrp.clusters
    
    return {
        "provided_portfolio": {
            "weights": allocations,
            "metrics": provided_metrics
        },
        "optimal_portfolio": {
            "weights": optimal_weights,
            "metrics": optimal_metrics
        },
        "clusters": clusters,
        "correlation_matrix": correlation_matrix,
        "method": "HIERARCHICAL_RISK_PARITY"
    }