# shell/app.py
from typing import Dict, List, Tuple, Any
import pandas as pd

def run_portfolio_optimization(
    data_file: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    method: str,
    allocations: Dict[str, float] = None,
    constraints: Dict[str, Tuple[float, float]] = None,
    risk_free_rate: float = 0.02,
    investor_views: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Coordinate portfolio optimization workflow."""
    from core.data import load_stock_data, filter_date_range, prepare_allocation_weights, prepare_constraints
    
    # Load and prepare data
    stock_data = load_stock_data(data_file)
    filtered_data = filter_date_range(stock_data, start_date, end_date)
    
    # Ensure data has required tickers
    missing_tickers = [ticker for ticker in tickers if ticker not in filtered_data.columns]
    if missing_tickers:
        raise ValueError(f"Missing tickers in data: {', '.join(missing_tickers)}")
    
    # Filter to only include requested tickers
    ticker_data = filtered_data[tickers].copy()
    
    # Prepare allocations and constraints
    if allocations is None:
        allocations = {}
    if constraints is None:
        constraints = {}
    
    complete_allocations = prepare_allocation_weights(tickers, allocations)
    complete_constraints = prepare_constraints(tickers, complete_allocations, constraints)
    
    # Run optimization based on method
    if method.lower() == 'mean_variance':
        from core.optimization.mean_variance import run_mean_variance_optimization
        results = run_mean_variance_optimization(
            ticker_data, tickers, complete_allocations, complete_constraints, risk_free_rate
        )
    
    elif method.lower() == 'black_litterman':
        from core.optimization.black_litterman import run_black_litterman_optimization
        results = run_black_litterman_optimization(
            ticker_data, tickers, complete_allocations, complete_constraints, 
            risk_free_rate, investor_views
        )
    
    elif method.lower() == 'hrp':
        from core.optimization.hrp import run_hierarchical_risk_parity
        results = run_hierarchical_risk_parity(
            ticker_data, tickers, complete_allocations, complete_constraints, risk_free_rate
        )
    
    else:
        raise ValueError(f"Unsupported optimization method: {method}")
    
    return results