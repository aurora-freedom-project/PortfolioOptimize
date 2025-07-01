# shell/cli.py
import argparse
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization CLI')
    
    # Input data file
    parser.add_argument('--data', type=str, default='merged_stock_prices.csv',
                        help='Path to stock price data CSV file')
    
    # Optimization method
    parser.add_argument('--method', type=str, default='mean_variance',
                        choices=['mean_variance', 'black_litterman', 'hrp'],
                        help='Optimization method to use')
    
    # Date range
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date (YYYY-MM-DD)')
    
    # Tickers
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of stock tickers')
    
    # Allocations (optional)
    parser.add_argument('--allocations', type=str, default='',
                        help='Comma-separated list of allocations (format: TICKER:WEIGHT,...)')
    
    # Constraints (optional)
    parser.add_argument('--constraints', type=str, default='',
                        help='Comma-separated list of constraints (format: TICKER:MIN:MAX,...)')
    
    # Risk-free rate
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Risk-free rate (default: 0.02)')
    
    # Investor views for Black-Litterman (optional)
    parser.add_argument('--views', type=str, default='',
                        help='Investor views for Black-Litterman (format: TICKER:VALUE:CONFIDENCE,...)')
    
    # Output file (optional)
    parser.add_argument('--output', type=str, default='',
                        help='Output file for results (JSON format)')
    
    return parser.parse_args()

def parse_tickers(tickers_str: str) -> List[str]:
    """Parse comma-separated ticker list."""
    return [ticker.strip() for ticker in tickers_str.split(',') if ticker.strip()]

def parse_allocations(allocations_str: str, tickers: List[str]) -> Dict[str, float]:
    """Parse allocation string into dictionary."""
    if not allocations_str:
        return {}
    
    allocations = {}
    for item in allocations_str.split(','):
        if ':' in item:
            ticker, weight = item.split(':')
            ticker = ticker.strip()
            
            if ticker in tickers:
                try:
                    allocations[ticker] = float(weight)
                except ValueError:
                    print(f"Warning: Invalid weight '{weight}' for ticker '{ticker}'. Using 0.")
                    allocations[ticker] = 0
    
    return allocations

def parse_constraints(constraints_str: str, tickers: List[str]) -> Dict[str, Tuple[float, float]]:
    """Parse constraints string into dictionary."""
    if not constraints_str:
        return {}
    
    constraints = {}
    for item in constraints_str.split(','):
        parts = item.split(':')
        if len(parts) == 3:
            ticker, min_val, max_val = parts
            ticker = ticker.strip()
            
            if ticker in tickers:
                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    constraints[ticker] = (min_val, max_val)
                except ValueError:
                    print(f"Warning: Invalid constraint values for ticker '{ticker}'. Using default (0,1).")
                    constraints[ticker] = (0, 1)
    
    return constraints

def parse_investor_views(views_str: str, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse investor views string into dictionary."""
    if not views_str:
        return None
    
    views = {}
    for item in views_str.split(','):
        parts = item.split(':')
        if len(parts) == 3:
            ticker, value, confidence = parts
            ticker = ticker.strip()
            
            if ticker in tickers:
                try:
                    value = float(value)
                    confidence = float(confidence)
                    views[ticker] = {
                        "view_type": "will return",
                        "value": value,
                        "confidence": confidence
                    }
                except ValueError:
                    print(f"Warning: Invalid view values for ticker '{ticker}'. Skipping.")
    
    return views if views else None

def display_portfolio_results(results: Dict[str, Any]):
    """Display portfolio optimization results."""
    print("\n======= Portfolio Optimization Results =======")
    
    # Method
    method = results.get("method", "Unknown")
    method_names = {
        "MEAN_VARIANCE": "Mean-Variance Optimization",
        "BLACK_LITTERMAN": "Black-Litterman Optimization",
        "HIERARCHICAL_RISK_PARITY": "Hierarchical Risk Parity"
    }
    print(f"Method: {method_names.get(method, method)}")
    
    # Provided portfolio
    print("\n--- Provided Portfolio ---")
    provided = results.get("provided_portfolio", {})
    provided_metrics = provided.get("metrics", {})
    print(f"Expected Return: {provided_metrics.get('expected_return', 0)*100:.2f}%")
    print(f"Standard Deviation: {provided_metrics.get('standard_deviation', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {provided_metrics.get('sharpe_ratio', 0):.4f}")
    if "sortino_ratio" in provided_metrics:
        print(f"Sortino Ratio: {provided_metrics.get('sortino_ratio', 0):.4f}")
    
    print("\nWeights:")
    for ticker, weight in provided.get("weights", {}).items():
        print(f"  {ticker}: {weight*100:.2f}%")
    
    # Optimal portfolio
    print("\n--- Optimal Portfolio ---")
    optimal = results.get("optimal_portfolio", {})
    optimal_metrics = optimal.get("metrics", {})
    print(f"Expected Return: {optimal_metrics.get('expected_return', 0)*100:.2f}%")
    print(f"Standard Deviation: {optimal_metrics.get('standard_deviation', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {optimal_metrics.get('sharpe_ratio', 0):.4f}")
    if "sortino_ratio" in optimal_metrics:
        print(f"Sortino Ratio: {optimal_metrics.get('sortino_ratio', 0):.4f}")
    
    print("\nWeights:")
    for ticker, weight in optimal.get("weights", {}).items():
        print(f"  {ticker}: {weight*100:.2f}%")
    
    # Additional method-specific information
    if method == "BLACK_LITTERMAN":
        print("\n--- Black-Litterman Information ---")
        bl_info = results.get("black_litterman_info", {})
        print(f"Views Applied: {'Yes' if bl_info.get('adjusted', False) else 'No'}")
        if not bl_info.get("adjusted", False):
            print(f"Reason: {bl_info.get('reason', 'Unknown')}")
            
    elif method == "HIERARCHICAL_RISK_PARITY":
        print("\n--- Hierarchical Risk Parity Information ---")
        print("Cluster-based optimization completed")
    
    print("\n==============================================")