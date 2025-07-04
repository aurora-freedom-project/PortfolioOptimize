# shell/cli.py
import argparse
import pandas as pd
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization CLI')
    
    # Data source configuration
    parser.add_argument('--data', type=str, default='merged_stock_prices.csv',
                        help='Path to stock price data file or database configuration')
    parser.add_argument('--data-source', type=str, default='file', 
                        choices=['file', 'yahoo', 'alphavantage', 'quandl', 'fred', 'crypto', 'database', 'auto'],
                        help='Primary data source to use')
    parser.add_argument('--fallback-sources', type=str, default='file,yahoo',
                        help='Comma-separated list of fallback data sources')
    parser.add_argument('--database-url', type=str,
                        help='Database URL for data storage (e.g., sqlite:///portfolio.db)')
    parser.add_argument('--api-keys', type=str,
                        help='JSON string with API keys for data sources')
    parser.add_argument('--update-cache', action='store_true',
                        help='Update database cache with fresh data')
    parser.add_argument('--cache-days', type=int, default=30,
                        help='Number of days to cache in database')
    
    # Real-time data options
    parser.add_argument('--real-time', action='store_true',
                        help='Use real-time market data')
    parser.add_argument('--stream-prices', action='store_true',
                        help='Start real-time price streaming')
    parser.add_argument('--stream-interval', type=int, default=60,
                        help='Price streaming interval in seconds')
    
    # Optimization method
    parser.add_argument('--method', type=str, default='mean_variance',
                        choices=['mean_variance', 'black_litterman', 'hrp'],
                        help='Optimization method to use')
    
    # Date range
    parser.add_argument('--start-date', type=str, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD)')
    
    # Tickers
    parser.add_argument('--tickers', type=str,
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
    
    # Configuration file support
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file with parameters')
    parser.add_argument('--save-config', type=str,
                        help='Save current parameters to JSON configuration file')
    
    # Investor views for Black-Litterman (optional)
    parser.add_argument('--views', type=str, default='',
                        help='Investor views for Black-Litterman (format: TICKER:VALUE:CONFIDENCE,...)')
    
    # Output file (optional)
    parser.add_argument('--output', type=str, default='',
                        help='Output file for results (JSON format)')
    
    # Show charts flag
    parser.add_argument('--show-charts', action='store_true',
                        help='Display interactive charts for portfolio analysis')
    
    # Charts from JSON file
    parser.add_argument('--charts-from-json', type=str, default='',
                        help='Generate charts from existing JSON results file')
    
    # Advanced optimization methods
    parser.add_argument('--advanced-method', type=str, choices=['max_sharpe_l2', 'min_cvar', 'semivariance', 'risk_parity', 'market_neutral', 'cla'],
                        help='Use advanced optimization methods for institutional investors')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='L2 regularization parameter for max_sharpe_l2 method')
    parser.add_argument('--confidence-level', type=float, default=0.05,
                        help='Confidence level for CVaR optimization')
    parser.add_argument('--target-volatility', type=float, default=0.15,
                        help='Target volatility for market neutral strategies')
    parser.add_argument('--long-short-ratio', type=float, default=1.0,
                        help='Long-short ratio for market neutral portfolios')
    
    # Backtesting parameters
    parser.add_argument('--run-backtest', action='store_true',
                        help='Run comprehensive backtesting analysis')
    parser.add_argument('--rebalance-frequency', type=str, default='M', choices=['D', 'W', 'M', 'Q'],
                        help='Rebalancing frequency for backtesting')
    parser.add_argument('--lookback-window', type=int, default=252,
                        help='Lookback window in days for optimization')
    
    # Reporting and analytics
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate comprehensive institutional report')
    parser.add_argument('--client-name', type=str, default='Institutional Client',
                        help='Client name for institutional reports')
    parser.add_argument('--report-type', type=str, default='QUARTERLY', choices=['MONTHLY', 'QUARTERLY', 'ANNUAL'],
                        help='Type of institutional report')
    
    # API server mode
    parser.add_argument('--start-api', action='store_true',
                        help='Start the portfolio optimization API server')
    parser.add_argument('--api-host', type=str, default='0.0.0.0',
                        help='API server host')
    parser.add_argument('--api-port', type=int, default=8000,
                        help='API server port')
    
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


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")


def save_config_file(args, config_path: str) -> None:
    """Save current arguments to JSON configuration file."""
    # Convert args to dictionary, excluding None values and internal attributes
    config = {}
    
    for key, value in vars(args).items():
        if not key.startswith('_') and value is not None:
            # Handle special cases
            if key in ['save_config', 'config']:
                continue  # Don't save these meta-parameters
            config[key] = value
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"✅ Configuration saved to: {config_path}")
    except Exception as e:
        raise ValueError(f"Error saving configuration file: {e}")


def merge_config_with_args(args, config: Dict[str, Any]):
    """Merge configuration file values with command line arguments.
    
    Command line arguments take precedence over config file values.
    """
    for key, value in config.items():
        # Only use config value if command line argument is default/None
        if hasattr(args, key):
            current_value = getattr(args, key)
            # Check if it's a default value that should be overridden
            if (current_value is None or 
                (key == 'risk_free_rate' and current_value == 0.02) or
                (isinstance(current_value, str) and current_value == '') or
                (isinstance(current_value, bool) and not current_value)):
                setattr(args, key, value)
                print(f"  {key}: {value} (from config)")


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration file structure."""
    return {
        "data": "merged_stock_prices.csv",
        "tickers": "ANZ.AX,CBA.AX,NAB.AX,WBC.AX",
        "method": "mean_variance",
        "start_date": "2021-01-01",
        "end_date": "2024-01-01", 
        "risk_free_rate": 0.0435,
        "allocations": "ANZ.AX:0.25,CBA.AX:0.25,NAB.AX:0.25,WBC.AX:0.25",
        "constraints": "ANZ.AX:0.15:0.35,CBA.AX:0.15:0.35",
        "output": "portfolio_results.json",
        "show_charts": True,
        "data_source": "file",
        "fallback_sources": "file,yahoo"
    }