# shell/app.py
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import os
import json

def run_portfolio_optimization(
    data_file: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    method: str,
    allocations: Dict[str, float] = None,
    constraints: Dict[str, Tuple[float, float]] = None,
    risk_free_rate: float = 0.02,
    investor_views: Dict[str, Dict[str, Any]] = None,
    advanced_method: Optional[str] = None,
    data_source: str = 'file',
    data_manager = None,
    validate_inputs: bool = True,
    **advanced_kwargs
) -> Dict[str, Any]:
    """Coordinate portfolio optimization workflow with multiple data sources."""
    
    # Input validation
    if validate_inputs:
        from core.validation_simple import validate_portfolio_inputs, ValidationError
        try:
            validated_data = validate_portfolio_inputs(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                method=advanced_method if advanced_method else method,
                allocations=allocations or {},
                constraints=constraints or {},
                risk_free_rate=risk_free_rate,
                investor_views=investor_views or {}
            )
            
            # Use validated data
            tickers = validated_data.tickers
            allocations = validated_data.allocations
            constraints = validated_data.constraints
            risk_free_rate = validated_data.risk_free_rate
            investor_views = validated_data.investor_views
            
            print(f"✅ Input validation passed for {method} optimization")
            
        except ValidationError as e:
            raise ValueError(f"Input validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected validation error: {e}")
    
    # Load and prepare data using new data manager
    if data_manager is None:
        from core.data import load_stock_data, filter_date_range, prepare_allocation_weights, prepare_constraints
        # Fallback to old method
        stock_data = load_stock_data(data_file)
        filtered_data = filter_date_range(stock_data, start_date, end_date)
    else:
        # Use new data manager
        filtered_data = data_manager.get_price_data(
            tickers, start_date, end_date,
            source_priority=[data_source] if data_source != 'auto' else None
        )
    
    # Ensure data has required tickers
    missing_tickers = [ticker for ticker in tickers if ticker not in filtered_data.columns]
    if missing_tickers:
        print(f"Warning: Missing tickers in data: {', '.join(missing_tickers)}")
        # Filter to only available tickers
        available_tickers = [t for t in tickers if t in filtered_data.columns]
        if not available_tickers:
            raise ValueError("No valid tickers found in data")
        tickers = available_tickers
    
    # Filter to only include requested tickers
    ticker_data = filtered_data[tickers].copy()
    
    # Prepare allocations and constraints
    if allocations is None:
        allocations = {}
    if constraints is None:
        constraints = {}
    
    # Use new utility functions or fallback to old ones
    if data_manager is None:
        from core.data import prepare_allocation_weights, prepare_constraints
        complete_allocations = prepare_allocation_weights(tickers, allocations)
        complete_constraints = prepare_constraints(tickers, complete_allocations, constraints)
    else:
        # Simple allocation preparation
        complete_allocations = {ticker: allocations.get(ticker, 1.0/len(tickers)) for ticker in tickers}
        # Normalize allocations
        total_weight = sum(complete_allocations.values())
        if total_weight > 0:
            complete_allocations = {k: v/total_weight for k, v in complete_allocations.items()}
        
        # Simple constraints preparation  
        complete_constraints = {ticker: constraints.get(ticker, (0.0, 1.0)) for ticker in tickers}
    
    # Run optimization based on method
    if advanced_method:
        # Use advanced optimization methods
        from core.optimization.advanced_optimization import run_advanced_optimization
        results = run_advanced_optimization(
            ticker_data, tickers, complete_allocations, advanced_method,
            constraints=complete_constraints, risk_free_rate=risk_free_rate,
            **advanced_kwargs
        )
    elif method.lower() == 'mean_variance':
        from core.optimization.mean_variance import run_mean_variance_optimization
        results = run_mean_variance_optimization(
            ticker_data, tickers, complete_allocations, complete_constraints,
            risk_free_rate
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
            ticker_data, tickers, complete_allocations, complete_constraints
        )
    
    else:
        raise ValueError(f"Unsupported optimization method: {method}")
    
    return results

def run_comprehensive_backtest_workflow(
    data_file: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_allocations: Dict[str, float],
    strategies: Dict[str, Dict[str, Any]],
    rebalance_frequency: str = 'M',
    lookback_window: int = 252
) -> Dict[str, Any]:
    """Run comprehensive backtesting workflow."""
    from core.data import load_stock_data, filter_date_range
    from core.backtesting import run_comprehensive_backtest
    
    # Load and prepare data
    stock_data = load_stock_data(data_file)
    filtered_data = filter_date_range(stock_data, start_date, end_date)
    
    # Ensure data has required tickers
    missing_tickers = [ticker for ticker in tickers if ticker not in filtered_data.columns]
    if missing_tickers:
        raise ValueError(f"Missing tickers in data: {', '.join(missing_tickers)}")
    
    # Filter to only include requested tickers
    ticker_data = filtered_data[tickers].copy()
    
    # Run comprehensive backtest
    results = run_comprehensive_backtest(
        ticker_data, initial_allocations, strategies,
        rebalance_frequency, lookback_window
    )
    
    return results

def generate_institutional_report_workflow(
    data_file: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    portfolio_weights: Dict[str, float],
    client_name: str = "Institutional Client",
    report_type: str = "QUARTERLY",
    backtest_config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Generate comprehensive institutional report workflow."""
    from core.data import load_stock_data, filter_date_range
    from core.reporting import generate_client_report
    from core.backtesting import run_comprehensive_backtest
    
    # Load and prepare data
    stock_data = load_stock_data(data_file)
    filtered_data = filter_date_range(stock_data, start_date, end_date)
    ticker_data = filtered_data[tickers].copy()
    
    # Run backtest if configuration provided
    backtest_results = {}
    optimization_results = []
    
    if backtest_config:
        backtest_results = run_comprehensive_backtest(
            ticker_data, 
            backtest_config.get('initial_allocations', {}),
            backtest_config.get('strategies', {}),
            backtest_config.get('rebalance_frequency', 'M'),
            backtest_config.get('lookback_window', 252)
        )
    
    # Generate comprehensive report
    output_dir = f"./reports/{client_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    deliverables = generate_client_report(
        ticker_data, backtest_results, optimization_results,
        portfolio_weights, client_name, output_dir
    )
    
    return deliverables

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the Portfolio Optimization API server."""
    try:
        import uvicorn
        from api.endpoints import app
        
        print(f"Starting Portfolio Optimization API server...")
        print(f"Server will be available at: http://{host}:{port}")
        print(f"API documentation: http://{host}:{port}/docs")
        print(f"Alternative docs: http://{host}:{port}/redoc")
        
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        print("Error: uvicorn is not installed. Install it with: pip install uvicorn[standard]")
    except Exception as e:
        print(f"Error starting API server: {e}")