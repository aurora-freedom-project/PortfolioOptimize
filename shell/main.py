#!/usr/bin/env python3
# shell/main.py
"""
Portfolio Optimization CLI - Enhanced Main Entry Point
=====================================================

Enhanced CLI supporting both offline files and real-time market data.

Features:
- Multiple data sources (files, Yahoo Finance, Alpha Vantage, etc.)
- Database caching and persistence
- Real-time price streaming
- Advanced optimization methods
- Comprehensive backtesting
- Institutional reporting
- API server mode

Usage Examples:
--------------

# Basic optimization with CSV file
python -m shell --data merged_stock_prices.csv --tickers AAPL,GOOGL,MSFT --method mean_variance

# Real-time optimization with Yahoo Finance
python -m shell --data-source yahoo --real-time --tickers AAPL,GOOGL,MSFT --method mean_variance

# Advanced optimization with database caching
python -m shell --data-source yahoo --database-url sqlite:///portfolio.db --advanced-method risk_parity --tickers AAPL,GOOGL,MSFT

# Multi-source fallback
python -m shell --data-source alphavantage --fallback-sources yahoo,file --api-keys '{"alphavantage": "YOUR_API_KEY"}' --tickers AAPL,GOOGL

# Real-time price streaming
python -m shell --stream-prices --tickers AAPL,GOOGL,TSLA --stream-interval 30

# Comprehensive backtest with database
python -m shell --run-backtest --database-url sqlite:///portfolio.db --data-source yahoo --tickers AAPL,GOOGL,MSFT,AMZN
"""

import sys
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .cli import (parse_args, parse_tickers, parse_allocations, parse_constraints, 
                 parse_investor_views, display_portfolio_results, load_config_file, 
                 save_config_file, merge_config_with_args, create_example_config)
from .app import (run_portfolio_optimization, run_comprehensive_backtest_workflow, 
                 generate_institutional_report_workflow, start_api_server)

def setup_data_manager(args) -> Optional[Any]:
    """
    Set up data manager based on CLI arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Configured DataManager instance or None for file fallback
    """
    
    try:
        from core.data_sources.data_manager import DataManager
        
        # Parse API keys if provided
        api_keys = {}
        if args.api_keys:
            try:
                api_keys = json.loads(args.api_keys)
            except json.JSONDecodeError:
                print("Warning: Invalid JSON format for API keys")
        
        # Configure data sources
        config = {}
        
        # Yahoo Finance (free, no API key needed)
        if args.data_source == 'yahoo' or 'yahoo' in args.fallback_sources:
            config['yahoo'] = {'rate_limit': 0.5}
        
        # Alpha Vantage (requires API key)
        if args.data_source == 'alphavantage' or 'alphavantage' in args.fallback_sources:
            if 'alphavantage' in api_keys:
                config['alphavantage'] = {
                    'api_key': api_keys['alphavantage'],
                    'rate_limit': 12.0  # Free tier: 5 calls per minute
                }
            else:
                print("Warning: Alpha Vantage requires API key")
        
        # Quandl (requires API key)
        if args.data_source == 'quandl' or 'quandl' in args.fallback_sources:
            if 'quandl' in api_keys:
                config['quandl'] = {
                    'api_key': api_keys['quandl'],
                    'rate_limit': 1.0
                }
            else:
                print("Warning: Quandl requires API key")
        
        # FRED (requires API key)
        if args.data_source == 'fred' or 'fred' in args.fallback_sources:
            if 'fred' in api_keys:
                config['fred'] = {
                    'api_key': api_keys['fred'],
                    'rate_limit': 1.0
                }
            else:
                print("Warning: FRED requires API key")
        
        # Crypto (free, no API key needed)
        if args.data_source == 'crypto' or 'crypto' in args.fallback_sources:
            config['crypto'] = {'rate_limit': 1.0}
        
        # Initialize data manager
        data_manager = DataManager(
            config=config,
            cache_directory="./data_cache/",
            database_url=args.database_url
        )
        
        # Update cache if requested
        if args.update_cache and args.tickers:
            tickers = parse_tickers(args.tickers)
            if tickers:
                print(f"Updating database cache for {len(tickers)} tickers...")
                records_updated = data_manager.update_database_cache(
                    tickers, 
                    source=args.data_source if args.data_source != 'auto' else 'yahoo',
                    days_back=args.cache_days
                )
                print(f"Updated {records_updated} records in database cache")
        
        return data_manager
        
    except ImportError:
        print("Warning: Enhanced data sources not available, falling back to file mode")
        return None
    except Exception as e:
        print(f"Warning: Error setting up data manager: {e}")
        return None

def handle_real_time_streaming(args, data_manager):
    """Handle real-time price streaming."""
    
    if not args.stream_prices:
        return
    
    if not data_manager:
        print("Error: Real-time streaming requires data manager setup")
        return
    
    tickers = parse_tickers(args.tickers) if args.tickers else ['AAPL', 'GOOGL', 'MSFT']
    
    def price_callback(prices: Dict[str, float]):
        """Callback function for price updates."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] Price Update:")
        for ticker, price in prices.items():
            print(f"  {ticker}: ${price:.2f}")
    
    print(f"Starting real-time price streaming for: {', '.join(tickers)}")
    print(f"Update interval: {args.stream_interval} seconds")
    print("Press Ctrl+C to stop...")
    
    try:
        data_source = args.data_source if args.data_source != 'auto' else 'yahoo'
        data_manager.start_real_time_streaming(
            tickers, price_callback, source=data_source, 
            interval_seconds=args.stream_interval
        )
        
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping price streaming...")
        data_manager.stop_real_time_streaming()
    except Exception as e:
        print(f"Streaming error: {e}")

def handle_data_quality_check(args, data_manager):
    """Generate data quality report."""
    
    if not data_manager:
        print("Data quality check requires data manager setup")
        return
    
    tickers = parse_tickers(args.tickers) if args.tickers else []
    if not tickers:
        print("No tickers specified for data quality check")
        return
    
    print(f"Generating data quality report for: {', '.join(tickers)}")
    report = data_manager.get_data_quality_report(tickers)
    
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Tickers Analyzed: {len(report['tickers_analyzed'])}")
    
    print("\nSource Coverage:")
    for source, info in report['source_coverage'].items():
        status = "✓ Available" if info.get('available', False) else "✗ Unavailable"
        print(f"  {source}: {status}")
        if 'error' in info:
            print(f"    Error: {info['error']}")
    
    if report.get('database_coverage'):
        print("\nDatabase Coverage:")
        for coverage in report['database_coverage']:
            print(f"  {coverage['ticker']}: {coverage['record_count']} records "
                  f"({coverage['start_date']} to {coverage['end_date']})")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

def main():
    """Main CLI entry point with enhanced data source support."""
    
    args = parse_args()
    
    # Handle configuration file loading
    if args.config:
        try:
            config = load_config_file(args.config)
            print("Merging configuration file with command line arguments...")
            merge_config_with_args(args, config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Handle saving configuration
    if args.save_config:
        try:
            save_config_file(args, args.save_config)
            print("Configuration saved successfully!")
            return
        except Exception as e:
            print(f"Error saving configuration: {e}")
            sys.exit(1)
    
    # Handle API server mode
    if args.start_api:
        start_api_server(args.api_host, args.api_port)
        return
    
    # Handle charts from JSON file
    if args.charts_from_json:
        if not os.path.exists(args.charts_from_json):
            print(f"Error: JSON file not found: {args.charts_from_json}")
            sys.exit(1)
        
        try:
            from shell.display.visualization import create_charts_from_json
            create_charts_from_json(args.charts_from_json)
            print(f"Charts generated from {args.charts_from_json}")
        except Exception as e:
            print(f"Error generating charts: {e}")
            sys.exit(1)
        return
    
    # Set up data manager
    data_manager = setup_data_manager(args)
    
    # Handle real-time streaming mode
    if args.stream_prices:
        handle_real_time_streaming(args, data_manager)
        return
    
    # Parse basic parameters
    if not args.tickers:
        print("Error: No tickers specified. Use --tickers parameter.")
        sys.exit(1)
    
    tickers = parse_tickers(args.tickers)
    allocations = parse_allocations(args.allocations, tickers)
    constraints = parse_constraints(args.constraints, tickers)
    investor_views = parse_investor_views(args.views, tickers)
    
    # Date range handling
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    if args.start_date is None:
        # Default to 2 years of data
        start_date = datetime.now() - timedelta(days=730)
        args.start_date = start_date.strftime('%Y-%m-%d')
    
    # Parse fallback sources
    fallback_sources = [s.strip() for s in args.fallback_sources.split(',') if s.strip()]
    
    try:
        
        # Data quality check mode
        if hasattr(args, 'data_quality_check') and args.data_quality_check:
            handle_data_quality_check(args, data_manager)
            return
        
        # Handle different workflow modes
        if args.run_backtest:
            # Comprehensive backtesting
            print("Running comprehensive backtesting analysis...")
            
            # Default strategies for backtesting
            strategies = {
                'mean_variance': {'method': 'mean_variance'},
                'hrp': {'method': 'hrp'},
                'risk_parity': {'method': 'risk_parity'}
            }
            
            if args.advanced_method:
                strategies[args.advanced_method] = {
                    'method': args.advanced_method,
                    'gamma': args.gamma,
                    'confidence_level': args.confidence_level,
                    'target_volatility': args.target_volatility,
                    'long_short_ratio': args.long_short_ratio
                }
            
            # Run backtest workflow
            if data_manager:
                # Enhanced backtest with data manager
                from core.backtesting import run_comprehensive_backtest
                
                # Get data
                price_data = data_manager.get_price_data(
                    tickers, args.start_date, args.end_date,
                    source_priority=[args.data_source] if args.data_source != 'auto' else fallback_sources
                )
                
                if price_data.empty:
                    print("Error: No price data available")
                    sys.exit(1)
                
                results = run_comprehensive_backtest(
                    price_data, allocations, strategies,
                    args.rebalance_frequency, args.lookback_window
                )
            else:
                # Fallback to file-based backtest
                results = run_comprehensive_backtest_workflow(
                    args.data, tickers, args.start_date, args.end_date,
                    allocations, strategies, args.rebalance_frequency, args.lookback_window
                )
            
            # Display results
            print("\n" + "="*60)
            print("COMPREHENSIVE BACKTESTING RESULTS")
            print("="*60)
            
            strategy_comparison = results.get('strategy_comparison', {})
            individual_results = strategy_comparison.get('individual_results', {})
            
            if individual_results:
                print(f"\nTested {len(individual_results)} strategies:")
                for strategy_name, strategy_result in individual_results.items():
                    if strategy_result and 'summary' in strategy_result:
                        summary = strategy_result['summary']
                        print(f"\n{strategy_name.upper()}:")
                        print(f"  Total Return: {summary.get('total_return', 0)*100:.2f}%")
                        print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
                        print(f"  Max Drawdown: {summary.get('max_drawdown', 0)*100:.2f}%")
                        print(f"  Avg Turnover: {summary.get('avg_turnover', 0)*100:.2f}%")
                
                best_strategy = results.get('best_strategy')
                if best_strategy:
                    print(f"\nBest Strategy (by Sharpe): {best_strategy}")
            
            # Save results if output specified
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nBacktest results saved to: {args.output}")
        
        elif args.generate_report:
            # Institutional report generation
            print("Generating comprehensive institutional report...")
            
            if data_manager:
                # Enhanced report with data manager
                from core.reporting import generate_client_report
                
                # Get data
                price_data = data_manager.get_price_data(
                    tickers, args.start_date, args.end_date,
                    source_priority=[args.data_source] if args.data_source != 'auto' else fallback_sources
                )
                
                if price_data.empty:
                    print("Error: No price data available")
                    sys.exit(1)
                
                # Generate reports
                output_dir = f"./reports/{args.client_name.replace(' ', '_')}"
                os.makedirs(output_dir, exist_ok=True)
                
                deliverables = generate_client_report(
                    price_data, {}, [], allocations, args.client_name, output_dir
                )
            else:
                # Fallback to file-based report
                deliverables = generate_institutional_report_workflow(
                    args.data, tickers, args.start_date, args.end_date,
                    allocations, args.client_name, args.report_type
                )
            
            print(f"\nGenerated {len(deliverables)} report deliverables:")
            for report_type, file_path in deliverables.items():
                print(f"  {report_type}: {file_path}")
        
        else:
            # Standard portfolio optimization
            print(f"Running {args.method} optimization...")
            
            # Prepare advanced method parameters
            advanced_kwargs = {}
            if args.advanced_method:
                advanced_kwargs.update({
                    'gamma': args.gamma,
                    'confidence_level': args.confidence_level,
                    'target_volatility': args.target_volatility,
                    'long_short_ratio': args.long_short_ratio
                })
            
            # Run optimization
            results = run_portfolio_optimization(
                data_file=args.data,
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                method=args.method,
                allocations=allocations,
                constraints=constraints,
                risk_free_rate=args.risk_free_rate,
                investor_views=investor_views,
                advanced_method=args.advanced_method,
                data_source=args.data_source,
                data_manager=data_manager,
                **advanced_kwargs
            )
            
            # Display results
            display_portfolio_results(results)
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nResults saved to: {args.output}")
            
            # Show charts if requested
            if args.show_charts:
                try:
                    from shell.display.visualization import PortfolioVisualizer
                    visualizer = PortfolioVisualizer()
                    
                    if args.method == 'mean_variance':
                        visualizer.plot_mean_variance_analysis(results)
                    elif args.method == 'black_litterman':
                        visualizer.plot_black_litterman_analysis(results)
                    elif args.method == 'hrp':
                        visualizer.plot_hrp_analysis(results)
                    
                    print("Charts displayed successfully")
                except Exception as e:
                    print(f"Error displaying charts: {e}")
        
        # Real-time price display if requested
        if args.real_time and data_manager:
            print("\nFetching real-time prices...")
            try:
                real_time_prices = data_manager.get_real_time_prices(
                    tickers, source=args.data_source if args.data_source != 'auto' else 'yahoo'
                )
                
                if real_time_prices:
                    print("\nReal-time Prices:")
                    for ticker, price in real_time_prices.items():
                        print(f"  {ticker}: ${price:.2f}")
                else:
                    print("No real-time prices available")
                    
            except Exception as e:
                print(f"Error fetching real-time prices: {e}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up data manager
        if data_manager:
            data_manager.close()

if __name__ == "__main__":
    main()