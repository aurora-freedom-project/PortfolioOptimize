#!/usr/bin/env python3
# tests/test_demo_scenarios.py
"""
Demo Scenarios for Portfolio Optimizer
======================================

Real-world scenarios demonstrating the complete capabilities
of the Portfolio Optimizer using the demo CSV files.
"""

import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import json

def create_demo_files():
    """Create realistic demo CSV files."""
    
    demo_dir = "./demo_data/"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create merged_stock_prices.csv with 3+ years of realistic data
    print("Creating merged_stock_prices.csv...")
    
    dates = pd.date_range('2021-01-01', '2024-06-01', freq='D')
    dates = dates[dates.weekday < 5]  # Business days only
    
    # Major ASX stocks
    tickers = ['ANZ.AX', 'CBA.AX', 'MQG.AX', 'NAB.AX', 'RIO.AX', 'WOW.AX', 'BHP.AX', 'CSL.AX', 'TLS.AX', 'WBC.AX']
    base_prices = {
        'ANZ.AX': 25, 'CBA.AX': 95, 'MQG.AX': 155, 'NAB.AX': 28, 'RIO.AX': 110,
        'WOW.AX': 35, 'BHP.AX': 45, 'CSL.AX': 280, 'TLS.AX': 4, 'WBC.AX': 22
    }
    
    # Generate realistic price movements
    np.random.seed(42)
    price_data = {}
    
    for ticker in tickers:
        # Different volatilities for different sectors
        if ticker in ['CSL.AX', 'TLS.AX']:
            vol = 0.035  # High volatility (healthcare/telecom)
        elif ticker in ['RIO.AX', 'BHP.AX']:
            vol = 0.030  # High volatility (mining)
        elif ticker in ['ANZ.AX', 'CBA.AX', 'NAB.AX', 'WBC.AX']:
            vol = 0.020  # Medium volatility (banking)
        else:
            vol = 0.025  # Medium volatility (other)
        
        # Generate returns with trends
        returns = np.random.normal(0.0008, vol, len(dates))
        
        # Add some trend for realism
        if ticker in ['CSL.AX', 'CBA.AX']:
            # Growth trend
            trend = np.linspace(0, 0.002, len(dates))
            returns += trend
        elif ticker in ['TLS.AX']:
            # Volatile with recovery
            trend = np.concatenate([
                np.linspace(0, -0.002, len(dates)//2),
                np.linspace(-0.002, 0.001, len(dates) - len(dates)//2)
            ])
            returns += trend
        
        # Calculate prices
        prices = [base_prices[ticker]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[ticker] = prices[1:]
    
    # Create DataFrame and save
    price_df = pd.DataFrame(price_data, index=dates)
    price_file = os.path.join(demo_dir, 'merged_stock_prices.csv')
    price_df.to_csv(price_file)
    
    print(f"✅ Created {price_file}")
    print(f"   📊 {len(dates)} trading days, {len(tickers)} stocks")
    print(f"   📈 Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    # Create market_caps.csv with detailed company information
    print("\nCreating market_caps.csv...")
    
    market_data = {
        'ticker': tickers,
        'company_name': [
            'Australia and New Zealand Banking Group', 'Commonwealth Bank of Australia', 'Macquarie Group', 'National Australia Bank', 'Rio Tinto',
            'Woolworths Group', 'BHP Group', 'CSL Limited', 'Telstra Corporation', 'Westpac Banking Corporation'
        ],
        'market_cap': [85, 180, 85, 125, 160, 40, 200, 150, 35, 70],  # Billions AUD
        'sector': [
            'Banking', 'Banking', 'Financial Services', 'Banking', 'Mining',
            'Consumer Staples', 'Mining', 'Healthcare', 'Telecommunications', 'Banking'
        ],
        'industry': [
            'Banking', 'Banking', 'Investment Banking', 'Banking', 'Iron Ore Mining',
            'Supermarkets', 'Iron Ore Mining', 'Biotechnology', 'Telecommunications', 'Banking'
        ],
        'exchange': ['ASX'] * 10,
        'country': ['Australia'] * 10,
        'currency': ['AUD'] * 10,
        'employees': [40000, 52000, 18000, 35000, 47000, 210000, 80000, 27000, 28000, 40000],
        'founded': [1835, 1911, 1969, 1858, 1873, 1924, 1885, 1916, 1975, 1817],
        'headquarters': [
            'Melbourne, VIC', 'Sydney, NSW', 'Sydney, NSW', 'Melbourne, VIC', 'London, UK',
            'Sydney, NSW', 'Melbourne, VIC', 'Melbourne, VIC', 'Melbourne, VIC', 'Sydney, NSW'
        ]
    }
    
    market_df = pd.DataFrame(market_data)
    market_file = os.path.join(demo_dir, 'market_caps.csv')
    market_df.to_csv(market_file, index=False)
    
    print(f"✅ Created {market_file}")
    print(f"   🏢 {len(tickers)} companies with detailed information")
    print(f"   🎯 Sectors: {len(set(market_data['sector']))} unique sectors")
    
    return demo_dir, price_file, market_file

def demo_scenario_1_basic_optimization():
    """Demo Scenario 1: Basic Portfolio Optimization with CSV."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 1: Basic Portfolio Optimization")
    print("="*60)
    print("Objective: Optimize Australian Big Four Banks portfolio using CSV data")
    print("Method: Mean-Variance Optimization")
    print("Data: merged_stock_prices.csv (ASX data)")
    
    cmd = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--tickers", "ANZ.AX,CBA.AX,NAB.AX",
        "--method", "mean_variance",
        "--start-date", "2023-01-01",
        "--end-date", "2023-12-31",
        "--allocations", "ANZ.AX:0.3,CBA.AX:0.4,NAB.AX:0.3",
        "--show-charts"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nExpected Output:")
    print("- Portfolio optimization results")
    print("- Comparison of provided vs optimal weights")
    print("- Risk-return metrics (Sharpe ratio, volatility, etc.)")
    print("- Efficient frontier visualization")
    
    return cmd

def demo_scenario_2_advanced_optimization():
    """Demo Scenario 2: Advanced Optimization Methods."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 2: Advanced Optimization Methods")
    print("="*60)
    print("Objective: Compare optimization approaches across ASX sectors")
    print("Methods: Risk Parity, CVaR, Market Neutral")
    print("Data: merged_stock_prices.csv (ASX diversified portfolio)")
    
    scenarios = []
    
    # Risk Parity
    cmd1 = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX",
        "--advanced-method", "risk_parity",
        "--start-date", "2023-01-01",
        "--show-charts"
    ]
    scenarios.append(("Risk Parity", cmd1))
    
    # CVaR Optimization
    cmd2 = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--tickers", "ANZ.AX,CBA.AX,NAB.AX,WBC.AX",
        "--advanced-method", "min_cvar",
        "--confidence-level", "0.05",
        "--start-date", "2023-01-01"
    ]
    scenarios.append(("CVaR Optimization", cmd2))
    
    # Market Neutral
    cmd3 = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--tickers", "CBA.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,MQG.AX",
        "--advanced-method", "market_neutral",
        "--target-volatility", "0.15",
        "--start-date", "2023-01-01"
    ]
    scenarios.append(("Market Neutral", cmd3))
    
    for name, cmd in scenarios:
        print(f"\n{name}:")
        print(" ".join(cmd))
    
    print("\nExpected Output:")
    print("- Different optimization approaches")
    print("- Risk-focused vs return-focused allocations") 
    print("- Long-short positions (market neutral)")
    print("- Tail risk management (CVaR)")
    
    return scenarios

def demo_scenario_3_backtesting():
    """Demo Scenario 3: Comprehensive Strategy Backtesting."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 3: Comprehensive Strategy Backtesting")
    print("="*60)
    print("Objective: Compare multiple strategies over time using ASX stocks")
    print("Methods: Mean-Variance, HRP, Risk Parity")
    print("Data: merged_stock_prices.csv (ASX 2022-2024)")
    
    cmd = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--run-backtest",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX,BHP.AX,MQG.AX",
        "--rebalance-frequency", "M",
        "--lookback-window", "252",
        "--start-date", "2022-06-01",
        "--allocations", "ANZ.AX:0.15,CBA.AX:0.25,RIO.AX:0.2,WOW.AX:0.15,BHP.AX:0.15,MQG.AX:0.1",
        "--output", "backtest_results.json"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nExpected Output:")
    print("- Multi-strategy performance comparison")
    print("- Out-of-sample validation results")
    print("- Rolling performance metrics")
    print("- Strategy correlation analysis")
    print("- Best performing strategy identification")
    
    return cmd

def demo_scenario_4_institutional_reporting():
    """Demo Scenario 4: Institutional Grade Reporting."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 4: Institutional Grade Reporting")
    print("="*60)
    print("Objective: Generate professional client reports for ASX investments")
    print("Target: Australian hedge fund quarterly report")
    print("Data: merged_stock_prices.csv + market_caps.csv (ASX data)")
    
    cmd = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--generate-report",
        "--client-name", "ASX Investment Fund",
        "--report-type", "QUARTERLY",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,MQG.AX,NAB.AX",
        "--start-date", "2023-01-01",
        "--allocations", "ANZ.AX:0.125,CBA.AX:0.2,RIO.AX:0.15,WOW.AX:0.1,BHP.AX:0.15,CSL.AX:0.125,MQG.AX:0.1,NAB.AX:0.125"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nExpected Output:")
    print("- Executive summary dashboard")
    print("- Comprehensive performance analytics") 
    print("- Risk attribution analysis")
    print("- Compliance monitoring")
    print("- Professional visualizations")
    print("- Client-ready PDF reports")
    
    return cmd

def demo_scenario_5_database_integration():
    """Demo Scenario 5: Database Integration and Caching."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 5: Database Integration & Caching")
    print("="*60)
    print("Objective: Demonstrate database caching for ASX data performance")
    print("Workflow: ASX CSV → Database → Fast retrieval")
    print("Data: merged_stock_prices.csv (ASX stocks)")
    
    scenarios = []
    
    # Step 1: Load CSV into database
    cmd1 = [
        "python", "-m", "shell",
        "--data", "demo_data/merged_stock_prices.csv",
        "--database-url", "sqlite:///demo_portfolio.db",
        "--update-cache",
        "--tickers", "ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX,BHP.AX,CSL.AX,TLS.AX,WBC.AX",
        "--cache-days", "730"
    ]
    scenarios.append(("Load CSV to Database", cmd1))
    
    # Step 2: Use database for fast optimization
    cmd2 = [
        "python", "-m", "shell",
        "--database-url", "sqlite:///demo_portfolio.db",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX",
        "--method", "mean_variance",
        "--start-date", "2023-01-01",
        "--show-charts"
    ]
    scenarios.append(("Fast Database Retrieval", cmd2))
    
    # Step 3: Hybrid approach (database + real-time fallback)
    cmd3 = [
        "python", "-m", "shell",
        "--data-source", "database",
        "--fallback-sources", "yahoo,file",
        "--database-url", "sqlite:///demo_portfolio.db",
        "--data", "demo_data/merged_stock_prices.csv",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX",
        "--method", "mean_variance"
    ]
    scenarios.append(("Hybrid Database + Real-time", cmd3))
    
    for name, cmd in scenarios:
        print(f"\n{name}:")
        print(" ".join(cmd))
    
    print("\nExpected Output:")
    print("- Fast data loading from database")
    print("- Automatic fallback to other sources")
    print("- Data quality and coverage reports")
    print("- Significant performance improvement")
    
    return scenarios

def demo_scenario_6_real_time_integration():
    """Demo Scenario 6: Real-time Data Integration."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 6: Real-time Data Integration")
    print("="*60)
    print("Objective: Combine ASX CSV historical data with live ASX prices")
    print("Sources: CSV (historical ASX data) + Yahoo Finance (real-time ASX)")
    print("Data: merged_stock_prices.csv + live ASX market data")
    
    scenarios = []
    
    # Real-time optimization
    cmd1 = [
        "python", "-m", "shell",
        "--data-source", "yahoo",
        "--real-time",
        "--tickers", "ANZ.AX,CBA.AX,NAB.AX",
        "--method", "mean_variance",
        "--show-charts"
    ]
    scenarios.append(("Real-time Optimization", cmd1))
    
    # Price streaming
    cmd2 = [
        "python", "-m", "shell",
        "--stream-prices",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX",
        "--stream-interval", "30"
    ]
    scenarios.append(("Real-time Price Streaming", cmd2))
    
    # Hybrid historical + real-time
    cmd3 = [
        "python", "-m", "shell",
        "--data-source", "file",
        "--data", "demo_data/merged_stock_prices.csv",
        "--fallback-sources", "yahoo",
        "--real-time",
        "--tickers", "ANZ.AX,CBA.AX,NAB.AX",
        "--method", "mean_variance"
    ]
    scenarios.append(("Hybrid Historical + Real-time", cmd3))
    
    for name, cmd in scenarios:
        print(f"\n{name}:")
        print(" ".join(cmd))
    
    print("\nExpected Output:")
    print("- Current market prices")
    print("- Live price streaming updates")
    print("- Optimization with latest data")
    print("- Seamless fallback between sources")
    
    return scenarios

def demo_scenario_7_multi_format_support():
    """Demo Scenario 7: Multiple File Format Support."""
    
    print("\n" + "="*60)
    print("DEMO SCENARIO 7: Multiple File Format Support")
    print("="*60)
    print("Objective: Demonstrate various file format capabilities")
    print("Formats: CSV, Excel, Parquet, JSON")
    print("Data: Convert and use merged_stock_prices in different formats")
    
    # First, create files in different formats using Python
    conversion_script = """
import pandas as pd
from core.data_sources.file_data import FileDataSource

# Load original CSV
fs = FileDataSource('./demo_data/')
data = fs.load_price_data('merged_stock_prices.csv')

# Convert to different formats
fs.save_data(data, 'demo_data/prices.xlsx', 'excel')
fs.save_data(data, 'demo_data/prices.parquet', 'parquet') 
fs.save_data(data, 'demo_data/prices.json', 'json')

print("Created files in multiple formats:")
print("✅ prices.xlsx (Excel)")
print("✅ prices.parquet (Parquet)")  
print("✅ prices.json (JSON)")
"""
    
    scenarios = []
    
    # Excel file usage
    cmd1 = [
        "python", "-m", "shell",
        "--data", "demo_data/prices.xlsx",
        "--tickers", "ANZ.AX,CBA.AX,NAB.AX",
        "--method", "mean_variance"
    ]
    scenarios.append(("Excel File", cmd1))
    
    # Parquet file usage (high performance)
    cmd2 = [
        "python", "-m", "shell",
        "--data", "demo_data/prices.parquet",
        "--tickers", "ANZ.AX,CBA.AX,RIO.AX,WOW.AX,MQG.AX",
        "--advanced-method", "risk_parity"
    ]
    scenarios.append(("Parquet File (Fast)", cmd2))
    
    print("Conversion Script:")
    print(conversion_script)
    
    for name, cmd in scenarios:
        print(f"\n{name}:")
        print(" ".join(cmd))
    
    print("\nExpected Output:")
    print("- Same results regardless of file format")
    print("- Automatic format detection")
    print("- Performance differences (Parquet fastest)")
    print("- Seamless format conversion")
    
    return scenarios

def run_all_demo_scenarios():
    """Run all demo scenarios and create documentation."""
    
    print("Portfolio Optimizer - Complete Demo Scenarios")
    print("=" * 80)
    print("Creating realistic demo files and showcasing all capabilities...")
    
    # Create demo files
    demo_dir, price_file, market_file = create_demo_files()
    
    # Run all scenarios
    scenarios = [
        demo_scenario_1_basic_optimization(),
        demo_scenario_2_advanced_optimization(), 
        demo_scenario_3_backtesting(),
        demo_scenario_4_institutional_reporting(),
        demo_scenario_5_database_integration(),
        demo_scenario_6_real_time_integration(),
        demo_scenario_7_multi_format_support()
    ]
    
    # Create demo script
    demo_script_path = os.path.join(demo_dir, "run_demos.sh")
    with open(demo_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Portfolio Optimizer Demo Script\n")
        f.write("# Generated automatically\n\n")
        
        for i, scenario in enumerate(scenarios, 1):
            f.write(f"echo 'Running Demo Scenario {i}...'\n")
            if isinstance(scenario, list) and len(scenario) > 0 and isinstance(scenario[0], str):
                # Single command scenario
                f.write(" ".join(scenario) + "\n\n")
            elif isinstance(scenario, list) and len(scenario) > 0 and isinstance(scenario[0], tuple):
                # Multiple command scenario
                for name, cmd in scenario:
                    f.write(f"echo '{name}:'\n")
                    f.write(" ".join(cmd) + "\n\n")
            else:
                # Fallback - skip if format is unclear
                f.write("# Scenario format not recognized\n\n")
    
    os.chmod(demo_script_path, 0o755)
    
    print(f"\n✅ Demo files created in: {demo_dir}")
    print(f"✅ Demo script created: {demo_script_path}")
    print("\nTo run the demos:")
    print(f"  cd {demo_dir}")
    print("  ./run_demos.sh")
    print("\nOr run individual scenarios using the commands shown above.")
    
    return demo_dir

if __name__ == "__main__":
    run_all_demo_scenarios()