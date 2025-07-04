# Portfolio Optimizer - Complete Documentation

A comprehensive, production-grade portfolio optimization platform designed for hedge funds, asset managers, brokers, and professional investors. Built with PyPortfolioOpt and enhanced with institutional-grade features including advanced optimization methods, real-time data integration, backtesting, analytics, and professional reporting.

## Table of Contents

1. [🎯 Overview](#-overview)
2. [✨ Key Features](#-key-features)
3. [🚀 Quick Start](#-quick-start)
4. [📁 CSV File Support](#-csv-file-support-primary-format)
5. [🌐 Data Sources](#-data-sources)
6. [📊 Optimization Methods](#-optimization-methods)
7. [🔄 Advanced Features](#-advanced-features)
8. [🎛️ Command Line Interface](#️-command-line-interface)
9. [🔧 Configuration Files](#-configuration-files)
10. [📈 Input Validation Rules](#-input-validation-rules)
11. [📚 Charts & Visualization](#-charts--visualization)
12. [🏗️ Project Architecture](#️-project-architecture)
13. [🚀 Production Deployment](#-production-deployment)
14. [🧪 Testing & Demo](#-testing--demo)
15. [🔐 Security & Best Practices](#-security--best-practices)
16. [🐛 Troubleshooting](#-troubleshooting)
17. [📄 License](#-license)

## 🎯 Overview

This production-ready tool combines the power of PyPortfolioOpt with modern data sources, advanced analytics, and institutional reporting capabilities. It supports both offline analysis with CSV files and real-time market data integration, making it suitable for everything from academic research to professional investment management.

The included demo files contain **Australian ASX-listed companies:**
- **ANZ.AX** - Australia and New Zealand Banking Group (Major Bank)
- **CBA.AX** - Commonwealth Bank of Australia (Largest Bank)
- **MQG.AX** - Macquarie Group (Investment Banking)
- **NAB.AX** - National Australia Bank (Big Four Bank)
- **RIO.AX** - Rio Tinto (Global Mining Giant)
- **WOW.AX** - Woolworths Group (Consumer Staples)

## ✨ Key Features

### 🔬 **Advanced Optimization Methods**
- **Mean-Variance Optimization:** Classic Markowitz model with L2 regularization
- **Black-Litterman Model:** Market equilibrium with investor views integration
- **Hierarchical Risk Parity (HRP):** Modern risk-based allocation
- **Risk Parity:** Equal risk contribution optimization
- **CVaR Optimization:** Conditional Value at Risk minimization
- **Market Neutral:** Long-short strategies with beta constraints
- **Semivariance:** Downside risk-focused optimization
- **L2 Regularized Max Sharpe:** Reduces concentration risk for institutional use

### 📊 **Data Sources & Integration**
- **CSV Files:** Primary format with intelligent parsing and auto-detection
- **Real-Time APIs:** Yahoo Finance, Alpha Vantage, Quandl, FRED, CoinGecko
- **Database Support:** SQLite, PostgreSQL, MySQL with automatic caching
- **Multiple Formats:** Excel, Parquet, JSON, HDF5 support
- **Hybrid Workflows:** Combine offline files with real-time data

### 📈 **Professional Analytics**
- **Comprehensive Metrics:** Risk-adjusted returns, drawdowns, VaR, CVaR
- **Risk Attribution:** Factor decomposition and contribution analysis
- **Performance Attribution:** Return decomposition by asset/sector
- **Stress Testing:** Scenario analysis and sensitivity testing
- **Correlation Analysis:** Dynamic correlation tracking

### 🔄 **Backtesting Framework**
- **Walk-Forward Analysis:** Out-of-sample validation
- **Multiple Strategies:** Compare different optimization methods
- **Rebalancing Control:** Flexible frequency and triggers
- **Transaction Costs:** Realistic performance estimation
- **Risk Management:** Drawdown limits and stop-loss integration

### 📋 **Institutional Reporting**
- **Client Reports:** Professional PDF reports with executive summaries
- **Compliance Monitoring:** Regulatory constraint tracking
- **Performance Dashboards:** Interactive visualization for clients
- **Risk Reports:** Detailed risk analysis and attribution
- **Custom Templates:** Branded reports for different client types

### 🌐 **API & Integration**
- **RESTful API:** FastAPI-based endpoints for system integration
- **Client Libraries:** Python SDK for programmatic access
- **Real-Time Streaming:** Live price monitoring and alerts
- **Webhook Support:** Event-driven integrations

## 🚀 Quick Start

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd portfolio_optimizer

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Your First Analysis (CSV Files)

Your existing CSV workflows work unchanged:

```bash
# Basic portfolio optimization with CSV data (Australian banks)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts

# Enhanced analysis with date filtering (Banking + Mining)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --tickers CBA.AX,RIO.AX,WOW.AX \
  --advanced-method risk_parity \
  --show-charts
```

### Real-Time Analysis

```bash
# Live market data analysis (ASX Big Four Banks)
python -m shell \
  --data-source yahoo \
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX \
  --method mean_variance \
  --real-time \
  --show-charts

# Multi-source with fallback (Diversified ASX portfolio)
python -m shell \
  --data-source alphavantage \
  --fallback-sources yahoo,file \
  --data demo_data/merged_stock_prices.csv \
  --api-keys '{"alphavantage": "YOUR_API_KEY"}' \
  --tickers CBA.AX,RIO.AX,WOW.AX
```

## 📁 CSV File Support (Primary Format)

CSV files are the **main** and **most important** data format. All new features work seamlessly with your existing CSV workflows.

### ✅ **Your Current Workflow - Still Works**

```bash
# This continues to work exactly as before
python -m shell --data demo_data/merged_stock_prices.csv --tickers ANZ.AX,CBA.AX,NAB.AX --method mean_variance --show-charts
```

### 📊 **Supported CSV Formats**

#### **Format A: Standard Format (Most Common)**
```csv
Date,ANZ.AX,CBA.AX,MQG.AX
2023-01-01,25.45,98.20,158.30
2023-01-02,25.68,99.15,159.45
2023-01-03,25.32,97.85,157.90
```

#### **Format B: Long Format**
```csv
date,ticker,price
2023-01-01,ANZ.AX,25.45
2023-01-01,CBA.AX,98.20
2023-01-01,MQG.AX,158.30
```

#### **Format C: European Format (Semicolon Separated)**
```csv
Date;ANZ.AX;CBA.AX;MQG.AX
01/01/2023;25,45;98,20;158,30
02/01/2023;25,68;99,15;159,45
```

### 🚀 **Enhanced CSV Features**

#### **Date Range Filtering**
```bash
# Filter CSV data by date range
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance
```

#### **Advanced Optimization Methods**
```bash
# Use advanced methods with CSV data
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --advanced-method risk_parity \
  --tickers ANZ.AX,CBA.AX,NAB.AX,MQG.AX \
  --show-charts

# CVaR optimization with CSV
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --advanced-method min_cvar \
  --confidence-level 0.05 \
  --tickers ANZ.AX,CBA.AX,NAB.AX
```

#### **CSV with Backtesting**
```bash
# Run comprehensive backtest using CSV data
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --run-backtest \
  --rebalance-frequency M \
  --lookback-window 252 \
  --tickers ANZ.AX,CBA.AX,NAB.AX,MQG.AX
```

#### **CSV with Institutional Reports**
```bash
# Generate professional reports from CSV data
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --generate-report \
  --client-name "ASX Investment Fund" \
  --report-type QUARTERLY \
  --tickers ANZ.AX,CBA.AX,NAB.AX,MQG.AX
```

### **Hybrid CSV + Real-Time Usage**

#### **CSV as Primary, Real-Time as Fallback**
```bash
# Use CSV first, fallback to Yahoo Finance for missing data
python -m shell \
  --data-source file \
  --data demo_data/merged_stock_prices.csv \
  --fallback-sources yahoo \
  --tickers ANZ.AX,CBA.AX,NAB.AX,NEWSTOCK.AX \
  --method mean_variance
```

#### **CSV with Database Caching**
```bash
# Load CSV into database for faster future access
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --database-url sqlite:///portfolio.db \
  --update-cache \
  --tickers ANZ.AX,CBA.AX,NAB.AX

# Use cached data for faster processing
python -m shell \
  --database-url sqlite:///portfolio.db \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance
```

## 🌐 Data Sources

### 📁 **Offline Data Sources**

| Format | Description | Usage |
|--------|-------------|--------|
| **CSV** | **Primary format - full support** | `--data demo_data/merged_stock_prices.csv` |
| Excel | .xlsx/.xls files | `--data asx_prices.xlsx` |
| Parquet | High-performance format | `--data asx_prices.parquet` |
| JSON | Structured data | `--data asx_prices.json` |
| HDF5 | Large datasets | `--data asx_prices.h5` |

### 🌐 **Real-Time Data Sources**

| Source | API Key | Rate Limit | ASX Coverage |
|--------|---------|------------|--------------|
| Yahoo Finance | Not required | No official limit | Full ASX coverage (.AX suffix) |
| Alpha Vantage | Required | 5 calls/min (free) | Major ASX stocks |
| Quandl | Required | 50 calls/day (free) | ASX economic datasets |
| FRED | Required | 120 calls/min | Australian economic data |
| CoinGecko | Not required | 10-50 calls/min | Cryptocurrency |

### 💾 **Database Support**

| Database | Usage | Best For |
|----------|-------|----------|
| SQLite | `sqlite:///asx_portfolio.db` | Development, single-user |
| PostgreSQL | `postgresql://user:pass@host/asx_db` | Production, multi-user |
| MySQL | `mysql://user:pass@host/asx_db` | Production, alternative |

### **Data Workflow Examples**

#### **1. Offline Analysis (CSV Files) - PRIMARY SUPPORT**
```bash
# Traditional workflow with CSV files (UNCHANGED - still works)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts
```

#### **2. Real-Time Analysis**
```bash
# Live market data analysis
python -m shell \
  --data-source yahoo \
  --real-time \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts
```

#### **3. Multi-Source Fallback**
```bash
# Try premium source, fallback to free sources
python -m shell \
  --data-source alphavantage \
  --fallback-sources yahoo,file \
  --api-keys '{"alphavantage": "YOUR_KEY"}' \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,NAB.AX
```

## 📊 Optimization Methods

### **Traditional Methods**

#### **Mean-Variance Optimization (Big Four Banks)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --method mean_variance \
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX \
  --show-charts
```

#### **Black-Litterman Model (Banking vs Resources)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --method black_litterman \
  --tickers CBA.AX,RIO.AX,WOW.AX \
  --views "CBA.AX:0.12:0.8,RIO.AX:0.18:0.6" \
  --show-charts
```

#### **Hierarchical Risk Parity (Diversified ASX)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --method hrp \
  --tickers ANZ.AX,CBA.AX,MQG.AX,RIO.AX \
  --show-charts
```

### **Advanced Methods**

#### **Risk Parity (All Sectors)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --advanced-method risk_parity \
  --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX \
  --show-charts
```

#### **CVaR Optimization (Conservative Banking)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --advanced-method min_cvar \
  --confidence-level 0.05 \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --show-charts
```

#### **Market Neutral Strategy (Banking vs Resources)**
```bash
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --advanced-method market_neutral \
  --target-volatility 0.15 \
  --tickers ANZ.AX,CBA.AX,RIO.AX,WOW.AX \
  --show-charts
```

| Method | Use Case | Key Features |
|--------|----------|--------------|
| **Mean-Variance** | Classic portfolio optimization | Efficient frontier, risk-return optimization |
| **Black-Litterman** | Incorporating market views | Bayesian approach, investor views integration |
| **Hierarchical Risk Parity** | Alternative diversification | Machine learning, cluster-based allocation |
| **L2 Regularized Max Sharpe** | Reducing concentration risk | Diversification penalty, institutional grade |
| **CVaR Optimization** | Tail risk management | Conditional Value at Risk minimization |
| **Semivariance** | Downside risk focus | Asymmetric risk measurement |
| **Risk Parity** | Equal risk contribution | Diversified risk budgeting |
| **Market Neutral** | Hedge fund strategies | Long-short, market exposure control |

## 🔄 Advanced Features

### **Comprehensive Backtesting**

```bash
# Multi-strategy backtesting with CSV data (Diversified ASX)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --run-backtest \
  --tickers ANZ.AX,CBA.AX,MQG.AX,RIO.AX \
  --rebalance-frequency M \
  --lookback-window 252 \
  --start-date 2022-01-01 \
  --output backtest_results.json
```

### **Institutional Reporting**

```bash
# Generate professional client reports (ASX Banking Sector)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --generate-report \
  --client-name "ASX Banking Fund" \
  --report-type QUARTERLY \
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX \
  --start-date 2023-01-01
```

### **Real-Time Data Integration**

```bash
# Real-time optimization (ASX Blue Chips)
python -m shell \
  --data-source yahoo \
  --real-time \
  --tickers CBA.AX,RIO.AX,WOW.AX \
  --method mean_variance \
  --show-charts

# Price streaming (ASX leaders)
python -m shell \
  --stream-prices \
  --tickers ANZ.AX,CBA.AX,RIO.AX \
  --stream-interval 30
```

### **Database Integration**

```bash
# Load CSV into database for faster access (All ASX stocks)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --database-url sqlite:///portfolio.db \
  --update-cache \
  --tickers ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX

# Use cached data for fast optimization (Banking focus)
python -m shell \
  --database-url sqlite:///portfolio.db \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts
```

## 🎛️ Command Line Interface

### Enhanced CLI Arguments

```bash
# Data Source Options
--data                    CSV file path (primary format)
--data-source            Data source (file, yahoo, alphavantage, etc.)
--fallback-sources       Comma-separated fallback sources
--database-url           Database URL for caching
--api-keys               JSON string with API keys
--real-time              Use real-time prices
--stream-prices          Start price streaming

# Optimization Options
--method                 Basic method (mean_variance, black_litterman, hrp)
--advanced-method        Advanced method (risk_parity, min_cvar, market_neutral)
--confidence-level       CVaR confidence level
--target-volatility      Target volatility for market neutral
--constraints            Weight constraints
--risk-free-rate         Risk-free rate (supports percentage conversion)

# Analysis Options
--run-backtest           Run comprehensive backtesting
--rebalance-frequency    Rebalancing frequency (D, W, M, Q)
--lookback-window        Lookback window for rolling optimization
--generate-report        Generate institutional client report
--client-name            Client name for reports
--report-type            Report type (MONTHLY, QUARTERLY, ANNUAL)

# Configuration Options
--config                 Path to JSON configuration file
--save-config            Save current parameters to JSON configuration file

# Output Options
--output                 JSON output file
--show-charts            Generate and display charts
--charts-from-json       Generate charts from existing results
```

## 🔧 Configuration Files

The CLI supports both command-line parameters AND configuration files for maximum flexibility.

### **✅ Risk-Free Rate Support**

The CLI fully supports risk-free rate configuration through:

#### **1. Command Line Parameter**
```bash
python -m shell --risk-free-rate 0.0435  # 4.35% as decimal
python -m shell --risk-free-rate 4.35    # 4.35% as percentage (auto-converted)
```

#### **2. Configuration File**
```json
{
  "risk_free_rate": 0.0435
}
```

#### **3. Hybrid Approach**
```bash
# Use config file but override risk-free rate
python -m shell --config conservative.json --risk-free-rate 5.0
```

### **Configuration File Format**

Configuration files use JSON format with the same parameter names as CLI arguments:

```json
{
  "data": "merged_stock_prices.csv",
  "tickers": "ANZ.AX,CBA.AX,NAB.AX,WBC.AX",
  "method": "mean_variance",
  "start_date": "2021-01-01", 
  "end_date": "2024-01-01",
  "risk_free_rate": 0.0435,
  "allocations": "ANZ.AX:0.25,CBA.AX:0.25,NAB.AX:0.25,WBC.AX:0.25",
  "constraints": "ANZ.AX:0.15:0.35,CBA.AX:0.15:0.35",
  "output": "portfolio_results.json",
  "show_charts": true,
  "data_source": "file"
}
```

### **Usage Examples**

#### **Loading Configuration Files**
```bash
# Use complete configuration from file
python -m shell --config examples/config_conservative.json

# Use config file with specific overrides
python -m shell --config examples/config_rba_rate.json --output my_results.json

# Override risk-free rate from config
python -m shell --config examples/config_conservative.json --risk-free-rate 3.5
```

#### **Saving Configuration Files**
```bash
# Save current command line parameters to config file
python -m shell \
    --tickers ANZ.AX,CBA.AX,NAB.AX \
    --method mean_variance \
    --risk-free-rate 4.35 \
    --start-date 2021-01-01 \
    --end-date 2024-01-01 \
    --save-config my_config.json
```

### **Pre-configured Examples**

#### **1. Conservative Portfolio (examples/config_conservative.json)**
```json
{
  "data": "merged_stock_prices.csv",
  "tickers": "ANZ.AX,CBA.AX,NAB.AX,WBC.AX",
  "method": "mean_variance",
  "start_date": "2021-01-01",
  "end_date": "2024-01-01",
  "risk_free_rate": 0.01,
  "allocations": "ANZ.AX:0.25,CBA.AX:0.25,NAB.AX:0.25,WBC.AX:0.25",
  "constraints": "ANZ.AX:0.20:0.30,CBA.AX:0.20:0.30,NAB.AX:0.20:0.30,WBC.AX:0.20:0.30",
  "output": "conservative_portfolio.json",
  "show_charts": false
}
```

#### **2. RBA Rate Portfolio (examples/config_rba_rate.json)**
```json
{
  "data": "merged_stock_prices.csv",
  "tickers": "ANZ.AX,CBA.AX,NAB.AX,WBC.AX,RIO.AX,BHP.AX",
  "method": "mean_variance",
  "start_date": "2021-01-01",
  "end_date": "2024-01-01",
  "risk_free_rate": 0.0435,
  "output": "rba_rate_portfolio.json",
  "show_charts": true
}
```

### **Parameter Precedence**

When using both config files and command line parameters:

1. **Command line parameters** take highest precedence
2. **Configuration file values** are used for unspecified CLI parameters  
3. **Default values** are used if neither CLI nor config specify a parameter

```bash
# config.json has: "risk_free_rate": 0.02
# This command will use 0.05 (CLI override)
python -m shell --config config.json --risk-free-rate 0.05

# This command will use 0.02 (from config file)
python -m shell --config config.json
```

## 📈 Input Validation Rules

The portfolio optimization system includes comprehensive input validation to ensure data quality and prevent invalid optimization attempts.

### **1. Date Range Validation**

#### **Format Requirements**
- **Format**: `YYYY-MM-DD` (ISO 8601 standard)
- **Example**: `2020-01-01`, `2023-12-31`
- **Invalid**: `01/01/2020`, `2020-1-1`, `Jan 1, 2020`

#### **Temporal Requirements**
- **Start Date < End Date**: Start date must be chronologically before end date
- **Mean-Variance**: Minimum 3 years required
- **Black-Litterman**: Minimum 5 years required (when investor views provided)

#### **Examples**
```python
# Valid
validate_portfolio_inputs(
    tickers=['ANZ.AX', 'CBA.AX'],
    start_date='2021-01-01',
    end_date='2024-01-01',  # 3 years
    method='mean_variance'
)

# Invalid - insufficient range
validate_portfolio_inputs(
    tickers=['ANZ.AX', 'CBA.AX'],
    start_date='2022-01-01',
    end_date='2023-12-31',  # Only 2 years
    method='mean_variance'
)
# Raises: "Date range must be at least 3 years for Mean-Variance"
```

### **2. Allocation Validation**

#### **Allocation Types**

**No Allocation**: Auto-distributes equally among tickers
```python
# Input
allocations = {}
tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX']

# Result: Each gets 33.33%
{'ANZ.AX': 0.3333, 'CBA.AX': 0.3333, 'NAB.AX': 0.3333}
```

**Partial Allocation**: 
- Total cannot exceed 100%
- Remaining distributed equally to unallocated tickers
```python
# Input
allocations = {'ANZ.AX': 0.4, 'CBA.AX': 0.3}  # 70% allocated
tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX', 'MQG.AX']

# Result: Remaining 30% split between NAB.AX and MQG.AX (15% each)
{'ANZ.AX': 0.4, 'CBA.AX': 0.3, 'NAB.AX': 0.15, 'MQG.AX': 0.15}
```

**Full Allocation**: Sum must equal 100% (±0.0001% tolerance)
```python
# Valid
allocations = {'ANZ.AX': 0.3, 'CBA.AX': 0.4, 'NAB.AX': 0.3}  # Exactly 100%

# Invalid
allocations = {'ANZ.AX': 0.3, 'CBA.AX': 0.4, 'NAB.AX': 0.2}  # Only 90%
```

### **3. Constraint Validation**

#### **Weight Bounds**
- Minimum and maximum values between 0.0 and 1.0
- Minimum ≤ Maximum for each ticker

#### **Feasibility Checks**
- **Existing allocations** must satisfy constraints
- **Equal distribution** must satisfy constraints (when no allocations provided)
- **Portfolio feasibility**: Sum of minimums ≤ 100%, Sum of maximums ≥ 100%

### **4. Risk-Free Rate Validation**

#### **Automatic Conversion**
- Automatically converted to float
- Default: 0.02 (2%)
- Percentage conversion: Values > 1 divided by 100

```python
# Valid conversions
risk_free_rate = 0.025      # 2.5% as decimal
risk_free_rate = 2.5        # 2.5% as percentage (converted to 0.025)
risk_free_rate = "0.03"     # String "0.03" (converted to 0.03)

# Invalid
risk_free_rate = "invalid"  # Non-numeric string
risk_free_rate = -0.01      # Negative rate
```

### **5. Investor Views Validation (Black-Litterman)**

#### **Format Requirements**
```python
investor_views = {
    "ticker": {
        "expected_return": float,  # Required
        "confidence": float        # Optional, defaults to 0.5
    }
}
```

#### **Examples**
```python
# Valid investor views
validate_portfolio_inputs(
    tickers=['ANZ.AX', 'CBA.AX'],
    start_date='2019-01-01',
    end_date='2024-01-01',  # 5 years
    method='black_litterman',
    investor_views={
        'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8},
        'CBA.AX': {'expected_return': 0.15}  # confidence defaults to 0.5
    }
)
```

## 📚 Charts & Visualization

### **Chart Types**

- **Efficient Frontier:** Risk-return visualization with envelope curves
- **Portfolio Allocation:** Pie charts comparing current vs optimal weights
- **Correlation Matrix:** Heatmap visualization of ASX asset correlations
- **Risk-Return Metrics:** Bar charts of performance indicators
- **Sector Analysis:** ASX sector-specific visualizations

### **Chart Generation**

```bash
# Generate charts during optimization (Banking sector)
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts

# Generate charts from existing results
python -m shell --charts-from-json asx_results.json
```

## 🏗️ Project Architecture

```
portfolio_optimizer/
├── core/                              # Core business logic
│   ├── data.py                       # Enhanced data processing
│   ├── models.py                     # Pydantic models
│   ├── utils.py                      # Financial utilities
│   ├── validation_simple.py         # Input validation system
│   ├── optimization/                 # Optimization algorithms
│   │   ├── mean_variance.py         # Classic Markowitz
│   │   ├── black_litterman.py       # Black-Litterman model
│   │   ├── hrp.py                   # Hierarchical Risk Parity
│   │   └── advanced_optimization.py # Advanced methods
│   ├── data_sources/                # Data integration layer
│   │   ├── file_data.py            # CSV/Excel/Parquet support
│   │   ├── market_feeds.py         # Real-time data sources
│   │   ├── database.py             # Database integration
│   │   └── data_manager.py         # Unified data management
│   ├── analytics.py                 # Professional analytics
│   ├── backtesting.py              # Backtesting framework
│   ├── reporting.py                # Institutional reporting
│   └── advanced_visualization.py   # Enhanced charting
├── shell/                           # CLI interface
│   ├── app.py                      # Application coordinator
│   ├── cli.py                      # Enhanced CLI with new features
│   ├── main.py                     # Main entry point
│   └── display/                    # Visualization and tools
│       └── visualization.py        # Chart generation and display
├── api/                            # REST API
│   ├── endpoints.py               # FastAPI endpoints
│   └── client.py                  # Python client SDK
├── tests/                          # Organized test suite
│   ├── unit/                      # Unit tests
│   │   └── test_validation.py     # Validation system tests
│   ├── integration/               # Integration tests
│   │   ├── test_csv_integration.py      # CSV integration tests
│   │   └── test_validation_integration.py # Validation integration tests
│   └── fixtures/                  # Test data and scenarios
│       └── test_demo_scenarios.py # Demo scenario tests
├── demo_data/                     # Sample data files
│   ├── merged_stock_prices.csv   # ASX sample data
│   ├── market_caps.csv          # ASX market cap data
│   └── run_demos.sh             # Demo script
├── docs/                         # Documentation
│   └── data_sources_guide.json   # Data source configuration guide
└── requirements.txt              # Dependencies
```

### **Core Modules**

| Module | Description | Key Features |
|--------|-------------|--------------|
| **core/optimization/** | Optimization algorithms | Mean-Variance, Black-Litterman, HRP, Risk Parity, CVaR |
| **core/data_sources/** | Data integration | CSV files, real-time APIs, database caching |
| **core/validation_simple.py** | Input validation | Date ranges, allocations, constraints, risk-free rate |
| **shell/** | Command-line interface | Enhanced CLI with configuration file support |
| **api/** | REST API | FastAPI endpoints and Python client SDK |

### **Key Architectural Improvements**

#### Test Organization (Updated Structure)
- **Unit Tests (`tests/unit/`)**: Isolated component testing with 27 comprehensive validation tests
- **Integration Tests (`tests/integration/`)**: End-to-end workflow testing with external dependencies (CSV, validation integration)
- **Test Fixtures (`tests/fixtures/`)**: Demo scenarios and realistic test data generation for ASX stocks

#### Chart Generation Enhancement
- **Standalone Chart Tool (`shell/display/chart_generator.py`)**: Independent chart generation from JSON optimization results
- **Flexible Usage**: Can be used separately from main CLI for post-processing and report generation
- **Multiple Chart Types**: Efficient frontier, portfolio weights, correlation matrix, risk-return metrics, and method-specific analyses

#### Validation System Consolidation  
- **Functional Validation (`core/validation_simple.py`)**: Streamlined validation approach with clear error messages
- **Comprehensive Coverage**: Date range validation (3+ years for Mean-Variance, 5+ years for Black-Litterman with views)
- **Smart Defaults**: Automatic percentage conversion for risk-free rates, feasibility checks for constraints

#### Demo Data Organization
- **Centralized Demo Data (`demo_data/`)**: All sample files and demo scripts in dedicated directory
- **ASX Market Focus**: Realistic Australian market data with Big Four banks, mining giants, and major sectors
- **Automated Demo Scripts**: `run_demos.sh` for complete workflow demonstrations

## 🚀 Production Deployment

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Environment Variables**
```bash
# API Configuration
PORTFOLIO_API_HOST=0.0.0.0
PORTFOLIO_API_PORT=8000
PORTFOLIO_API_KEY=your_api_key_here

# Database Configuration (optional)
PORTFOLIO_DB_URL=postgresql://user:pass@localhost/portfolio_db

# API Keys
ALPHAVANTAGE_API_KEY=your_api_key
QUANDL_API_KEY=your_quandl_key
FRED_API_KEY=your_fred_key

# Logging Configuration
PORTFOLIO_LOG_LEVEL=INFO
```

## 🧪 Testing & Demo

### **Organized Test Suite**

The test suite is now organized into three categories for better maintainability:

```bash
# Unit Tests (27 tests) - Fast validation and component testing
python -m pytest tests/unit/ -v

# Integration Tests (8 tests) - End-to-end workflow testing
python -m pytest tests/integration/ -v

# Test Fixtures and Demo Scenarios
python tests/fixtures/test_demo_scenarios.py

# Run all tests
python -m pytest tests/ -v
```

### **Test Categories**

| Category | Location | Purpose | Test Count |
|----------|----------|---------|------------|
| **Unit Tests** | `tests/unit/` | Component validation, input checking | 27 tests |
| **Integration Tests** | `tests/integration/` | CSV integration, workflow validation | 8 tests |
| **Fixtures** | `tests/fixtures/` | Demo data generation, scenarios | Demo scripts |

### **Chart Generation**

Charts can be generated from optimization results:

```bash
# Generate charts from optimization results using CLI
python -m shell --charts-from-json results.json

# Generate charts during optimization
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,NAB.AX \
  --method mean_variance \
  --show-charts
```

### **Demo Data Generation**

The project includes demo data generators that create realistic ASX CSV files:

```python
# Create demo ASX CSV files
from tests.fixtures.test_demo_scenarios import create_demo_files
demo_dir, price_file, market_file = create_demo_files()
# Creates: merged_stock_prices.csv, market_caps.csv with ASX data
```

## 🔐 Security & Best Practices

### **API Key Management**

```bash
# Store API keys securely in environment variables
export ALPHAVANTAGE_API_KEY="your_key"
export QUANDL_API_KEY="your_key"

# Use in CLI without exposing keys
python -m shell --data-source alphavantage --tickers CBA.AX,RIO.AX
```

### **Database Security**

```bash
# Use environment variables for database URLs
export PORTFOLIO_DB_URL="postgresql://user:pass@localhost/asx_portfolio_db"

# Enable SSL for production databases
export PORTFOLIO_DB_URL="postgresql://user:pass@localhost/asx_portfolio_db?sslmode=require"
```

### **Rate Limiting**

```python
# Configure appropriate rate limits
config = {
    'alphavantage': {'rate_limit': 12.0},  # 5 calls/minute = 12 seconds
    'yahoo': {'rate_limit': 0.5}           # Conservative limit
}
```

## 🐛 Troubleshooting

### **Common Issues**

#### **CSV File Issues (ASX Data)**
```bash
# Check CSV structure for ASX data
python -c "
from core.data_sources.file_data import preview_data_file
preview = preview_data_file('merged_stock_prices.csv')
print('Columns:', preview['columns'])
print('Date columns:', preview['potential_date_columns'])
print('ASX Ticker columns:', preview['potential_ticker_columns'])
"
```

#### **No Data Found (ASX Tickers)**
```bash
# Verify ASX ticker symbols and date ranges
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers CBA.AX \
  --start-date 2023-01-01 \
  --end-date 2023-12-31
```

#### **Performance Issues (Large ASX Dataset)**
```bash
# Use database caching for large ASX datasets
python -m shell \
  --data demo_data/large_asx_price_data.csv \
  --database-url sqlite:///asx_portfolio.db \
  --update-cache \
  --tickers ANZ.AX,CBA.AX,RIO.AX
```

#### **Validation Errors**
```bash
# Check validation requirements
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX \
  --start-date 2021-01-01 \
  --end-date 2024-01-01 \
  --method mean_variance
```

### **Data Quality Checks**

```bash
# Generate data quality report for ASX stocks
python -m shell \
  --data demo_data/merged_stock_prices.csv \
  --tickers ANZ.AX,CBA.AX,MQG.AX \
  --database-url sqlite:///asx_portfolio.db \
  --update-cache
```

## 📈 Performance Benchmarks

### **Optimization Speed (ASX Data)**
- **Small ASX portfolios (3-5 stocks):** < 1 second
- **Medium ASX portfolios (5-15 stocks):** < 5 seconds
- **Large ASX portfolios (15+ stocks):** < 30 seconds

### **Data Loading Speed**
- **merged_stock_prices.csv (ASX data):** < 1 second
- **Large ASX CSV files (1-10MB):** < 5 seconds
- **Database retrieval (ASX stocks):** < 0.5 seconds
- **Real-time ASX API calls:** 1-3 seconds per ticker

## 🎯 Use Cases

### **Academic Research (Australian Markets)**
- **Data Sources:** merged_stock_prices.csv + Australian economic data
- **Methods:** All optimization algorithms
- **Focus:** ASX market analysis, Australian banking sector studies

### **Individual Investors (ASX Focus)**
- **Data Sources:** Yahoo Finance (.AX tickers) + CSV files
- **Methods:** Mean-variance, risk parity
- **Output:** Personal ASX portfolio optimization

### **Australian Wealth Management**
- **Data Sources:** Alpha Vantage ASX data + database caching
- **Methods:** Black-Litterman with Australian market views
- **Output:** Client reports, ASX performance tracking

### **Australian Hedge Funds**
- **Data Sources:** Multiple APIs + real-time ASX streaming
- **Methods:** Advanced optimization, backtesting
- **Output:** Institutional reports, ASX risk management

### **Australian Brokers & Platforms**
- **Integration:** REST API + Python SDK
- **Data Sources:** Real-time ASX feeds + database
- **Output:** Automated ASX portfolio recommendations

## 🔮 Future Enhancements

### **Planned Features**
- **ASX-Specific Features:** Sector rotation models for Australian markets
- **Dividend Focus:** Australian dividend yield optimization
- **Currency Hedging:** AUD currency exposure management
- **ESG Integration:** ASX ESG scoring and sustainable investing
- **Small Cap Support:** Extended ASX small cap coverage

### **Australian Market Enhancements**
- **Sector Analysis:** Banking, Mining, Resources, Consumer sector models
- **Regulatory Compliance:** ASIC compliance reporting features
- **Tax Optimization:** Australian capital gains tax considerations
- **Franking Credits:** Dividend imputation credit integration

## 🏦 ASX Market Coverage

### **Included ASX Stocks (Sample Data)**

| Ticker | Company | Sector | Market Cap (AUD) |
|--------|---------|--------|------------------|
| **ANZ.AX** | Australia and New Zealand Banking Group | Banking | $85B |
| **CBA.AX** | Commonwealth Bank of Australia | Banking | $180B |
| **MQG.AX** | Macquarie Group | Financial Services | $85B |
| **NAB.AX** | National Australia Bank | Banking | $125B |
| **RIO.AX** | Rio Tinto | Mining | $160B |
| **WOW.AX** | Woolworths Group | Consumer Staples | $40B |

### **Sector Distribution**
- **Banking (66%):** ANZ.AX, CBA.AX, NAB.AX, MQG.AX
- **Mining (17%):** RIO.AX
- **Consumer Staples (17%):** WOW.AX

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🤝 Support & Community

### **Getting Help**
- **Documentation:** This README and inline code documentation
- **Issues:** GitHub Issues for bug reports and feature requests
- **Discussions:** GitHub Discussions for questions and ASX-specific support

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with ASX data
5. Submit a pull request

## 📋 Changelog

### **Version 2.0.0 (Current)**
- ✅ **Enhanced CSV support** with intelligent parsing (ASX optimized)
- ✅ **Advanced optimization methods** (Risk Parity, CVaR, Market Neutral)
- ✅ **Real-time ASX data integration** with .AX ticker support
- ✅ **Comprehensive backtesting** framework for ASX markets
- ✅ **Professional analytics** and risk management
- ✅ **Institutional reporting** system with ASX focus
- ✅ **Database integration** with caching for ASX data
- ✅ **REST API** with Python SDK
- ✅ **ASX sector analysis** capabilities
- ✅ **Input validation system** with comprehensive error handling
- ✅ **Configuration file support** for parameter management

### **Version 1.0.0 (Previous)**
- Basic portfolio optimization
- CSV file support
- Simple visualization
- Command-line interface

---

**Portfolio Optimizer - Production-grade investment tool for hedge funds, brokers, and investment professionals. Optimized for Australian ASX markets with your CSV files working perfectly with all new features!**

*Built with PyPortfolioOpt • Enhanced for institutional use • ASX market focused • Supports both offline and real-time workflows*