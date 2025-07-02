# Portfolio Optimization CLI with Advanced Visualization

## 📊 Overview

This command-line tool performs advanced portfolio optimization using various financial models with comprehensive interactive visualization capabilities. It takes a set of stock tickers, historical price data, and optional user-defined constraints to calculate optimal asset allocation while generating professional charts for analysis.

The tool is designed with a modular architecture, separating data processing, financial modeling, command-line interaction, and visualization into distinct components.

## ✨ Key Features

### 🎯 **Multiple Optimization Methods**
- **Mean-Variance Optimization:** Classic Markowitz model to find the portfolio with maximum Sharpe ratio
- **Black-Litterman Model:** Incorporates investor's subjective views on asset performance for stable allocation
- **Hierarchical Risk Parity (HRP):** Modern approach using graph theory and machine learning for risk distribution

### 📈 **Interactive Chart Generation**
- **Efficient Frontier Plots:** Comprehensive envelope visualization with 100+ portfolios
- **Portfolio Allocation Charts:** Side-by-side pie charts comparing current vs optimal weights
- **Correlation Matrix Heatmaps:** Professional visualization of asset correlations
- **Risk-Return Metrics:** Bar charts comparing key performance indicators
- **Method-specific charts:** Specialized visualizations for each optimization method

### 🔧 **Flexible Analysis Options**
- Custom date ranges for historical analysis
- Portfolio constraint settings (min/max weights per asset)
- Investor views integration (Black-Litterman)
- Multiple output formats (JSON, PNG charts)
- Standalone chart generation from existing results

## 🏗️ Project Structure

```
portfolio_optimizer/
├── core/                          # Core business logic
│   ├── __init__.py               # Package exports
│   ├── data.py                   # Data loading and preprocessing
│   ├── models.py                 # Pydantic models for validation
│   ├── utils.py                  # Financial calculation utilities
│   └── optimization/             # Optimization algorithms
│       ├── __init__.py          # Optimization exports
│       ├── mean_variance.py     # Mean-Variance optimization
│       ├── black_litterman.py   # Black-Litterman model
│       └── hrp.py               # Hierarchical Risk Parity
├── shell/                        # CLI interface
│   ├── __init__.py              # Shell exports  
│   ├── app.py                   # Application workflow coordinator
│   ├── cli.py                   # Command line interface
│   └── display/                 # Visualization components
│       ├── __init__.py         # Display exports
│       └── visualization.py    # Chart generation
├── main.py                      # Main CLI entry point
├── chart_generator.py           # Standalone chart generator
├── requirements.txt             # Dependencies
├── CHARTS_GUIDE.md             # Comprehensive visualization guide
└── README.md                   # This file
```

## 🚀 Installation

1. **Clone or download the project**

2. **Set up Python virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Usage

The tool comes with sample Australian stock data (`merged_stock_prices.csv`) containing ASX-listed companies from 2020-2025:

- **ANZ.AX** - Australia and New Zealand Banking Group (Major Bank)
- **CBA.AX** - Commonwealth Bank of Australia (Major Bank) 
- **MQG.AX** - Macquarie Group (Investment Bank)
- **NAB.AX** - National Australia Bank (Major Bank)
- **RIO.AX** - Rio Tinto (Mining/Resources)
- **WOW.AX** - Woolworths Group (Retail/Consumer)

Here are diverse usage examples showcasing different sectors and analysis approaches:

### 1. Basic Mean-Variance Optimization (Financial Sector Focus)

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method mean_variance \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --tickers "ANZ.AX,CBA.AX,MQG.AX,NAB.AX" \
  --allocations "ANZ.AX:0.25,CBA.AX:0.25,MQG.AX:0.25,NAB.AX:0.25" \
  --output financial_portfolio.json
```

### 2. Black-Litterman with Investor Views (Mixed Sector Portfolio)

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method black_litterman \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --tickers "CBA.AX,RIO.AX,WOW.AX" \
  --allocations "CBA.AX:0.4,RIO.AX:0.3,WOW.AX:0.3" \
  --views "CBA.AX:0.12:0.8,RIO.AX:0.15:0.6,WOW.AX:0.08:0.7" \
  --risk-free-rate 0.035 \
  --output mixed_sector.json \
  --show-charts
```

### 3. HRP Optimization with Constraints (Full Diversification)

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method hrp \
  --start-date 2021-06-01 \
  --end-date 2024-06-01 \
  --tickers "ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX" \
  --allocations "ANZ.AX:0.16,CBA.AX:0.17,MQG.AX:0.17,NAB.AX:0.16,RIO.AX:0.17,WOW.AX:0.17" \
  --constraints "RIO.AX:0.1:0.3,MQG.AX:0.05:0.25" \
  --output diversified_hrp.json \
  --show-charts
```

### 4. COVID Recovery Analysis (2020-2022)

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method mean_variance \
  --start-date 2020-03-01 \
  --end-date 2022-03-01 \
  --tickers "ANZ.AX,CBA.AX,RIO.AX,WOW.AX" \
  --allocations "ANZ.AX:0.2,CBA.AX:0.3,RIO.AX:0.25,WOW.AX:0.25" \
  --risk-free-rate 0.01 \
  --output covid_recovery.json \
  --show-charts
```

### 5. Generate Charts from Existing Results

```bash
# From main CLI
python main.py --charts-from-json financial_portfolio.json

# Using standalone chart generator for specific charts
python chart_generator.py --input mixed_sector.json --chart-type frontier
python chart_generator.py --input diversified_hrp.json --chart-type all
```

## 🎛️ Command Line Arguments

### Required Arguments
- `--tickers`: Comma-separated stock tickers (e.g., "AAPL,GOOGL,MSFT")
- `--start-date`: Analysis start date (YYYY-MM-DD format)
- `--end-date`: Analysis end date (YYYY-MM-DD format)

### Optional Arguments
- `--data`: CSV file path (default: merged_stock_prices.csv)
- `--method`: Optimization method (mean_variance|black_litterman|hrp, default: mean_variance)
- `--allocations`: Current portfolio weights (TICKER:WEIGHT,...)
- `--constraints`: Weight constraints (TICKER:MIN:MAX,...)
- `--risk-free-rate`: Annual risk-free rate (default: 0.02)
- `--views`: Investor views for Black-Litterman (TICKER:RETURN:CONFIDENCE,...)
- `--output`: JSON output file path
- `--show-charts`: Generate and display charts
- `--charts-from-json`: Generate charts from existing JSON results

## 📊 Optimization Methods Deep Dive

### 1. Mean-Variance Optimization
- **Objective:** Maximize Sharpe ratio or minimize volatility
- **Output:** Efficient frontier with 100+ portfolios, optimal weights
- **Best for:** Traditional portfolio optimization with clear risk-return tradeoffs

### 2. Black-Litterman Model
- **Objective:** Incorporate market views with personal insights
- **Features:** Market-implied returns, investor views integration, posterior return estimates
- **Output:** Adjusted efficient frontier, view impact analysis
- **Best for:** Investors with specific market opinions or forecasts

### 3. Hierarchical Risk Parity (HRP)
- **Objective:** Risk-based allocation using asset clustering
- **Features:** No covariance matrix inversion, robust to estimation errors
- **Output:** Cluster-based weights, hierarchical structure analysis  
- **Best for:** Risk parity strategies, avoiding estimation errors

## 📈 Chart Types and Features

### Efficient Frontier Chart
- **Envelope visualization** with 100+ portfolio points
- **Smooth curve fitting** for proper frontier display
- **Color-coded Sharpe ratios** for performance visualization
- **Special portfolio highlighting** (max Sharpe, min volatility)
- **Current portfolio positioning** for comparison

### Portfolio Allocation Charts
- **Side-by-side pie charts** comparing current vs optimal weights
- **Professional color schemes** with clear percentage labels
- **Automatic legend generation** for easy interpretation

### Correlation Matrix Heatmap
- **Triangular heatmap design** to avoid redundancy
- **Color-coded correlation values** with numerical labels
- **Professional styling** using seaborn visualization

### Risk-Return Metrics
- **Bar chart comparisons** of key performance metrics
- **Expected return, volatility, Sharpe and Sortino ratios**
- **Side-by-side portfolio analysis** with numerical labels

## 🎨 Advanced Chart Features

### Method-Specific Visualizations
- **Black-Litterman:** Prior vs posterior returns analysis
- **HRP:** Hierarchical clustering dendrogram (future enhancement)
- **Mean-Variance:** Traditional efficient frontier with optimal points

### Chart Customization
- **High-resolution output** (300 DPI PNG files)
- **Professional styling** with consistent color schemes
- **Interactive display** with zoom and pan capabilities
- **Automatic chart saving** with organized directory structure

## 📁 Output Files

### JSON Results Structure
```json
{
  "provided_portfolio": {
    "weights": {...},
    "metrics": {...}
  },
  "optimal_portfolio": {
    "weights": {...}, 
    "metrics": {...}
  },
  "efficient_frontier": [...],
  "correlation_matrix": {...},
  "method": "OPTIMIZATION_METHOD"
}
```

### Chart Files
- `efficient_frontier.png` - Efficient frontier visualization
- `portfolio_weights.png` - Portfolio allocation comparison
- `correlation_matrix.png` - Asset correlation heatmap
- `risk_return_metrics.png` - Performance metrics comparison

## 🧮 Example Workflows

### Complete Analysis Workflow (Big 4 Banks)
```bash
# 1. Run comprehensive optimization with all features
python main.py \
  --data merged_stock_prices.csv \
  --method mean_variance \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --tickers "ANZ.AX,CBA.AX,MQG.AX,NAB.AX,WOW.AX" \
  --allocations "ANZ.AX:0.2,CBA.AX:0.25,MQG.AX:0.2,NAB.AX:0.2,WOW.AX:0.15" \
  --constraints "WOW.AX:0.1:0.25,MQG.AX:0.1:0.3" \
  --risk-free-rate 0.04 \
  --output asx_portfolio.json \
  --show-charts

# 2. Generate specific chart types for presentation
python chart_generator.py \
  --input asx_portfolio.json \
  --chart-type frontier \
  --output-dir presentation_charts
```

### Black-Litterman with Market Views (Resource vs Banking)
```bash
python main.py \
  --data merged_stock_prices.csv \
  --method black_litterman \
  --start-date 2023-06-01 \
  --end-date 2024-12-31 \
  --tickers "CBA.AX,RIO.AX,MQG.AX" \
  --allocations "CBA.AX:0.4,RIO.AX:0.35,MQG.AX:0.25" \
  --views "CBA.AX:0.10:0.8,RIO.AX:0.18:0.6,MQG.AX:0.12:0.7" \
  --risk-free-rate 0.045 \
  --output sector_rotation.json \
  --show-charts
```

### Risk Parity Analysis (All Sectors)
```bash
python main.py \
  --data merged_stock_prices.csv \
  --method hrp \
  --start-date 2021-01-01 \
  --end-date 2024-06-30 \
  --tickers "ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX,WOW.AX" \
  --allocations "ANZ.AX:0.16,CBA.AX:0.17,MQG.AX:0.17,NAB.AX:0.16,RIO.AX:0.17,WOW.AX:0.17" \
  --output full_diversification.json \
  --show-charts
```

## 📦 Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing operations
- **PyPortfolioOpt** - Portfolio optimization algorithms
- **pydantic** - Data validation and settings management

### Visualization Libraries  
- **matplotlib** - Core plotting functionality
- **seaborn** - Statistical data visualization

### Development Libraries
- **pytest** - Testing framework
- **pytest-mock** - Mocking for tests
- **pytest-cov** - Coverage reporting

## 🔧 Technical Notes

### Performance Optimization
- **Efficient frontier generation:** 100+ portfolios for smooth curves
- **Duplicate portfolio removal:** Automatic cleanup for clean visualizations
- **Memory-efficient processing:** Optimized for typical portfolio sizes (5-50 assets)

### Error Handling
- **Graceful degradation:** Charts will skip if visualization packages missing
- **Validation checks:** Comprehensive input validation before processing
- **Fallback mechanisms:** Alternative strategies when optimization fails

### Platform Compatibility
- **Cross-platform:** Works on Windows, macOS, and Linux
- **Headless support:** Chart generation works without GUI (saves files only)
- **Virtual environment friendly:** Clean dependency management

## 📚 Additional Resources

- **CHARTS_GUIDE.md** - Comprehensive visualization documentation
- **Requirements.txt** - Complete dependency specifications
- **Example data files** - Sample stock price data for testing

For detailed chart usage instructions, see [CHARTS_GUIDE.md](CHARTS_GUIDE.md).

---

**Portfolio Optimization CLI** - Professional portfolio analysis with advanced visualization capabilities.