# Portfolio Optimizer

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_minimal.txt
```

### 2. Run Example
```bash
python run_example.py
```

### 3. View Results
- **Charts**: Open HTML files in `charts/` folder in your browser
- **Data**: Check `streamlined_results.json` for raw data
- **Success**: You should see "✅ Optimization completed successfully!"

## Files

- `streamlined_optimizer.py` - Main optimizer (691 lines)
- `run_example.py` - Comprehensive test suite (210 lines)
- `migrate_from_legacy.py` - Migration tool for upgrading (805 lines)
- `data/market_caps.csv` - Market cap data (provided)
- `data/merged_stock_prices.csv` - Stock price data

## Generated Charts

1. **Efficient Frontier** (`charts/efficient_frontier.html`)
   - Risk vs Return with Sharpe ratio colors
   - Max Sharpe portfolio highlighted

2. **Mean-Variance Comparison** (`charts/mv_comparison.html`)
   - Provided vs Optimal portfolio pie charts

3. **Black-Litterman Comparison** (`charts/bl_comparison.html`)
   - Prior vs Posterior vs Views bar chart

## Usage

```python
from streamlined_optimizer import create_optimizer

portfolio_data = {
    "tickers": ["ANZ.AX", "CBA.AX", "RIO.AX"],
    "allocations": {"ANZ.AX": 0.3, "CBA.AX": 0.4, "RIO.AX": 0.3},
    "investor_views": {
        "CBA.AX": {"expected_return": 0.10, "confidence": 0.8}
    },
    "start_date": "2019-01-01",
    "end_date": "2024-12-31"
}

optimizer = create_optimizer()
result = optimizer.optimize_portfolio(portfolio_data)
```

## Output JSON Structure

```json
{
  "mean_variance": {
    "optimal_weights": {...},
    "efficient_frontier": [...]
  },
  "black_litterman": {
    "market_implied_returns": {...},
    "posterior_returns": {...},
    "view_details": {...}
  },
  "charts": {...}
}
```

## Requirements

- Python 3.8+
- PyPortfolioOpt>=1.5.4
- pandas>=1.3.0
- numpy>=1.21.0
- plotly>=5.0.0
- pydantic>=2.0.0

## Test Coverage

All functionality is tested with 14 comprehensive test scenarios:
- ✅ 6 Happy path scenarios (valid inputs)
- ✅ 8 Error handling scenarios (validation & constraints)

Run tests with: `python run_example.py`

## Date Range Requirements

- **Mean-Variance**: Minimum 3 years of data
- **Black-Litterman**: Minimum 5 years of data (when using investor views)

## Migration Tool

Use `migrate_from_legacy.py` to migrate from older portfolio optimizer versions.