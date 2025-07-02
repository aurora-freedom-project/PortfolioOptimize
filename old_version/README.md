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
- **Data**: Check `optimization_results.json` for raw data
- **Success**: You should see "✅ Optimization completed successfully!"

## Files

- `streamlined_optimizer.py` - Main optimizer
- `run_example.py` - Example script
- `market_caps.csv` - Market cap data (provided)
- `merged_stock_prices.csv` - Your price data

## Generated Charts

1. **Efficient Frontier** (`charts/efficient_frontier.html`)
   - Risk vs Return with Sharpe ratio colors
   - Max Sharpe portfolio highlighted

2. **Mean-Variance Comparison** (`charts/mv_comparison.html`)
   - Provided vs Optimal portfolio pie charts

3. **Black-Litterman Comparison** (`charts/bl_comparison.html`)
   - Prior vs Posterior vs Views bar chart

4. **HRP Correlation** (`charts/hrp_correlation.html`)
   - Correlation heatmap for clustering

## Usage

```python
from streamlined_optimizer import create_optimizer

portfolio_data = {
    "tickers": ["ANZ.AX", "CBA.AX", "RIO.AX"],
    "allocations": {"ANZ.AX": 0.3, "CBA.AX": 0.4, "RIO.AX": 0.3},
    "investor_views": {
        "CBA.AX": {"expected_return": 0.10, "confidence": 0.8}
    },
    "start_date": "2022-01-01",
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
  "hrp": {
    "optimal_weights": {...},
    "correlation_matrix": {...}
  },
  "charts": {...}
}
```

## Requirements

- Python 3.7+
- PyPortfolioOpt
- pandas
- numpy
- plotly