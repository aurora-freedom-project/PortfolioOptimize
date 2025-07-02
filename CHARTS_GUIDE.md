# Portfolio Optimization Charts Guide

This guide explains how to use the new `--show-charts` functionality in the Portfolio Optimization CLI.

## Overview

The portfolio optimization tool now includes comprehensive visualization capabilities using matplotlib and seaborn. You can generate professional-looking charts to analyze your portfolio optimization results.

## Chart Types Available

### 1. Efficient Frontier Chart
- **Purpose**: Visualize the risk-return tradeoff for different portfolio allocations
- **Features**: 
  - Scatter plot of portfolios colored by Sharpe ratio
  - Highlights the optimal (Max Sharpe) portfolio
  - Shows your current portfolio position
  - Interactive color scale for Sharpe ratios

### 2. Portfolio Allocation Charts
- **Purpose**: Compare current vs optimal portfolio weights
- **Features**: 
  - Side-by-side pie charts
  - Clear percentage labels for each asset
  - Professional color schemes

### 3. Correlation Matrix Heatmap
- **Purpose**: Visualize asset correlations
- **Features**: 
  - Color-coded correlation matrix
  - Numerical correlation values
  - Triangular heatmap to avoid redundancy

### 4. Risk-Return Metrics Comparison
- **Purpose**: Compare key performance metrics
- **Features**: 
  - Bar charts for Expected Return, Risk, Sharpe Ratio, Sortino Ratio
  - Side-by-side comparison of current vs optimal portfolios
  - Clear numerical labels

## Usage Methods

### Method 1: Generate Charts During Optimization

Run your portfolio optimization with the `--show-charts` flag:

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method black_litterman \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --tickers "ANZ.AX,CBA.AX,MQG.AX,NAB.AX,RIO.AX" \
  --allocations "ANZ.AX:0.2,CBA.AX:0.2,MQG.AX:0.2,NAB.AX:0.2,RIO.AX:0.2" \
  --output results.json \
  --show-charts
```

This will:
1. Run the optimization
2. Display results in the terminal
3. Save results to `results.json`
4. Generate and display all charts
5. Save chart images to `results_charts/` directory

### Method 2: Generate Charts from Existing JSON Results

If you already have optimization results saved in a JSON file:

```bash
python main.py --charts-from-json bl_results.json
```

This will:
1. Load results from `bl_results.json`
2. Generate all charts
3. Save charts to `bl_results_charts/` directory

### Method 3: Use the Standalone Chart Generator

For more control over chart generation:

```bash
python chart_generator.py --input bl_results.json --output-dir my_charts --chart-type all
```

Options for `--chart-type`:
- `all`: Generate all chart types (default)
- `frontier`: Only efficient frontier chart
- `weights`: Only portfolio allocation charts
- `correlation`: Only correlation matrix
- `metrics`: Only risk-return metrics chart

## Chart Output

### Display Behavior
- Charts are displayed interactively using matplotlib
- On macOS/Linux with GUI: Charts open in separate windows
- You can zoom, pan, and save charts manually from the matplotlib interface

### Saved Files
Charts are automatically saved as high-resolution PNG files (300 DPI) with the following names:
- `efficient_frontier.png`
- `portfolio_weights.png` 
- `correlation_matrix.png`
- `risk_return_metrics.png`

## Requirements

The charting functionality requires additional Python packages:

```bash
pip install matplotlib seaborn
```

These are included in the updated `requirements.txt`.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'seaborn'**
   ```bash
   pip install matplotlib seaborn
   ```

2. **Charts not displaying on headless systems**
   - Charts will still be saved as PNG files
   - Set `MPLBACKEND=Agg` environment variable for headless operation

3. **Memory issues with large datasets**
   - The visualization module is optimized for typical portfolio sizes (5-50 assets)
   - For very large portfolios, consider using `--chart-type` to generate specific charts

### Performance Tips

- Use `--chart-type` to generate only needed charts for faster execution
- Charts are generated after optimization completes, so they don't affect optimization performance
- PNG files are saved at high resolution but are typically small (< 1MB each)

## Integration with Existing Workflows

### Batch Processing
Generate charts for multiple result files:

```bash
for file in *.json; do
  python main.py --charts-from-json "$file"
done
```

### Custom Chart Directories
Organize charts by date or strategy:

```bash
python chart_generator.py \
  --input results_2024.json \
  --output-dir "charts/2024_analysis" \
  --chart-type all
```

## Chart Customization

The visualization module uses professional defaults, but you can modify:

- Colors: Edit the `colors` property in `PortfolioVisualizer` class
- Figure sizes: Modify `figsize` parameter
- Styles: Change seaborn style in the `__init__` method
- Chart types: Add new plotting methods to the `PortfolioVisualizer` class

## Examples

### Complete Workflow Example
```bash
# 1. Run optimization with charts
python main.py \
  --data merged_stock_prices.csv \
  --method mean_variance \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --tickers "AAPL,GOOGL,MSFT,TSLA,NVDA" \
  --allocations "AAPL:0.2,GOOGL:0.2,MSFT:0.2,TSLA:0.2,NVDA:0.2" \
  --output tech_portfolio.json \
  --show-charts

# 2. Later, regenerate just the efficient frontier chart
python chart_generator.py \
  --input tech_portfolio.json \
  --chart-type frontier \
  --output-dir presentation_charts
```

This comprehensive charting system provides professional-quality visualizations for portfolio analysis, making your optimization results easy to understand and present.