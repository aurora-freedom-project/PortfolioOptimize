# Portfolio Optimizer CLI

## 1. Overview

This command-line tool performs advanced portfolio optimization using various financial models. It takes a set of stock tickers, historical price data, and optional user-defined constraints to calculate the optimal asset allocation for different objectives, such as maximizing the Sharpe ratio or applying sophisticated models like Black-Litterman and Hierarchical Risk Parity (HRP).

The tool is designed with a modular architecture, separating data processing, financial modeling, and command-line interaction into distinct components.

## 2. Features

- **Multiple Optimization Methods:**
  - **Mean-Variance Optimization:** Classic Markowitz model to find the portfolio with the maximum Sharpe ratio.
  - **Black-Litterman Model:** Incorporates investor's subjective views on asset performance to produce a more stable and intuitive allocation.
  - **Hierarchical Risk Parity (HRP):** A modern approach that uses graph theory and machine learning to distribute risk based on asset correlations, avoiding the instability issues of quadratic optimizers.
- **Flexible Inputs:**
  - Define custom date ranges for analysis.
  - Provide existing portfolio allocations to compare against the optimal one.
  - Set min/max weight constraints for each asset.
  - Input a custom risk-free rate.
- **Detailed Output:**
  - Comparison between the user's provided portfolio and the calculated optimal portfolio.
  - Key performance metrics: Expected Return, Standard Deviation, Sharpe Ratio, and Sortino Ratio.
  - Asset correlation matrix.
  - A detailed breakdown of the efficient frontier, showing 100 different portfolios and their expected performance.

## 3. Project Structure

The project is organized into three main packages: `core`, `shell`, and the root directory for primary scripts.

```
portfolio_optimizer/
├── core/
│   ├── data.py             # Data loading and preprocessing
│   ├── models.py           # Pydantic models for input validation
│   ├── utils.py            # Financial calculation utilities
│   └── optimization/
│       ├── mean_variance.py  # Mean-Variance optimization logic
│       ├── black_litterman.py# Black-Litterman model logic
│       └── hrp.py            # Hierarchical Risk Parity logic
├── shell/
│   ├── app.py              # Main application workflow coordinator
│   └── cli.py              # CLI argument parsing and output display
├── main.py                 # Main entry point for the application
├── requirements.txt        # Project dependencies
└── merged_stock_prices.csv # Example data file
```

## 4. Installation

1.  **Clone the repository** (if applicable).

2.  **Set up a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 5. Usage

The tool is run from the command line using `python main.py`.

### Basic Example (Mean-Variance)

This command runs a Mean-Variance optimization for 5 Australian stocks from 2018 to 2021.

```bash
python main.py \
  --data merged_stock_prices.csv \
  --method mean_variance \
  --start-date 2018-01-01 \
  --end-date 2021-01-01 \
  --tickers ANZ.AX,CBA.AX,NAB.AX,RIO.AX,WOW.AX \
  --risk-free-rate 0.02
```

### Arguments

- `--data`: Path to the CSV file containing stock prices. The file must have a `date` column and columns for each stock ticker.
- `--method`: The optimization model to use. Choices: `mean_variance`, `black_litterman`, `hrp`.
- `--start-date`/`--end-date`: The date range for historical data analysis (format: `YYYY-MM-DD`).
- `--tickers`: A comma-separated list of stock tickers to include in the portfolio.
- `--allocations` (Optional): Your current portfolio weights. Format: `TICKER1:WEIGHT1,TICKER2:WEIGHT2`.
- `--constraints` (Optional): Min/max weight limits for tickers. Format: `TICKER1:MIN:MAX,TICKER2:MIN:MAX`.
- `--risk-free-rate` (Optional): The annual risk-free rate. Default is `0.02`.
- `--views` (Optional, for Black-Litterman): Your subjective views on asset returns. Format: `TICKER1:RETURN:CONFIDENCE,...`.
- `--output` (Optional): Path to a JSON file to save the full results.

## 6. Core Components Deep Dive

### a. Data Handling (`core/data.py`)

- **`load_stock_data`**: Reads the CSV file into a pandas DataFrame, converts the `date` column to datetime objects, and sets it as the index.
- **`calculate_returns_and_covariance`**: Uses the `pypfopt` library to calculate the annualized expected returns (`mu`) and the sample covariance matrix (`S`) from the price data. It includes a check to ensure the covariance matrix is positive semi-definite, a requirement for many optimization algorithms.

### b. Input Validation (`core/models.py`)

- **`PortfolioModel`**: A Pydantic model that defines the structure and validation rules for all inputs.
- **Validators**:
  - `@field_validator`: Ensures individual fields are correct (e.g., no duplicate tickers, allocation weights sum to 1).
  - `@model_validator`: Performs cross-field validation (e.g., start date is before end date, allocation for a ticker is within its specified constraints).
  - This rigorous validation prevents errors during the optimization process.

### c. Optimization Models (`core/optimization/`)

- **`mean_variance.py`**:
  - Implements the classic Markowitz model.
  - **`optimize_max_sharpe`**: Finds the portfolio weights that maximize the Sharpe ratio.
  - **`generate_efficient_frontier`**: Calculates a series of optimal portfolios for a range of target returns to plot the efficient frontier. It now generates 100 portfolios and ensures the max Sharpe portfolio is included.

- **`black_litterman.py`**:
  - Calculates market-implied prior returns based on a benchmark (e.g., an equally weighted portfolio).
  - **`apply_investor_views`**: Adjusts these prior returns based on the user's subjective views, creating a blended set of posterior returns.
  - Uses the posterior returns as the input for a Mean-Variance optimization, resulting in a more stable and customized portfolio.

- **`hrp.py`**:
  - Does not require a covariance matrix inversion, making it robust to noisy or collinear data.
  - **Process**:
    1.  **Tree Clustering**: Groups assets based on their correlation.
    2.  **Quasi-Diagonalization**: Reorders the covariance matrix based on the clusters.
    3.  **Recursive Bisection**: Distributes weights top-down based on the inverse of the variance within each cluster.

### d. Utility Functions (`core/utils.py`)

- **`calculate_portfolio_metrics`**: A central function to compute expected return, volatility, and Sharpe ratio for any given set of weights.
- **`calculate_sortino_ratio`**: Calculates the Sortino ratio, which is a modification of the Sharpe ratio that only penalizes for downside volatility (returns below the risk-free rate), providing a more realistic measure of risk for many investors. The implementation has been corrected to properly use the `daily_risk_free` rate.

## 7. Dependencies

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **pydantic**: For data validation and settings management.
- **PyPortfolioOpt (pypfopt)**: The core library providing the financial modeling algorithms.
