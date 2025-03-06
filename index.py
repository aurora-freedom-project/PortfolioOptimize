from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import efficient_frontier, risk_models, expected_returns, black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from functools import partial, reduce
import copy
import re

class OptimizationMethod:
    """Enumeration of available optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"


class PortfolioModel(BaseModel):
    """Unified model for portfolio optimization inputs."""
    
    tickers: List[str] = Field(
        description="List of stock tickers in the portfolio")
    
    allocations: Dict[str, float] = Field(
        default_factory=dict,
        description="Weight allocation for each ticker (total = 1)")
    
    constraints: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Minimum and maximum weight limits for each ticker")
    
    start_date: Union[int, str] = Field(
        description="Analysis start date (Unix timestamp or ISO string)")
    
    end_date: Optional[Union[int, str]] = Field(
        default=None,
        description="Analysis end date (Unix timestamp or ISO string)")

    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate of return (e.g., 0.02 for 2%)")
    
    optimization_method: str = Field(
        default=OptimizationMethod.MEAN_VARIANCE,
        description="Optimization method to use")
    
    investor_views: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Investor views for Black-Litterman model")
    
    # Field validators from ScenarioModel
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Check that the ticker list is not empty and has no duplicates."""
        if not v:
            raise ValueError("Ticker list cannot be empty")
        
        if len(v) != len(set(v)):
            raise ValueError("Ticker list contains duplicate values")
        
        return v
    
    @field_validator('allocations')
    @classmethod
    def validate_allocations(cls, v, info):
        """Validate allocation weights."""
        tickers = info.data.get('tickers', [])
        
        for ticker in v.keys():
            if ticker not in tickers:
                raise ValueError(f"Ticker '{ticker}' in allocations is not in the tickers list")
        
        if v and sum(v.values()) > 1.0001:  
            raise ValueError(f"Total allocation weight ({sum(v.values())}) exceeds 1")
        
        for ticker, allocation in v.items():
            if allocation <= 0 or allocation > 1:
                raise ValueError(f"Weight for '{ticker}' ({allocation}) must be > 0 and <= 1")
        
        return v
    
    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v, info):
        """Validate weight constraints."""
        tickers = info.data.get('tickers', [])
        
        for ticker in v.keys():
            if ticker not in tickers:
                raise ValueError(f"Ticker '{ticker}' in constraints is not in the tickers list")
        
        for ticker, (min_val, max_val) in v.items():
            if min_val < 0 or min_val > 1:
                raise ValueError(f"Minimum limit for '{ticker}' ({min_val}) must be in range [0, 1]")
            
            if max_val < 0 or max_val > 1:
                raise ValueError(f"Maximum limit for '{ticker}' ({max_val}) must be in range [0, 1]")
            
            if min_val > max_val:
                raise ValueError(f"Minimum limit ({min_val}) for '{ticker}' is greater than maximum limit ({max_val})")
        
        return v
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return None
            
        if isinstance(v, str) and v.isdigit():
            v = int(v)
            
        if isinstance(v, int):
            try:
                return datetime.fromtimestamp(v)
            except ValueError as e:
                raise ValueError(f"Timestamp not valid: {e}")
                
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                try:
                    return datetime.strptime(v, "%Y-%m-%d")
                except ValueError as e:
                    raise ValueError(f"Day format not valid: {e}")
        
        return v

    @model_validator(mode='after')
    def validate_dates_order(self):
        """Validate date order and minimum time period based on optimization method."""
        if self.start_date is None:
            raise ValueError("Start date is required and cannot be empty")
                
        if self.end_date is None:
            raise ValueError("End date is required and cannot be empty")
                
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        
        if isinstance(self.start_date, datetime) and isinstance(self.end_date, datetime):
            # Xác định số năm tối thiểu dựa trên phương pháp tối ưu hóa
            if self.optimization_method == OptimizationMethod.MEAN_VARIANCE:
                min_required_years = 3
                method_name = "Mean-Variance"
            elif self.optimization_method == OptimizationMethod.BLACK_LITTERMAN:
                min_required_years = 5
                method_name = "Black-Litterman"
            else:
                min_required_years = 3  # Mặc định là 3 năm
                method_name = "Default"
            
            # Tính toán thời điểm kết thúc tối thiểu yêu cầu
            min_end_date_year = self.start_date.year + min_required_years
            
            try:
                min_end_date = datetime(
                    min_end_date_year,
                    self.start_date.month,
                    self.start_date.day,
                    self.start_date.hour,
                    self.start_date.minute,
                    self.start_date.second
                )
            except ValueError:
                if self.start_date.month == 2 and self.start_date.day == 29:
                    min_end_date = datetime(
                        min_end_date_year,
                        2,
                        28,
                        self.start_date.hour,
                        self.start_date.minute,
                        self.start_date.second
                    )
                else:
                    if self.start_date.month == 12:
                        next_month = datetime(min_end_date_year + 1, 1, 1)
                    else:
                        next_month = datetime(min_end_date_year, self.start_date.month + 1, 1)
                    
                    last_day = next_month - timedelta(days=1)
                    min_end_date = datetime(
                        last_day.year,
                        last_day.month,
                        last_day.day,
                        self.start_date.hour,
                        self.start_date.minute,
                        self.start_date.second
                    )
            
            # Kiểm tra chỉ điều kiện tối thiểu, không yêu cầu khoảng thời gian chính xác
            if self.end_date < min_end_date:
                days_short = (min_end_date - self.end_date).days
                raise ValueError(
                    f"Time period for {method_name} optimization must be at least {min_required_years} years. "
                    f"Currently missing {days_short} days to reach {min_required_years} years."
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_allocations_with_constraints(self):
        """Check allocations comply with constraints."""
        for ticker, allocation in self.allocations.items():
            if ticker in self.constraints:
                min_val, max_val = self.constraints[ticker]
                if allocation < min_val:
                    raise ValueError(f"Weight for '{ticker}' ({allocation}) is less than minimum limit ({min_val})")
                if allocation > max_val:
                    raise ValueError(f"Weight for '{ticker}' ({allocation}) is greater than maximum limit ({max_val})")
        return self
    
    def get_remaining_allocation(self) -> float:
        """Calculate remaining unallocated weight."""
        return 1.0 - sum(self.allocations.values())
    
    def get_remaining_tickers(self) -> List[str]:
        """Get list of tickers without allocations."""
        return [ticker for ticker in self.tickers if ticker not in self.allocations]
    
    def is_fully_allocated(self) -> bool:
        """Check if all tickers have allocations."""
        return len(self.allocations) == len(self.tickers)        
    

    def to_dict(self) -> Dict:
        """Convert model to dictionary format."""
        # Ensure allocations are rounded
        allocations = {ticker: round(alloc, 3) for ticker, alloc in self.allocations.items()}
        
        # Convert constraints to appropriate format
        adjusted_constraints = {}
        for ticker, (min_val, max_val) in self.constraints.items():
            alloc = allocations.get(ticker, 0)
            adjusted_min = min(round(min_val, 2), round(alloc, 2))
            adjusted_max = max(round(max_val, 2), round(alloc, 2))
            adjusted_constraints[ticker] = (adjusted_min, adjusted_max)
        
        result = {
            "tickers": self.tickers,
            "allocations": allocations,
            "constraints": adjusted_constraints,
            "risk_free_rate": self.risk_free_rate,
            "optimization_method": self.optimization_method
        }
        
        # Handle dates
        if isinstance(self.start_date, datetime):
            result["start_date"] = str(int(self.start_date.timestamp()))
        else:
            result["start_date"] = self.start_date
            
        if self.end_date is None:
            result["end_date"] = None
        elif isinstance(self.end_date, datetime):
            result["end_date"] = str(int(self.end_date.timestamp()))
        else:
            result["end_date"] = self.end_date
        
        # Include investor views if present
        if self.investor_views:
            result["investor_views"] = self.investor_views
            
        return result


def auto_allocate(tickers: List[str], allocations: Dict[str, float], 
                  constraints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Automatically allocate weights to unallocated tickers.
    
    Args:
        tickers: List of stock tickers
        allocations: Current allocations
        constraints: Weight constraints
        
    Returns:
        Updated allocations
    """
    # Calculate remaining allocation
    allocated_sum = sum(allocations.values())
    remaining_allocation = 1.0 - allocated_sum
    
    # Get unallocated tickers
    remaining_tickers = [ticker for ticker in tickers if ticker not in allocations]
    
    # Return current allocations if all tickers are allocated
    if not remaining_tickers:
        return allocations
    
    # Create constraints summary
    constraints_summary = {}
    for ticker in tickers:
        ticker_info = {
            "allocation": allocations.get(ticker, 0),
            "min": 0,
            "max": 1
        }
        
        if ticker in constraints:
            ticker_info["min"], ticker_info["max"] = constraints[ticker]
            
        constraints_summary[ticker] = ticker_info
    
    # Calculate total minimum requirement
    total_min_required = sum(constraints_summary[ticker]["min"] for ticker in remaining_tickers)
    
    # Adjust constraints if total minimum exceeds remaining allocation
    adjusted_constraints = constraints.copy()
    if total_min_required > remaining_allocation:
        ratio = remaining_allocation / total_min_required
        for ticker in remaining_tickers:
            min_val = constraints_summary[ticker]["min"] * ratio
            max_val = constraints_summary[ticker]["max"]
            adjusted_constraints[ticker] = (round(min_val, 2), round(max_val, 2))
        
        # Update constraints summary
        for ticker, (min_val, max_val) in adjusted_constraints.items():
            if ticker in constraints_summary:
                constraints_summary[ticker]["min"] = min_val
                constraints_summary[ticker]["max"] = max_val
        
        # Recalculate total minimum required
        total_min_required = sum(constraints_summary[ticker]["min"] for ticker in remaining_tickers)
    
    # Allocate minimum weights
    updated_allocations = allocations.copy()
    for ticker in remaining_tickers:
        min_val = constraints_summary[ticker]["min"]
        updated_allocations[ticker] = min_val
    
    # Calculate remaining allocation after minimums
    remaining_after_min = remaining_allocation - total_min_required
    
    # Calculate total space for additional allocation
    total_space = sum(
        constraints_summary[ticker]["max"] - constraints_summary[ticker]["min"]
        for ticker in remaining_tickers
    )
    
    # If no space for additional allocation, return allocations with minimums
    if total_space <= 0.0001:
        return ensure_total_is_one(updated_allocations)
    
    # Distribute remaining allocation proportionally
    for ticker in remaining_tickers:
        min_val = constraints_summary[ticker]["min"]
        max_val = constraints_summary[ticker]["max"]
        space = max_val - min_val
        
        ratio = space / total_space
        additional = remaining_after_min * ratio
        
        updated_allocations[ticker] = min(min_val + additional, max_val)
    
    # Ensure total allocation is exactly 1
    return ensure_total_is_one(updated_allocations)


def ensure_total_is_one(allocations: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure allocations sum to exactly 1.0.
    
    Args:
        allocations: Current allocations
        
    Returns:
        Adjusted allocations
    """
    adjusted = {ticker: round(alloc, 4) for ticker, alloc in allocations.items()}
    
    total = sum(adjusted.values())
    difference = 1.0 - total
    
    if abs(difference) > 0.0001:
        sorted_tickers = sorted(adjusted.keys(), key=lambda t: adjusted[t])
        
        if sorted_tickers:
            adjusted[sorted_tickers[-1]] += difference
    
    result = {ticker: round(alloc, 3) for ticker, alloc in adjusted.items()}
    
    total = sum(result.values())
    if abs(total - 1.0) > 0.001 and sorted_tickers:
        result[sorted_tickers[-1]] += round(1.0 - total, 3)
    
    return result


def auto_calculate_constraints(tickers: List[str], allocations: Dict[str, float], 
                               existing_constraints: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Automatically calculate constraints for tickers without constraints.
    
    Args:
        tickers: List of stock tickers
        allocations: Current allocations
        existing_constraints: Existing constraints
        
    Returns:
        Complete constraints
    """
    constraints = existing_constraints.copy()
    
    for ticker in tickers:
        if ticker in constraints:
            continue
            
        if ticker in allocations:
            allocation = allocations[ticker]
            min_val = min(allocation, max(0, allocation - 0.05))
            max_val = max(allocation, min(1, allocation + 0.05))
            constraints[ticker] = (round(min_val, 2), round(max_val, 2))
        else:
            constraints[ticker] = (0.05, 0.4)
            
    return constraints


def convert_timestamp(timestamp: Union[str, int]) -> str:
    """Convert timestamp to YYYY-MM-DD format required by YFinance."""
    try:
        # Process timestamp as string of digits
        if isinstance(timestamp, str) and timestamp.isdigit():
            timestamp = int(timestamp)
        
        # Process timestamp as integer (Unix timestamp)
        if isinstance(timestamp, int):
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d')
        
        # Process timestamp as date string
        elif isinstance(timestamp, str):
            # If already in YYYY-MM-DD format, return as is
            if re.match(r'^\d{4}-\d{2}-\d{2}$', timestamp):
                return timestamp
            
            # Try parse ISO format
            try:
                dt = datetime.fromisoformat(timestamp)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                # Try other common formats
                for fmt in ['%Y%m%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                
                raise ValueError(f"Cannot parse timestamp: {timestamp}")
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
            
    except Exception as e:
        raise ValueError(f"Invalid date format ({timestamp}): {e}")


def parse_portfolio_data(portfolio: Dict) -> Dict:
    """Phân tích dữ liệu danh mục và chuyển đổi timestamp."""
    parsed_data = portfolio.copy()
    
    # Chuyển đổi ngày bắt đầu
    try:
        parsed_data["start_date"] = convert_timestamp(portfolio["start_date"])
    except Exception as e:
        print(f"Lỗi chuyển đổi start_date: {e}")
        parsed_data["start_date"] = portfolio["start_date"]  # Giữ nguyên để xử lý lỗi sau
        
    # Chuyển đổi ngày kết thúc
    try:
        parsed_data["end_date"] = convert_timestamp(portfolio["end_date"])
    except Exception as e:
        print(f"Lỗi chuyển đổi end_date: {e}")
        parsed_data["end_date"] = portfolio["end_date"]  # Giữ nguyên để xử lý lỗi sau
    
    return parsed_data


def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch stock price data and exchange information.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Tuple of price data and ticker information
    """
    try:
        # Convert date formats to YYYY-MM-DD
        try:
            start_date_str = convert_timestamp(start_date)
            end_date_str = convert_timestamp(end_date)
            print(f"Loading data for {tickers} from {start_date_str} to {end_date_str}")
        except Exception as date_error:
            print(f"Error converting date format: {date_error}")
            raise ValueError(f"Invalid date format: {date_error}")
        
        # Load data
        data = yf.download(tickers, start=start_date_str, end=end_date_str)
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"Could not load data for period {start_date_str} to {end_date_str}")
        
        # Get closing price data
        close_data = data['Close']
        
        # Check for missing tickers
        missing_tickers = []
        for ticker in tickers:
            if ticker not in close_data.columns:
                missing_tickers.append(ticker)
            elif close_data[ticker].isna().all():
                missing_tickers.append(ticker)
        
        if missing_tickers:
            print(f"Warning: No data available for tickers: {missing_tickers}")
        
        # Get exchange information for each ticker
        ticker_info = {}
        for ticker in tickers:
            try:
                if ticker in close_data.columns:
                    info = yf.Ticker(ticker).info
                    exchange = info.get('exchange', 'N/A')
                    ticker_info[ticker] = exchange
                else:
                    ticker_info[ticker] = 'N/A'
            except Exception as e:
                print(f"Unable to get information for {ticker}: {e}")
                ticker_info[ticker] = 'N/A'
        
        # Handle missing data if any
        if close_data.isna().any().any():
            close_data = close_data.ffill().bfill()
            print("Filled missing values in data")
        
        return close_data, ticker_info
        
    except Exception as e:
        print(f"Error loading stock data: {e}")
        # Return empty DataFrame and default information
        empty_df = pd.DataFrame(columns=tickers)
        ticker_info = {ticker: 'N/A' for ticker in tickers}
        return empty_df, ticker_info


def calculate_returns_and_covariance(price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Calculate expected returns and covariance matrix."""
    # Check for empty data
    if price_data.empty:
        raise ValueError("Cannot calculate on empty data")
    
    # Calculate expected returns
    mu = expected_returns.mean_historical_return(price_data)
    
    # Replace NaN values if any
    mu = mu.fillna(0.02)  # Replace with 2%
    
    # Calculate covariance matrix
    S = risk_models.sample_cov(price_data)
    
    # Force covariance matrix to be symmetric
    S = 0.5 * (S + S.T)
    
    # Check if covariance matrix is positive definite
    try:
        # Try Cholesky decomposition (only succeeds if matrix is positive definite)
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # If not positive definite, add a small amount to the main diagonal
        print("Warning: Covariance matrix is not positive definite. Making adjustments.")
        min_eig = np.min(np.linalg.eigvals(S))
        if min_eig < 0:
            S = S + (-min_eig + 1e-5) * np.eye(len(S))
    
    return mu, S


def calculate_portfolio_metrics(
    weights: np.ndarray, 
    mu: pd.Series, 
    S: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate portfolio performance metrics including Sortino ratio."""
    portfolio_return = weights @ mu
    portfolio_volatility = np.sqrt(weights @ S @ weights)
    
    sharpe_ratio = 0.0
    if portfolio_volatility > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Calculate Sortino ratio
    # We need to calculate downside deviation for Sortino ratio
    # This requires daily returns, so we'll compute it separately in the analysis functions
    
    return {
        "expected_return": float(portfolio_return),
        "standard_deviation": float(portfolio_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "risk_free_rate": float(risk_free_rate)
    }


def calculate_sortino_ratio(
    weights: np.ndarray,
    stock_data: pd.DataFrame,
    risk_free_rate: float = 0.02,
    target_return: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Calculate the Sortino ratio for a portfolio.
    
    Args:
        weights: Portfolio weights
        stock_data: Daily price data
        risk_free_rate: Risk-free rate (annualized)
        target_return: Minimum acceptable return (default: 0)
        annualization_factor: Number of trading days in a year
        
    Returns:
        Sortino ratio
    """
    # Convert weights array to dictionary for easier handling
    if isinstance(weights, np.ndarray):
        weights_dict = {stock_data.columns[i]: weights[i] for i in range(len(weights))}
    else:
        weights_dict = weights
    
    # Calculate daily portfolio returns
    daily_returns = stock_data.pct_change().dropna()
    portfolio_returns = pd.Series(0, index=daily_returns.index)
    
    for ticker, weight in weights_dict.items():
        if ticker in daily_returns.columns:
            portfolio_returns += daily_returns[ticker] * weight
    
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    daily_target = (1 + target_return) ** (1 / annualization_factor) - 1
    
    # Calculate excess returns over target
    excess_returns = portfolio_returns - daily_target
    
    # Calculate downside returns (returns below target)
    downside_returns = excess_returns[excess_returns < 0]
    
    # Calculate downside deviation (annualized)
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2) * annualization_factor)
    else:
        # If no downside returns, use a small value to avoid division by zero
        downside_deviation = 1e-6
    
    # Calculate expected annual return
    expected_annual_return = (1 + portfolio_returns.mean()) ** annualization_factor - 1
    
    # Calculate Sortino ratio
    sortino_ratio = (expected_annual_return - risk_free_rate) / downside_deviation
    
    return float(sortino_ratio)


def create_weight_dict(tickers: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    """Create weight dictionary from tickers and weights."""
    return {ticker: float(weights.get(ticker, 0)) for ticker in tickers}


def weights_to_array(tickers: List[str], weights: Dict[str, float]) -> np.ndarray:
    """Convert weights dictionary to numpy array."""
    return np.array([weights.get(ticker, 0) for ticker in tickers])


def add_constraints_to_ef(
    ef: efficient_frontier.EfficientFrontier,
    tickers: List[str],
    constraints: Dict[str, Tuple[float, float]]
) -> efficient_frontier.EfficientFrontier:
    """Add constraints to efficient frontier object."""
    for ticker in tickers:
        lower, upper = constraints.get(ticker, (0, 1))
        ef.add_constraint(
            lambda w, ticker=ticker, idx=tickers.index(ticker): w[idx] >= lower
        )
        ef.add_constraint(
            lambda w, ticker=ticker, idx=tickers.index(ticker): w[idx] <= upper
        )
    return ef


def create_efficient_frontier(
    mu: pd.Series, 
    S: pd.DataFrame, 
    tickers: List[str], 
    constraints: Dict[str, Tuple[float, float]]
) -> efficient_frontier.EfficientFrontier:
    """Create and configure efficient frontier object."""
    ef = efficient_frontier.EfficientFrontier(mu, S)
    return add_constraints_to_ef(ef, tickers, constraints)


# Optimization Function Chain
def try_max_sharpe(mu, S, tickers, constraints, risk_free_rate, stock_data=None):
    """Try max Sharpe ratio optimization strategy."""
    try:
        ef = create_efficient_frontier(mu, S, tickers, constraints)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        weights_array = weights_to_array(tickers, weights)
        metrics = calculate_portfolio_metrics(weights_array, mu, S, risk_free_rate)
        
        # Calculate Sortino ratio if stock_data is provided
        if stock_data is not None:
            sortino_ratio = calculate_sortino_ratio(weights_array, stock_data, risk_free_rate)
            metrics["sortino_ratio"] = sortino_ratio
            
        return weights, metrics
    except Exception as e:
        print(f"Max Sharpe optimization failed: {e}")
        return None


def try_min_volatility(mu, S, tickers, constraints, risk_free_rate, stock_data=None):
    """Try minimum volatility optimization strategy."""
    try:
        ef = create_efficient_frontier(mu, S, tickers, constraints)
        ef.min_volatility()
        weights = ef.clean_weights()
        weights_array = weights_to_array(tickers, weights)
        metrics = calculate_portfolio_metrics(weights_array, mu, S, risk_free_rate)
        
        # Calculate Sortino ratio if stock_data is provided
        if stock_data is not None:
            sortino_ratio = calculate_sortino_ratio(weights_array, stock_data, risk_free_rate)
            metrics["sortino_ratio"] = sortino_ratio
            
        return weights, metrics
    except Exception as e:
        print(f"Min volatility optimization failed: {e}")
        return None


def try_equal_weights(mu, S, tickers, risk_free_rate, stock_data=None):
    """Use equal weights strategy (fallback)."""
    equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
    equal_weights_array = weights_to_array(tickers, equal_weights)
    equal_metrics = calculate_portfolio_metrics(equal_weights_array, mu, S, risk_free_rate)
    
    # Calculate Sortino ratio if stock_data is provided
    if stock_data is not None:
        sortino_ratio = calculate_sortino_ratio(equal_weights_array, stock_data, risk_free_rate)
        equal_metrics["sortino_ratio"] = sortino_ratio
        
    return equal_weights, equal_metrics


def find_optimal_portfolio(
    mu: pd.Series, 
    S: pd.DataFrame, 
    tickers: List[str], 
    constraints: Dict[str, Tuple[float, float]],
    risk_free_rate: float = 0.02,
    stock_data: pd.DataFrame = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Find optimal portfolio using a chain of optimization strategies.
    
    Args:
        mu: Expected returns
        S: Covariance matrix
        tickers: List of tickers
        constraints: Weight constraints
        risk_free_rate: Risk-free rate
        stock_data: Price data for Sortino ratio calculation
        
    Returns:
        Tuple of optimal weights and metrics
    """
    # Try different optimization strategies in sequence
    optimization_strategies = [
        lambda: try_max_sharpe(mu, S, tickers, constraints, risk_free_rate, stock_data),
        lambda: try_min_volatility(mu, S, tickers, constraints, risk_free_rate, stock_data),
        lambda: try_equal_weights(mu, S, tickers, risk_free_rate, stock_data)
    ]
    
    # Try each strategy until one succeeds
    for strategy in optimization_strategies:
        result = strategy()
        if result:
            return result
    
    # Final fallback to equal weights (should never reach here)
    return try_equal_weights(mu, S, tickers, risk_free_rate, stock_data)


def get_min_max_return(mu: pd.Series, S: pd.DataFrame, tickers: List[str], constraints: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Determine reasonable minimum and maximum return values."""
    try:
        # Calculate average and standard deviation of returns
        mean_return = mu.mean()
        
        # Check and handle NaN values
        if np.isnan(mean_return):
            print("Warning: Mean return is NaN, using default value")
            mean_return = 0.05  # Reasonable fallback value
        
        # Set return range based on mean value
        min_ret = max(-0.2, mean_return - 0.2)  # Not less than -20%
        max_ret = min(0.5, mean_return + 0.2)   # Not more than 50%
        
        # Ensure range is wide enough
        if max_ret - min_ret < 0.1:
            mid_point = (max_ret + min_ret) / 2
            min_ret = mid_point - 0.05
            max_ret = mid_point + 0.05
        
        # Ensure range isn't too negative
        min_ret = max(min_ret, -0.5)  # Absolute lower limit is -50%
        
        return min_ret, max_ret
        
    except Exception as e:
        print(f"Error in get_min_max_return function: {e}. Using fallback values.")
        # Return safe fallback values
        return -0.1, 0.2  # Range from -10% to 20%


def generate_frontier_portfolio(
    target_return: float,
    mu: pd.Series, 
    S: pd.DataFrame, 
    tickers: List[str], 
    constraints: Dict[str, Tuple[float, float]],
    optimal_sharpe: float,
    stock_data: pd.DataFrame = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """Generate a portfolio with specified target return."""
    ef = create_efficient_frontier(mu, S, tickers, constraints)
    try:
        ef.efficient_return(target_return)
        weights = ef.clean_weights()
        weights_array = weights_to_array(tickers, weights)
        metrics = calculate_portfolio_metrics(weights_array, mu, S, risk_free_rate)
        
        is_max_sharpe = abs(metrics["sharpe_ratio"] - optimal_sharpe) < 0.0001
        
        # Calculate Sortino ratio if stock_data is provided
        sortino_ratio = None
        if stock_data is not None:
            sortino_ratio = calculate_sortino_ratio(weights_array, stock_data, risk_free_rate)
        
        result = {
            "weights": create_weight_dict(tickers, weights),
            "expected_return": metrics["expected_return"],
            "standard_deviation": metrics["standard_deviation"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "is_max_sharpe": is_max_sharpe
        }
        
        # Add Sortino ratio if calculated
        if sortino_ratio is not None:
            result["sortino_ratio"] = sortino_ratio
            
        return result
    except ValueError as e:
        raise ValueError(f"Could not generate portfolio with return {target_return}: {e}")


def generate_frontier_portfolios(
    mu: pd.Series, 
    S: pd.DataFrame, 
    tickers: List[str], 
    constraints: Dict[str, Tuple[float, float]],
    optimal_metrics: Dict[str, float],
    optimal_weights: Dict[str, float] = None,
    num_portfolios: int = 100,
    stock_data: pd.DataFrame = None,
    risk_free_rate: float = 0.02
) -> List[Dict[str, Any]]:
    """
    Generate portfolios along the efficient frontier.
    
    Args:
        mu: Expected returns
        S: Covariance matrix
        tickers: List of stock tickers
        constraints: Weight constraints
        optimal_metrics: Metrics of the optimal portfolio
        optimal_weights: Weights of the optimal portfolio (optional)
        num_portfolios: Number of portfolios to create
        stock_data: Stock price data for Sortino ratio calculation
        risk_free_rate: Risk-free rate
        
    Returns:
        List of portfolios along the efficient frontier
    """
    # Get optimal parameters from optimal_metrics
    optimal_return = optimal_metrics["expected_return"]
    optimal_sharpe = optimal_metrics["sharpe_ratio"]
    
    # Print information for checking
    print(f"Optimal return: {optimal_return}, Sharpe ratio: {optimal_sharpe}")
    
    # Calculate optimal_weights if not provided
    if optimal_weights is None:
        try:
            # Only calculate optimal_weights if needed
            optimal_weights, _ = find_optimal_portfolio(mu, S, tickers, constraints)
            print("Calculated optimal weights")
        except Exception as e:
            print(f"Error calculating optimal weights: {e}")
            # Create equal weights as fallback
            optimal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
    
    # Find min/max return values from historical data (using improved version)
    try:
        min_ret, max_ret = get_min_max_return(mu, S, tickers, constraints)
        
        # Check for unusual values
        if min_ret < -0.5 or max_ret > 0.5 or np.isnan(min_ret) or np.isnan(max_ret):
            print(f"Warning: Unreasonable return range: [{min_ret}, {max_ret}]. Using default values.")
            mean_return = mu.mean()
            if np.isnan(mean_return):
                mean_return = 0.05
            min_ret = max(-0.2, mean_return - 0.1)
            max_ret = min(0.3, mean_return + 0.1)
    except Exception as e:
        print(f"Error calculating return range: {e}. Using default values.")
        # Use safe fallback values
        min_ret = -0.1
        max_ret = 0.2
    
    print(f"Calculated return range: [{min_ret:.4f}, {max_ret:.4f}]")
    
    # Ensure optimal_return is within min_ret and max_ret range
    if not np.isnan(optimal_return):
        min_ret = min(min_ret, optimal_return * 0.7)
        max_ret = max(max_ret, optimal_return * 1.3)
    
    # Ensure optimal_return is not at the edges
    # Goal: position optimal_return at around 50% in the list
    target_position_ratio = 0.5  # Set in the middle of the list
    
    if not np.isnan(optimal_return):
        # Adjust min_ret and max_ret to position optimal_return at desired location
        current_position_ratio = (optimal_return - min_ret) / (max_ret - min_ret) if max_ret != min_ret else 0.5

        if current_position_ratio < 0.3:  # Too close to lower bound
            min_ret = optimal_return - (max_ret - optimal_return) * target_position_ratio / (1 - target_position_ratio)
        elif current_position_ratio > 0.7:  # Too close to upper bound
            max_ret = optimal_return + (optimal_return - min_ret) * (1 - target_position_ratio) / target_position_ratio
    
    # Number of portfolios on each side of the optimal portfolio
    left_points = int(num_portfolios * target_position_ratio)
    right_points = num_portfolios - left_points - 1  # subtract 1 for optimal portfolio
    
    # Ensure at least 10 points on each side
    if left_points < 10:
        left_points = 10
        right_points = num_portfolios - left_points - 1
    if right_points < 10:
        right_points = 10
        left_points = num_portfolios - right_points - 1
    
    # Create target return arrays
    try:
        left_returns = np.linspace(min_ret, optimal_return, left_points, endpoint=False)
        right_returns = np.linspace(optimal_return, max_ret, right_points + 1)
        
        # Combine target returns
        target_returns = np.concatenate([left_returns, right_returns])
        target_returns = np.unique(target_returns)
        
        print(f"Created {len(target_returns)} target return levels from {np.min(target_returns):.4f} to {np.max(target_returns):.4f}")
    except Exception as e:
        print(f"Error creating target return array: {e}. Using fallback method.")
        # Simple fallback method
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    # Create optimal portfolio using metrics from outside
    optimal_portfolio = {
        "weights": create_weight_dict(tickers, optimal_weights),
        "expected_return": optimal_return,
        "standard_deviation": optimal_metrics["standard_deviation"],
        "sharpe_ratio": optimal_sharpe,
        "is_max_sharpe": True
    }
    
    # Add Sortino ratio if stock_data is provided
    if stock_data is not None:
        optimal_weights_array = weights_to_array(tickers, optimal_weights)
        optimal_sortino = calculate_sortino_ratio(optimal_weights_array, stock_data, risk_free_rate)
        optimal_portfolio["sortino_ratio"] = optimal_sortino
    
    # Add optimal portfolio directly - must be included
    frontier_portfolios = [optimal_portfolio]
    
    # Create other portfolios along the frontier
    for target_return in target_returns:
        # Skip target_return close to optimal_return to avoid duplicates
        if abs(target_return - optimal_return) < 0.0000001:
            continue
            
        try:
            # Create portfolio with target return
            portfolio = generate_frontier_portfolio(
                target_return=target_return,
                mu=mu, 
                S=S, 
                tickers=tickers, 
                constraints=constraints,
                optimal_sharpe=optimal_sharpe,
                stock_data=stock_data,
                risk_free_rate=risk_free_rate
            )
            portfolio["is_max_sharpe"] = False  # Ensure no other portfolio is marked as optimal
            frontier_portfolios.append(portfolio)
        except Exception as e:
            # Try to create interpolated portfolio if enough portfolios exist
            if len(frontier_portfolios) >= 2:
                # Find nearest portfolios with returns lower and higher
                lower_port = None
                higher_port = None
                
                for p in frontier_portfolios:
                    if p["expected_return"] < target_return and (lower_port is None or p["expected_return"] > lower_port["expected_return"]):
                        lower_port = p
                    if p["expected_return"] > target_return and (higher_port is None or p["expected_return"] < higher_port["expected_return"]):
                        higher_port = p
                
                # If both portfolios found, create interpolated portfolio
                if lower_port and higher_port:
                    # Calculate interpolation coefficient
                    weight = (target_return - lower_port["expected_return"]) / (higher_port["expected_return"] - lower_port["expected_return"])
                    
                    # Create interpolated portfolio
                    interpolated_portfolio = {
                        "weights": {t: (1-weight) * lower_port["weights"][t] + weight * higher_port["weights"][t] for t in tickers},
                        "expected_return": target_return,
                        "standard_deviation": (1-weight) * lower_port["standard_deviation"] + weight * higher_port["standard_deviation"],
                        "sharpe_ratio": (1-weight) * lower_port["sharpe_ratio"] + weight * higher_port["sharpe_ratio"],
                        "is_max_sharpe": False,
                        "is_interpolated": True
                    }
                    
                    # Add Sortino ratio if present in both portfolios
                    if "sortino_ratio" in lower_port and "sortino_ratio" in higher_port:
                        interpolated_portfolio["sortino_ratio"] = (1-weight) * lower_port["sortino_ratio"] + weight * higher_port["sortino_ratio"]
                    elif stock_data is not None:
                        # Calculate Sortino directly if stock_data is available
                        weights_array = weights_to_array(tickers, interpolated_portfolio["weights"])
                        interpolated_portfolio["sortino_ratio"] = calculate_sortino_ratio(weights_array, stock_data, risk_free_rate)
                    frontier_portfolios.append(interpolated_portfolio)
            continue
    
    # Sắp xếp lại các portfolio theo expected_return
    frontier_portfolios.sort(key=lambda x: x["expected_return"])
    
    # Đảm bảo rằng danh sách có đúng num_portfolios
    if len(frontier_portfolios) > num_portfolios:
        # Nếu có quá nhiều portfolios, giữ lại optimal và loại bỏ các portfolios khác một cách đều đặn
        optimal_index = next((i for i, p in enumerate(frontier_portfolios) if p.get("is_max_sharpe", False)), None)
        
        if optimal_index is not None:
            # Giữ optimal portfolio và các portfolios xung quanh
            keep_indices = [optimal_index]
            
            # Tính số lượng cần loại bỏ từ mỗi bên
            left_excess = optimal_index
            right_excess = len(frontier_portfolios) - optimal_index - 1
            left_to_keep = min(left_points, left_excess)
            right_to_keep = min(right_points, right_excess)
            
            # Chọn các indices cần giữ lại ở phía trái
            if left_excess > 0 and left_to_keep > 0:
                left_stride = max(1, left_excess // left_to_keep)
                for i in range(optimal_index - 1, -1, -left_stride):
                    if len(keep_indices) < num_portfolios:
                        keep_indices.append(i)
            
            # Chọn các indices cần giữ lại ở phía phải
            if right_excess > 0 and right_to_keep > 0:
                right_stride = max(1, right_excess // right_to_keep)
                for i in range(optimal_index + 1, len(frontier_portfolios), right_stride):
                    if len(keep_indices) < num_portfolios:
                        keep_indices.append(i)
            
            # Lấy các portfolio cần giữ lại
            keep_indices.sort()
            frontier_portfolios = [frontier_portfolios[i] for i in keep_indices]
    
    # Xử lý trường hợp không đủ portfolios
    while len(frontier_portfolios) < num_portfolios:
        # Thêm portfolio ở giữa các cặp portfolios có khoảng cách expected_return lớn nhất
        max_gap = 0
        insert_index = -1
        
        for i in range(len(frontier_portfolios) - 1):
            gap = frontier_portfolios[i+1]["expected_return"] - frontier_portfolios[i]["expected_return"]
            if gap > max_gap:
                max_gap = gap
                insert_index = i
        
        # Nếu không tìm thấy khoảng cách lớn, thoát khỏi vòng lặp
        if insert_index < 0 or max_gap < 0.00001:
            break
            
        # Tạo portfolio ở giữa
        mid_return = (frontier_portfolios[insert_index]["expected_return"] + 
                      frontier_portfolios[insert_index+1]["expected_return"]) / 2
        
        try:
            # Thử tạo portfolio với target return ở giữa
            portfolio = generate_frontier_portfolio(
                target_return=mid_return,
                mu=mu, S=S, tickers=tickers, constraints=constraints,
                optimal_sharpe=optimal_sharpe
            )
            portfolio["is_max_sharpe"] = False
            frontier_portfolios.insert(insert_index + 1, portfolio)
        except Exception:
            # Nếu không thể tạo portfolio với target return, nội suy giữa hai danh mục
            left = frontier_portfolios[insert_index]
            right = frontier_portfolios[insert_index+1]
            
            # Nội suy tuyến tính
            weight = 0.5  # Điểm giữa
            interpolated_portfolio = {
                "weights": {t: (1-weight) * left["weights"][t] + weight * right["weights"][t] for t in tickers},
                "expected_return": mid_return,
                "standard_deviation": (1-weight) * left["standard_deviation"] + weight * right["standard_deviation"],
                "sharpe_ratio": (1-weight) * left["sharpe_ratio"] + weight * right["sharpe_ratio"],
                "is_max_sharpe": False,
                "is_interpolated": True
            }
            frontier_portfolios.insert(insert_index + 1, interpolated_portfolio)
        
        # Sắp xếp lại theo expected_return
        frontier_portfolios.sort(key=lambda x: x["expected_return"])
    
    # Đảm bảo không vượt quá num_portfolios
    if len(frontier_portfolios) > num_portfolios:
        frontier_portfolios = frontier_portfolios[:num_portfolios]
    
    # Kiểm tra một lần cuối có đúng một portfolio là is_max_sharpe
    max_sharpe_count = sum(1 for p in frontier_portfolios if p.get("is_max_sharpe", False))
    if max_sharpe_count != 1:
        # Tìm vị trí của portfolio tối ưu
        optimal_index = None
        max_sharpe = -float('inf')
        for i, port in enumerate(frontier_portfolios):
            if port["sharpe_ratio"] > max_sharpe:
                max_sharpe = port["sharpe_ratio"]
                optimal_index = i
        
        # Reset tất cả flag is_max_sharpe
        for i, port in enumerate(frontier_portfolios):
            port["is_max_sharpe"] = (i == optimal_index)
    
    return frontier_portfolios


# Black-Litterman Model Functions
def prepare_black_litterman_data(
    price_data: pd.DataFrame,
    benchmark_weights: Dict[str, float],
    risk_free_rate: float = 0.02,
    market_risk_aversion: float = 2.5
) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    Prepare data for Black-Litterman model.
    """
    # Đảm bảo price_data có chỉ mục và hình dạng đúng
    if not isinstance(price_data, pd.DataFrame):
        raise ValueError("price_data phải là DataFrame")
    
    # Lấy danh sách mã chứng khoán từ cột của price_data
    tickers = price_data.columns.tolist()
    
    # Tính ma trận hiệp phương sai
    cov_matrix = risk_models.sample_cov(price_data)
    
    # Đảm bảo cov_matrix là DataFrame với chỉ mục và cột đúng
    if not isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    
    # Tạo Series trọng số với cùng chỉ mục như cov_matrix
    benchmark_weights_series = pd.Series(
        {ticker: benchmark_weights.get(ticker, 0) for ticker in tickers},
        index=tickers
    )
    
    # Chuẩn hóa trọng số
    if abs(benchmark_weights_series.sum() - 1.0) > 0.001:
        benchmark_weights_series = benchmark_weights_series / benchmark_weights_series.sum()
    
    try:
        # Sử dụng hàm đúng để tính lợi nhuận ẩn định thị trường
        market_implied_returns = black_litterman.market_implied_prior_returns(
            market_prices=benchmark_weights_series,  # Sử dụng trọng số chuẩn làm giá thị trường
            delta=market_risk_aversion,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate
        )
        
        # Đảm bảo market_implied_returns có cùng chỉ mục như cov_matrix
        market_implied_returns = pd.Series(
            market_implied_returns, 
            index=cov_matrix.index
        )
    except Exception as e:
        # Fallback nếu phương pháp trên không hoạt động
        # Sử dụng lợi nhuận lịch sử làm dự phòng
        historical_returns = expected_returns.mean_historical_return(price_data)
        market_implied_returns = historical_returns
        print(f"Không thể tính lợi nhuận ẩn định: {e}. Sử dụng lợi nhuận lịch sử.")
    
    # Tính lợi nhuận lịch sử để so sánh
    historical_returns = expected_returns.mean_historical_return(price_data)
    
    # Lưu trữ thông tin bổ sung
    additional_info = {
        "market_risk_aversion": market_risk_aversion,
        "risk_free_rate": risk_free_rate,
        "historical_returns": historical_returns.to_dict()
    }
    
    return market_implied_returns, cov_matrix, additional_info


def apply_investor_views(
    market_implied_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    view_dict: Dict[str, Dict],
    confidence_dict: Dict[str, float] = None,
    tau: float = 0.05
) -> Tuple[pd.Series, Dict]:
    """
    Apply investor views to Black-Litterman model.
    
    Args:
        market_implied_returns: Market implied returns
        cov_matrix: Covariance matrix
        view_dict: Investor views on assets
        confidence_dict: Confidence levels for views
        tau: Black-Litterman tau parameter
        
    Returns:
        Tuple of posterior returns and view application info
    """
    # Initialize view dictionary
    viewdict = {}
    
    # Use default confidence if not provided
    if confidence_dict is None:
        confidence_dict = {}

    # Convert views to Black-Litterman format
    for ticker, view_info in view_dict.items():
        if ticker in market_implied_returns.index:
            if view_info.get("view_type") == "will return":
                value = view_info.get("value", 0) / 100  # Convert percentage to decimal
                viewdict[ticker] = value
    
    # If no valid views, return market implied returns
    if not viewdict:
        return market_implied_returns, {"adjusted": False, "reason": "No valid views provided"}
    
    try:
        # Apply Black-Litterman model
        bl = BlackLittermanModel(
            cov_matrix,
            pi=market_implied_returns,
            absolute_views=viewdict,
            tau=tau
        )
        
        # Get posterior returns
        posterior_returns = bl.bl_returns()
        
        # Return results with metadata
        additional_info = {
            "adjusted": True,
            "prior_returns": market_implied_returns.to_dict(),
            "posterior_returns": posterior_returns.to_dict(),
            "views": viewdict,
            "confidence": confidence_dict
        }
        
        return posterior_returns, additional_info
        
    except Exception as e:
        print(f"Error in Black-Litterman model: {e}")
        return market_implied_returns, {
            "adjusted": False, 
            "reason": f"Error in Black-Litterman calculation: {str(e)}"
        }


# Asset Analysis Functions
def analyze_ticker_performance(ticker: str, stock_data: pd.DataFrame, lookback_period=None) -> Dict[str, Any]:
    """
    Analyze individual stock performance using Compound Annual Return.
    
    Args:
        ticker: Stock ticker
        stock_data: Price data
        lookback_period: Number of days to analyze
        
    Returns:
        Performance metrics
    """
    if ticker not in stock_data.columns:
        return {"status": "missing_data"}
    
    ticker_data = stock_data[ticker].dropna()
    if len(ticker_data) < 30:
        return {"status": "insufficient_data"}
    
    # Limit time period if specified
    if lookback_period is not None and lookback_period < len(ticker_data):
        ticker_data = ticker_data[-lookback_period:]
    
    # Calculate daily returns for volatility and other calculations
    returns = ticker_data.pct_change().dropna()
    
    # Calculate CAGR - Compound Annual Growth Rate
    trading_days = len(ticker_data)
    years = trading_days / 252  # 252 trading days per year
    initial_price = ticker_data.iloc[0]
    final_price = ticker_data.iloc[-1]
    annualized_return = (final_price / initial_price) ** (1 / years) - 1
    
    # Use logarithmic returns for more accurate volatility calculation
    log_returns = np.log(1 + returns)
    annualized_vol = log_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio based on CAGR
    risk_free_rate = 0.02  # Default risk-free rate
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # Calculate Sortino ratio 
    # Get negative returns only for downside deviation
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-6
    sortino = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    max_drawdown = (ticker_data / ticker_data.cummax() - 1).min()

    # Debug line - can be removed after testing
    print(f"Ticker: {ticker}")
    print(f"Initial price: {initial_price}, Final price: {final_price}, Years: {years}")
    print(f"CAGR: {annualized_return}")
    print(f"Volatility (log): {annualized_vol}")
    print(f"Sharpe: {sharpe}")
    print(f"Sortino: {sortino}")
    print(f"Max drawdown: {max_drawdown}")

    # Evaluate weakness criteria - now including Sortino ratio
    is_weak = (
        (annualized_return < -0.05) or
        (annualized_vol > 0.5) or
        (sharpe < -0.2) or
        (sortino < -0.1) or
        (max_drawdown < -0.4)
    )
    
    # Calculate weakness score - updated to include Sortino
    weakness_score = 0
    if annualized_return < -0.05: weakness_score += 1
    if annualized_return < -0.15: weakness_score += 1
    if annualized_vol > 0.5: weakness_score += 1
    if annualized_vol > 0.7: weakness_score += 1
    if sharpe < -0.2: weakness_score += 1
    if sharpe < -0.5: weakness_score += 1
    if sortino < -0.1: weakness_score += 1
    if sortino < -0.3: weakness_score += 1
    if max_drawdown < -0.4: weakness_score += 1
    if max_drawdown < -0.6: weakness_score += 1
    
    # Classify weakness level
    weakness_level = "none"
    if weakness_score >= 1: weakness_level = "mild"
    if weakness_score >= 3: weakness_level = "moderate"
    if weakness_score >= 5: weakness_level = "severe"
    
    return {
        "status": "analyzed",
        "annualized_return": float(annualized_return),  # CAGR
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "is_weak": is_weak,
        "weakness_score": weakness_score,
        "weakness_level": weakness_level
    }

def filter_weak_tickers(tickers: List[str], stock_data: pd.DataFrame) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Analyze and filter out weak tickers.
    
    Args:
        tickers: List of tickers
        stock_data: Price data
        
    Returns:
        Tuple of weak tickers list and analysis results
    """
    analysis_results = {}
    weak_tickers = []
    
    for ticker in tickers:
        analysis = analyze_ticker_performance(ticker, stock_data)
        analysis_results[ticker] = analysis
        
        if analysis.get("status") == "analyzed" and analysis.get("is_weak", False):
            weak_tickers.append(ticker)
    
    # Sort by weakness score
    weak_tickers.sort(
        key=lambda ticker: analysis_results[ticker].get("weakness_score", 0), 
        reverse=True
    )
    
    return weak_tickers, analysis_results


# Portfolio Analysis Functions
def analyze_portfolio(portfolio_data: Dict, analysis_type: str = OptimizationMethod.MEAN_VARIANCE) -> Dict[str, Any]:
    """
    Analyze a portfolio using either Mean-Variance or Black-Litterman method.
    
    Args:
        portfolio_data: Portfolio data
        analysis_type: Analysis method
        
    Returns:
        Analysis results
    """
    # Extract portfolio parameters
    tickers = portfolio_data["tickers"]
    allocations = portfolio_data["allocations"]
    constraints = portfolio_data["constraints"]
    start_date = portfolio_data["start_date"]
    end_date = portfolio_data["end_date"]
    risk_free_rate = portfolio_data.get("risk_free_rate", 0.02)
    investor_views = portfolio_data.get("investor_views", None)
    
    try:
        # Fetch stock data
        stock_data, ticker_info = fetch_stock_data(tickers, start_date, end_date)
        
        # Check for missing data
        missing_data_tickers = []
        for ticker in tickers:
            if ticker not in stock_data.columns or stock_data[ticker].isna().sum() > len(stock_data) * 0.1:
                missing_data_tickers.append(ticker)
        
        if missing_data_tickers:
            return {
                "error": True,
                "message": "Insufficient data for optimization",
                "problematic_tickers": missing_data_tickers
            }
        
        # Branch based on analysis type
        if analysis_type == OptimizationMethod.BLACK_LITTERMAN:
            return analyze_portfolio_black_litterman(
                tickers, allocations, constraints, stock_data, ticker_info, risk_free_rate, investor_views
            )
        else:  # Default to Mean-Variance
            return analyze_portfolio_mean_variance(
                tickers, allocations, constraints, stock_data, ticker_info, risk_free_rate
            )
            
    except Exception as e:
        # General error handling
        error_message = str(e)
        problematic_tickers = []
        
        # Try to identify problematic tickers
        for ticker in tickers:
            if ticker in error_message:
                problematic_tickers.append(ticker)
        
        # If no specific tickers identified, include all
        if not problematic_tickers:
            problematic_tickers = tickers
            
        return {
            "error": True,
            "message": f"Failed to analyze portfolio: {error_message}",
            "problematic_tickers": problematic_tickers
        }


def analyze_portfolio_mean_variance(
    tickers: List[str],
    allocations: Dict[str, float],
    constraints: Dict[str, Tuple[float, float]],
    stock_data: pd.DataFrame,
    ticker_info: Dict[str, str],
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Analyze portfolio using Mean-Variance optimization.
    
    Args:
        tickers: List of tickers
        allocations: Current allocations
        constraints: Weight constraints
        stock_data: Price data
        ticker_info: Ticker information
        risk_free_rate: Risk-free rate
        
    Returns:
        Analysis results
    """
    # Calculate expected returns and covariance
    mu, S = calculate_returns_and_covariance(stock_data)
    
    # Check for singularity in covariance matrix
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Find highly correlated pairs
        highly_correlated = []
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j and abs(S.loc[ticker1, ticker2] / np.sqrt(S.loc[ticker1, ticker1] * S.loc[ticker2, ticker2])) > 0.95:
                    highly_correlated.append((ticker1, ticker2))
        
        problematic_tickers = []
        for pair in highly_correlated:
            if pair[0] not in problematic_tickers:
                problematic_tickers.append(pair[0])
            if pair[1] not in problematic_tickers:
                problematic_tickers.append(pair[1])
        
        return {
            "error": True,
            "message": "Covariance matrix is singular, cannot optimize portfolio",
            "problematic_tickers": problematic_tickers
        }
    
    # Analyze provided portfolio
    provided_weights_array = weights_to_array(tickers, allocations)
    provided_metrics = calculate_portfolio_metrics(provided_weights_array, mu, S, risk_free_rate)
    
    # Calculate Sortino ratio for provided portfolio
    provided_sortino = calculate_sortino_ratio(provided_weights_array, stock_data, risk_free_rate)
    provided_metrics["sortino_ratio"] = provided_sortino
    
    # Find optimal portfolio
    optimal_weights, optimal_metrics = find_optimal_portfolio(mu, S, tickers, constraints, risk_free_rate)
    
    # Calculate Sortino ratio for optimal portfolio
    optimal_weights_array = weights_to_array(tickers, optimal_weights)
    optimal_sortino = calculate_sortino_ratio(optimal_weights_array, stock_data, risk_free_rate)
    optimal_metrics["sortino_ratio"] = optimal_sortino
    
    # Use provided portfolio if it has better Sharpe ratio
    if provided_metrics["sharpe_ratio"] > optimal_metrics["sharpe_ratio"]:
        print("Provided portfolio has better Sharpe ratio than computed optimal portfolio.")
        optimal_weights = {ticker: allocations.get(ticker, 0) for ticker in tickers}
        optimal_metrics = provided_metrics
    
    # Generate efficient frontier
    frontier_portfolios = generate_frontier_portfolios(
        mu, S, tickers, constraints, optimal_metrics, optimal_weights
    )
    
    # Calculate Sortino ratio for each frontier portfolio
    for portfolio in frontier_portfolios:
        portfolio_weights = portfolio["weights"]
        portfolio_weights_array = weights_to_array(tickers, portfolio_weights)
        portfolio["sortino_ratio"] = calculate_sortino_ratio(portfolio_weights_array, stock_data, risk_free_rate)
    
    # Calculate asset correlations
    correlation_matrix = stock_data.corr().to_dict()
    
    # Return compiled results
    return {        
        "provided_portfolio": {
            "weights": create_weight_dict(tickers, allocations),
            "metrics": provided_metrics
        },
        "efficient_frontier_portfolios": frontier_portfolios,
        "efficient_frontier_assets": create_weight_dict(tickers, optimal_weights),
        "asset_correlations": correlation_matrix,
        "optimal_portfolio": {
            "weights": create_weight_dict(tickers, optimal_weights),
            "metrics": optimal_metrics
        },
        "ticker_info": ticker_info,
        "method": OptimizationMethod.MEAN_VARIANCE
    }


def analyze_portfolio_black_litterman(
    tickers: List[str],
    allocations: Dict[str, float],
    constraints: Dict[str, Tuple[float, float]],
    stock_data: pd.DataFrame,
    ticker_info: Dict[str, str],
    risk_free_rate: float = 0.02,
    investor_views: Optional[Dict[str, Dict]] = None
) -> Dict[str, Any]:
    """
    Analyze portfolio using Black-Litterman model.
    
    Args:
        tickers: List of tickers
        allocations: Current allocations
        constraints: Weight constraints
        stock_data: Price data
        ticker_info: Ticker information
        risk_free_rate: Risk-free rate
        investor_views: Investor views
        
    Returns:
        Analysis results
    """
    # Ensure all tickers have allocations
    complete_allocations = {ticker: allocations.get(ticker, 0) for ticker in tickers}
    total_allocation = sum(complete_allocations.values())
    
    if total_allocation > 0:
        # Normalize allocations
        complete_allocations = {ticker: alloc / total_allocation for ticker, alloc in complete_allocations.items()}
    else:
        # Equal allocation if none provided
        complete_allocations = {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    # Prepare data for Black-Litterman
    market_implied_returns, cov_matrix, bl_info = prepare_black_litterman_data(
        stock_data, complete_allocations, risk_free_rate
    )
    
    # Calculate benchmark metrics
    benchmark_weights_array = weights_to_array(tickers, complete_allocations)
    benchmark_metrics = calculate_portfolio_metrics(
        benchmark_weights_array, market_implied_returns, cov_matrix, risk_free_rate
    )
    
    # Calculate Sortino ratio for benchmark portfolio
    benchmark_sortino = calculate_sortino_ratio(benchmark_weights_array, stock_data, risk_free_rate)
    benchmark_metrics["sortino_ratio"] = benchmark_sortino
    
    # Process investor views
    bl_returns = market_implied_returns
    view_application_info = {"adjusted": False, "reason": "No views provided"}
    
    if investor_views and len(investor_views) > 0:
        # Extract views and confidence levels
        view_dict = {}
        confidence_dict = {}
        
        for ticker, view_info in investor_views.items():
            if ticker in tickers:
                view_dict[ticker] = {
                    "view_type": view_info.get("view_type", "will return"),
                    "value": view_info.get("value", 0)
                }
                confidence_dict[ticker] = view_info.get("confidence", 50)
        
        # Apply views using Black-Litterman
        bl_returns, view_application_info = apply_investor_views(
            market_implied_returns, cov_matrix, view_dict, confidence_dict
        )
    
    try:
        # Find optimal portfolio using posterior returns
        optimal_weights, optimal_metrics = find_optimal_portfolio(
            bl_returns, cov_matrix, tickers, constraints, risk_free_rate
        )
        
        # Calculate Sortino ratio for optimal portfolio
        optimal_weights_array = weights_to_array(tickers, optimal_weights)
        optimal_sortino = calculate_sortino_ratio(optimal_weights_array, stock_data, risk_free_rate)
        optimal_metrics["sortino_ratio"] = optimal_sortino
        
        # Generate constrained efficient frontier
        frontier_portfolios = generate_frontier_portfolios(
            bl_returns, cov_matrix, tickers, constraints, optimal_metrics, optimal_weights
        )
        
        # Calculate Sortino ratio for each frontier portfolio
        for portfolio in frontier_portfolios:
            portfolio_weights = portfolio["weights"]
            portfolio_weights_array = weights_to_array(tickers, portfolio_weights)
            portfolio["sortino_ratio"] = calculate_sortino_ratio(portfolio_weights_array, stock_data, risk_free_rate)
        
        # Generate unconstrained efficient frontier for comparison
        unconstrained_constraints = {ticker: (0, 1) for ticker in tickers}
        unconstrained_optimal_weights, unconstrained_optimal_metrics = find_optimal_portfolio(
            bl_returns, cov_matrix, tickers, unconstrained_constraints, risk_free_rate
        )
        
        # Calculate Sortino ratio for unconstrained optimal portfolio
        unconstrained_weights_array = weights_to_array(tickers, unconstrained_optimal_weights)
        unconstrained_sortino = calculate_sortino_ratio(unconstrained_weights_array, stock_data, risk_free_rate)
        unconstrained_optimal_metrics["sortino_ratio"] = unconstrained_sortino
        
        unconstrained_frontier = generate_frontier_portfolios(
            bl_returns, cov_matrix, tickers, unconstrained_constraints, 
            unconstrained_optimal_metrics, unconstrained_optimal_weights
        )
        
        # Calculate Sortino ratio for each unconstrained frontier portfolio
        for portfolio in unconstrained_frontier:
            portfolio_weights = portfolio["weights"]
            portfolio_weights_array = weights_to_array(tickers, portfolio_weights)
            portfolio["sortino_ratio"] = calculate_sortino_ratio(portfolio_weights_array, stock_data, risk_free_rate)
            
    except Exception as e:
        print(f"Optimization error: {e}. Using equal weights fallback.")
        
        # Fallback to equal weights
        equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
        equal_weights_array = weights_to_array(tickers, equal_weights)
        equal_metrics = calculate_portfolio_metrics(
            equal_weights_array, bl_returns, cov_matrix, risk_free_rate
        )
        
        # Calculate Sortino ratio for equal weights portfolio
        equal_sortino = calculate_sortino_ratio(equal_weights_array, stock_data, risk_free_rate)
        equal_metrics["sortino_ratio"] = equal_sortino
        
        # Use equal weights for everything
        optimal_weights = equal_weights
        optimal_metrics = equal_metrics
        frontier_portfolios = [{
            "weights": equal_weights,
            "expected_return": equal_metrics["expected_return"],
            "standard_deviation": equal_metrics["standard_deviation"],
            "sharpe_ratio": equal_metrics["sharpe_ratio"],
            "sortino_ratio": equal_sortino,
            "is_max_sharpe": True
        }]
        
        unconstrained_optimal_weights = equal_weights
        unconstrained_optimal_metrics = equal_metrics
        unconstrained_frontier = frontier_portfolios.copy()
    
    # Calculate correlations
    correlation_matrix = stock_data.corr().to_dict()
    
    # Return results
    return {
        "benchmark_portfolio": {
            "weights": create_weight_dict(tickers, complete_allocations),
            "metrics": benchmark_metrics,
            "market_implied_returns": market_implied_returns.to_dict()
        },
        "black_litterman_info": {
            **bl_info,
            **view_application_info
        },
        "constrained_portfolio": {
            "weights": create_weight_dict(tickers, optimal_weights),
            "metrics": optimal_metrics,
            "efficient_frontier": frontier_portfolios
        },
        "unconstrained_portfolio": {
            "weights": create_weight_dict(tickers, unconstrained_optimal_weights),
            "metrics": unconstrained_optimal_metrics,
            "efficient_frontier": unconstrained_frontier
        },
        "asset_correlations": correlation_matrix,
        "ticker_info": ticker_info,
        "method": OptimizationMethod.BLACK_LITTERMAN
    }


def optimize_with_ticker_filtering(portfolio_data: Dict) -> Dict[str, Any]:
    """
    Optimize portfolio with automatic ticker filtering.
    
    Args:
        portfolio_data: Portfolio data
        
    Returns:
        Optimization results with recommendations
    """
    tickers = portfolio_data["tickers"]
    allocations = portfolio_data["allocations"]
    constraints = portfolio_data["constraints"]
    start_date = portfolio_data["start_date"]
    end_date = portfolio_data["end_date"]
    risk_free_rate = portfolio_data.get("risk_free_rate", 0.02)
    optimization_method = portfolio_data.get("optimization_method", OptimizationMethod.MEAN_VARIANCE)
    
    # Fetch stock data
    stock_data, ticker_info = fetch_stock_data(tickers, start_date, end_date)
    
    # Analyze and identify weak tickers
    weak_tickers, ticker_analysis = filter_weak_tickers(tickers, stock_data)
    
    # Generate recommendations
    recommendations = []
    if weak_tickers:
        for ticker in weak_tickers:
            analysis = ticker_analysis[ticker]
            weakness_level = analysis.get("weakness_level")
            if weakness_level == "severe":
                recommendations.append(f"Recommend removing '{ticker}' due to poor performance (level: severe)")
            elif weakness_level == "moderate":
                recommendations.append(f"Consider removing '{ticker}' due to suboptimal performance (level: moderate)")
            elif weakness_level == "mild":
                recommendations.append(f"Consider reducing allocation for '{ticker}' due to below-average performance (level: mild)")
    
    # Try optimization with all tickers
    original_result = analyze_portfolio(portfolio_data, optimization_method)
    
    # If successful, return results with recommendations
    if not original_result.get("error", False):
        return {
            "status": "success",
            "result": original_result,
            "weak_tickers": weak_tickers,
            "ticker_analysis": {t: ticker_analysis[t] for t in weak_tickers} if weak_tickers else {},
            "recommendations": recommendations
        }
    
    # If optimization failed, identify problematic tickers
    problematic_tickers = original_result.get("problematic_tickers", [])
    
    # Find priority tickers to remove
    priority_tickers_to_remove = [t for t in problematic_tickers if t in weak_tickers]
    
    if not priority_tickers_to_remove and problematic_tickers:
        priority_tickers_to_remove = problematic_tickers
    
    if not priority_tickers_to_remove and weak_tickers:
        priority_tickers_to_remove = weak_tickers[:min(2, len(weak_tickers))]
    
    # Try alternative portfolios by removing problematic tickers
    alternative_portfolios = []
    removed_combinations = []
    
    if priority_tickers_to_remove:
        # Try removing single tickers
        for ticker_to_remove in priority_tickers_to_remove:
            remaining_tickers = [t for t in tickers if t != ticker_to_remove]
            
            if len(remaining_tickers) >= 2:
                # Create new portfolio without problematic ticker
                new_allocations = {t: allocations.get(t, 0) for t in remaining_tickers}
                new_constraints = {t: constraints.get(t, (0, 1)) for t in remaining_tickers}
                
                # Normalize allocations
                if sum(new_allocations.values()) > 0:
                    total = sum(new_allocations.values())
                    new_allocations = {t: v/total for t, v in new_allocations.items()}
                
                # Create new portfolio data
                new_portfolio_data = {
                    "tickers": remaining_tickers,
                    "allocations": new_allocations,
                    "constraints": new_constraints,
                    "start_date": start_date,
                    "end_date": end_date,
                    "risk_free_rate": risk_free_rate,
                    "optimization_method": optimization_method
                }
                
                # Try optimization
                alternative_result = analyze_portfolio(new_portfolio_data, optimization_method)
                
                if not alternative_result.get("error", False):
                    alternative_portfolios.append({
                        "removed_tickers": [ticker_to_remove],
                        "result": alternative_result
                    })
                    removed_combinations.append([ticker_to_remove])
        
        # Try removing pairs of tickers
        if len(priority_tickers_to_remove) > 1:
            for i in range(len(priority_tickers_to_remove)):
                for j in range(i+1, len(priority_tickers_to_remove)):
                    tickers_to_remove = [priority_tickers_to_remove[i], priority_tickers_to_remove[j]]
                    remaining_tickers = [t for t in tickers if t not in tickers_to_remove]
                    
                    if len(remaining_tickers) >= 2:
                        # Create new portfolio without problematic tickers
                        new_allocations = {t: allocations.get(t, 0) for t in remaining_tickers}
                        new_constraints = {t: constraints.get(t, (0, 1)) for t in remaining_tickers}
                        
                        # Normalize allocations
                        if sum(new_allocations.values()) > 0:
                            total = sum(new_allocations.values())
                            new_allocations = {t: v/total for t, v in new_allocations.items()}
                        
                        # Create new portfolio data
                        new_portfolio_data = {
                            "tickers": remaining_tickers,
                            "allocations": new_allocations,
                            "constraints": new_constraints,
                            "start_date": start_date,
                            "end_date": end_date,
                            "risk_free_rate": risk_free_rate,
                            "optimization_method": optimization_method
                        }
                        
                        # Try optimization
                        alternative_result = analyze_portfolio(new_portfolio_data, optimization_method)
                        
                        if not alternative_result.get("error", False):
                            alternative_portfolios.append({
                                "removed_tickers": tickers_to_remove,
                                "result": alternative_result
                            })
                            removed_combinations.append(tickers_to_remove)
    
    # Return results based on alternative portfolios
    if alternative_portfolios:
        # Find best alternative by Sharpe ratio
        best_alternative = max(
            alternative_portfolios,
            key=lambda p: p["result"]["optimal_portfolio"]["metrics"]["sharpe_ratio"]
        )
        
        # Add recommendations for removed tickers
        for removed_tickers in removed_combinations:
            removed_str = ", ".join(f"'{t}'" for t in removed_tickers)
            recommendations.append(f"Attempted portfolio optimization after removing {removed_str}")

        best_removed_str = ", ".join(f"'{t}'" for t in best_alternative["removed_tickers"])
        recommendations.append(f"Recommend using the optimal portfolio after removing {best_removed_str}")
        
        return {
            "status": "alternative_success",
            "original_error": original_result.get("message", "Unknown error"),
            "result": best_alternative["result"],
            "weak_tickers": weak_tickers,
            "problematic_tickers": problematic_tickers,
            "removed_tickers": best_alternative["removed_tickers"],
            "ticker_analysis": {t: ticker_analysis[t] for t in weak_tickers} if weak_tickers else {},
            "alternative_portfolios": [
                {
                    "removed_tickers": p["removed_tickers"],
                    "sharpe_ratio": p["result"]["optimal_portfolio"]["metrics"]["sharpe_ratio"]
                }
                for p in alternative_portfolios
            ],
            "recommendations": recommendations
        }
    
    # Return failure status if no alternatives worked
    return {
        "status": "failure",
        "original_error": original_result.get("message", "Unknown error"),
        "weak_tickers": weak_tickers,
        "problematic_tickers": problematic_tickers,
        "ticker_analysis": {t: ticker_analysis[t] for t in weak_tickers} if weak_tickers else {},
        "recommendations": recommendations + ["Could not optimize portfolio even after attempting to remove problematic tickers"]
    }


# Main Execution Functions
def process_portfolio(portfolio_data: Dict) -> Dict[str, Any]:
    """
    Process a single portfolio with automatic optimization selection.
    
    Args:
        portfolio_data: Portfolio data
        
    Returns:
        Processed portfolio results
    """
    try:
        # Parse dates
        parsed_data = parse_portfolio_data(portfolio_data)
        
        # Create model and validate
        portfolio_model = PortfolioModel(**parsed_data)
        
        # Auto-calculate constraints if needed
        if not all(ticker in portfolio_model.constraints for ticker in portfolio_model.tickers):
            portfolio_model.constraints = auto_calculate_constraints(
                portfolio_model.tickers, 
                portfolio_model.allocations, 
                portfolio_model.constraints
            )
        
        # Auto-allocate weights if needed
        if not portfolio_model.is_fully_allocated():
            portfolio_model.allocations = auto_allocate(
                portfolio_model.tickers,
                portfolio_model.allocations,
                portfolio_model.constraints
            )
        
        # Convert to dictionary
        processed_data = portfolio_model.to_dict()
        
        # Optimize with ticker filtering
        return optimize_with_ticker_filtering(processed_data)
    
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "original_data": portfolio_data
        }


def process_portfolios(portfolios_data: List[Dict]) -> List[Dict]:
    """
    Process multiple portfolios.
    
    Args:
        portfolios_data: List of portfolio data
        
    Returns:
        List of processed results
    """
    portfolio_results = []
    
    for i, portfolio_data in enumerate(portfolios_data):
        # Process portfolio
        result = process_portfolio(portfolio_data)
        
        # Add portfolio index
        result_with_index = {
            "portfolio_index": i + 1,
            "original_portfolio": portfolio_data,
            **result
        }
        
        portfolio_results.append(result_with_index)
    
    return portfolio_results


def convert_to_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def optimize_portfolios(json_data: str) -> str:
    """
    Main function to optimize portfolios from JSON input.
    
    Args:
        json_data: Portfolio data in JSON format
        
    Returns:
        Optimization results in JSON format
    """
    # Parse input JSON
    if isinstance(json_data, str):
        try:
            portfolios_data = json.loads(json_data)
        except json.JSONDecodeError:
            portfolios_data = json_data
    else:
        portfolios_data = json_data
    
    # Process portfolios
    results = process_portfolios(portfolios_data)
    
    # Return JSON formatted results
    return json.dumps(results, indent=2, default=convert_to_serializable, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    # Dữ liệu test cho Mean-Variance
    # portfolio_data_mv = {
    #     "tickers": ["GE", "NFLX", "NVDA", "AMZN", "TSLA", "META", "GM"],
    #     "allocations": {
    #         "GE": 0.02,
    #         "NFLX": 0.042,
    #         "NVDA": 0.37,
    #         "AMZN": 0.26,
    #         "TSLA": 0.13,
    #         "META": 0.17,
    #         "GM": 0.006
    #     },
    #     # "constraints": {
    #     #     "AAPL": [0.10, 0.30],
    #     #     "MSFT": [0.10, 0.30],
    #     #     "JNJ": [0.10, 0.30],
    #     #     "PG": [0.05, 0.25],
    #     #     "KO": [0.05, 0.25]
    #     # },
    #     "start_date": "2015-01-02",
    #     "end_date": "2025-01-02",
    #     "risk_free_rate": 4.19,
    #     "optimization_method": OptimizationMethod.MEAN_VARIANCE
    # }

    portfolio_data_mv = {
        "tickers": ["BHP.AX", "CBA.AX", "CSL.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "FMG.AX", "RIO.AX", "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX", "GMG.AX", "XRO.AX", "ALL.AX", "COL.AX", "MIN.AX", "RMD.AX", "SHL.AX", "REA.AX"],
        "allocations": {
            "BHP.AX": 0.11,
            "CBA.AX": 0.095,
            "CSL.AX": 0.085,
            "WBC.AX": 0.065,
            "NAB.AX": 0.065,
            "ANZ.AX": 0.06,
            "FMG.AX": 0.055,
            "RIO.AX": 0.05,
            "MQG.AX": 0.045,
            "WES.AX": 0.04,
            "WOW.AX": 0.035,
            "TLS.AX": 0.03,
            "GMG.AX": 0.025,
            "XRO.AX": 0.025,
            "ALL.AX": 0.02,
            "COL.AX": 0.02,
            "MIN.AX": 0.015,
            "RMD.AX": 0.015,
            "SHL.AX": 0.01,
            "REA.AX": 0.01
        },
        "constraints": {
            "BHP.AX": [0.08, 0.15],
            "CBA.AX": [0.07, 0.13],
            "CSL.AX": [0.06, 0.12],
            "WBC.AX": [0.04, 0.10],
            "NAB.AX": [0.04, 0.10],
            "ANZ.AX": [0.04, 0.10],
            "FMG.AX": [0.03, 0.08],
            "RIO.AX": [0.03, 0.08],
            "MQG.AX": [0.03, 0.07],
            "WES.AX": [0.02, 0.06],
            "WOW.AX": [0.02, 0.05],
            "TLS.AX": [0.01, 0.04],
            "GMG.AX": [0.01, 0.04],
            "XRO.AX": [0.01, 0.04],
            "ALL.AX": [0.01, 0.03],
            "COL.AX": [0.01, 0.03],
            "MIN.AX": [0.005, 0.02],
            "RMD.AX": [0.005, 0.02],
            "SHL.AX": [0.005, 0.02],
            "REA.AX": [0.005, 0.02]
        },
        "start_date": "2015-01-02",
        "end_date": "2025-01-02",
        "risk_free_rate": 0.0383,  # Lãi suất phi rủi ro cập nhật theo RBA 2023
        "optimization_method": OptimizationMethod.MEAN_VARIANCE
    }

    # Example portfolio data with Black-Litterman investor views
    portfolio_data = {
        "tickers": ["BHP.AX", "CBA.AX", "CSL.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "FMG.AX", "RIO.AX", "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX", "GMG.AX", "XRO.AX", "ALL.AX", "COL.AX", "MIN.AX", "RMD.AX", "SHL.AX", "REA.AX"],
        "allocations": {
            "BHP.AX": 0.11,
            "CBA.AX": 0.095,
            "CSL.AX": 0.085,
            "WBC.AX": 0.065,
            "NAB.AX": 0.065,
            "ANZ.AX": 0.06,
            "FMG.AX": 0.055,
            "RIO.AX": 0.05,
            "MQG.AX": 0.045,
            "WES.AX": 0.04,
            "WOW.AX": 0.035,
            "TLS.AX": 0.03,
            "GMG.AX": 0.025,
            "XRO.AX": 0.025,
            "ALL.AX": 0.02,
            "COL.AX": 0.02,
            "MIN.AX": 0.015,
            "RMD.AX": 0.015,
            "SHL.AX": 0.01,
            "REA.AX": 0.01
        },
        "start_date": "2015-01-02",
        "end_date": "2025-01-02",
        "risk_free_rate": 0.0383,
        "optimization_method": OptimizationMethod.BLACK_LITTERMAN,
        "investor_views": {
            "CSL.AX": {
                "view_type": "will return",
                "value": 0.07,  # Kỳ vọng vượt trội 7% so với benchmark
                "confidence": 70,
                "comparison_asset": "S&P/ASX 200"
            },
            "FMG.AX": {
                "view_type": "will return",
                "value": -0.12,  # Dự báo giảm 12% do giá quặng sắt giảm
                "confidence": 65
            },
            "XRO.AX": {
                "view_type": "will return",
                "value": 0.15,  # Kỳ vọng tăng trưởng 15% nhờ mở rộng thị trường châu Á
                "confidence": 60,
                "comparison_asset": "S&P/ASX All Technology"
            }
        }
    }

    # Để chạy thử nghiệm
    result_mv = process_portfolio(portfolio_data_mv)
    
    # Process a single portfolio
    result = process_portfolio(portfolio_data)
    
    # Add index
    result_with_index = {
        "mean_variance": {
            "portfolio_index": 1,
            **result_mv
        },
        "black_litterman": {
            "portfolio_index": 2,
            **result,
        }
    }

    # Print or save results
    print(json.dumps([result_with_index], indent=2, default=convert_to_serializable, ensure_ascii=False))
    
    # Save to file
    with open('portfolio_optimization_result.json', 'w', encoding='utf-8') as f:
        json.dump([result_with_index], f, indent=2, default=convert_to_serializable, ensure_ascii=False)
