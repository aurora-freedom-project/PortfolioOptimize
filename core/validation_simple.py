# core/validation_simple.py
"""
Portfolio Optimization Input Validation
=======================================

Simplified validation system for portfolio optimization parameters.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom validation error for portfolio optimization."""
    pass


def validate_date_format(date_str: str) -> None:
    """Validate date format is YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValidationError(f"Date format must be YYYY-MM-DD, got: {date_str}")


def validate_date_range(start_date: str, end_date: str, method: str) -> None:
    """Validate date range requirements."""
    validate_date_format(start_date)
    validate_date_format(end_date)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Start date must be before end date
    if start >= end:
        raise ValidationError("Start date must be before end date")
    
    # Calculate years difference
    years_diff = (end - start).days / 365.25
    
    # Method-specific minimum requirements (with small tolerance for floating point)
    if method.lower() == 'mean_variance' and years_diff < 2.99:
        raise ValidationError("Date range must be at least 3 years for Mean-Variance")
    
    if method.lower() == 'black_litterman' and years_diff < 4.99:
        raise ValidationError("Date range must be at least 5 years for Black-Litterman")


def validate_tickers(tickers: List[str]) -> None:
    """Validate tickers list."""
    if not tickers:
        raise ValidationError("Tickers list must not be empty")


def validate_allocations(tickers: List[str], allocations: Dict[str, float]) -> Dict[str, float]:
    """Validate and complete allocations."""
    validate_tickers(tickers)
    
    # Filter allocations to only include valid tickers
    valid_allocations = {k: v for k, v in allocations.items() if k in tickers}
    
    # Validate individual allocation values
    for ticker, allocation in valid_allocations.items():
        if not (0 <= allocation <= 1):
            raise ValidationError(f"{ticker} allocation {allocation*100:.1f}% not in range [0%, 100%]")
    
    total_allocated = sum(valid_allocations.values())
    num_tickers = len(tickers)
    num_allocated = len(valid_allocations)
    
    # Case 1: No allocations - auto-distribute equally
    if num_allocated == 0:
        equal_weight = 1.0 / num_tickers
        return {ticker: equal_weight for ticker in tickers}
    
    # Case 2: Partial allocations
    if num_allocated < num_tickers:
        if total_allocated > 1.0:
            raise ValidationError(f"Total allocation {total_allocated*100:.1f}% exceeds 100%")
        
        # Distribute remaining equally to unallocated tickers
        unallocated_tickers = [t for t in tickers if t not in valid_allocations]
        remaining_weight = 1.0 - total_allocated
        
        if remaining_weight > 0 and unallocated_tickers:
            equal_remaining = remaining_weight / len(unallocated_tickers)
            for ticker in unallocated_tickers:
                valid_allocations[ticker] = equal_remaining
    
    # Case 3: Full allocations - must sum to 100% (with tolerance)
    else:
        tolerance = 0.000001  # 0.0001% tolerance
        if abs(total_allocated - 1.0) > tolerance:
            raise ValidationError(f"Allocation sum must be 100%, got {total_allocated*100:.4f}%")
    
    # Normalize to ensure exact 100%
    total = sum(valid_allocations.values())
    if total > 0:
        valid_allocations = {k: v/total for k, v in valid_allocations.items()}
    
    return valid_allocations


def validate_constraints(tickers: List[str], constraints: Dict[str, Tuple[float, float]], 
                        allocations: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Validate and complete constraints."""
    validate_tickers(tickers)
    
    # Validate constraint bounds
    for ticker, (min_weight, max_weight) in constraints.items():
        if not (0.0 <= min_weight <= 1.0):
            raise ValidationError(f"{ticker} minimum weight {min_weight} not in range [0.0, 1.0]")
        if not (0.0 <= max_weight <= 1.0):
            raise ValidationError(f"{ticker} maximum weight {max_weight} not in range [0.0, 1.0]")
        if min_weight > max_weight:
            raise ValidationError(f"{ticker} minimum weight {min_weight} > maximum weight {max_weight}")
    
    # Check if existing allocations satisfy constraints
    for ticker in tickers:
        if ticker in allocations and ticker in constraints:
            allocation = allocations[ticker]
            min_weight, max_weight = constraints[ticker]
            if not (min_weight <= allocation <= max_weight):
                raise ValidationError(f"{ticker} allocation {allocation*100:.1f}% violates constraint [{min_weight*100:.1f}%, {max_weight*100:.1f}%]")
    
    # Check if equal distribution satisfies constraints (for empty allocations)
    if not allocations:
        equal_weight = 1.0 / len(tickers)
        for ticker in tickers:
            if ticker in constraints:
                min_weight, max_weight = constraints[ticker]
                if not (min_weight <= equal_weight <= max_weight):
                    raise ValidationError(f"Equal distribution {equal_weight*100:.1f}% violates {ticker} constraint [{min_weight*100:.1f}%, {max_weight*100:.1f}%]")
    
    # Check if constraints allow for feasible portfolio
    total_min = sum(constraints.get(ticker, (0.0, 1.0))[0] for ticker in tickers)
    total_max = sum(constraints.get(ticker, (0.0, 1.0))[1] for ticker in tickers)
    
    if total_min > 1.0:
        raise ValidationError("Constraint is too tight. Minimum weights sum to more than 100%")
    if total_max < 1.0:
        raise ValidationError("Constraint is too tight. Maximum weights sum to less than 100%")
    
    # Return complete constraints with defaults
    return {ticker: constraints.get(ticker, (0.0, 1.0)) for ticker in tickers}


def validate_risk_free_rate(rate: Any) -> float:
    """Validate and convert risk-free rate."""
    try:
        rate_float = float(rate)
        if rate_float < 0:
            raise ValidationError("Risk-free rate cannot be negative")
        if rate_float > 1:
            # Assume percentage was provided, convert to decimal
            rate_float = rate_float / 100
        return rate_float
    except (ValueError, TypeError):
        raise ValidationError(f"Risk-free rate must be a number, got: {rate}")


def validate_investor_views(views: Dict[str, Dict[str, float]], tickers: List[str],
                           date_range_years: float) -> Dict[str, Dict[str, float]]:
    """Validate investor views format."""
    if not views:
        return views
    
    # Only validate if views are provided
    if views and date_range_years < 4.99:
        raise ValidationError("Date range must be at least 5 years when investor views are provided")
    
    validated_views = {}
    for ticker, view_data in views.items():
        if not isinstance(view_data, dict):
            raise ValidationError(f"Investor view for {ticker} must be a dictionary")
        
        if 'expected_return' not in view_data:
            raise ValidationError(f"Investor view for {ticker} must include 'expected_return'")
        
        expected_return = view_data['expected_return']
        if not isinstance(expected_return, (int, float)):
            raise ValidationError(f"Expected return for {ticker} must be a number")
        
        # Set default confidence if not provided
        confidence = view_data.get('confidence', 0.5)
        if not (0 < confidence <= 1):
            raise ValidationError(f"Confidence for {ticker} must be between 0 and 1")
        
        validated_views[ticker] = {
            'expected_return': expected_return,
            'confidence': confidence
        }
    
    return validated_views


class ValidatedPortfolioData:
    """Container for validated portfolio optimization data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, method: str,
                 allocations: Dict[str, float], constraints: Dict[str, Tuple[float, float]],
                 risk_free_rate: float, investor_views: Dict[str, Dict[str, float]]):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.method = method
        self.allocations = allocations
        self.constraints = constraints
        self.risk_free_rate = risk_free_rate
        self.investor_views = investor_views


def validate_portfolio_inputs(
    tickers: List[str],
    start_date: str,
    end_date: str,
    method: str,
    allocations: Dict[str, float] = None,
    constraints: Dict[str, Tuple[float, float]] = None,
    risk_free_rate: Any = 0.02,
    investor_views: Dict[str, Dict[str, float]] = None
) -> ValidatedPortfolioData:
    """
    Validate all portfolio optimization inputs.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        method: Optimization method
        allocations: Portfolio allocations (optional)
        constraints: Weight constraints (optional)
        risk_free_rate: Risk-free rate (optional)
        investor_views: Investor views for Black-Litterman (optional)
    
    Returns:
        ValidatedPortfolioData object
    
    Raises:
        ValidationError: If any validation fails
    """
    try:
        # Set defaults
        allocations = allocations or {}
        constraints = constraints or {}
        investor_views = investor_views or {}
        
        # Calculate date range for investor views validation
        date_range_years = 0.0
        if start_date and end_date:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                date_range_years = (end - start).days / 365.25
            except ValueError:
                pass
        
        # Validate all components
        validate_date_range(start_date, end_date, method)
        validate_tickers(tickers)
        
        validated_allocations = validate_allocations(tickers, allocations)
        validated_constraints = validate_constraints(tickers, constraints, validated_allocations)
        validated_rate = validate_risk_free_rate(risk_free_rate)
        validated_views = validate_investor_views(investor_views, tickers, date_range_years)
        
        return ValidatedPortfolioData(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            method=method,
            allocations=validated_allocations,
            constraints=validated_constraints,
            risk_free_rate=validated_rate,
            investor_views=validated_views
        )
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}")


def example_validation_usage():
    """Example usage of the validation system."""
    
    print("Portfolio Optimization Input Validation Examples")
    print("=" * 50)
    
    # Example 1: Valid inputs
    try:
        valid_data = validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            allocations={'ANZ.AX': 0.4, 'CBA.AX': 0.6},  # NAB.AX will get 0%
            risk_free_rate=0.025
        )
        print("✅ Valid inputs accepted")
        print(f"   Complete allocations: {valid_data.allocations}")
        print(f"   Complete constraints: {valid_data.constraints}")
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    # Example 2: Date range too short for Black-Litterman
    try:
        validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX'],
            start_date='2022-01-01',
            end_date='2023-12-31',
            method='black_litterman',
            investor_views={'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8}}
        )
        print("✅ Should not reach here")
        
    except ValidationError as e:
        print(f"✅ Correctly caught short date range: {e}")
    
    # Example 3: Invalid allocation sum
    try:
        validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            allocations={'ANZ.AX': 0.5, 'CBA.AX': 0.4, 'NAB.AX': 0.2}  # Sums to 110%
        )
        print("❌ Should not reach here")
        
    except ValidationError as e:
        print(f"✅ Correctly caught invalid allocation: {e}")
    
    print("\nValidation system working correctly! ✅")


if __name__ == "__main__":
    example_validation_usage()