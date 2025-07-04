#!/usr/bin/env python3
# tests/test_validation.py
"""
Comprehensive Tests for Portfolio Optimization Input Validation
==============================================================

Tests for all validation rules using the simplified validation system.
Updated to use ASX tickers and functional validation approach.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.validation_simple import (
    validate_portfolio_inputs,
    ValidationError,
    validate_date_range,
    validate_allocations,
    validate_constraints,
    validate_investor_views,
    validate_risk_free_rate,
    ValidatedPortfolioData
)


class TestDateRangeValidation:
    """Test date range validation rules."""
    
    def test_valid_date_format(self):
        """Test valid date format YYYY-MM-DD."""
        # Should not raise any exception
        validate_date_range('2020-01-01', '2023-12-31', 'mean_variance')
    
    def test_invalid_date_format(self):
        """Test invalid date formats are rejected."""
        with pytest.raises(ValidationError, match="Date format must be YYYY-MM-DD, got:"):
            validate_date_range('01/01/2020', '2023-12-31', 'mean_variance')
        
        with pytest.raises(ValidationError, match="Date format must be YYYY-MM-DD, got:"):
            validate_date_range('2020-Jan-01', '2023-12-31', 'mean_variance')
    
    def test_start_date_before_end_date(self):
        """Test start date must be before end date."""
        with pytest.raises(ValidationError, match="Start date must be before end date"):
            validate_date_range('2023-12-31', '2020-01-01', 'mean_variance')
    
    def test_mean_variance_minimum_3_years(self):
        """Test Mean-Variance requires minimum 3 years."""
        # Valid: exactly 3 years
        validate_date_range('2020-01-01', '2023-01-01', 'mean_variance')
        
        # Invalid: less than 3 years
        with pytest.raises(ValidationError, match="Date range must be at least 3 years for Mean-Variance"):
            validate_date_range('2021-01-01', '2023-06-01', 'mean_variance')
    
    def test_black_litterman_minimum_5_years(self):
        """Test Black-Litterman requires minimum 5 years."""
        # Valid: exactly 5 years
        validate_date_range('2018-01-01', '2023-01-01', 'black_litterman')
        
        # Invalid: less than 5 years
        with pytest.raises(ValidationError, match="Date range must be at least 5 years for Black-Litterman"):
            validate_date_range('2020-01-01', '2023-12-31', 'black_litterman')


class TestAllocationValidation:
    """Test allocation validation rules."""
    
    def test_no_allocation_auto_distribute(self):
        """Test no allocation auto-distributes equally."""
        tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX']
        allocations = {}
        complete = validate_allocations(tickers, allocations)
        
        expected_weight = 1.0 / 3
        for ticker in tickers:
            assert abs(complete[ticker] - expected_weight) < 0.0001
    
    def test_partial_allocation_valid(self):
        """Test partial allocation with remaining distributed."""
        tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX', 'MQG.AX']
        allocations = {'ANZ.AX': 0.3, 'CBA.AX': 0.4}  # 70% allocated
        complete = validate_allocations(tickers, allocations)
        
        assert complete['ANZ.AX'] == 0.3
        assert complete['CBA.AX'] == 0.4
        # Remaining 30% split between NAB.AX and MQG.AX
        assert abs(complete['NAB.AX'] - 0.15) < 0.0001
        assert abs(complete['MQG.AX'] - 0.15) < 0.0001
    
    def test_partial_allocation_exceeds_100_percent(self):
        """Test partial allocation exceeding 100% is rejected."""
        tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX']
        allocations = {'ANZ.AX': 0.7, 'CBA.AX': 0.6}  # 130% total
        
        with pytest.raises(ValidationError, match="Total allocation.*exceeds 100%"):
            validate_allocations(tickers, allocations)
    
    def test_full_allocation_exact_100_percent(self):
        """Test full allocation must sum to exactly 100%."""
        # Valid: exactly 100%
        tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX']
        allocations = {'ANZ.AX': 0.3, 'CBA.AX': 0.4, 'NAB.AX': 0.3}
        complete = validate_allocations(tickers, allocations)
        assert abs(sum(complete.values()) - 1.0) < 0.0001
    
    def test_full_allocation_not_100_percent(self):
        """Test full allocation not summing to 100% is rejected."""
        tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX']
        allocations = {'ANZ.AX': 0.3, 'CBA.AX': 0.4, 'NAB.AX': 0.2}  # 90% total
        
        with pytest.raises(ValidationError, match="Allocation sum must be 100%"):
            validate_allocations(tickers, allocations)
    
    def test_allocation_bounds_0_to_1(self):
        """Test all allocations must be between 0 and 1."""
        tickers = ['ANZ.AX', 'CBA.AX']
        
        # Negative allocation
        with pytest.raises(ValidationError, match="allocation.*not in range"):
            validate_allocations(tickers, {'ANZ.AX': -0.1, 'CBA.AX': 0.5})
        
        # > 100% allocation
        with pytest.raises(ValidationError, match="allocation.*not in range"):
            validate_allocations(tickers, {'ANZ.AX': 1.5, 'CBA.AX': 0.5})
    
    def test_empty_tickers_list(self):
        """Test empty tickers list is rejected."""
        with pytest.raises(ValidationError, match="Tickers list must not be empty"):
            validate_allocations([], {})


class TestConstraintValidation:
    """Test constraint validation rules."""
    
    def test_valid_constraint_bounds(self):
        """Test valid constraint bounds are accepted."""
        tickers = ['ANZ.AX', 'CBA.AX']
        constraints = {'ANZ.AX': (0.2, 0.8), 'CBA.AX': (0.1, 0.9)}
        allocations = {'ANZ.AX': 0.5, 'CBA.AX': 0.5}
        
        complete = validate_constraints(tickers, constraints, allocations)
        assert complete['ANZ.AX'] == (0.2, 0.8)
        assert complete['CBA.AX'] == (0.1, 0.9)
    
    def test_invalid_constraint_bounds(self):
        """Test invalid constraint bounds are rejected."""
        tickers = ['ANZ.AX']
        allocations = {}
        
        # Minimum weight > 1.0
        with pytest.raises(ValidationError, match="minimum weight.*not in range"):
            validate_constraints(tickers, {'ANZ.AX': (1.5, 2.0)}, allocations)
        
        # Minimum > Maximum
        with pytest.raises(ValidationError, match="minimum weight.*> maximum weight"):
            validate_constraints(tickers, {'ANZ.AX': (0.8, 0.2)}, allocations)
    
    def test_allocation_violates_constraints(self):
        """Test allocation violating constraints is rejected."""
        tickers = ['ANZ.AX', 'CBA.AX']
        constraints = {'ANZ.AX': (0.1, 0.3)}  # Max 30%
        allocations = {'ANZ.AX': 0.5, 'CBA.AX': 0.5}  # ANZ.AX has 50%
        
        with pytest.raises(ValidationError, match="allocation.*violates constraint"):
            validate_constraints(tickers, constraints, allocations)
    
    def test_constraints_too_tight_minimum(self):
        """Test constraints too tight on minimum side."""
        tickers = ['ANZ.AX', 'CBA.AX']
        constraints = {'ANZ.AX': (0.6, 1.0), 'CBA.AX': (0.6, 1.0)}  # Min sum = 120%
        allocations = {}
        
        with pytest.raises(ValidationError, match="Equal distribution.*violates.*constraint|Constraint is too tight"):
            validate_constraints(tickers, constraints, allocations)


class TestInvestorViewValidation:
    """Test investor view validation rules."""
    
    def test_valid_investor_views(self):
        """Test valid investor views format."""
        views = {
            'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8},
            'CBA.AX': {'expected_return': 0.15, 'confidence': 0.6}
        }
        tickers = ['ANZ.AX', 'CBA.AX']
        date_range_years = 5.0
        
        validated = validate_investor_views(views, tickers, date_range_years)
        assert 'ANZ.AX' in validated
        assert validated['ANZ.AX']['expected_return'] == 0.12
    
    def test_default_confidence(self):
        """Test default confidence is set when not provided."""
        views = {'ANZ.AX': {'expected_return': 0.12}}  # No confidence
        validated = validate_investor_views(views, [], 5.0)
        assert validated['ANZ.AX']['confidence'] == 0.5
    
    def test_invalid_view_format(self):
        """Test invalid view formats are rejected."""
        # View not a dictionary
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validate_investor_views({'ANZ.AX': 0.12}, [], 5.0)
        
        # Missing expected_return
        with pytest.raises(ValidationError, match="must include 'expected_return'"):
            validate_investor_views({'ANZ.AX': {'confidence': 0.8}}, [], 5.0)
        
        # Invalid confidence range
        with pytest.raises(ValidationError, match="Confidence.*must be between 0 and 1"):
            validate_investor_views({'ANZ.AX': {'expected_return': 0.12, 'confidence': 1.5}}, [], 5.0)
    
    def test_views_require_5_years(self):
        """Test investor views require at least 5 years of data."""
        views = {'ANZ.AX': {'expected_return': 0.12}}
        
        with pytest.raises(ValidationError, match="Date range must be at least 5 years when investor views"):
            validate_investor_views(views, [], 3.0)  # Less than 5 years


class TestRiskFreeRateValidation:
    """Test risk-free rate validation rules."""
    
    def test_valid_risk_free_rate(self):
        """Test valid risk-free rate conversion."""
        # Float
        rate = validate_risk_free_rate(0.025)
        assert rate == 0.025
        
        # String convertible to float
        rate = validate_risk_free_rate("0.03")
        assert rate == 0.03
        
        # Integer
        rate = validate_risk_free_rate(2)
        assert rate == 0.02  # Converted from percentage
    
    def test_percentage_conversion(self):
        """Test percentage values are converted to decimal."""
        rate = validate_risk_free_rate(2.5)  # 2.5%
        assert rate == 0.025
    
    def test_invalid_risk_free_rate(self):
        """Test invalid risk-free rates are rejected."""
        with pytest.raises(ValidationError, match="Risk-free rate must be a number"):
            validate_risk_free_rate("invalid")
        
        with pytest.raises(ValidationError, match="Risk-free rate cannot be negative"):
            validate_risk_free_rate(-0.01)


class TestPortfolioDataIntegration:
    """Test integrated portfolio data validation."""
    
    def test_valid_portfolio_data(self):
        """Test valid complete portfolio data."""
        data = validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            allocations={'ANZ.AX': 0.4, 'CBA.AX': 0.3, 'NAB.AX': 0.3},
            constraints={'ANZ.AX': (0.2, 0.6)},
            risk_free_rate=0.025
        )
        
        assert len(data.tickers) == 3
        assert abs(sum(data.allocations.values()) - 1.0) < 0.0001
        assert data.risk_free_rate == 0.025
        assert data.constraints['ANZ.AX'] == (0.2, 0.6)
        assert data.constraints['CBA.AX'] == (0.0, 1.0)  # Default
    
    def test_black_litterman_with_views(self):
        """Test Black-Litterman with investor views."""
        data = validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX'],
            start_date='2018-01-01',
            end_date='2023-12-31',  # 5+ years
            method='black_litterman',
            investor_views={
                'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8}
            }
        )
        
        assert 'ANZ.AX' in data.investor_views
        assert data.investor_views['ANZ.AX']['expected_return'] == 0.12
    
    def test_error_propagation(self):
        """Test validation errors are properly propagated."""
        # Date range too short
        with pytest.raises(ValidationError, match="Date range must be at least 3 years"):
            validate_portfolio_inputs(
                tickers=['ANZ.AX', 'CBA.AX'],
                start_date='2022-01-01',
                end_date='2023-12-31',
                method='mean_variance'
            )
        
        # Invalid allocation
        with pytest.raises(ValidationError, match="Total allocation.*exceeds 100%"):
            validate_portfolio_inputs(
                tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
                start_date='2020-01-01',
                end_date='2023-12-31',
                method='mean_variance',
                allocations={'ANZ.AX': 0.7, 'CBA.AX': 0.6}  # 130%
            )


def test_validation_examples():
    """Test the example validation scenarios with ASX tickers."""
    print("\n" + "="*60)
    print("PORTFOLIO OPTIMIZATION VALIDATION TESTS (ASX)")
    print("="*60)
    
    # Test 1: Valid Mean-Variance
    try:
        data = validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            allocations={'ANZ.AX': 0.4, 'CBA.AX': 0.6}  # NAB.AX gets 0%
        )
        print("✅ Test 1: Valid Mean-Variance optimization")
        print(f"   Complete allocations: {data.allocations}")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Date range too short for Black-Litterman
    try:
        validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX'],
            start_date='2022-01-01',
            end_date='2023-12-31',
            method='black_litterman',
            investor_views={'ANZ.AX': {'expected_return': 0.12}}
        )
        print("❌ Test 2: Should have failed")
    except ValidationError as e:
        print("✅ Test 2: Correctly caught short date range for Black-Litterman")
        print(f"   Error: {e}")
    
    # Test 3: Allocation exceeding 100%
    try:
        validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'NAB.AX'],
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            allocations={'ANZ.AX': 0.5, 'CBA.AX': 0.6}  # 110% for partial
        )
        print("❌ Test 3: Should have failed")
    except ValidationError as e:
        print("✅ Test 3: Correctly caught allocation exceeding 100%")
        print(f"   Error: {e}")
    
    # Test 4: ASX-specific example with RBA rate
    try:
        data = validate_portfolio_inputs(
            tickers=['ANZ.AX', 'CBA.AX', 'WBC.AX', 'NAB.AX'],  # Big 4 banks
            start_date='2020-01-01',
            end_date='2023-12-31',
            method='mean_variance',
            risk_free_rate=4.35,  # Current RBA cash rate
            constraints={'ANZ.AX': (0.2, 0.3), 'CBA.AX': (0.2, 0.3)}
        )
        print("✅ Test 4: ASX Big 4 Banks portfolio with RBA rate")
        print(f"   Risk-free rate: {data.risk_free_rate}")
        print(f"   Allocations: {data.allocations}")
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    print("\n✅ All validation tests completed successfully!")


if __name__ == "__main__":
    test_validation_examples()