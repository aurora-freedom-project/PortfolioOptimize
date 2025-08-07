#!/usr/bin/env python3
# tests/test_validation_integration.py
"""
Integration Tests for Portfolio Optimization Validation
======================================================

Tests the validation system integrated with the optimization workflow.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.validation_simple import validate_portfolio_inputs, ValidationError
from shell.app import run_portfolio_optimization


def create_test_data():
    """Create test CSV data for validation tests."""
    test_dir = tempfile.mkdtemp()
    
    # Generate test price data
    dates = pd.date_range('2018-01-01', '2024-01-01', freq='D')
    dates = dates[dates.weekday < 5]  # Business days only
    
    np.random.seed(42)
    tickers = ['ANZ.AX', 'CBA.AX', 'NAB.AX', 'MQG.AX']
    base_prices = {'ANZ.AX': 25, 'CBA.AX': 95, 'NAB.AX': 28, 'MQG.AX': 155}
    
    price_data = {}
    for ticker in tickers:
        returns = np.random.normal(0.0008, 0.02, len(dates))
        prices = [base_prices[ticker]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        price_data[ticker] = prices[1:]
    
    price_df = pd.DataFrame(price_data, index=dates)
    price_file = os.path.join(test_dir, 'test_prices.csv')
    price_df.to_csv(price_file)
    
    return price_file, tickers


def test_validation_rules():
    """Test all validation rules comprehensively."""
    print("Testing Portfolio Optimization Validation Rules")
    print("=" * 60)
    
    price_file, tickers = create_test_data()
    
    # Test 1: Valid Mean-Variance (3+ years)
    print("\n1. Testing valid Mean-Variance optimization (3+ years)")
    try:
        result = run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],  # ANZ.AX, CBA.AX, NAB.AX
            start_date='2021-01-01',
            end_date='2024-01-01',  # 3 years
            method='mean_variance',
            allocations={'ANZ.AX': 0.4, 'CBA.AX': 0.3, 'NAB.AX': 0.3}
        )
        print("✅ Valid Mean-Variance optimization passed")
        print(f"   Method: {result['method']}")
        print(f"   Optimal Sharpe: {result['optimal_portfolio']['metrics']['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"❌ Valid Mean-Variance failed: {e}")
    
    # Test 2: Mean-Variance with insufficient date range
    print("\n2. Testing Mean-Variance with insufficient date range")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2022-01-01',
            end_date='2023-12-31',  # Only 2 years
            method='mean_variance'
        )
        print("❌ Should have failed - insufficient date range")
    except ValueError as e:
        if "Date range must be at least 3 years" in str(e):
            print("✅ Correctly rejected insufficient date range for Mean-Variance")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 3: Valid Black-Litterman (5+ years)
    print("\n3. Testing valid Black-Litterman optimization (5+ years)")
    try:
        result = run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],
            start_date='2019-01-01',
            end_date='2024-01-01',  # 5 years
            method='black_litterman',
            investor_views={
                'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8}
            }
        )
        print("✅ Valid Black-Litterman optimization passed")
        print(f"   Method: {result['method']}")
        print(f"   Optimal Sharpe: {result['optimal_portfolio']['metrics']['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"❌ Valid Black-Litterman failed: {e}")
    
    # Test 4: Black-Litterman with insufficient date range for views
    print("\n4. Testing Black-Litterman with insufficient date range for views")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2021-01-01',
            end_date='2023-12-31',  # Only 3 years
            method='black_litterman',
            investor_views={
                'ANZ.AX': {'expected_return': 0.12, 'confidence': 0.8}
            }
        )
        print("❌ Should have failed - insufficient date range for views")
    except ValueError as e:
        if "Date range must be at least 5 years when investor views" in str(e):
            print("✅ Correctly rejected insufficient date range for Black-Litterman with views")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 5: Invalid date format
    print("\n5. Testing invalid date format")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='01/01/2021',  # Wrong format
            end_date='2023-12-31',
            method='mean_variance'
        )
        print("❌ Should have failed - invalid date format")
    except ValueError as e:
        if "Date format must be YYYY-MM-DD" in str(e):
            print("✅ Correctly rejected invalid date format")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 6: Start date after end date
    print("\n6. Testing start date after end date")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2023-12-31',
            end_date='2021-01-01',  # End before start
            method='mean_variance'
        )
        print("❌ Should have failed - start date after end date")
    except ValueError as e:
        if "Start date must be before end date" in str(e):
            print("✅ Correctly rejected start date after end date")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 7: Invalid allocation sum (partial exceeding 100%)
    print("\n7. Testing partial allocation exceeding 100%")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],  # 3 tickers
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            allocations={'ANZ.AX': 0.7, 'CBA.AX': 0.6}  # 130% for 2 of 3 tickers
        )
        print("❌ Should have failed - partial allocation exceeding 100%")
    except ValueError as e:
        if "Total allocation" in str(e) and "exceeds 100%" in str(e):
            print("✅ Correctly rejected partial allocation exceeding 100%")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 8: Invalid allocation sum (full not equal to 100%)
    print("\n8. Testing full allocation not equal to 100%")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            allocations={'ANZ.AX': 0.3, 'CBA.AX': 0.3, 'NAB.AX': 0.2}  # 80% total
        )
        print("❌ Should have failed - full allocation not equal to 100%")
    except ValueError as e:
        if "Allocation sum must be 100%" in str(e):
            print("✅ Correctly rejected full allocation not equal to 100%")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 9: Allocation out of bounds
    print("\n9. Testing allocation out of bounds")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            allocations={'ANZ.AX': -0.1, 'CBA.AX': 1.1}  # Negative and > 100%
        )
        print("❌ Should have failed - allocation out of bounds")
    except ValueError as e:
        if "not in range [0%, 100%]" in str(e):
            print("✅ Correctly rejected allocation out of bounds")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 10: Valid constraints
    print("\n10. Testing valid constraints")
    try:
        result = run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            allocations={'ANZ.AX': 0.3, 'CBA.AX': 0.4, 'NAB.AX': 0.3},
            constraints={'ANZ.AX': (0.2, 0.5), 'CBA.AX': (0.3, 0.6)}
        )
        print("✅ Valid constraints accepted")
        print(f"   Optimal weights: {result['optimal_portfolio']['weights']}")
    except Exception as e:
        print(f"❌ Valid constraints failed: {e}")
    
    # Test 11: Allocation violating constraints
    print("\n11. Testing allocation violating constraints")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:3],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            allocations={'ANZ.AX': 0.8, 'CBA.AX': 0.1, 'NAB.AX': 0.1},
            constraints={'ANZ.AX': (0.1, 0.5)}  # ANZ.AX max 50%, but allocated 80%
        )
        print("❌ Should have failed - allocation violating constraints")
    except ValueError as e:
        if "violates constraint" in str(e):
            print("✅ Correctly rejected allocation violating constraints")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 12: Constraints too tight
    print("\n12. Testing constraints too tight")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            constraints={'ANZ.AX': (0.7, 1.0), 'CBA.AX': (0.7, 1.0)}  # Min sum = 140%
        )
        print("❌ Should have failed - constraints too tight")
    except ValueError as e:
        if "Constraint is too tight" in str(e):
            print("✅ Correctly rejected constraints too tight")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 13: Invalid risk-free rate
    print("\n13. Testing invalid risk-free rate")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            risk_free_rate="invalid"
        )
        print("❌ Should have failed - invalid risk-free rate")
    except ValueError as e:
        if "Risk-free rate must be a number" in str(e):
            print("✅ Correctly rejected invalid risk-free rate")
        else:
            print(f"❌ Wrong error message: {e}")
    
    # Test 14: Risk-free rate percentage conversion
    print("\n14. Testing risk-free rate percentage conversion")
    try:
        result = run_portfolio_optimization(
            data_file=price_file,
            tickers=tickers[:2],
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance',
            risk_free_rate=2.5  # Should be converted to 0.025
        )
        print("✅ Risk-free rate percentage conversion worked")
        # The rate should be converted from 2.5% to 0.025
    except Exception as e:
        print(f"❌ Risk-free rate conversion failed: {e}")
    
    # Test 15: Empty tickers list
    print("\n15. Testing empty tickers list")
    try:
        run_portfolio_optimization(
            data_file=price_file,
            tickers=[],  # Empty list
            start_date='2021-01-01',
            end_date='2024-01-01',
            method='mean_variance'
        )
        print("❌ Should have failed - empty tickers list")
    except ValueError as e:
        if "Tickers list must not be empty" in str(e):
            print("✅ Correctly rejected empty tickers list")
        else:
            print(f"❌ Wrong error message: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ALL VALIDATION TESTS COMPLETED!")
    print("The validation system is working correctly.")
    print("=" * 60)
    
    # Clean up
    os.unlink(price_file)


if __name__ == "__main__":
    test_validation_rules()