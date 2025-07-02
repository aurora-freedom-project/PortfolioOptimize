#!/usr/bin/env python3
"""
Comprehensive test scenarios for portfolio optimizer validation
"""

from streamlined_optimizer import create_optimizer
import json

def test_scenario(name, portfolio_data, should_pass=True):
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"{'='*60}")
    
    # Create unique chart prefix for this test
    chart_prefix = f"{name.replace(' ', '_').replace('-', '').lower()}_"
    optimizer = create_optimizer(chart_prefix=chart_prefix)
    try:
        result = optimizer.optimize_portfolio(portfolio_data)
        if should_pass:
            print("✅ PASS - Optimization completed successfully")
            print(f"📊 Charts: {list(result['charts'].keys())}")
            
            # Display key results
            if "mean_variance" in result:
                mv_weights = result["mean_variance"]["optimal_weights"]
                print("  MV Weights:", {k: f"{v:.3f}" for k, v in mv_weights.items() if v > 0.001})
            
            if "black_litterman" in result:
                bl_weights = result["black_litterman"]["optimal_weights"]
                print("  BL Weights:", {k: f"{v:.3f}" for k, v in bl_weights.items() if v > 0.001})
            
            # Save results for successful cases
            filename = f"test_results_{name.replace(' ', '_').replace('-', '').lower()}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {filename}")
            
            return True
        else:
            print("❌ FAIL - Expected error but got success")
            return False
    except ValueError as e:
        if not should_pass:
            print(f"✅ PASS - Expected error: {e}")
            return True
        else:
            print(f"❌ FAIL - Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ ERROR - Unexpected exception: {e}")
        return False

def main():
    print("🎯 Portfolio Optimizer - Comprehensive Test Scenarios")
    
    base_tickers = ["ANZ.AX", "CBA.AX", "MQG.AX", "NAB.AX"]
    
    # HAPPY CASES
    scenarios = [
        # Case 1: No allocation (auto equal distribution)
        ("Case 1 - No Allocation (Auto Equal)", {
            "tickers": base_tickers,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "risk_free_rate": 0.02
        }, True),
        
        # Case 2: Partial allocation
        ("Case 2 - Partial Allocation", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.3, "CBA.AX": 0.2},  # 50% remaining for others
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, True),
        
        # Case 3: Full allocation
        ("Case 3 - Full Allocation", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.25, "CBA.AX": 0.25, "MQG.AX": 0.25, "NAB.AX": 0.25},
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, True),
        
        # Case 4: With constraints (valid)
        ("Case 4 - Valid Constraints", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.3, "CBA.AX": 0.3, "MQG.AX": 0.2, "NAB.AX": 0.2},
            "constraints": {
                "ANZ.AX": {"min": 0.2, "max": 0.4},
                "CBA.AX": {"min": 0.1, "max": 0.5}
            },
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, True),
        
        # Case 5: Black-Litterman with 5+ years
        ("Case 5 - Black-Litterman Valid", {
            "tickers": base_tickers,
            "investor_views": {
                "ANZ.AX": {"expected_return": 0.08, "confidence": 0.7}
            },
            "start_date": "2018-01-01",
            "end_date": "2024-12-31"
        }, True),
        
        # Case 6: Risk-free rate as integer (small value)
        ("Case 6 - Risk-Free Rate as Integer", {
            "tickers": base_tickers,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "risk_free_rate": 0  # Integer 0 should convert to 0.0 float
        }, True),
    ]
    
    # EDGE CASES (Should Fail)
    edge_cases = [
        # Date range too short for Mean-Variance
        ("Edge 1 - Date Range Too Short (MV)", {
            "tickers": base_tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-12-31"  # Only 2 years
        }, False),
        
        # Date range too short for Black-Litterman
        ("Edge 2 - Date Range Too Short (BL)", {
            "tickers": base_tickers,
            "investor_views": {"ANZ.AX": {"expected_return": 0.08, "confidence": 0.7}},
            "start_date": "2021-01-01",
            "end_date": "2024-12-31"  # Only 4 years
        }, False),
        
        # Allocation sum > 100%
        ("Edge 3 - Allocation Sum > 100%", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.6, "CBA.AX": 0.5},  # 110% total
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False),
        
        # Full allocation sum ≠ 100%
        ("Edge 4 - Full Allocation ≠ 100%", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.3, "CBA.AX": 0.3, "MQG.AX": 0.2, "NAB.AX": 0.15},  # 95%
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False),
        
        # Constraint violation - existing allocation
        ("Edge 5 - Constraint Violation (Existing)", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.5},  # 50% violates max 40%
            "constraints": {"ANZ.AX": {"min": 0.1, "max": 0.4}},
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False),
        
        # Constraint violation - equal distribution
        ("Edge 6 - Constraint Violation (Equal)", {
            "tickers": base_tickers,  # 25% each violates min 30%
            "constraints": {"ANZ.AX": {"min": 0.3, "max": 0.5}},
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False),
        
        # Constraint violation - remaining allocation
        ("Edge 7 - Constraint Violation (Remaining)", {
            "tickers": base_tickers,
            "allocations": {"ANZ.AX": 0.2, "CBA.AX": 0.2},  # 60% remaining = 30% each for others
            "constraints": {"MQG.AX": {"min": 0.1, "max": 0.25}},  # 30% > 25% max
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False),
        
        # Constraints too tight for optimization
        ("Edge 8 - Constraints Too Tight", {
            "tickers": base_tickers,
            "constraints": {
                "ANZ.AX": {"min": 0.4, "max": 0.45},
                "CBA.AX": {"min": 0.4, "max": 0.45},
                "MQG.AX": {"min": 0.4, "max": 0.45},
                "NAB.AX": {"min": 0.4, "max": 0.45}
            },
            "start_date": "2020-01-01",
            "end_date": "2024-12-31"
        }, False)
    ]
    
    # Run all scenarios
    all_scenarios = scenarios + edge_cases
    passed = 0
    total = len(all_scenarios)
    
    for name, data, should_pass in all_scenarios:
        if test_scenario(name, data, should_pass):
            passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY: {passed}/{total} scenarios passed")
    print(f"💾 Generated {passed} JSON result files")
    print(f"📈 Generated HTML chart files in charts/ directory")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    main()