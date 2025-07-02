#!/usr/bin/env python3
"""
Simple example to run the streamlined optimizer with specific charts
"""

from streamlined_optimizer import create_optimizer
import json

def main():
    print("🎯 Streamlined Portfolio Optimizer - Specific Charts")
    print("=" * 55)
    
    # Portfolio specification with all required data
    portfolio_data = {
        "tickers": ["ANZ.AX", "CBA.AX", "MQG.AX", "NAB.AX", "RIO.AX", "WOW.AX"],
        
        # Provided allocation for Mean-Variance pie chart comparison
        "allocations": {
            "ANZ.AX": 0.15,
            "CBA.AX": 0.25,
            "MQG.AX": 0.15,
            "NAB.AX": 0.15,
            "RIO.AX": 0.20,
            "WOW.AX": 0.10
        },
        
        # Investor views for Black-Litterman
        "investor_views": {
            "CBA.AX": {"expected_return": 0.10, "confidence": 0.8},
            "RIO.AX": {"expected_return": 0.06, "confidence": 0.6}
        },
        
        "start_date": "2022-01-01",
        "end_date": "2024-12-31"
    }
    
    # Run optimization
    optimizer = create_optimizer()
    result = optimizer.optimize_portfolio(portfolio_data)
    
    if result["status"] == "success":
        print("✅ Optimization completed successfully!")
        
        # Display chart information
        print("\n📊 Generated Charts:")
        for chart_name, file_path in result["charts"].items():
            print(f"  {chart_name}: {file_path}")
        
        # Display key results
        print("\n🎯 Optimization Results:")
        
        # Mean-Variance weights
        if "mean_variance" in result:
            mv_weights = result["mean_variance"]["optimal_weights"]
            print("\nMean-Variance Optimal Weights:")
            for ticker, weight in sorted(mv_weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.001:
                    print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        # Black-Litterman weights
        if "black_litterman" in result:
            bl_weights = result["black_litterman"]["optimal_weights"]
            print("\nBlack-Litterman Optimal Weights:")
            for ticker, weight in sorted(bl_weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.001:
                    print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        # HRP removed from this version
        
        # Save complete results
        with open("optimization_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Complete results saved to: optimization_results.json")
        print("🌐 Open HTML chart files in your browser to view visualizations")
        
    else:
        print(f"❌ Optimization failed: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()