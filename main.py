# main.py
import json
import sys
from datetime import datetime

def main():
    """Main entry point for portfolio optimization CLI."""
    from shell.cli import parse_args, parse_tickers, parse_allocations, parse_constraints, parse_investor_views, display_portfolio_results
    from shell.app import run_portfolio_optimization
    
    # Parse command line arguments
    args = parse_args()
    
    # Parse tickers, allocations, and constraints
    tickers = parse_tickers(args.tickers)
    allocations = parse_allocations(args.allocations, tickers)
    constraints = parse_constraints(args.constraints, tickers)
    investor_views = parse_investor_views(args.views, tickers)
    
    try:
        # Run portfolio optimization
        results = run_portfolio_optimization(
            args.data,
            tickers,
            args.start_date,
            args.end_date,
            args.method,
            allocations,
            constraints,
            args.risk_free_rate,
            investor_views
        )
        
        # Display results
        display_portfolio_results(results)
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()