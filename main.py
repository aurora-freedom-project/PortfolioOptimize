# main.py
import json
import sys
from datetime import datetime

def main():
    """Main entry point for portfolio optimization CLI."""
    from shell.cli import parse_args, parse_tickers, parse_allocations, parse_constraints, parse_investor_views, display_portfolio_results
    from shell.app import run_portfolio_optimization
    from shell.display import show_charts_from_results, show_charts_from_json
    
    # Parse command line arguments
    args = parse_args()
    
    # Handle charts-from-json mode
    if args.charts_from_json:
        try:
            print(f"📂 Loading results from {args.charts_from_json}...")
            output_dir = f"{args.charts_from_json.rsplit('.', 1)[0]}_charts"
            show_charts_from_json(args.charts_from_json, output_dir)
            return
        except Exception as e:
            print(f"❌ Error loading charts from JSON: {e}")
            sys.exit(1)
    
    # Validate required arguments for optimization
    if not args.tickers or not args.start_date or not args.end_date:
        print("❌ Error: --tickers, --start-date, and --end-date are required for portfolio optimization")
        print("Use --charts-from-json to generate charts from existing results")
        sys.exit(1)
    
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
        
        # Show charts if requested
        if args.show_charts:
            try:
                print("\n🔄 Generating portfolio analysis charts...")
                output_dir = "charts" if not args.output else f"{args.output.rsplit('.', 1)[0]}_charts"
                show_charts_from_results(results, output_dir)
            except Exception as chart_error:
                print(f"⚠️  Warning: Could not generate charts: {chart_error}")
                print("Charts require matplotlib and seaborn. Install with: pip install matplotlib seaborn")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()