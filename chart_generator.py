#!/usr/bin/env python3
"""
Standalone chart generator for portfolio optimization results.
Can be used to generate charts from existing JSON result files.
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for chart generator."""
    parser = argparse.ArgumentParser(description='Generate charts from portfolio optimization results')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to JSON results file')
    parser.add_argument('--output-dir', '-o', type=str, default='charts',
                        help='Output directory for chart files (default: charts)')
    parser.add_argument('--chart-type', '-t', type=str, 
                        choices=['all', 'frontier', 'weights', 'correlation', 'metrics'],
                        default='all',
                        help='Type of chart to generate (default: all)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    try:
        from shell.display import load_results_from_json, PortfolioVisualizer
        
        # Load results
        print(f"📂 Loading results from {args.input}...")
        results = load_results_from_json(args.input)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = PortfolioVisualizer()
        
        print(f"📊 Generating charts in {args.output_dir}...")
        
        # Generate requested charts
        if args.chart_type == 'all':
            visualizer.generate_all_charts(results, str(output_dir))
        elif args.chart_type == 'frontier':
            print("📈 Generating Efficient Frontier chart...")
            visualizer.plot_efficient_frontier(results, str(output_dir / "efficient_frontier.png"))
        elif args.chart_type == 'weights':
            print("🥧 Generating Portfolio Weights chart...")
            visualizer.plot_portfolio_weights(results, str(output_dir / "portfolio_weights.png"))
        elif args.chart_type == 'correlation':
            print("🔗 Generating Correlation Matrix chart...")
            visualizer.plot_correlation_matrix(results, str(output_dir / "correlation_matrix.png"))
        elif args.chart_type == 'metrics':
            print("📊 Generating Risk-Return Metrics chart...")
            visualizer.plot_risk_return_metrics(results, str(output_dir / "risk_return_metrics.png"))
        
        print("✅ Chart generation completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error: Missing required packages: {e}")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()