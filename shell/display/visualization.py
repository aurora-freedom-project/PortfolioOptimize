# shell/display/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

class PortfolioVisualizer:
    """Portfolio analysis visualization suite using matplotlib and seaborn."""
    
    def __init__(self, figsize: tuple = (12, 8), style: str = 'whitegrid'):
        """Initialize visualizer with default settings."""
        self.figsize = figsize
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_style(style)
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def plot_efficient_frontier(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot the efficient frontier with portfolio positions."""
        if 'efficient_frontier' not in results:
            print("No efficient frontier data available for plotting")
            return
            
        frontier_data = results['efficient_frontier']
        if not frontier_data:
            print("Empty efficient frontier data")
            return
            
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data for plotting
        returns = [p['expected_return'] * 100 for p in frontier_data]
        volatilities = [p['standard_deviation'] * 100 for p in frontier_data]
        sharpe_ratios = [p['sharpe_ratio'] for p in frontier_data]
        
        # Sort by volatility for proper line plotting
        sorted_data = sorted(zip(volatilities, returns, sharpe_ratios), key=lambda x: x[0])
        vol_sorted, ret_sorted, sharpe_sorted = zip(*sorted_data)
        
        # Plot efficient frontier as a line for better envelope visualization
        ax.plot(vol_sorted, ret_sorted, 'b-', linewidth=2, alpha=0.8, label='Efficient Frontier')
        
        # Plot efficient frontier points colored by Sharpe ratio
        scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, 
                           cmap='viridis', s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Highlight special portfolios
        for i, portfolio in enumerate(frontier_data):
            if portfolio.get('is_max_sharpe', False):
                ax.scatter(volatilities[i], returns[i], 
                          marker='*', s=400, c='red', label='Max Sharpe Portfolio', 
                          edgecolors='black', linewidth=2, zorder=5)
            elif portfolio.get('is_min_volatility', False):
                ax.scatter(volatilities[i], returns[i], 
                          marker='s', s=200, c='green', label='Min Volatility Portfolio', 
                          edgecolors='black', linewidth=1, zorder=5)
        
        # Plot provided portfolio if available
        if 'provided_portfolio' in results:
            provided_metrics = results['provided_portfolio']['metrics']
            provided_ret = provided_metrics['expected_return'] * 100
            provided_vol = provided_metrics['standard_deviation'] * 100
            ax.scatter(provided_vol, provided_ret, marker='D', s=250, 
                      c='orange', label='Current Portfolio', 
                      edgecolors='black', linewidth=2, zorder=4)
        
        # Add method information to title
        method = results.get('method', 'Unknown')
        method_names = {
            'MEAN_VARIANCE': 'Mean-Variance',
            'BLACK_LITTERMAN': 'Black-Litterman', 
            'HIERARCHICAL_RISK_PARITY': 'Hierarchical Risk Parity'
        }
        method_name = method_names.get(method, method)
        
        # Formatting
        ax.set_xlabel('Risk (Standard Deviation %)', fontsize=12)
        ax.set_ylabel('Expected Return (%)', fontsize=12)
        ax.set_title(f'Efficient Frontier Analysis - {method_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', fontsize=11)
        
        # Add statistics text
        if frontier_data:
            stats_text = f'Portfolios: {len(frontier_data)}\n'
            max_sharpe = max(sharpe_ratios)
            min_vol = min(volatilities)
            stats_text += f'Max Sharpe: {max_sharpe:.3f}\n'
            stats_text += f'Min Vol: {min_vol:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Efficient frontier chart saved to {save_path}")
        
        plt.show()
    
    def plot_portfolio_weights(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot portfolio weight comparisons."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Current portfolio weights
        if 'provided_portfolio' in results:
            provided_weights = results['provided_portfolio']['weights']
            tickers = list(provided_weights.keys())
            weights = [provided_weights[ticker] * 100 for ticker in tickers]
            
            colors1 = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
            wedges1, texts1, autotexts1 = ax1.pie(weights, labels=tickers, autopct='%1.1f%%',
                                                  colors=colors1, startangle=90)
            ax1.set_title('Current Portfolio Allocation', fontsize=14, fontweight='bold', pad=20)
        
        # Optimal portfolio weights
        if 'optimal_portfolio' in results:
            optimal_weights = results['optimal_portfolio']['weights']
            tickers = list(optimal_weights.keys())
            weights = [optimal_weights[ticker] * 100 for ticker in tickers]
            
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
            wedges2, texts2, autotexts2 = ax2.pie(weights, labels=tickers, autopct='%1.1f%%',
                                                  colors=colors2, startangle=90)
            ax2.set_title('Optimal Portfolio Allocation', fontsize=14, fontweight='bold', pad=20)
        
        # Styling
        for autotext in autotexts1 + autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio weights chart saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap."""
        if 'correlation_matrix' not in results:
            print("No correlation matrix data available for plotting")
            return
            
        correlation_data = results['correlation_matrix']
        
        # Convert to DataFrame for easier plotting
        df_corr = pd.DataFrame(correlation_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        sns.heatmap(df_corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, 
                   fmt='.2f', ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Assets', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix chart saved to {save_path}")
        
        plt.show()
    
    def plot_risk_return_metrics(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot risk-return metrics comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        provided_metrics = results.get('provided_portfolio', {}).get('metrics', {})
        optimal_metrics = results.get('optimal_portfolio', {}).get('metrics', {})
        
        if not provided_metrics or not optimal_metrics:
            print("Insufficient metrics data for comparison plot")
            return
        
        portfolios = ['Current', 'Optimal']
        
        # Expected Returns
        returns = [provided_metrics.get('expected_return', 0) * 100,
                  optimal_metrics.get('expected_return', 0) * 100]
        ax1.bar(portfolios, returns, color=['orange', 'green'], alpha=0.7)
        ax1.set_title('Expected Annual Returns', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        for i, v in enumerate(returns):
            ax1.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Risk (Standard Deviation)
        risks = [provided_metrics.get('standard_deviation', 0) * 100,
                optimal_metrics.get('standard_deviation', 0) * 100]
        ax2.bar(portfolios, risks, color=['orange', 'green'], alpha=0.7)
        ax2.set_title('Risk (Standard Deviation)', fontweight='bold')
        ax2.set_ylabel('Standard Deviation (%)')
        for i, v in enumerate(risks):
            ax2.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # Sharpe Ratio
        sharpe_ratios = [provided_metrics.get('sharpe_ratio', 0),
                        optimal_metrics.get('sharpe_ratio', 0)]
        ax3.bar(portfolios, sharpe_ratios, color=['orange', 'green'], alpha=0.7)
        ax3.set_title('Sharpe Ratio', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        for i, v in enumerate(sharpe_ratios):
            ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Sortino Ratio (if available)
        if 'sortino_ratio' in provided_metrics and 'sortino_ratio' in optimal_metrics:
            sortino_ratios = [provided_metrics.get('sortino_ratio', 0),
                             optimal_metrics.get('sortino_ratio', 0)]
            ax4.bar(portfolios, sortino_ratios, color=['orange', 'green'], alpha=0.7)
            ax4.set_title('Sortino Ratio', fontweight='bold')
            ax4.set_ylabel('Sortino Ratio')
            for i, v in enumerate(sortino_ratios):
                ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Sortino Ratio\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14, alpha=0.5)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk-return metrics chart saved to {save_path}")
        
        plt.show()
    
    def generate_all_charts(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
        """Generate all available charts for the portfolio analysis."""
        print("\n📊 Generating Portfolio Analysis Charts...")
        
        save_dir = None
        if output_dir:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Efficient Frontier
            print("📈 Plotting Efficient Frontier...")
            frontier_path = save_dir / "efficient_frontier.png" if save_dir else None
            self.plot_efficient_frontier(results, str(frontier_path) if frontier_path else None)
            
            # Portfolio Weights
            print("🥧 Plotting Portfolio Allocations...")
            weights_path = save_dir / "portfolio_weights.png" if save_dir else None
            self.plot_portfolio_weights(results, str(weights_path) if weights_path else None)
            
            # Correlation Matrix
            print("🔗 Plotting Correlation Matrix...")
            corr_path = save_dir / "correlation_matrix.png" if save_dir else None
            self.plot_correlation_matrix(results, str(corr_path) if corr_path else None)
            
            # Risk-Return Metrics
            print("📊 Plotting Risk-Return Metrics...")
            metrics_path = save_dir / "risk_return_metrics.png" if save_dir else None
            self.plot_risk_return_metrics(results, str(metrics_path) if metrics_path else None)
            
            print("✅ All charts generated successfully!")
            
        except Exception as e:
            print(f"❌ Error generating charts: {e}")


def load_results_from_json(json_path: str) -> Dict[str, Any]:
    """Load portfolio optimization results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")


def show_charts_from_json(json_path: str, output_dir: Optional[str] = None) -> None:
    """Load results from JSON and display all charts."""
    results = load_results_from_json(json_path)
    visualizer = PortfolioVisualizer()
    visualizer.generate_all_charts(results, output_dir)


def show_charts_from_results(results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """Display all charts from results dictionary."""
    visualizer = PortfolioVisualizer()
    visualizer.generate_all_charts(results, output_dir)