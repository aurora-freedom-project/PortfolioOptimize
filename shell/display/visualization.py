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
    
    def plot_black_litterman_analysis(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot Black-Litterman specific analysis (Prior vs Posterior vs Views)."""
        if results.get('method') != 'BLACK_LITTERMAN':
            return
            
        bl_info = results.get('black_litterman_info', {})
        if not bl_info.get('adjusted', False):
            print("No Black-Litterman views applied - skipping BL analysis chart")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get data
        historical = bl_info.get('historical_returns', {})
        prior = bl_info.get('prior_returns', {})
        posterior = bl_info.get('posterior_returns', {})
        views = bl_info.get('views', {})
        
        if not all([historical, prior, posterior]):
            print("Insufficient Black-Litterman data for analysis chart")
            return
            
        tickers = list(historical.keys())
        
        # Plot 1: Returns Comparison (Prior vs Posterior vs Historical)
        x = np.arange(len(tickers))
        width = 0.25
        
        hist_returns = [historical[ticker] * 100 for ticker in tickers]
        prior_returns = [prior[ticker] * 100 for ticker in tickers]
        posterior_returns = [posterior[ticker] * 100 for ticker in tickers]
        
        ax1.bar(x - width, hist_returns, width, label='Historical Returns', alpha=0.8, color='skyblue')
        ax1.bar(x, prior_returns, width, label='Prior (Market-Implied)', alpha=0.8, color='orange')
        ax1.bar(x + width, posterior_returns, width, label='Posterior (Adjusted)', alpha=0.8, color='green')
        
        ax1.set_xlabel('Assets')
        ax1.set_ylabel('Expected Return (%)')
        ax1.set_title('Black-Litterman Returns Analysis', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: View Impact Analysis
        view_impact = []
        view_labels = []
        colors = []
        
        for ticker in tickers:
            if ticker in views:
                # Calculate impact: posterior - prior
                impact = (posterior[ticker] - prior[ticker]) * 100
                view_impact.append(impact)
                view_labels.append(f"{ticker}\n({views[ticker]*100:+.1f}% view)")
                colors.append('green' if impact > 0 else 'red')
            else:
                view_impact.append(0)
                view_labels.append(f"{ticker}\n(no view)")
                colors.append('gray')
        
        bars = ax2.bar(range(len(tickers)), view_impact, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Return Adjustment (%)')
        ax2.set_title('Impact of Investor Views', fontweight='bold')
        ax2.set_xticks(range(len(tickers)))
        ax2.set_xticklabels(view_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, view_impact):
            if abs(value) > 0.01:  # Only show significant impacts
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if value > 0 else -0.15),
                        f'{value:+.2f}%', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Black-Litterman analysis chart saved to {save_path}")
        
        plt.show()
    
    def plot_hrp_analysis(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot HRP specific analysis (Dendrogram and Risk Contribution)."""
        if results.get('method') != 'HIERARCHICAL_RISK_PARITY':
            return
            
        clusters = results.get('clusters', [])
        weights = results.get('optimal_portfolio', {}).get('weights', {})
        correlation_matrix = results.get('correlation_matrix', {})
        
        if not all([clusters, weights, correlation_matrix]):
            print("Insufficient HRP data for analysis chart")
            return
            
        # Import scipy for dendrogram
        try:
            from scipy.cluster.hierarchy import dendrogram
            from scipy.spatial.distance import squareform
        except ImportError:
            print("scipy required for HRP dendrogram - install with: pip install scipy")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        tickers = list(weights.keys())
        
        # Plot 1: Hierarchical Clustering Dendrogram
        try:
            # Convert correlation to distance matrix
            corr_df = pd.DataFrame(correlation_matrix)
            distance_matrix = np.sqrt(0.5 * (1 - corr_df.values))
            
            # Create dendrogram
            dendro = dendrogram(clusters, labels=tickers, ax=ax1, leaf_rotation=90, leaf_font_size=10)
            ax1.set_title('Asset Clustering Hierarchy', fontweight='bold')
            ax1.set_ylabel('Distance')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Dendrogram Error:\n{str(e)}', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Asset Clustering Hierarchy (Error)', fontweight='bold')
        
        # Plot 2: Portfolio Weights
        weight_values = [weights[ticker] * 100 for ticker in tickers]
        colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
        
        bars = ax2.bar(tickers, weight_values, color=colors, alpha=0.8)
        ax2.set_title('HRP Asset Allocation', fontweight='bold')
        ax2.set_ylabel('Weight (%)')
        ax2.set_xlabel('Assets')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, weight_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Risk Contribution Estimate
        # Calculate simple risk contribution based on weights and volatility
        corr_df = pd.DataFrame(correlation_matrix)
        volatilities = np.sqrt(np.diag(corr_df.values))
        risk_contrib = []
        
        for i, ticker in enumerate(tickers):
            weight = weights[ticker]
            vol = volatilities[i] if i < len(volatilities) else 0.2  # fallback
            contrib = weight * vol
            risk_contrib.append(contrib * 100)
        
        ax3.pie(risk_contrib, labels=tickers, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Estimated Risk Contribution', fontweight='bold')
        
        # Plot 4: Diversification Analysis
        # Calculate concentration metrics
        weight_values_norm = np.array(weight_values) / 100
        herfindahl_index = np.sum(weight_values_norm ** 2)
        effective_assets = 1 / herfindahl_index
        max_weight = np.max(weight_values_norm) * 100
        
        metrics = ['Herfindahl\nIndex', 'Effective\nAssets', 'Max Weight\n(%)', 'Number of\nAssets']
        values = [herfindahl_index, effective_assets, max_weight, len(tickers)]
        
        bars = ax4.bar(metrics, values, color=['red', 'green', 'orange', 'blue'], alpha=0.7)
        ax4.set_title('Diversification Metrics', fontweight='bold')
        ax4.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"HRP analysis chart saved to {save_path}")
        
        plt.show()

    def generate_all_charts(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
        """Generate all available charts for the portfolio analysis."""
        print("\n📊 Generating Portfolio Analysis Charts...")
        
        save_dir = None
        if output_dir:
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        method = results.get('method', 'Unknown')
        
        try:
            # Core Charts (for all methods)
            print("📈 Plotting Efficient Frontier...")
            frontier_path = save_dir / "efficient_frontier.png" if save_dir else None
            self.plot_efficient_frontier(results, str(frontier_path) if frontier_path else None)
            
            print("🥧 Plotting Portfolio Allocations...")
            weights_path = save_dir / "portfolio_weights.png" if save_dir else None
            self.plot_portfolio_weights(results, str(weights_path) if weights_path else None)
            
            print("🔗 Plotting Correlation Matrix...")
            corr_path = save_dir / "correlation_matrix.png" if save_dir else None
            self.plot_correlation_matrix(results, str(corr_path) if corr_path else None)
            
            print("📊 Plotting Risk-Return Metrics...")
            metrics_path = save_dir / "risk_return_metrics.png" if save_dir else None
            self.plot_risk_return_metrics(results, str(metrics_path) if metrics_path else None)
            
            # Method-Specific Charts
            if method == 'BLACK_LITTERMAN':
                print("🎯 Plotting Black-Litterman Analysis...")
                bl_path = save_dir / "black_litterman_analysis.png" if save_dir else None
                self.plot_black_litterman_analysis(results, str(bl_path) if bl_path else None)
                
            elif method == 'HIERARCHICAL_RISK_PARITY':
                print("🌳 Plotting HRP Analysis...")
                hrp_path = save_dir / "hrp_analysis.png" if save_dir else None
                self.plot_hrp_analysis(results, str(hrp_path) if hrp_path else None)
            
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