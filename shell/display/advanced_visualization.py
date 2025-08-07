# shell/display/advanced_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AdvancedPortfolioVisualizer:
    """
    Professional-grade visualization dashboard for institutional investors.
    Creates interactive and static visualizations for comprehensive portfolio analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_comprehensive_dashboard(self, backtest_results: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # Extract data from backtest results
        strategy_comparison = backtest_results.get('strategy_comparison', {})
        individual_results = strategy_comparison.get('individual_results', {})
        
        if not individual_results:
            print("No strategy results found for visualization")
            return
        
        # 1. Strategy Performance Comparison (top left)
        ax1 = plt.subplot(4, 3, 1)
        self._plot_strategy_returns_comparison(individual_results, ax1)
        
        # 2. Risk-Return Scatter (top middle)
        ax2 = plt.subplot(4, 3, 2)
        self._plot_risk_return_scatter(individual_results, ax2)
        
        # 3. Drawdown Analysis (top right)
        ax3 = plt.subplot(4, 3, 3)
        self._plot_drawdown_analysis(individual_results, ax3)
        
        # 4. Rolling Sharpe Ratios (second row left)
        ax4 = plt.subplot(4, 3, 4)
        self._plot_rolling_sharpe_ratios(individual_results, ax4)
        
        # 5. Turnover Analysis (second row middle)
        ax5 = plt.subplot(4, 3, 5)
        self._plot_turnover_analysis(individual_results, ax5)
        
        # 6. Weight Evolution (second row right)
        ax6 = plt.subplot(4, 3, 6)
        best_strategy = backtest_results.get('best_strategy')
        if best_strategy and best_strategy in individual_results:
            self._plot_weight_evolution(individual_results[best_strategy], ax6)
        
        # 7. Performance Metrics Heatmap (third row, spanning 2 columns)
        ax7 = plt.subplot(4, 3, (7, 8))
        self._plot_performance_heatmap(individual_results, ax7)
        
        # 8. Out-of-Sample Performance (third row right)
        ax8 = plt.subplot(4, 3, 9)
        self._plot_out_of_sample_performance(individual_results, ax8)
        
        # 9. Strategy Correlation Matrix (fourth row left)
        ax9 = plt.subplot(4, 3, 10)
        self._plot_strategy_correlations(strategy_comparison, ax9)
        
        # 10. Risk Attribution Analysis (fourth row middle and right)
        ax10 = plt.subplot(4, 3, (11, 12))
        if best_strategy and best_strategy in individual_results:
            self._plot_risk_attribution(individual_results[best_strategy], ax10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, backtest_results: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        """
        strategy_comparison = backtest_results.get('strategy_comparison', {})
        individual_results = strategy_comparison.get('individual_results', {})
        
        if not individual_results:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Cumulative Returns', 'Risk-Return Analysis', 'Rolling Metrics',
                          'Weight Evolution', 'Performance Heatmap', 'Drawdown Analysis',
                          'Turnover Analysis', 'Out-of-Sample Performance', 'Strategy Correlations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Add traces for each visualization
        self._add_interactive_cumulative_returns(fig, individual_results, row=1, col=1)
        self._add_interactive_risk_return(fig, individual_results, row=1, col=2)
        self._add_interactive_rolling_metrics(fig, individual_results, row=1, col=3)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Portfolio Optimization Strategy Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    def _plot_strategy_returns_comparison(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot cumulative returns comparison."""
        for strategy_name, results in individual_results.items():
            if results and 'returns_series' in results:
                dates = pd.to_datetime(results['returns_series']['dates'])
                cum_returns = results['returns_series']['cumulative_returns']
                ax.plot(dates, cum_returns, label=strategy_name, linewidth=2)
        
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_return_scatter(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot risk-return scatter plot."""
        strategies = []
        returns = []
        risks = []
        
        for strategy_name, results in individual_results.items():
            if results and 'summary' in results:
                strategies.append(strategy_name)
                returns.append(results['summary']['annualized_return'])
                risks.append(results['summary']['annualized_volatility'])
        
        scatter = ax.scatter(risks, returns, s=100, alpha=0.7, c=range(len(strategies)), cmap='viridis')
        
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.set_xlabel('Annualized Volatility')
        ax.set_ylabel('Annualized Return')
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown_analysis(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot maximum drawdown comparison."""
        strategies = []
        drawdowns = []
        
        for strategy_name, results in individual_results.items():
            if results and 'summary' in results:
                strategies.append(strategy_name)
                drawdowns.append(abs(results['summary']['max_drawdown']))
        
        bars = ax.bar(strategies, drawdowns, alpha=0.7, color=self.colors[:len(strategies)])
        ax.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Max Drawdown')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, drawdowns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_rolling_sharpe_ratios(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot rolling Sharpe ratios (simplified version)."""
        for i, (strategy_name, results) in enumerate(individual_results.items()):
            if results and 'returns_series' in results:
                returns = pd.Series(results['returns_series']['returns'])
                
                # Calculate 30-day rolling Sharpe ratio
                rolling_mean = returns.rolling(30).mean() * 252
                rolling_std = returns.rolling(30).std() * np.sqrt(252)
                rolling_sharpe = rolling_mean / rolling_std
                
                ax.plot(rolling_sharpe.dropna(), label=strategy_name, alpha=0.7)
        
        ax.set_title('30-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_turnover_analysis(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot turnover analysis."""
        strategies = []
        turnovers = []
        
        for strategy_name, results in individual_results.items():
            if results and 'summary' in results:
                strategies.append(strategy_name)
                turnovers.append(results['summary']['avg_turnover'])
        
        bars = ax.bar(strategies, turnovers, alpha=0.7, color=self.colors[:len(strategies)])
        ax.set_title('Average Portfolio Turnover', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Turnover')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, turnovers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_weight_evolution(self, strategy_results: Dict[str, Any], ax) -> None:
        """Plot weight evolution for best strategy."""
        if 'weight_analysis' not in strategy_results:
            ax.text(0.5, 0.5, 'No weight evolution data', ha='center', va='center', transform=ax.transAxes)
            return
        
        weight_evolution = strategy_results['weight_analysis'].get('weight_evolution', {})
        dates = pd.to_datetime(strategy_results['weight_analysis'].get('dates', []))
        
        if not weight_evolution or len(dates) == 0:
            ax.text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot top 5 assets by average weight
        avg_weights = {ticker: np.mean(weights) for ticker, weights in weight_evolution.items()}
        top_assets = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for ticker, _ in top_assets:
            if ticker in weight_evolution:
                ax.plot(dates, weight_evolution[ticker], label=ticker, linewidth=2)
        
        ax.set_title('Weight Evolution (Top 5 Assets)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_heatmap(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot performance metrics heatmap."""
        metrics = ['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown', 'avg_turnover']
        strategies = list(individual_results.keys())
        
        data = []
        for strategy in strategies:
            if individual_results[strategy] and 'summary' in individual_results[strategy]:
                row = []
                for metric in metrics:
                    value = individual_results[strategy]['summary'].get(metric, 0)
                    row.append(value)
                data.append(row)
        
        if data:
            data = np.array(data)
            
            # Normalize each column (metric) for better visualization
            data_normalized = data.copy()
            for i in range(data.shape[1]):
                col = data[:, i]
                if col.std() > 0:
                    data_normalized[:, i] = (col - col.mean()) / col.std()
            
            im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(strategies)
            
            # Add text annotations
            for i in range(len(strategies)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", 
                                 color="white" if abs(data_normalized[i, j]) > 1 else "black")
            
            ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Normalized Score')
    
    def _plot_out_of_sample_performance(self, individual_results: Dict[str, Any], ax) -> None:
        """Plot out-of-sample performance."""
        strategies = []
        oos_sharpes = []
        
        for strategy_name, results in individual_results.items():
            if (results and 'out_of_sample_analysis' in results and 
                results['out_of_sample_analysis']):
                strategies.append(strategy_name)
                oos_sharpe = results['out_of_sample_analysis'].get('oos_sharpe', 0)
                oos_sharpes.append(oos_sharpe)
        
        if strategies:
            bars = ax.bar(strategies, oos_sharpes, alpha=0.7, color=self.colors[:len(strategies)])
            ax.set_title('Out-of-Sample Sharpe Ratios', fontsize=14, fontweight='bold')
            ax.set_ylabel('Sharpe Ratio')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, oos_sharpes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No out-of-sample data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_strategy_correlations(self, strategy_comparison: Dict[str, Any], ax) -> None:
        """Plot strategy correlation matrix."""
        comparison = strategy_comparison.get('comparison', {})
        correlation_analysis = comparison.get('correlation_analysis', {})
        correlation_matrix = correlation_analysis.get('correlation_matrix', {})
        
        if correlation_matrix:
            strategies = list(correlation_matrix.keys())
            corr_data = np.array([[correlation_matrix[s1][s2] for s2 in strategies] for s1 in strategies])
            
            im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45)
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(strategies)
            
            # Add correlation values
            for i in range(len(strategies)):
                for j in range(len(strategies)):
                    text = ax.text(j, i, f'{corr_data[i, j]:.2f}', ha="center", va="center",
                                 color="white" if abs(corr_data[i, j]) > 0.5 else "black")
            
            ax.set_title('Strategy Return Correlations', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Correlation')
        else:
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_risk_attribution(self, strategy_results: Dict[str, Any], ax) -> None:
        """Plot risk attribution analysis."""
        # This is a simplified version - in practice you'd use actual risk attribution data
        if 'performance_metrics' in strategy_results:
            # Create a sample risk attribution visualization
            risk_sources = ['Market Risk', 'Specific Risk', 'Currency Risk', 'Interest Rate Risk']
            risk_contributions = [0.4, 0.3, 0.2, 0.1]  # Sample data
            
            wedges, texts, autotexts = ax.pie(risk_contributions, labels=risk_sources, autopct='%1.1f%%',
                                            colors=self.colors[:len(risk_sources)])
            
            ax.set_title('Risk Attribution (Sample)', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No risk attribution data', ha='center', va='center', transform=ax.transAxes)
    
    def _add_interactive_cumulative_returns(self, fig, individual_results: Dict[str, Any], row: int, col: int) -> None:
        """Add interactive cumulative returns plot."""
        for strategy_name, results in individual_results.items():
            if results and 'returns_series' in results:
                dates = results['returns_series']['dates']
                cum_returns = results['returns_series']['cumulative_returns']
                
                fig.add_trace(
                    go.Scatter(x=dates, y=cum_returns, name=strategy_name, mode='lines'),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Cumulative Return", row=row, col=col)
    
    def _add_interactive_risk_return(self, fig, individual_results: Dict[str, Any], row: int, col: int) -> None:
        """Add interactive risk-return scatter plot."""
        strategies = []
        returns = []
        risks = []
        
        for strategy_name, results in individual_results.items():
            if results and 'summary' in results:
                strategies.append(strategy_name)
                returns.append(results['summary']['annualized_return'])
                risks.append(results['summary']['annualized_volatility'])
        
        fig.add_trace(
            go.Scatter(x=risks, y=returns, mode='markers+text', text=strategies,
                      textposition="top center", name="Strategies"),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Annualized Volatility", row=row, col=col)
        fig.update_yaxes(title_text="Annualized Return", row=row, col=col)
    
    def _add_interactive_rolling_metrics(self, fig, individual_results: Dict[str, Any], row: int, col: int) -> None:
        """Add interactive rolling metrics plot."""
        for strategy_name, results in individual_results.items():
            if results and 'returns_series' in results:
                returns = pd.Series(results['returns_series']['returns'])
                rolling_sharpe = (returns.rolling(30).mean() * 252) / (returns.rolling(30).std() * np.sqrt(252))
                
                fig.add_trace(
                    go.Scatter(y=rolling_sharpe.dropna(), name=f"{strategy_name} Sharpe", mode='lines'),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Trading Days", row=row, col=col)
        fig.update_yaxes(title_text="30-Day Rolling Sharpe", row=row, col=col)
    
    def create_advanced_efficient_frontier(self, optimization_results: List[Dict[str, Any]], 
                                         save_path: Optional[str] = None) -> None:
        """
        Create advanced efficient frontier visualization with multiple methods.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colors for different methods
        method_colors = {
            'MEAN_VARIANCE': '#1f77b4',
            'BLACK_LITTERMAN': '#ff7f0e', 
            'HRP': '#2ca02c',
            'ADVANCED_MAX_SHARPE_L2': '#d62728',
            'ADVANCED_MIN_CVAR': '#9467bd',
            'ADVANCED_RISK_PARITY': '#8c564b'
        }
        
        # 1. Main efficient frontier comparison
        for result in optimization_results:
            method = result.get('method', 'Unknown')
            if 'efficient_frontier_portfolios' in result:
                portfolios = result['efficient_frontier_portfolios']
                volatilities = [p['standard_deviation'] for p in portfolios]
                returns = [p['expected_return'] for p in portfolios]
                
                color = method_colors.get(method, '#000000')
                ax1.plot(volatilities, returns, label=method, color=color, linewidth=2, alpha=0.8)
                
                # Mark optimal portfolio
                if 'optimal_portfolio' in result:
                    opt_vol = result['optimal_portfolio']['metrics']['standard_deviation']
                    opt_ret = result['optimal_portfolio']['metrics']['expected_return']
                    ax1.scatter([opt_vol], [opt_ret], color=color, s=100, marker='*', 
                              edgecolors='black', linewidth=1, zorder=5)
        
        ax1.set_xlabel('Volatility (Standard Deviation)')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Efficient Frontier Comparison Across Methods', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe ratio visualization
        for result in optimization_results:
            method = result.get('method', 'Unknown')
            if 'efficient_frontier_portfolios' in result:
                portfolios = result['efficient_frontier_portfolios']
                sharpe_ratios = [p.get('sharpe_ratio', 0) for p in portfolios]
                volatilities = [p['standard_deviation'] for p in portfolios]
                
                color = method_colors.get(method, '#000000')
                ax2.plot(volatilities, sharpe_ratios, label=method, color=color, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Volatility (Standard Deviation)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio Along Efficient Frontier', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Weight concentration analysis
        concentration_data = []
        method_names = []
        
        for result in optimization_results:
            method = result.get('method', 'Unknown')
            if 'optimal_portfolio' in result:
                weights = list(result['optimal_portfolio']['weights'].values())
                # Calculate Herfindahl index (concentration measure)
                herfindahl = sum(w**2 for w in weights)
                concentration_data.append(herfindahl)
                method_names.append(method.replace('ADVANCED_', '').replace('_', ' '))
        
        if concentration_data:
            bars = ax3.bar(range(len(method_names)), concentration_data, 
                          color=[method_colors.get(m, '#000000') for m in optimization_results])
            ax3.set_xticks(range(len(method_names)))
            ax3.set_xticklabels(method_names, rotation=45, ha='right')
            ax3.set_ylabel('Concentration Index')
            ax3.set_title('Portfolio Concentration Comparison', fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, concentration_data):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Risk decomposition (simplified)
        if optimization_results:
            best_result = max(optimization_results, 
                            key=lambda x: x.get('optimal_portfolio', {}).get('metrics', {}).get('sharpe_ratio', 0))
            
            if 'optimal_portfolio' in best_result:
                weights = best_result['optimal_portfolio']['weights']
                tickers = list(weights.keys())
                weight_values = list(weights.values())
                
                # Create pie chart for weight distribution
                ax4.pie(weight_values, labels=tickers, autopct='%1.1f%%', startangle=90)
                ax4.set_title(f'Optimal Weight Distribution\n({best_result.get("method", "Best Strategy")})', 
                            fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced efficient frontier saved to {save_path}")
        
        plt.show()


def create_institutional_report_visualizations(backtest_results: Dict[str, Any],
                                             optimization_results: List[Dict[str, Any]],
                                             save_directory: str = "./") -> Dict[str, str]:
    """
    Create a complete set of institutional-grade visualizations.
    
    Returns:
        Dictionary mapping visualization names to file paths
    """
    visualizer = AdvancedPortfolioVisualizer()
    saved_files = {}
    
    try:
        # 1. Comprehensive Dashboard
        dashboard_path = f"{save_directory}/comprehensive_dashboard.png"
        visualizer.create_comprehensive_dashboard(backtest_results, dashboard_path)
        saved_files['comprehensive_dashboard'] = dashboard_path
        
        # 2. Advanced Efficient Frontier
        if optimization_results:
            frontier_path = f"{save_directory}/advanced_efficient_frontier.png"
            visualizer.create_advanced_efficient_frontier(optimization_results, frontier_path)
            saved_files['efficient_frontier'] = frontier_path
        
        # 3. Interactive Dashboard (HTML)
        interactive_fig = visualizer.create_interactive_dashboard(backtest_results)
        if interactive_fig.data:
            interactive_path = f"{save_directory}/interactive_dashboard.html"
            interactive_fig.write_html(interactive_path)
            saved_files['interactive_dashboard'] = interactive_path
        
        print(f"Created {len(saved_files)} institutional-grade visualizations")
        return saved_files
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return saved_files