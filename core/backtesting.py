# core/backtesting.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime, timedelta
import warnings
from collections import defaultdict

from core.optimization.mean_variance import run_mean_variance_optimization
from core.optimization.black_litterman import run_black_litterman_optimization  
from core.optimization.hrp import run_hierarchical_risk_parity
from core.optimization.advanced_optimization import run_advanced_optimization
from core.analytics import PortfolioAnalytics
from core.utils import weights_to_array, calculate_portfolio_metrics

class PortfolioBacktester:
    """
    Professional backtesting framework for portfolio optimization strategies.
    Designed for hedge funds and institutional investors requiring robust
    historical performance analysis and strategy validation.
    """
    
    def __init__(self, price_data: pd.DataFrame, rebalance_frequency: str = 'M',
                 lookback_window: int = 252, min_history: int = 126):
        """
        Initialize backtester with historical price data.
        
        Args:
            price_data: Historical price data with datetime index
            rebalance_frequency: 'D', 'W', 'M', 'Q' for daily, weekly, monthly, quarterly
            lookback_window: Number of periods to use for optimization
            min_history: Minimum periods required before starting backtest
        """
        self.price_data = price_data.sort_index()
        self.returns = price_data.pct_change().dropna()
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.min_history = min_history
        self.tickers = list(price_data.columns)
        
        # Generate rebalancing dates
        self.rebalance_dates = self._generate_rebalance_dates()
        
    def _generate_rebalance_dates(self) -> List[datetime]:
        """Generate rebalancing dates based on frequency."""
        start_date = self.price_data.index[self.min_history]
        end_date = self.price_data.index[-1]
        
        if self.rebalance_frequency == 'D':
            dates = pd.date_range(start_date, end_date, freq='D')
        elif self.rebalance_frequency == 'W':
            dates = pd.date_range(start_date, end_date, freq='W')
        elif self.rebalance_frequency == 'M':
            dates = pd.date_range(start_date, end_date, freq='M')
        elif self.rebalance_frequency == 'Q':
            dates = pd.date_range(start_date, end_date, freq='Q')
        else:
            raise ValueError("Invalid rebalance frequency. Use 'D', 'W', 'M', or 'Q'.")
        
        # Filter dates that exist in our data
        valid_dates = [d for d in dates if d in self.price_data.index]
        return valid_dates
    
    def backtest_strategy(self, 
                         optimization_method: str,
                         initial_allocations: Dict[str, float],
                         **method_kwargs) -> Dict[str, Any]:
        """
        Backtest a portfolio optimization strategy with walk-forward analysis.
        
        Args:
            optimization_method: 'mean_variance', 'black_litterman', 'hrp', or advanced methods
            initial_allocations: Starting portfolio allocations
            **method_kwargs: Method-specific parameters
        """
        
        # Initialize tracking variables
        portfolio_weights_history = []
        portfolio_returns_history = []
        turnover_history = []
        optimization_results_history = []
        
        # Current weights start with initial allocations
        current_weights = initial_allocations.copy()
        
        for i, rebalance_date in enumerate(self.rebalance_dates):
            try:
                # Get historical data window
                end_idx = self.price_data.index.get_loc(rebalance_date)
                start_idx = max(0, end_idx - self.lookback_window)
                
                historical_data = self.price_data.iloc[start_idx:end_idx+1]
                
                if len(historical_data) < self.min_history:
                    continue
                
                # Run optimization
                new_weights, optimization_result = self._run_optimization(
                    historical_data, optimization_method, current_weights, **method_kwargs
                )
                
                # Calculate turnover
                turnover = self._calculate_turnover(current_weights, new_weights)
                
                # Store results
                portfolio_weights_history.append({
                    'date': rebalance_date,
                    'weights': new_weights.copy(),
                    'turnover': turnover
                })
                
                optimization_results_history.append({
                    'date': rebalance_date,
                    'optimization_result': optimization_result
                })
                
                # Calculate returns until next rebalance
                if i < len(self.rebalance_dates) - 1:
                    next_rebalance = self.rebalance_dates[i + 1]
                    period_returns = self._calculate_period_returns(
                        rebalance_date, next_rebalance, new_weights
                    )
                else:
                    # Last period - calculate until end of data
                    period_returns = self._calculate_period_returns(
                        rebalance_date, self.price_data.index[-1], new_weights
                    )
                
                portfolio_returns_history.extend(period_returns)
                
                # Update current weights
                current_weights = new_weights
                
            except Exception as e:
                print(f"Error on {rebalance_date}: {e}")
                continue
        
        # Compile backtest results
        backtest_results = self._compile_backtest_results(
            portfolio_weights_history,
            portfolio_returns_history,
            optimization_results_history,
            optimization_method
        )
        
        return backtest_results
    
    def compare_strategies(self, 
                          strategies: Dict[str, Dict[str, Any]],
                          initial_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare multiple optimization strategies side by side.
        
        Args:
            strategies: Dict mapping strategy names to their parameters
            initial_allocations: Starting allocations for all strategies
        """
        strategy_results = {}
        
        for strategy_name, strategy_config in strategies.items():
            method = strategy_config.pop('method')
            try:
                result = self.backtest_strategy(
                    method, initial_allocations, **strategy_config
                )
                strategy_results[strategy_name] = result
            except Exception as e:
                print(f"Error backtesting {strategy_name}: {e}")
                strategy_results[strategy_name] = None
        
        # Generate comparison metrics
        comparison = self._generate_strategy_comparison(strategy_results)
        
        return {
            'individual_results': strategy_results,
            'comparison': comparison,
            'backtest_period': {
                'start': self.rebalance_dates[0].strftime('%Y-%m-%d'),
                'end': self.rebalance_dates[-1].strftime('%Y-%m-%d'),
                'rebalance_frequency': self.rebalance_frequency
            }
        }
    
    def rolling_optimization_analysis(self,
                                    optimization_method: str,
                                    initial_allocations: Dict[str, float],
                                    analysis_window: int = 126,
                                    **method_kwargs) -> Dict[str, Any]:
        """
        Analyze how optimization results change over time with rolling windows.
        """
        rolling_results = []
        
        # Create overlapping windows for analysis
        for i in range(self.min_history, len(self.price_data) - analysis_window, 21):  # Every ~month
            window_start = i
            window_end = i + analysis_window
            
            historical_data = self.price_data.iloc[window_start:window_end]
            analysis_date = historical_data.index[-1]
            
            try:
                weights, optimization_result = self._run_optimization(
                    historical_data, optimization_method, initial_allocations, **method_kwargs
                )
                
                rolling_results.append({
                    'date': analysis_date,
                    'weights': weights,
                    'metrics': optimization_result.get('optimal_portfolio', {}).get('metrics', {}),
                    'window_start': historical_data.index[0],
                    'window_end': analysis_date
                })
                
            except Exception as e:
                print(f"Error in rolling analysis on {analysis_date}: {e}")
                continue
        
        # Analyze weight stability and consistency
        stability_analysis = self._analyze_weight_stability(rolling_results)
        
        return {
            'rolling_results': rolling_results,
            'stability_analysis': stability_analysis,
            'method': optimization_method,
            'analysis_window_days': analysis_window
        }
    
    def _run_optimization(self, historical_data: pd.DataFrame, 
                         method: str, current_weights: Dict[str, float], 
                         **kwargs) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Run optimization based on method."""
        
        if method == 'mean_variance':
            result = run_mean_variance_optimization(historical_data, self.tickers, current_weights)
            new_weights = result['optimal_portfolio']['weights']
            
        elif method == 'black_litterman':
            views = kwargs.get('views', {})
            view_confidences = kwargs.get('view_confidences', {})
            result = run_black_litterman_optimization(
                historical_data, self.tickers, current_weights, views, view_confidences
            )
            new_weights = result['optimal_portfolio']['weights']
            
        elif method == 'hrp':
            result = run_hierarchical_risk_parity(historical_data, self.tickers, current_weights)
            new_weights = result['optimal_portfolio']['weights']
            
        elif method in ['max_sharpe_l2', 'min_cvar', 'semivariance', 'risk_parity', 'market_neutral', 'cla']:
            result = run_advanced_optimization(
                historical_data, self.tickers, current_weights, method, **kwargs
            )
            new_weights = result['optimal_portfolio']['weights']
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return new_weights, result
    
    def _calculate_turnover(self, old_weights: Dict[str, float], 
                          new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        turnover = 0
        for ticker in self.tickers:
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            turnover += abs(new_w - old_w)
        return turnover / 2  # Divide by 2 for standard turnover definition
    
    def _calculate_period_returns(self, start_date: datetime, end_date: datetime,
                                weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate portfolio returns for a period."""
        try:
            start_idx = self.price_data.index.get_loc(start_date)
            end_idx = self.price_data.index.get_loc(end_date)
        except KeyError:
            return []
        
        period_returns = []
        weights_array = weights_to_array(self.tickers, weights)
        
        for i in range(start_idx + 1, end_idx + 1):
            date = self.price_data.index[i]
            daily_returns = self.returns.iloc[i]
            portfolio_return = np.sum(weights_array * daily_returns)
            
            period_returns.append({
                'date': date,
                'portfolio_return': portfolio_return,
                'weights': weights.copy()
            })
        
        return period_returns
    
    def _compile_backtest_results(self, weights_history: List[Dict],
                                returns_history: List[Dict],
                                optimization_history: List[Dict],
                                method: str) -> Dict[str, Any]:
        """Compile comprehensive backtest results."""
        
        # Create portfolio returns series
        portfolio_returns = pd.Series(
            [r['portfolio_return'] for r in returns_history],
            index=[r['date'] for r in returns_history]
        )
        
        # Calculate comprehensive performance metrics
        analytics = PortfolioAnalytics(self.price_data)
        
        # Get final weights for analysis
        final_weights = weights_history[-1]['weights'] if weights_history else {}
        performance_metrics = analytics.calculate_comprehensive_metrics(final_weights)
        
        # Calculate additional backtest-specific metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Turnover analysis
        turnovers = [w['turnover'] for w in weights_history]
        avg_turnover = np.mean(turnovers) if turnovers else 0
        
        # Weights consistency analysis
        weight_changes = self._analyze_weight_changes(weights_history)
        
        # Out-of-sample performance
        oos_analysis = self._calculate_out_of_sample_performance(
            returns_history, optimization_history
        )
        
        return {
            'summary': {
                'method': method,
                'total_return': total_return,
                'annualized_return': portfolio_returns.mean() * 252,
                'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'avg_turnover': avg_turnover,
                'num_rebalances': len(weights_history)
            },
            'performance_metrics': performance_metrics,
            'returns_series': {
                'dates': portfolio_returns.index.strftime('%Y-%m-%d').tolist(),
                'returns': portfolio_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist()
            },
            'weights_history': weights_history,
            'weight_analysis': weight_changes,
            'out_of_sample_analysis': oos_analysis,
            'rebalancing_analysis': {
                'frequency': self.rebalance_frequency,
                'avg_turnover': avg_turnover,
                'turnover_history': turnovers
            }
        }
    
    def _generate_strategy_comparison(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics across strategies."""
        
        valid_strategies = {k: v for k, v in strategy_results.items() if v is not None}
        
        if not valid_strategies:
            return {}
        
        comparison_metrics = {}
        
        for metric in ['total_return', 'annualized_return', 'annualized_volatility', 
                      'sharpe_ratio', 'max_drawdown', 'avg_turnover']:
            comparison_metrics[metric] = {
                strategy: result['summary'][metric] 
                for strategy, result in valid_strategies.items()
            }
        
        # Find best performing strategies
        best_sharpe = max(valid_strategies.items(), 
                         key=lambda x: x[1]['summary']['sharpe_ratio'])
        best_return = max(valid_strategies.items(), 
                         key=lambda x: x[1]['summary']['total_return'])
        lowest_vol = min(valid_strategies.items(), 
                        key=lambda x: x[1]['summary']['annualized_volatility'])
        
        return {
            'metrics_comparison': comparison_metrics,
            'best_performers': {
                'highest_sharpe': {'strategy': best_sharpe[0], 'value': best_sharpe[1]['summary']['sharpe_ratio']},
                'highest_return': {'strategy': best_return[0], 'value': best_return[1]['summary']['total_return']},
                'lowest_volatility': {'strategy': lowest_vol[0], 'value': lowest_vol[1]['summary']['annualized_volatility']}
            },
            'correlation_analysis': self._calculate_strategy_correlations(valid_strategies)
        }
    
    def _analyze_weight_stability(self, rolling_results: List[Dict]) -> Dict[str, Any]:
        """Analyze stability of weights over time."""
        if len(rolling_results) < 2:
            return {}
        
        # Calculate weight volatility for each asset
        weight_volatilities = {}
        weight_means = {}
        
        for ticker in self.tickers:
            weights_series = [r['weights'].get(ticker, 0) for r in rolling_results]
            weight_volatilities[ticker] = np.std(weights_series)
            weight_means[ticker] = np.mean(weights_series)
        
        # Calculate turnover between consecutive periods
        turnovers = []
        for i in range(1, len(rolling_results)):
            turnover = self._calculate_turnover(
                rolling_results[i-1]['weights'],
                rolling_results[i]['weights']
            )
            turnovers.append(turnover)
        
        return {
            'weight_volatilities': weight_volatilities,
            'weight_means': weight_means,
            'avg_weight_volatility': np.mean(list(weight_volatilities.values())),
            'avg_turnover': np.mean(turnovers) if turnovers else 0,
            'stability_score': 1 / (1 + np.mean(list(weight_volatilities.values())))  # Higher is more stable
        }
    
    def _analyze_weight_changes(self, weights_history: List[Dict]) -> Dict[str, Any]:
        """Analyze how weights change over the backtest period."""
        if len(weights_history) < 2:
            return {}
        
        # Track weight evolution
        weight_evolution = defaultdict(list)
        dates = []
        
        for entry in weights_history:
            dates.append(entry['date'])
            for ticker in self.tickers:
                weight_evolution[ticker].append(entry['weights'].get(ticker, 0))
        
        # Calculate statistics
        weight_stats = {}
        for ticker in self.tickers:
            weights = weight_evolution[ticker]
            weight_stats[ticker] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'final': weights[-1] if weights else 0
            }
        
        return {
            'weight_evolution': dict(weight_evolution),
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'weight_statistics': weight_stats
        }
    
    def _calculate_out_of_sample_performance(self, returns_history: List[Dict],
                                           optimization_history: List[Dict]) -> Dict[str, Any]:
        """Calculate out-of-sample performance metrics."""
        
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        oos_periods = []
        
        for i, opt_result in enumerate(optimization_history):
            opt_date = opt_result['date']
            
            # Find returns after this optimization date
            future_returns = [r for r in returns_history if r['date'] > opt_date]
            
            if future_returns:
                # Take next 21 days (approximately 1 month) as out-of-sample period
                oos_period = future_returns[:21]
                if len(oos_period) >= 10:  # Minimum 10 days for analysis
                    oos_returns = [r['portfolio_return'] for r in oos_period]
                    oos_periods.append({
                        'optimization_date': opt_date,
                        'oos_return': np.sum(oos_returns),
                        'oos_volatility': np.std(oos_returns) * np.sqrt(252),
                        'num_days': len(oos_returns)
                    })
        
        if not oos_periods:
            return {}
        
        avg_oos_return = np.mean([p['oos_return'] for p in oos_periods])
        avg_oos_volatility = np.mean([p['oos_volatility'] for p in oos_periods])
        
        return {
            'average_oos_return': avg_oos_return,
            'average_oos_volatility': avg_oos_volatility,
            'oos_sharpe': avg_oos_return / avg_oos_volatility if avg_oos_volatility > 0 else 0,
            'num_oos_periods': len(oos_periods),
            'oos_periods': oos_periods
        }
    
    def _calculate_strategy_correlations(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlations between strategy returns."""
        
        # Extract return series for each strategy
        strategy_returns = {}
        
        for strategy_name, results in strategy_results.items():
            returns_data = results['returns_series']
            dates = pd.to_datetime(returns_data['dates'])
            returns = pd.Series(returns_data['returns'], index=dates)
            strategy_returns[strategy_name] = returns
        
        # Align all series to common dates
        if len(strategy_returns) > 1:
            returns_df = pd.DataFrame(strategy_returns)
            returns_df = returns_df.dropna()
            
            correlation_matrix = returns_df.corr().to_dict()
            
            return {
                'correlation_matrix': correlation_matrix,
                'avg_correlation': returns_df.corr().values[np.triu_indices_from(returns_df.corr().values, k=1)].mean()
            }
        
        return {}


def run_comprehensive_backtest(price_data: pd.DataFrame,
                             initial_allocations: Dict[str, float],
                             strategies_config: Dict[str, Dict[str, Any]],
                             rebalance_frequency: str = 'M',
                             lookback_window: int = 252) -> Dict[str, Any]:
    """
    Run comprehensive backtesting analysis with multiple strategies.
    
    Args:
        price_data: Historical price data
        initial_allocations: Starting portfolio allocations
        strategies_config: Configuration for each strategy to test
        rebalance_frequency: Rebalancing frequency
        lookback_window: Lookback window for optimization
    
    Returns:
        Comprehensive backtesting results
    """
    
    backtester = PortfolioBacktester(
        price_data=price_data,
        rebalance_frequency=rebalance_frequency,
        lookback_window=lookback_window
    )
    
    # Run strategy comparison
    strategy_comparison = backtester.compare_strategies(strategies_config, initial_allocations)
    
    # Run rolling analysis for the best performing strategy (by Sharpe ratio)
    best_strategy = None
    best_sharpe = -float('inf')
    
    for strategy_name, results in strategy_comparison['individual_results'].items():
        if results and results['summary']['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['summary']['sharpe_ratio']
            best_strategy = strategy_name
    
    rolling_analysis = None
    if best_strategy and best_strategy in strategies_config:
        strategy_config = strategies_config[best_strategy].copy()
        method = strategy_config.pop('method')
        
        rolling_analysis = backtester.rolling_optimization_analysis(
            method, initial_allocations, **strategy_config
        )
    
    return {
        'strategy_comparison': strategy_comparison,
        'rolling_analysis': rolling_analysis,
        'backtest_config': {
            'rebalance_frequency': rebalance_frequency,
            'lookback_window': lookback_window,
            'backtest_period': {
                'start': price_data.index[0].strftime('%Y-%m-%d'),
                'end': price_data.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(price_data)
            }
        },
        'best_strategy': best_strategy
    }