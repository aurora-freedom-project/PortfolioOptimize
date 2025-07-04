# core/analytics.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings

class PortfolioAnalytics:
    """
    Professional portfolio analytics for hedge funds and institutional investors.
    Provides comprehensive performance attribution, risk decomposition, and advanced metrics.
    """
    
    def __init__(self, price_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None):
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.returns = price_data.pct_change().dropna()
        
        if benchmark_data is not None:
            self.benchmark_returns = benchmark_data.pct_change().dropna()
        else:
            self.benchmark_returns = None
    
    def calculate_comprehensive_metrics(self, weights: Dict[str, float], 
                                      risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio performance metrics for institutional analysis.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.price_data.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Advanced risk metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        upside_returns = portfolio_returns[portfolio_returns > 0]
        
        # Sortino Ratio
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio (Annual Return / Max Drawdown)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        var_99 = np.percentile(portfolio_returns, 1) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        cvar_99 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean() * np.sqrt(252)
        
        # Tail ratios and skewness
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        tail_ratio = len(upside_returns) / len(downside_returns) if len(downside_returns) > 0 else float('inf')
        
        # Capture ratios
        if self.benchmark_returns is not None:
            benchmark_portfolio_returns = (self.benchmark_returns * weights_array).sum(axis=1)
            upside_capture = (upside_returns.mean() / benchmark_portfolio_returns[benchmark_portfolio_returns > 0].mean()) if len(benchmark_portfolio_returns[benchmark_portfolio_returns > 0]) > 0 else 1
            downside_capture = (downside_returns.mean() / benchmark_portfolio_returns[benchmark_portfolio_returns < 0].mean()) if len(benchmark_portfolio_returns[benchmark_portfolio_returns < 0]) > 0 else 1
        else:
            upside_capture = None
            downside_capture = None
        
        # Maximum consecutive gains/losses
        consecutive_gains = self._calculate_consecutive_periods(portfolio_returns > 0)
        consecutive_losses = self._calculate_consecutive_periods(portfolio_returns < 0)
        
        # Volatility clustering (GARCH-like measure)
        volatility_clustering = self._calculate_volatility_clustering(portfolio_returns)
        
        # Information ratio (if benchmark available)
        if self.benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_portfolio_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        else:
            tracking_error = None
            information_ratio = None
        
        return {
            # Basic metrics
            'expected_return': annual_return,
            'standard_deviation': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            
            # Distribution metrics
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            
            # Capture ratios
            'upside_capture': upside_capture,
            'downside_capture': downside_capture,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            
            # Persistence metrics
            'max_consecutive_gains': consecutive_gains,
            'max_consecutive_losses': consecutive_losses,
            'volatility_clustering': volatility_clustering,
            
            # Period information
            'total_periods': len(portfolio_returns),
            'positive_periods': len(upside_returns),
            'negative_periods': len(downside_returns),
            'win_rate': len(upside_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        }
    
    def risk_attribution_analysis(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive risk attribution analysis for portfolio construction.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.price_data.columns])
        tickers = list(self.price_data.columns)
        
        # Covariance matrix
        cov_matrix = self.returns.cov() * 252
        
        # Portfolio variance
        portfolio_variance = weights_array.T @ cov_matrix @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal contribution to risk
        marginal_contrib = cov_matrix @ weights_array
        
        # Component contribution to risk
        component_contrib = weights_array * marginal_contrib
        
        # Percentage contribution to risk
        risk_contrib_pct = component_contrib / portfolio_variance
        
        # Active weights (vs equal weight)
        equal_weights = np.ones(len(tickers)) / len(tickers)
        active_weights = weights_array - equal_weights
        
        # Concentration metrics
        herfindahl_index = np.sum(weights_array ** 2)
        effective_assets = 1 / herfindahl_index
        concentration_ratio = np.sum(np.sort(weights_array)[-3:])  # Top 3 concentration
        
        # Diversification ratio
        weighted_avg_volatility = np.sum(weights_array * np.sqrt(np.diag(cov_matrix)))
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'marginal_contributions': dict(zip(tickers, marginal_contrib)),
            'component_contributions': dict(zip(tickers, component_contrib)),
            'risk_contribution_pct': dict(zip(tickers, risk_contrib_pct)),
            'active_weights': dict(zip(tickers, active_weights)),
            'herfindahl_index': herfindahl_index,
            'effective_assets': effective_assets,
            'concentration_ratio': concentration_ratio,
            'diversification_ratio': diversification_ratio
        }
    
    def performance_attribution(self, weights: Dict[str, float], 
                              factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Performance attribution analysis including factor decomposition.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.price_data.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        tickers = list(self.price_data.columns)
        
        # Asset-level contribution
        asset_returns = self.returns.mean() * 252
        asset_contributions = weights_array * asset_returns
        
        # Interaction effects (diversification benefit)
        sum_individual = np.sum(asset_contributions)
        portfolio_return = portfolio_returns.mean() * 252
        interaction_effect = portfolio_return - sum_individual
        
        # Factor attribution (if factor data provided)
        if factor_returns is not None:
            try:
                factor_attribution = self._calculate_factor_attribution(portfolio_returns, factor_returns)
            except:
                factor_attribution = {}
        else:
            factor_attribution = {}
        
        # Sector/asset class attribution (simplified)
        sector_attribution = self._estimate_sector_attribution(weights, asset_returns)
        
        return {
            'total_return': portfolio_return,
            'asset_contributions': dict(zip(tickers, asset_contributions)),
            'interaction_effect': interaction_effect,
            'factor_attribution': factor_attribution,
            'sector_attribution': sector_attribution,
            'top_contributors': self._get_top_contributors(tickers, asset_contributions, 3),
            'bottom_contributors': self._get_bottom_contributors(tickers, asset_contributions, 3)
        }
    
    def stress_test_analysis(self, weights: Dict[str, float], 
                           scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Stress testing analysis for different market scenarios.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.price_data.columns])
        tickers = list(self.price_data.columns)
        
        stress_results = {}
        
        for scenario_name, scenario_returns in scenarios.items():
            # Apply scenario returns
            scenario_array = np.array([scenario_returns.get(ticker, 0) for ticker in tickers])
            portfolio_scenario_return = np.sum(weights_array * scenario_array)
            
            # Calculate impact
            normal_return = (self.returns * weights_array).sum(axis=1).mean() * 252
            scenario_impact = portfolio_scenario_return - normal_return
            
            stress_results[scenario_name] = {
                'portfolio_return': portfolio_scenario_return,
                'impact_vs_normal': scenario_impact,
                'asset_contributions': dict(zip(tickers, weights_array * scenario_array))
            }
        
        return stress_results
    
    def rolling_performance_analysis(self, weights: Dict[str, float], 
                                   window_days: int = 252) -> Dict[str, Any]:
        """
        Rolling performance analysis for trend identification.
        """
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.price_data.columns])
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Rolling metrics
        rolling_returns = portfolio_returns.rolling(window=window_days).mean() * 252
        rolling_volatility = portfolio_returns.rolling(window=window_days).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_volatility
        
        # Rolling max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window_days).max()
        rolling_drawdown = cumulative_returns / rolling_max - 1
        rolling_max_dd = rolling_drawdown.rolling(window=window_days).min()
        
        return {
            'dates': rolling_returns.index.tolist(),
            'rolling_returns': rolling_returns.tolist(),
            'rolling_volatility': rolling_volatility.tolist(),
            'rolling_sharpe': rolling_sharpe.tolist(),
            'rolling_max_drawdown': rolling_max_dd.tolist(),
            'window_days': window_days
        }
    
    def _calculate_consecutive_periods(self, condition_series: pd.Series) -> int:
        """Calculate maximum consecutive periods meeting a condition."""
        groups = (condition_series != condition_series.shift()).cumsum()
        consecutive = condition_series.groupby(groups).cumsum()
        return consecutive[condition_series].max() if any(condition_series) else 0
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering measure."""
        abs_returns = returns.abs()
        correlation = abs_returns.corr(abs_returns.shift(1))
        return correlation if not np.isnan(correlation) else 0
    
    def _calculate_factor_attribution(self, portfolio_returns: pd.Series, 
                                    factor_returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor attribution using regression analysis."""
        from sklearn.linear_model import LinearRegression
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        port_aligned = portfolio_returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        # Run regression
        model = LinearRegression()
        model.fit(factors_aligned, port_aligned)
        
        # Calculate factor contributions
        factor_contributions = {}
        for i, factor in enumerate(factors_aligned.columns):
            factor_contrib = model.coef_[i] * factors_aligned[factor].mean() * 252
            factor_contributions[factor] = factor_contrib
        
        factor_contributions['alpha'] = model.intercept_ * 252
        factor_contributions['r_squared'] = model.score(factors_aligned, port_aligned)
        
        return factor_contributions
    
    def _estimate_sector_attribution(self, weights: Dict[str, float], 
                                   asset_returns: pd.Series) -> Dict[str, float]:
        """Estimate sector attribution (simplified based on ticker patterns)."""
        # Simple sector classification based on ticker patterns
        sectors = {
            'financials': [t for t in weights.keys() if any(x in t.upper() for x in ['ANZ', 'CBA', 'NAB', 'MQG'])],
            'resources': [t for t in weights.keys() if any(x in t.upper() for x in ['RIO', 'BHP', 'FMG'])],
            'retail': [t for t in weights.keys() if any(x in t.upper() for x in ['WOW', 'COL', 'WES'])],
            'other': []
        }
        
        # Assign remaining tickers to 'other'
        all_classified = sum(sectors.values(), [])
        sectors['other'] = [t for t in weights.keys() if t not in all_classified]
        
        sector_attribution = {}
        for sector, tickers in sectors.items():
            if tickers:
                sector_weight = sum(weights.get(t, 0) for t in tickers)
                sector_return = np.mean([asset_returns.get(t, 0) for t in tickers if t in asset_returns.index])
                sector_attribution[sector] = sector_weight * sector_return
            else:
                sector_attribution[sector] = 0
        
        return sector_attribution
    
    def _get_top_contributors(self, tickers: List[str], contributions: np.ndarray, n: int) -> List[Dict[str, float]]:
        """Get top N contributors to performance."""
        sorted_indices = np.argsort(contributions)[::-1]
        return [{'ticker': tickers[i], 'contribution': contributions.iloc[i] if hasattr(contributions, 'iloc') else contributions[i]} 
                for i in sorted_indices[:n]]
    
    def _get_bottom_contributors(self, tickers: List[str], contributions: np.ndarray, n: int) -> List[Dict[str, float]]:
        """Get bottom N contributors to performance."""
        sorted_indices = np.argsort(contributions)
        return [{'ticker': tickers[i], 'contribution': contributions.iloc[i] if hasattr(contributions, 'iloc') else contributions[i]} 
                for i in sorted_indices[:n]]


def generate_comprehensive_analytics_report(price_data: pd.DataFrame, 
                                          weights: Dict[str, float],
                                          benchmark_data: Optional[pd.DataFrame] = None,
                                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Generate comprehensive analytics report for institutional investors.
    """
    analytics = PortfolioAnalytics(price_data, benchmark_data)
    
    # Core performance metrics
    performance_metrics = analytics.calculate_comprehensive_metrics(weights, risk_free_rate)
    
    # Risk attribution
    risk_attribution = analytics.risk_attribution_analysis(weights)
    
    # Performance attribution
    performance_attribution = analytics.performance_attribution(weights)
    
    # Rolling analysis
    rolling_analysis = analytics.rolling_performance_analysis(weights, window_days=252)
    
    # Stress test scenarios (predefined common scenarios)
    stress_scenarios = {
        'market_crash': {ticker: -0.3 for ticker in weights.keys()},  # 30% decline
        'financial_crisis': {ticker: -0.5 if 'financial' in ticker.lower() else -0.2 for ticker in weights.keys()},
        'recession': {ticker: -0.25 for ticker in weights.keys()},
        'black_swan': {ticker: -0.6 for ticker in weights.keys()}
    }
    stress_results = analytics.stress_test_analysis(weights, stress_scenarios)
    
    return {
        'performance_metrics': performance_metrics,
        'risk_attribution': risk_attribution,
        'performance_attribution': performance_attribution,
        'rolling_analysis': rolling_analysis,
        'stress_test_results': stress_results,
        'analysis_date': datetime.now().isoformat(),
        'analytics_version': '1.0.0'
    }