# core/optimization/advanced_optimization.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pypfopt import efficient_frontier, expected_returns, risk_models, objective_functions
from pypfopt.cla import CLA
from pypfopt.exceptions import OptimizationError
import warnings

from core.utils import weights_to_array, calculate_portfolio_metrics

class AdvancedPortfolioOptimizer:
    """
    Enterprise-grade portfolio optimization with advanced risk models and methods.
    Designed for hedge funds, asset managers, and institutional investors.
    """
    
    def __init__(self, price_data: pd.DataFrame, tickers: List[str], risk_free_rate: float = 0.02):
        self.price_data = price_data
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.returns = price_data.pct_change().dropna()
        
        # Pre-compute common data
        self.mu = expected_returns.mean_historical_return(price_data)
        self.S = risk_models.sample_cov(price_data)
        
    def optimize_max_sharpe_l2_regularized(self, gamma: float = 0.1, 
                                         constraints: Dict[str, Tuple[float, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Optimize for maximum Sharpe ratio with L2 regularization for weight diversification.
        Reduces concentration risk and encourages more balanced allocations.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ef = efficient_frontier.EfficientFrontier(self.mu, self.S)
            
            # Add weight constraints
            if constraints:
                for ticker, (min_weight, max_weight) in constraints.items():
                    if ticker in self.tickers:
                        idx = self.tickers.index(ticker)
                        ef.add_constraint(lambda w, i=idx: w[i] >= min_weight)
                        ef.add_constraint(lambda w, i=idx: w[i] <= max_weight)
            
            # Add L2 regularization to encourage diversification
            ef.add_objective(objective_functions.L2_reg, gamma=gamma)
            
            try:
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                weights = ef.clean_weights()
                
                # Calculate performance metrics
                weights_array = weights_to_array(self.tickers, weights)
                metrics = calculate_portfolio_metrics(
                    weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
                )
                
                return weights, metrics
                
            except OptimizationError as e:
                print(f"L2 regularized optimization failed: {e}")
                return self._fallback_equal_weights()
    
    def optimize_min_cvar(self, confidence_level: float = 0.05, 
                         constraints: Dict[str, Tuple[float, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Minimize Conditional Value at Risk (CVaR) - focuses on tail risk management.
        Essential for hedge funds and risk-conscious investors.
        """
        try:
            from pypfopt import CVAROpt
            
            # CVaR optimization using historical returns
            cvar_optimizer = CVAROpt(self.returns, beta=confidence_level)
            
            # Add constraints
            if constraints:
                for ticker, (min_weight, max_weight) in constraints.items():
                    if ticker in self.tickers:
                        cvar_optimizer.add_constraint(
                            lambda w, ticker=ticker, min_w=min_weight, max_w=max_weight:
                            min_w <= w[self.tickers.index(ticker)] <= max_w
                        )
            
            weights = cvar_optimizer.min_cvar()
            weights = cvar_optimizer.clean_weights()
            
            # Calculate performance metrics
            weights_array = weights_to_array(self.tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
            )
            
            # Add CVaR-specific metrics
            portfolio_returns = (self.returns * weights_array).sum(axis=1)
            cvar_value = -portfolio_returns.quantile(confidence_level)
            var_value = -portfolio_returns.quantile(confidence_level)
            
            metrics.update({
                'cvar': cvar_value,
                'var': var_value,
                'confidence_level': confidence_level
            })
            
            return weights, metrics
            
        except ImportError:
            print("CVaR optimization requires pypfopt[optional] - falling back to min volatility")
            return self._optimize_min_volatility_fallback(constraints)
    
    def optimize_mean_semivariance(self, constraints: Dict[str, Tuple[float, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Optimize using semivariance (downside risk) instead of full variance.
        Focuses only on negative returns, preferred by many institutional investors.
        """
        # Calculate semivariance matrix (downside risk only)
        negative_returns = self.returns.copy()
        negative_returns[negative_returns > 0] = 0
        
        # Semivariance covariance matrix
        S_semi = negative_returns.cov() * 252  # Annualized
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ef = efficient_frontier.EfficientFrontier(self.mu, S_semi)
            
            # Add constraints
            if constraints:
                for ticker, (min_weight, max_weight) in constraints.items():
                    if ticker in self.tickers:
                        idx = self.tickers.index(ticker)
                        ef.add_constraint(lambda w, i=idx: w[i] >= min_weight)
                        ef.add_constraint(lambda w, i=idx: w[i] <= max_weight)
            
            try:
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                weights = ef.clean_weights()
                
                # Calculate performance metrics with semivariance
                weights_array = weights_to_array(self.tickers, weights)
                metrics = calculate_portfolio_metrics(
                    weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
                )
                
                # Add semivariance-specific metrics
                portfolio_returns = (self.returns * weights_array).sum(axis=1)
                downside_returns = portfolio_returns[portfolio_returns < 0]
                semivariance = downside_returns.var() * 252 if len(downside_returns) > 0 else 0
                downside_deviation = np.sqrt(semivariance)
                
                metrics.update({
                    'semivariance': semivariance,
                    'downside_deviation': downside_deviation,
                    'sortino_ratio': (metrics['expected_return'] - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
                })
                
                return weights, metrics
                
            except OptimizationError as e:
                print(f"Semivariance optimization failed: {e}")
                return self._fallback_equal_weights()
    
    def optimize_risk_parity(self, method: str = 'equal_marginal_contrib') -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Risk Parity optimization - equal risk contribution from each asset.
        Popular among institutional investors for its diversification properties.
        """
        try:
            from pypfopt import RiskParityOpt
            
            rp = RiskParityOpt(self.S)
            
            if method == 'equal_marginal_contrib':
                weights = rp.equal_marginal_contrib()
            else:
                weights = rp.equal_risk_contrib()
            
            weights = rp.clean_weights()
            
            # Calculate performance metrics
            weights_array = weights_to_array(self.tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
            )
            
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights_array.T @ self.S @ weights_array)
            marginal_contrib = self.S @ weights_array / portfolio_vol
            risk_contrib = weights_array * marginal_contrib / portfolio_vol
            
            metrics.update({
                'risk_contributions': dict(zip(self.tickers, risk_contrib)),
                'risk_parity_method': method
            })
            
            return weights, metrics
            
        except ImportError:
            print("Risk Parity requires pypfopt[optional] - falling back to equal weights")
            return self._fallback_equal_weights()
    
    def optimize_market_neutral(self, target_volatility: float = 0.15,
                               long_short_ratio: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Market neutral portfolio optimization with target volatility.
        Essential for hedge funds pursuing alpha generation strategies.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ef = efficient_frontier.EfficientFrontier(self.mu, self.S, weight_bounds=(-1, 1))
            
            # Market neutral constraint (weights sum to 0)
            ef.add_constraint(lambda w: np.sum(w) == 0)
            
            # Long-short balance constraint
            ef.add_constraint(lambda w: np.sum(w[w > 0]) / abs(np.sum(w[w < 0])) <= long_short_ratio + 0.1)
            ef.add_constraint(lambda w: np.sum(w[w > 0]) / abs(np.sum(w[w < 0])) >= long_short_ratio - 0.1)
            
            try:
                # Target volatility with maximum expected return
                ef.efficient_risk(target_volatility)
                weights = ef.clean_weights()
                
                # Calculate performance metrics
                weights_array = weights_to_array(self.tickers, weights)
                metrics = calculate_portfolio_metrics(
                    weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
                )
                
                # Add market neutral specific metrics
                long_exposure = sum(w for w in weights_array if w > 0)
                short_exposure = abs(sum(w for w in weights_array if w < 0))
                gross_exposure = long_exposure + short_exposure
                net_exposure = long_exposure - short_exposure
                
                metrics.update({
                    'long_exposure': long_exposure,
                    'short_exposure': short_exposure,
                    'gross_exposure': gross_exposure,
                    'net_exposure': net_exposure,
                    'long_short_ratio': long_exposure / short_exposure if short_exposure > 0 else 0
                })
                
                return weights, metrics
                
            except OptimizationError as e:
                print(f"Market neutral optimization failed: {e}")
                return self._fallback_equal_weights()
    
    def optimize_critical_line_algorithm(self, target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Critical Line Algorithm for complete efficient frontier calculation.
        Provides exact solution for mean-variance optimization problem.
        """
        try:
            cla = CLA(self.mu, self.S)
            
            if target_return:
                weights = cla.efficient_return(target_return)
            else:
                weights = cla.max_sharpe(risk_free_rate=self.risk_free_rate)
            
            weights = cla.clean_weights()
            
            # Get full efficient frontier
            efficient_frontier_data = []
            mu_range = np.linspace(self.mu.min(), self.mu.max(), 50)
            
            for target_mu in mu_range:
                try:
                    cla_temp = CLA(self.mu, self.S)
                    temp_weights = cla_temp.efficient_return(target_mu)
                    temp_weights = cla_temp.clean_weights()
                    
                    weights_array = weights_to_array(self.tickers, temp_weights)
                    temp_metrics = calculate_portfolio_metrics(
                        weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
                    )
                    
                    efficient_frontier_data.append({
                        'target_return': target_mu,
                        'weights': temp_weights,
                        'metrics': temp_metrics
                    })
                except:
                    continue
            
            # Calculate metrics for main portfolio
            weights_array = weights_to_array(self.tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
            )
            
            return {
                'weights': weights,
                'metrics': metrics,
                'efficient_frontier_cla': efficient_frontier_data,
                'method': 'Critical Line Algorithm'
            }
            
        except Exception as e:
            print(f"CLA optimization failed: {e}")
            weights, metrics = self._fallback_equal_weights()
            return {
                'weights': weights,
                'metrics': metrics,
                'efficient_frontier_cla': [],
                'method': 'Critical Line Algorithm (Failed - Equal Weights)'
            }
    
    def _optimize_min_volatility_fallback(self, constraints: Dict[str, Tuple[float, float]] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Fallback to minimum volatility optimization."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ef = efficient_frontier.EfficientFrontier(self.mu, self.S)
            
            if constraints:
                for ticker, (min_weight, max_weight) in constraints.items():
                    if ticker in self.tickers:
                        idx = self.tickers.index(ticker)
                        ef.add_constraint(lambda w, i=idx: w[i] >= min_weight)
                        ef.add_constraint(lambda w, i=idx: w[i] <= max_weight)
            
            ef.min_volatility()
            weights = ef.clean_weights()
            
            weights_array = weights_to_array(self.tickers, weights)
            metrics = calculate_portfolio_metrics(
                weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
            )
            
            return weights, metrics
    
    def _fallback_equal_weights(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Fallback to equal weights when optimization fails."""
        weights = {ticker: 1.0/len(self.tickers) for ticker in self.tickers}
        weights_array = weights_to_array(self.tickers, weights)
        metrics = calculate_portfolio_metrics(
            weights_array, self.mu, self.S, self.price_data, self.risk_free_rate
        )
        return weights, metrics


def run_advanced_optimization(
    price_data: pd.DataFrame,
    tickers: List[str],
    allocations: Dict[str, float],
    method: str,
    constraints: Dict[str, Tuple[float, float]] = None,
    risk_free_rate: float = 0.02,
    **kwargs
) -> Dict[str, Any]:
    """
    Run advanced portfolio optimization with institutional-grade methods.
    
    Args:
        price_data: Historical price data
        tickers: List of asset tickers
        allocations: Current portfolio allocations
        method: Optimization method ('max_sharpe_l2', 'min_cvar', 'semivariance', 'risk_parity', 'market_neutral', 'cla')
        constraints: Weight constraints per asset
        risk_free_rate: Risk-free rate
        **kwargs: Additional method-specific parameters
    """
    
    optimizer = AdvancedPortfolioOptimizer(price_data, tickers, risk_free_rate)
    
    # Calculate provided portfolio metrics
    provided_weights_array = weights_to_array(tickers, allocations)
    provided_metrics = calculate_portfolio_metrics(
        provided_weights_array, optimizer.mu, optimizer.S, price_data, risk_free_rate
    )
    
    # Run optimization based on method
    if method == 'max_sharpe_l2':
        gamma = kwargs.get('gamma', 0.1)
        optimal_weights, optimal_metrics = optimizer.optimize_max_sharpe_l2_regularized(gamma, constraints)
        method_info = {'gamma': gamma, 'regularization': 'L2'}
        
    elif method == 'min_cvar':
        confidence_level = kwargs.get('confidence_level', 0.05)
        optimal_weights, optimal_metrics = optimizer.optimize_min_cvar(confidence_level, constraints)
        method_info = {'confidence_level': confidence_level, 'focus': 'tail_risk'}
        
    elif method == 'semivariance':
        optimal_weights, optimal_metrics = optimizer.optimize_mean_semivariance(constraints)
        method_info = {'risk_measure': 'downside_only', 'focus': 'semivariance'}
        
    elif method == 'risk_parity':
        rp_method = kwargs.get('rp_method', 'equal_marginal_contrib')
        optimal_weights, optimal_metrics = optimizer.optimize_risk_parity(rp_method)
        method_info = {'risk_parity_method': rp_method, 'focus': 'equal_risk_contribution'}
        
    elif method == 'market_neutral':
        target_vol = kwargs.get('target_volatility', 0.15)
        ls_ratio = kwargs.get('long_short_ratio', 1.0)
        optimal_weights, optimal_metrics = optimizer.optimize_market_neutral(target_vol, ls_ratio)
        method_info = {'target_volatility': target_vol, 'long_short_ratio': ls_ratio, 'market_exposure': 'neutral'}
        
    elif method == 'cla':
        target_return = kwargs.get('target_return', None)
        result = optimizer.optimize_critical_line_algorithm(target_return)
        optimal_weights = result['weights']
        optimal_metrics = result['metrics']
        method_info = {'algorithm': 'Critical Line Algorithm', 'exact_solution': True}
        
    else:
        raise ValueError(f"Unknown advanced optimization method: {method}")
    
    # Create correlation matrix
    correlation_matrix = price_data.corr().to_dict()
    
    return {
        "provided_portfolio": {
            "weights": allocations,
            "metrics": provided_metrics
        },
        "optimal_portfolio": {
            "weights": optimal_weights,
            "metrics": optimal_metrics
        },
        "method_info": method_info,
        "correlation_matrix": correlation_matrix,
        "method": f"ADVANCED_{method.upper()}",
        "risk_free_rate": risk_free_rate
    }