"""
Streamlined Portfolio Optimizer with Specific Charts
==================================================

Minimal implementation focusing on:
1. Efficient Frontier from portfolio data
2. Pie charts for Mean-Variance (Provided vs Optimal)
3. Black-Litterman Prior/Posterior/Views comparison
4. HRP optimization with specific charts
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    csv_file: str = "merged_stock_prices.csv"
    market_cap_file: str = "market_caps.csv"
    risk_free_rate: float = 0.0383
    tau: float = 0.05


class MarketCapLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def get_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        df = pd.read_csv(self.file_path)
        caps = {}
        for ticker in tickers:
            if ticker in df['ticker'].values:
                cap = df[df['ticker'] == ticker]['market_cap_billions_aud'].iloc[0] * 1e9
                caps[ticker] = cap
            else:
                caps[ticker] = 1e9  # Default
        return caps


class ChartGenerator:
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_efficient_frontier_chart(self, frontier_data: List[Dict]) -> str:
        """Efficient frontier from portfolio data"""
        # Sort by risk for proper line connection
        sorted_data = sorted(frontier_data, key=lambda x: x["standard_deviation"])
        
        returns = [p["expected_return"] for p in sorted_data]
        risks = [p["standard_deviation"] for p in sorted_data]
        sharpe_ratios = [p["sharpe_ratio"] for p in sorted_data]
        
        # Find optimal portfolio in sorted data
        optimal_idx = None
        for i, p in enumerate(sorted_data):
            if p.get("is_optimal", False):
                optimal_idx = i
                break
        
        if optimal_idx is None:
            optimal_idx = np.argmax(sharpe_ratios)
        
        # Create marker sizes - larger for optimal
        marker_sizes = [15 if i == optimal_idx else 6 for i in range(len(sorted_data))]
        marker_symbols = ['star' if i == optimal_idx else 'circle' for i in range(len(sorted_data))]
        
        fig = go.Figure()
        
        # Single trace for efficient frontier with optimal highlighted
        fig.add_trace(go.Scatter(
            x=risks, y=returns,
            mode='lines+markers',
            name=f'Efficient Frontier ({len(frontier_data)} portfolios)',
            line=dict(color='blue', width=2),
            marker=dict(
                size=marker_sizes,
                color=sharpe_ratios,
                colorscale='Viridis',
                showscale=True,
                symbol=marker_symbols,
                line=dict(width=1, color='red')
            ),
            hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}<br>Sharpe: %{marker.color:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Efficient Frontier Envelope ({len(frontier_data)} Portfolios)',
            xaxis_title='Risk (Standard Deviation)',
            yaxis_title='Expected Return',
            template='plotly_white'
        )
        
        file_path = self.output_dir / "efficient_frontier.html"
        fig.write_html(str(file_path))
        return str(file_path)
    
    def create_mv_pie_charts(self, provided_weights: Dict, optimal_weights: Dict) -> str:
        """Pie charts for Mean-Variance: Provided vs Optimal"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=["Provided Portfolio", "Optimal Portfolio"]
        )
        
        # Provided portfolio
        provided_filtered = {k: v for k, v in provided_weights.items() if v > 0.001}
        fig.add_trace(go.Pie(
            labels=list(provided_filtered.keys()),
            values=list(provided_filtered.values()),
            name="Provided",
            textinfo='label+percent'
        ), row=1, col=1)
        
        # Optimal portfolio
        optimal_filtered = {k: v for k, v in optimal_weights.items() if v > 0.001}
        fig.add_trace(go.Pie(
            labels=list(optimal_filtered.keys()),
            values=list(optimal_filtered.values()),
            name="Optimal",
            textinfo='label+percent'
        ), row=1, col=2)
        
        fig.update_layout(title="Mean-Variance: Provided vs Optimal Allocation")
        
        file_path = self.output_dir / "mv_comparison.html"
        fig.write_html(str(file_path))
        return str(file_path)
    
    def create_bl_comparison_chart(self, bl_data: Dict, optimal_weights: Dict) -> str:
        """Combined chart: Bar chart for Prior/Posterior/Views + Pie chart for Optimal"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=["Prior vs Posterior vs Views", "Optimal Portfolio"]
        )
        
        # Bar chart for returns comparison
        tickers = list(bl_data["market_implied_returns"].keys())
        prior_returns = [bl_data["market_implied_returns"][t] for t in tickers]
        posterior_returns = [bl_data["posterior_returns"][t] for t in tickers]
        
        # Prior returns
        fig.add_trace(go.Bar(
            x=tickers, y=prior_returns,
            name='Prior (Market Implied)',
            marker_color='lightblue'
        ), row=1, col=1)
        
        # Posterior returns
        fig.add_trace(go.Bar(
            x=tickers, y=posterior_returns,
            name='Posterior (Updated)',
            marker_color='darkblue'
        ), row=1, col=1)
        
        # Views (where available)
        view_returns = []
        for ticker in tickers:
            if bl_data.get("views_applied") and ticker in bl_data.get("view_details", {}):
                view_returns.append(bl_data["view_details"][ticker]["expected_return"])
            else:
                view_returns.append(None)
        
        view_tickers = [t for t, v in zip(tickers, view_returns) if v is not None]
        view_values = [v for v in view_returns if v is not None]
        if view_values:
            fig.add_trace(go.Bar(
                x=view_tickers, y=view_values,
                name='Investor Views',
                marker_color='red'
            ), row=1, col=1)
        
        # Pie chart for optimal portfolio
        optimal_filtered = {k: v for k, v in optimal_weights.items() if v > 0.001}
        fig.add_trace(go.Pie(
            labels=list(optimal_filtered.keys()),
            values=list(optimal_filtered.values()),
            name="Optimal",
            textinfo='label+percent'
        ), row=1, col=2)
        
        fig.update_layout(title='Black-Litterman Analysis')
        
        file_path = self.output_dir / "bl_comparison.html"
        fig.write_html(str(file_path))
        return str(file_path)
    



class StreamlinedOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.market_cap_loader = MarketCapLoader(config.market_cap_file)
        self.chart_generator = ChartGenerator()
    
    def optimize_portfolio(self, portfolio_data: Dict) -> Dict[str, Any]:
        """Main optimization function"""
        # Load price data
        price_data = pd.read_csv(self.config.csv_file, parse_dates=['date'], index_col='date')
        
        tickers = portfolio_data["tickers"]
        start_date = pd.to_datetime(portfolio_data["start_date"])
        end_date = pd.to_datetime(portfolio_data["end_date"])
        
        # Filter data
        price_data = price_data[tickers]
        price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
        price_data = price_data.dropna()
        
        result = {"status": "success", "charts": {}}
        
        # Mean-Variance Optimization
        mv_result = self._optimize_mean_variance(price_data, portfolio_data)
        result["mean_variance"] = mv_result
        
        # Create MV pie charts if provided weights exist
        if portfolio_data.get("allocations"):
            chart_file = self.chart_generator.create_mv_pie_charts(
                portfolio_data["allocations"], 
                mv_result["optimal_weights"]
            )
            result["charts"]["mv_comparison"] = chart_file
        
        # Efficient frontier chart
        if mv_result.get("efficient_frontier"):
            chart_file = self.chart_generator.create_efficient_frontier_chart(
                mv_result["efficient_frontier"]
            )
            result["charts"]["efficient_frontier"] = chart_file
        
        # Black-Litterman Optimization
        if portfolio_data.get("investor_views"):
            bl_result = self._optimize_black_litterman(price_data, portfolio_data)
            result["black_litterman"] = bl_result
            
            # BL comparison chart with optimal pie chart
            chart_file = self.chart_generator.create_bl_comparison_chart(
                bl_result, bl_result["optimal_weights"]
            )
            result["charts"]["bl_comparison"] = chart_file
        
        return result
    
    def _optimize_mean_variance(self, price_data: pd.DataFrame, portfolio_data: Dict) -> Dict:
        """Mean-Variance optimization"""
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        optimal_weights = ef.clean_weights()
        
        # Generate efficient frontier with exactly 100 portfolios
        frontier_portfolios = []
        
        # Get optimal portfolio info
        ef_optimal = EfficientFrontier(mu, S)
        ef_optimal.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        optimal_perf = ef_optimal.portfolio_performance(verbose=False, risk_free_rate=self.config.risk_free_rate)
        
        # Generate 100 portfolios using risk targets
        min_vol_ef = EfficientFrontier(mu, S)
        min_vol_ef.min_volatility()
        min_vol_perf = min_vol_ef.portfolio_performance(verbose=False)
        
        # Use volatility range
        min_vol = min_vol_perf[1]
        max_vol = min_vol * 3
        target_vols = np.linspace(min_vol, max_vol, 100)
        
        for target_vol in target_vols:
            try:
                ef_temp = EfficientFrontier(mu, S)
                ef_temp.efficient_risk(target_vol)
                weights = ef_temp.clean_weights()
                perf = ef_temp.portfolio_performance(verbose=False, risk_free_rate=self.config.risk_free_rate)
                
                # Don't mark as optimal here - we'll do it later
                is_optimal = False
                
                frontier_portfolios.append({
                    "expected_return": perf[0],
                    "standard_deviation": perf[1],
                    "sharpe_ratio": perf[2],
                    "weights": weights,
                    "is_optimal": is_optimal
                })
            except:
                continue
        
        # Always ensure we have the exact optimal portfolio
        if frontier_portfolios:
            # Find the portfolio with highest Sharpe ratio and replace it with exact optimal
            max_sharpe_idx = max(range(len(frontier_portfolios)), 
                               key=lambda i: frontier_portfolios[i]["sharpe_ratio"])
            frontier_portfolios[max_sharpe_idx] = {
                "expected_return": optimal_perf[0],
                "standard_deviation": optimal_perf[1],
                "sharpe_ratio": optimal_perf[2],
                "weights": optimal_weights,
                "is_optimal": True
            }
        
        # Ensure exactly 100 portfolios
        frontier_portfolios = frontier_portfolios[:100]
        
        # Calculate metrics for provided and optimal portfolios
        provided_portfolio = None
        if portfolio_data.get("allocations"):
            provided_metrics = self._calculate_portfolio_metrics(
                portfolio_data["allocations"], mu, S, self.config.risk_free_rate
            )
            provided_portfolio = {
                "weights": portfolio_data["allocations"],
                "metrics": provided_metrics
            }
        
        # Use the same performance values from PyPortfolioOpt for consistency
        ef_perf = EfficientFrontier(mu, S)
        ef_perf.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        perf_values = ef_perf.portfolio_performance(verbose=False, risk_free_rate=self.config.risk_free_rate)
        
        optimal_portfolio = {
            "weights": optimal_weights,
            "metrics": {
                "expected_return": perf_values[0],
                "standard_deviation": perf_values[1],
                "sharpe_ratio": perf_values[2],
                "risk_free_rate": self.config.risk_free_rate
            }
        }
        
        result = {
            "optimal_weights": optimal_weights,
            "efficient_frontier": frontier_portfolios,
            "expected_returns": mu.to_dict(),
            "covariance_matrix": S.to_dict(),
            "optimal_portfolio": optimal_portfolio
        }
        
        if provided_portfolio:
            result["provided_portfolio"] = provided_portfolio
        
        return result
    
    def _optimize_black_litterman(self, price_data: pd.DataFrame, portfolio_data: Dict) -> Dict:
        """Black-Litterman optimization"""
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        
        # Get market caps and calculate market weights
        market_caps = self.market_cap_loader.get_market_caps(portfolio_data["tickers"])
        total_cap = sum(market_caps.values())
        market_weights = pd.Series({t: market_caps[t]/total_cap for t in portfolio_data["tickers"]})
        
        # Calculate market-implied returns
        risk_aversion = 3.0
        pi = risk_aversion * S @ market_weights
        
        # Prepare views
        views_dict = {}
        confidences = {}
        for ticker, view_data in portfolio_data.get("investor_views", {}).items():
            if isinstance(view_data, dict):
                views_dict[ticker] = view_data["expected_return"]
                confidences[ticker] = view_data.get("confidence", 0.5)
        
        # Create Q and P matrices for Black-Litterman
        if views_dict:
            # Q: view returns, P: picking matrix
            tickers = list(pi.index)
            Q = np.array([views_dict[ticker] for ticker in views_dict.keys()])
            P = np.zeros((len(views_dict), len(tickers)))
            
            for i, ticker in enumerate(views_dict.keys()):
                ticker_idx = tickers.index(ticker)
                P[i, ticker_idx] = 1.0
            
            # Omega: view uncertainty matrix
            omega_diag = [1.0 / confidences[ticker] for ticker in views_dict.keys()]
            omega = np.diag(omega_diag)
            
            bl = BlackLittermanModel(S, pi=pi, Q=Q, P=P, omega=omega, tau=self.config.tau)
        else:
            # No views - use market equilibrium
            bl = BlackLittermanModel(S, pi=pi, tau=self.config.tau)
        
        # Get posterior estimates
        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        # Optimize
        ef = EfficientFrontier(mu_bl, S_bl)
        ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        optimal_weights = ef.clean_weights()
        
        # Calculate metrics for provided and optimal portfolios
        provided_portfolio = None
        if portfolio_data.get("allocations"):
            provided_metrics = self._calculate_portfolio_metrics(
                portfolio_data["allocations"], mu_bl, S_bl, self.config.risk_free_rate
            )
            provided_portfolio = {
                "weights": portfolio_data["allocations"],
                "metrics": provided_metrics
            }
        
        optimal_metrics = self._calculate_portfolio_metrics(
            optimal_weights, mu_bl, S_bl, self.config.risk_free_rate
        )
        optimal_portfolio = {
            "weights": optimal_weights,
            "metrics": optimal_metrics
        }
        
        result = {
            "optimal_weights": optimal_weights,
            "market_caps": market_caps,
            "market_weights": market_weights.to_dict(),
            "market_implied_returns": pi.to_dict(),
            "posterior_returns": mu_bl.to_dict(),
            "views_applied": len(views_dict) > 0,
            "view_details": {
                ticker: {"expected_return": views_dict[ticker], "confidence": confidences[ticker]}
                for ticker in views_dict
            } if views_dict else {},
            "optimal_portfolio": optimal_portfolio
        }
        
        if provided_portfolio:
            result["provided_portfolio"] = provided_portfolio
        
        return result
    
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            # Convert weights to array
            tickers = list(expected_returns.index)
            weights_array = np.array([weights.get(ticker, 0) for ticker in tickers])
            
            # Normalize weights
            if weights_array.sum() > 0:
                weights_array = weights_array / weights_array.sum()
            
            # Calculate metrics
            portfolio_return = weights_array @ expected_returns
            portfolio_variance = weights_array @ cov_matrix @ weights_array
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                "expected_return": float(portfolio_return),
                "standard_deviation": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "risk_free_rate": float(risk_free_rate)
            }
        except Exception as e:
            return {
                "expected_return": 0.0,
                "standard_deviation": 0.0,
                "sharpe_ratio": 0.0,
                "risk_free_rate": float(risk_free_rate)
            }
    



def create_optimizer(config: Optional[Config] = None) -> StreamlinedOptimizer:
    """Factory function"""
    return StreamlinedOptimizer(config or Config())


def main_example():
    """Streamlined example"""
    print("🚀 Streamlined Portfolio Optimizer")
    
    portfolio_data = {
        "tickers": ["ANZ.AX", "CBA.AX", "MQG.AX", "NAB.AX", "RIO.AX", "WOW.AX"],
        "allocations": {  # Provided portfolio for comparison
            "ANZ.AX": 0.15,
            "CBA.AX": 0.25,
            "MQG.AX": 0.15,
            "NAB.AX": 0.15,
            "RIO.AX": 0.20,
            "WOW.AX": 0.10
        },
        "investor_views": {  # For Black-Litterman
            "CBA.AX": {"expected_return": 0.10, "confidence": 0.8},
            "RIO.AX": {"expected_return": 0.06, "confidence": 0.6}
        },
        "start_date": "2022-01-01",
        "end_date": "2024-12-31"
    }
    
    optimizer = create_optimizer()
    result = optimizer.optimize_portfolio(portfolio_data)
    
    if result["status"] == "success":
        print("✅ Optimization completed")
        print(f"📊 Charts generated: {list(result['charts'].keys())}")
        
        # Save results
        with open("streamlined_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print("💾 Results saved to streamlined_results.json")
    
    return result


if __name__ == "__main__":
    main_example()