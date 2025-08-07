# core/reporting.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import warnings

from core.analytics import generate_comprehensive_analytics_report
from shell.display.advanced_visualization import create_institutional_report_visualizations

@dataclass
class ExecutiveSummary:
    """Executive summary for institutional reports."""
    portfolio_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    tracking_error: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]
    information_ratio: Optional[float]
    
@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    expected_shortfall: float
    maximum_drawdown: float
    downside_deviation: float
    sortino_ratio: float
    calmar_ratio: float
    tail_ratio: float
    skewness: float
    kurtosis: float

@dataclass
class PerformanceAttribution:
    """Performance attribution analysis."""
    asset_contributions: Dict[str, float]
    sector_contributions: Dict[str, float]
    factor_contributions: Dict[str, float]
    interaction_effects: float
    selection_effect: float
    allocation_effect: float

class InstitutionalReportGenerator:
    """
    Professional reporting system for hedge funds, asset managers, and institutional investors.
    Generates comprehensive reports with executive summaries, detailed analytics, and visualizations.
    """
    
    def __init__(self, price_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None):
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.report_timestamp = datetime.now()
        
    def generate_comprehensive_report(self, 
                                    backtest_results: Dict[str, Any],
                                    optimization_results: List[Dict[str, Any]],
                                    portfolio_weights: Dict[str, float],
                                    client_name: str = "Institutional Client",
                                    report_type: str = "QUARTERLY") -> Dict[str, Any]:
        """
        Generate a comprehensive institutional report.
        
        Args:
            backtest_results: Results from backtesting analysis
            optimization_results: Results from various optimization methods
            portfolio_weights: Current portfolio weights
            client_name: Name of the client/fund
            report_type: Type of report (MONTHLY, QUARTERLY, ANNUAL)
        """
        
        # Generate analytics
        analytics_report = generate_comprehensive_analytics_report(
            self.price_data, portfolio_weights, self.benchmark_data
        )
        
        # Create executive summary
        executive_summary = self._create_executive_summary(analytics_report, backtest_results)
        
        # Risk analysis
        risk_metrics = self._create_risk_metrics(analytics_report)
        
        # Performance attribution
        performance_attribution = self._create_performance_attribution(analytics_report, portfolio_weights)
        
        # Strategy comparison
        strategy_analysis = self._analyze_strategy_performance(backtest_results, optimization_results)
        
        # Stress testing results
        stress_test_results = self._compile_stress_test_results(analytics_report)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            executive_summary, risk_metrics, strategy_analysis
        )
        
        # Market outlook and commentary
        market_commentary = self._generate_market_commentary(analytics_report)
        
        # Compliance and regulatory
        compliance_metrics = self._calculate_compliance_metrics(portfolio_weights, analytics_report)
        
        # Create the main report structure
        comprehensive_report = {
            "report_metadata": {
                "client_name": client_name,
                "report_type": report_type,
                "reporting_period": {
                    "start": self.price_data.index[0].strftime('%Y-%m-%d'),
                    "end": self.price_data.index[-1].strftime('%Y-%m-%d')
                },
                "generation_timestamp": self.report_timestamp.isoformat(),
                "report_version": "2.0.0"
            },
            
            "executive_summary": asdict(executive_summary),
            
            "performance_overview": {
                "portfolio_performance": analytics_report['performance_metrics'],
                "benchmark_comparison": self._create_benchmark_comparison(analytics_report),
                "attribution_analysis": asdict(performance_attribution),
                "rolling_performance": analytics_report.get('rolling_analysis', {})
            },
            
            "risk_analysis": {
                "risk_metrics": asdict(risk_metrics),
                "risk_attribution": analytics_report['risk_attribution'],
                "stress_testing": stress_test_results,
                "scenario_analysis": analytics_report.get('stress_test_results', {})
            },
            
            "strategy_analysis": strategy_analysis,
            
            "portfolio_construction": {
                "current_weights": portfolio_weights,
                "weight_constraints": self._analyze_weight_constraints(portfolio_weights),
                "diversification_metrics": self._calculate_diversification_metrics(portfolio_weights, analytics_report),
                "turnover_analysis": self._analyze_turnover(backtest_results)
            },
            
            "market_outlook": market_commentary,
            
            "recommendations": recommendations,
            
            "compliance_and_risk_management": compliance_metrics,
            
            "appendices": {
                "methodology": self._create_methodology_appendix(),
                "data_sources": self._create_data_sources_appendix(),
                "glossary": self._create_glossary(),
                "disclaimers": self._create_disclaimers()
            }
        }
        
        return comprehensive_report
    
    def generate_executive_dashboard(self, comprehensive_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a high-level executive dashboard for C-suite consumption.
        """
        exec_summary = comprehensive_report['executive_summary']
        
        dashboard = {
            "key_metrics": {
                "portfolio_value": exec_summary['portfolio_value'],
                "ytd_return": exec_summary['total_return'],
                "sharpe_ratio": exec_summary['sharpe_ratio'],
                "max_drawdown": exec_summary['max_drawdown'],
                "var_95": exec_summary['var_95']
            },
            
            "performance_vs_benchmark": {
                "outperformance": exec_summary.get('alpha', 0),
                "tracking_error": exec_summary.get('tracking_error', 0),
                "information_ratio": exec_summary.get('information_ratio', 0)
            },
            
            "risk_indicators": {
                "risk_level": self._categorize_risk_level(exec_summary),
                "concentration_risk": self._assess_concentration_risk(comprehensive_report),
                "liquidity_risk": self._assess_liquidity_risk(comprehensive_report)
            },
            
            "top_recommendations": comprehensive_report['recommendations'][:3],
            
            "alerts": self._generate_alerts(comprehensive_report)
        }
        
        return dashboard
    
    def export_to_pdf(self, comprehensive_report: Dict[str, Any], 
                     output_path: str, include_charts: bool = True) -> str:
        """
        Export comprehensive report to PDF format.
        Note: This is a placeholder - actual implementation would use reportlab or similar.
        """
        
        # In a real implementation, this would generate a professional PDF
        # For now, we'll create a detailed text summary
        
        report_content = self._format_report_for_export(comprehensive_report)
        
        # Save as text file (PDF generation would require additional libraries)
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w') as f:
            f.write(report_content)
        
        if include_charts:
            # Generate visualizations
            try:
                chart_dir = output_path.replace('.pdf', '_charts')
                import os
                os.makedirs(chart_dir, exist_ok=True)
                
                # This would call the visualization module
                print(f"Charts would be generated in: {chart_dir}")
                
            except Exception as e:
                print(f"Error generating charts: {e}")
        
        return txt_path
    
    def _create_executive_summary(self, analytics_report: Dict[str, Any], 
                                backtest_results: Dict[str, Any]) -> ExecutiveSummary:
        """Create executive summary from analytics."""
        
        perf_metrics = analytics_report['performance_metrics']
        
        # Extract benchmark-related metrics if available
        tracking_error = perf_metrics.get('tracking_error')
        information_ratio = perf_metrics.get('information_ratio')
        
        # Estimate portfolio value (simplified)
        portfolio_value = 10_000_000  # $10M base assumption
        
        return ExecutiveSummary(
            portfolio_value=portfolio_value,
            total_return=perf_metrics['expected_return'],
            annualized_return=perf_metrics['expected_return'],
            volatility=perf_metrics['standard_deviation'],
            sharpe_ratio=perf_metrics['sharpe_ratio'],
            max_drawdown=perf_metrics['max_drawdown'],
            var_95=perf_metrics['var_95'],
            tracking_error=tracking_error,
            alpha=information_ratio * tracking_error if (information_ratio and tracking_error) else None,
            beta=None,  # Would need benchmark correlation
            information_ratio=information_ratio
        )
    
    def _create_risk_metrics(self, analytics_report: Dict[str, Any]) -> RiskMetrics:
        """Create comprehensive risk metrics."""
        
        perf_metrics = analytics_report['performance_metrics']
        
        return RiskMetrics(
            value_at_risk_95=perf_metrics['var_95'],
            value_at_risk_99=perf_metrics['var_99'],
            conditional_var_95=perf_metrics['cvar_95'],
            conditional_var_99=perf_metrics['cvar_99'],
            expected_shortfall=perf_metrics['cvar_95'],  # ES is typically CVaR
            maximum_drawdown=perf_metrics['max_drawdown'],
            downside_deviation=perf_metrics['downside_deviation'],
            sortino_ratio=perf_metrics['sortino_ratio'],
            calmar_ratio=perf_metrics['calmar_ratio'],
            tail_ratio=perf_metrics['tail_ratio'],
            skewness=perf_metrics['skewness'],
            kurtosis=perf_metrics['kurtosis']
        )
    
    def _create_performance_attribution(self, analytics_report: Dict[str, Any], 
                                      portfolio_weights: Dict[str, float]) -> PerformanceAttribution:
        """Create performance attribution analysis."""
        
        perf_attr = analytics_report.get('performance_attribution', {})
        
        return PerformanceAttribution(
            asset_contributions=perf_attr.get('asset_contributions', {}),
            sector_contributions=perf_attr.get('sector_attribution', {}),
            factor_contributions=perf_attr.get('factor_attribution', {}),
            interaction_effects=perf_attr.get('interaction_effect', 0),
            selection_effect=0,  # Would need benchmark weights for calculation
            allocation_effect=0   # Would need benchmark weights for calculation
        )
    
    def _analyze_strategy_performance(self, backtest_results: Dict[str, Any],
                                    optimization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance across different strategies."""
        
        strategy_comparison = backtest_results.get('strategy_comparison', {})
        individual_results = strategy_comparison.get('individual_results', {})
        
        strategy_rankings = []
        
        for strategy_name, results in individual_results.items():
            if results and 'summary' in results:
                summary = results['summary']
                strategy_rankings.append({
                    'strategy': strategy_name,
                    'total_return': summary['total_return'],
                    'sharpe_ratio': summary['sharpe_ratio'],
                    'max_drawdown': summary['max_drawdown'],
                    'volatility': summary['annualized_volatility'],
                    'turnover': summary['avg_turnover']
                })
        
        # Sort by Sharpe ratio
        strategy_rankings.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return {
            'strategy_rankings': strategy_rankings,
            'best_strategy': backtest_results.get('best_strategy'),
            'strategy_comparison_summary': strategy_comparison.get('comparison', {}),
            'rolling_analysis': backtest_results.get('rolling_analysis', {})
        }
    
    def _compile_stress_test_results(self, analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Compile stress testing results."""
        
        stress_results = analytics_report.get('stress_test_results', {})
        
        # Add interpretation and risk assessment
        interpreted_results = {}
        
        for scenario, results in stress_results.items():
            portfolio_return = results.get('portfolio_return', 0)
            impact = results.get('impact_vs_normal', 0)
            
            # Categorize severity
            if impact < -0.20:
                severity = "SEVERE"
            elif impact < -0.10:
                severity = "MODERATE"
            elif impact < -0.05:
                severity = "MILD"
            else:
                severity = "MINIMAL"
            
            interpreted_results[scenario] = {
                **results,
                'severity': severity,
                'risk_rating': self._calculate_scenario_risk_rating(portfolio_return, impact)
            }
        
        return {
            'scenario_results': interpreted_results,
            'worst_case_scenario': min(stress_results.items(), 
                                     key=lambda x: x[1].get('portfolio_return', 0))[0] if stress_results else None,
            'stress_test_summary': {
                'total_scenarios': len(stress_results),
                'severe_scenarios': len([s for s in interpreted_results.values() if s['severity'] == 'SEVERE']),
                'avg_impact': np.mean([s.get('impact_vs_normal', 0) for s in stress_results.values()]) if stress_results else 0
            }
        }
    
    def _generate_recommendations(self, executive_summary: ExecutiveSummary,
                                risk_metrics: RiskMetrics,
                                strategy_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on analysis."""
        
        recommendations = []
        
        # Risk-based recommendations
        if executive_summary.sharpe_ratio < 0.5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk Management',
                'title': 'Improve Risk-Adjusted Returns',
                'description': f'Current Sharpe ratio of {executive_summary.sharpe_ratio:.2f} is below optimal levels. Consider rebalancing towards higher risk-adjusted return assets.',
                'action_items': [
                    'Review asset allocation strategy',
                    'Consider alternative risk models',
                    'Evaluate factor exposures'
                ]
            })
        
        if abs(executive_summary.max_drawdown) > 0.20:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk Management',
                'title': 'Reduce Maximum Drawdown Risk',
                'description': f'Maximum drawdown of {executive_summary.max_drawdown:.1%} exceeds risk tolerance. Implement downside protection strategies.',
                'action_items': [
                    'Consider hedging strategies',
                    'Implement stop-loss mechanisms',
                    'Evaluate portfolio diversification'
                ]
            })
        
        # Volatility-based recommendations
        if executive_summary.volatility > 0.25:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Portfolio Construction',
                'title': 'Optimize Portfolio Volatility',
                'description': f'Portfolio volatility of {executive_summary.volatility:.1%} may be higher than necessary for the return profile.',
                'action_items': [
                    'Consider risk parity approaches',
                    'Evaluate correlation structure',
                    'Review position sizing'
                ]
            })
        
        # Strategy-based recommendations
        best_strategy = strategy_analysis.get('best_strategy')
        if best_strategy:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Strategy Enhancement',
                'title': f'Consider {best_strategy} Strategy Implementation',
                'description': f'Backtesting indicates {best_strategy} provides superior risk-adjusted returns.',
                'action_items': [
                    f'Evaluate implementation costs for {best_strategy}',
                    'Conduct forward-testing',
                    'Assess operational requirements'
                ]
            })
        
        # Performance-based recommendations
        if executive_summary.information_ratio and executive_summary.information_ratio < 0.5:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Performance Enhancement',
                'title': 'Enhance Alpha Generation',
                'description': 'Information ratio suggests limited alpha generation relative to benchmark.',
                'action_items': [
                    'Review factor exposures',
                    'Consider alternative data sources',
                    'Evaluate active management strategies'
                ]
            })
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_market_commentary(self, analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market outlook and commentary."""
        
        perf_metrics = analytics_report['performance_metrics']
        
        # Market regime analysis based on portfolio metrics
        volatility = perf_metrics['standard_deviation']
        skewness = perf_metrics['skewness']
        kurtosis = perf_metrics['kurtosis']
        
        if volatility > 0.25 and kurtosis > 3:
            market_regime = "HIGH_VOLATILITY"
            regime_description = "Markets are experiencing elevated volatility with increased tail risk."
        elif volatility < 0.15 and abs(skewness) < 0.5:
            market_regime = "LOW_VOLATILITY"
            regime_description = "Markets are in a relatively stable, low-volatility environment."
        else:
            market_regime = "TRANSITIONAL"
            regime_description = "Markets are in a transitional phase with mixed signals."
        
        return {
            'market_regime': market_regime,
            'regime_description': regime_description,
            'key_observations': [
                f"Portfolio volatility at {volatility:.1%} reflects current market conditions",
                f"Skewness of {skewness:.2f} indicates {'negative' if skewness < 0 else 'positive'} return distribution bias",
                f"Kurtosis of {kurtosis:.2f} suggests {'higher' if kurtosis > 3 else 'lower'} than normal tail risk"
            ],
            'outlook': self._generate_market_outlook(market_regime),
            'risk_factors': [
                'Geopolitical tensions',
                'Central bank policy changes',
                'Economic growth uncertainty',
                'Market liquidity conditions'
            ]
        }
    
    def _calculate_compliance_metrics(self, portfolio_weights: Dict[str, float],
                                    analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance and regulatory metrics."""
        
        # Concentration limits
        max_single_position = max(portfolio_weights.values()) if portfolio_weights else 0
        top_5_concentration = sum(sorted(portfolio_weights.values(), reverse=True)[:5])
        
        # Risk metrics for compliance
        risk_metrics = analytics_report['performance_metrics']
        
        compliance_status = {
            'concentration_compliance': {
                'max_single_position': max_single_position,
                'max_position_limit': 0.20,  # 20% limit
                'compliant': max_single_position <= 0.20,
                'top_5_concentration': top_5_concentration,
                'top_5_limit': 0.60,  # 60% limit
                'top_5_compliant': top_5_concentration <= 0.60
            },
            'risk_compliance': {
                'var_95': risk_metrics['var_95'],
                'var_limit': -0.05,  # 5% daily VaR limit
                'var_compliant': risk_metrics['var_95'] >= -0.05,
                'max_drawdown': risk_metrics['max_drawdown'],
                'drawdown_limit': -0.25,  # 25% max drawdown limit
                'drawdown_compliant': risk_metrics['max_drawdown'] >= -0.25
            },
            'operational_metrics': {
                'turnover_compliance': True,  # Simplified
                'liquidity_compliance': True,  # Simplified
                'leverage_compliance': True   # Simplified
            }
        }
        
        # Overall compliance score
        all_checks = [
            compliance_status['concentration_compliance']['compliant'],
            compliance_status['concentration_compliance']['top_5_compliant'],
            compliance_status['risk_compliance']['var_compliant'],
            compliance_status['risk_compliance']['drawdown_compliant']
        ]
        
        compliance_score = sum(all_checks) / len(all_checks)
        
        return {
            'compliance_status': compliance_status,
            'overall_compliance_score': compliance_score,
            'compliance_rating': 'COMPLIANT' if compliance_score >= 0.8 else 'NEEDS_ATTENTION',
            'violations': [check for check, status in zip(['Max Position', 'Top 5 Concentration', 'VaR', 'Max Drawdown'], all_checks) if not status]
        }
    
    def _categorize_risk_level(self, exec_summary) -> str:
        """Categorize overall risk level."""
        
        risk_score = 0
        
        # Handle both dict and object input
        volatility = exec_summary.get('volatility') if isinstance(exec_summary, dict) else getattr(exec_summary, 'volatility', 0)
        max_drawdown = exec_summary.get('max_drawdown') if isinstance(exec_summary, dict) else getattr(exec_summary, 'max_drawdown', 0)
        sharpe_ratio = exec_summary.get('sharpe_ratio') if isinstance(exec_summary, dict) else getattr(exec_summary, 'sharpe_ratio', 0)
        
        # Volatility component
        if volatility and volatility > 0.25:
            risk_score += 2
        elif volatility and volatility > 0.15:
            risk_score += 1
        
        # Drawdown component
        if max_drawdown and abs(max_drawdown) > 0.20:
            risk_score += 2
        elif max_drawdown and abs(max_drawdown) > 0.10:
            risk_score += 1
        
        # Sharpe ratio component (inverse)
        if sharpe_ratio and sharpe_ratio < 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_concentration_risk(self, comprehensive_report: Dict[str, Any]) -> str:
        """Assess portfolio concentration risk."""
        
        portfolio_construction = comprehensive_report.get('portfolio_construction', {})
        diversification_metrics = portfolio_construction.get('diversification_metrics', {})
        
        herfindahl_index = diversification_metrics.get('herfindahl_index', 0)
        
        if herfindahl_index > 0.25:
            return "HIGH"
        elif herfindahl_index > 0.15:
            return "MEDIUM" 
        else:
            return "LOW"
    
    def _assess_liquidity_risk(self, comprehensive_report: Dict[str, Any]) -> str:
        """Assess portfolio liquidity risk (simplified)."""
        # This is simplified - real implementation would analyze actual liquidity metrics
        return "MEDIUM"
    
    def _generate_alerts(self, comprehensive_report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate important alerts for executive attention."""
        
        alerts = []
        
        exec_summary = comprehensive_report['executive_summary']
        compliance = comprehensive_report['compliance_and_risk_management']
        
        # Risk alerts
        if exec_summary['sharpe_ratio'] < 0.3:
            alerts.append({
                'type': 'RISK',
                'severity': 'HIGH',
                'message': f"Sharpe ratio of {exec_summary['sharpe_ratio']:.2f} is critically low"
            })
        
        # Compliance alerts
        if compliance['compliance_rating'] == 'NEEDS_ATTENTION':
            alerts.append({
                'type': 'COMPLIANCE',
                'severity': 'HIGH',
                'message': f"Compliance violations detected: {', '.join(compliance['violations'])}"
            })
        
        # Performance alerts
        if exec_summary['total_return'] < -0.10:
            alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'MEDIUM',
                'message': f"Portfolio down {exec_summary['total_return']:.1%} for the period"
            })
        
        return alerts
    
    # Additional helper methods
    def _create_benchmark_comparison(self, analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create benchmark comparison analysis."""
        return {
            'relative_performance': analytics_report['performance_metrics'].get('information_ratio', 0),
            'tracking_error': analytics_report['performance_metrics'].get('tracking_error', 0),
            'active_share': 0.5,  # Simplified
            'beta': 1.0  # Simplified
        }
    
    def _analyze_weight_constraints(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze weight constraints and concentration."""
        weights = list(portfolio_weights.values())
        return {
            'max_weight': max(weights) if weights else 0,
            'min_weight': min(weights) if weights else 0,
            'weight_range': max(weights) - min(weights) if weights else 0,
            'num_positions': len([w for w in weights if w > 0.01])  # Positions > 1%
        }
    
    def _calculate_diversification_metrics(self, portfolio_weights: Dict[str, float],
                                         analytics_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate diversification metrics."""
        weights = list(portfolio_weights.values())
        
        herfindahl_index = sum(w**2 for w in weights) if weights else 0
        effective_assets = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        risk_attribution = analytics_report.get('risk_attribution', {})
        
        return {
            'herfindahl_index': herfindahl_index,
            'effective_assets': effective_assets,
            'diversification_ratio': risk_attribution.get('diversification_ratio', 1.0),
            'concentration_ratio': risk_attribution.get('concentration_ratio', 0)
        }
    
    def _analyze_turnover(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio turnover patterns."""
        strategy_comparison = backtest_results.get('strategy_comparison', {})
        individual_results = strategy_comparison.get('individual_results', {})
        
        turnover_data = {}
        for strategy, results in individual_results.items():
            if results and 'summary' in results:
                turnover_data[strategy] = results['summary'].get('avg_turnover', 0)
        
        return {
            'strategy_turnovers': turnover_data,
            'avg_turnover': np.mean(list(turnover_data.values())) if turnover_data else 0,
            'turnover_range': max(turnover_data.values()) - min(turnover_data.values()) if turnover_data else 0
        }
    
    def _calculate_scenario_risk_rating(self, portfolio_return: float, impact: float) -> str:
        """Calculate risk rating for stress test scenarios."""
        if portfolio_return < -0.30 or impact < -0.25:
            return "EXTREME_RISK"
        elif portfolio_return < -0.20 or impact < -0.15:
            return "HIGH_RISK"
        elif portfolio_return < -0.10 or impact < -0.10:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"
    
    def _generate_market_outlook(self, market_regime: str) -> str:
        """Generate market outlook based on regime."""
        outlooks = {
            "HIGH_VOLATILITY": "Expect continued volatility with defensive positioning recommended",
            "LOW_VOLATILITY": "Stable environment supports growth-oriented strategies",
            "TRANSITIONAL": "Mixed signals suggest cautious optimism with active monitoring"
        }
        return outlooks.get(market_regime, "Market conditions require careful monitoring")
    
    def _create_methodology_appendix(self) -> Dict[str, str]:
        """Create methodology appendix."""
        return {
            'optimization_methods': 'Mean-Variance, Black-Litterman, HRP, and advanced optimization techniques',
            'risk_models': 'Historical covariance, factor models, and stress testing scenarios',
            'performance_attribution': 'Asset-level and factor-based attribution analysis',
            'backtesting': 'Walk-forward analysis with out-of-sample validation'
        }
    
    def _create_data_sources_appendix(self) -> Dict[str, str]:
        """Create data sources appendix."""
        return {
            'price_data': 'Historical price data from reliable market data providers',
            'risk_factors': 'Standard risk factor models and custom factor analysis',
            'benchmarks': 'Industry-standard benchmark indices',
            'economic_data': 'Macroeconomic indicators from central banks and statistical agencies'
        }
    
    def _create_glossary(self) -> Dict[str, str]:
        """Create glossary of terms."""
        return {
            'Sharpe Ratio': 'Risk-adjusted return measure (excess return / volatility)',
            'VaR': 'Value at Risk - potential loss at given confidence level',
            'CVaR': 'Conditional Value at Risk - expected loss beyond VaR threshold',
            'Maximum Drawdown': 'Largest peak-to-trough decline in portfolio value',
            'Information Ratio': 'Excess return relative to benchmark per unit of tracking error',
            'Sortino Ratio': 'Risk-adjusted return using downside deviation instead of total volatility'
        }
    
    def _create_disclaimers(self) -> List[str]:
        """Create standard disclaimers."""
        return [
            "Past performance is not indicative of future results",
            "This analysis is based on historical data and model assumptions",
            "Risk metrics are estimates and actual risk may differ",
            "Investment decisions should consider individual circumstances and risk tolerance",
            "This report is for informational purposes only and does not constitute investment advice"
        ]
    
    def _format_report_for_export(self, comprehensive_report: Dict[str, Any]) -> str:
        """Format comprehensive report for text export."""
        
        report_content = f"""
INSTITUTIONAL PORTFOLIO ANALYSIS REPORT
=====================================

Client: {comprehensive_report['report_metadata']['client_name']}
Report Type: {comprehensive_report['report_metadata']['report_type']}
Period: {comprehensive_report['report_metadata']['reporting_period']['start']} to {comprehensive_report['report_metadata']['reporting_period']['end']}
Generated: {comprehensive_report['report_metadata']['generation_timestamp']}

EXECUTIVE SUMMARY
=================
Portfolio Value: ${comprehensive_report['executive_summary']['portfolio_value']:,.0f}
Total Return: {comprehensive_report['executive_summary']['total_return']:.2%}
Sharpe Ratio: {comprehensive_report['executive_summary']['sharpe_ratio']:.2f}
Maximum Drawdown: {comprehensive_report['executive_summary']['max_drawdown']:.2%}
VaR (95%): {comprehensive_report['executive_summary']['var_95']:.2%}

RISK ANALYSIS
=============
Value at Risk (95%): {comprehensive_report['risk_analysis']['risk_metrics']['value_at_risk_95']:.2%}
Conditional VaR (95%): {comprehensive_report['risk_analysis']['risk_metrics']['conditional_var_95']:.2%}
Sortino Ratio: {comprehensive_report['risk_analysis']['risk_metrics']['sortino_ratio']:.2f}
Calmar Ratio: {comprehensive_report['risk_analysis']['risk_metrics']['calmar_ratio']:.2f}

STRATEGY ANALYSIS
=================
Best Strategy: {comprehensive_report['strategy_analysis']['best_strategy']}

TOP RECOMMENDATIONS
===================
"""
        
        for i, rec in enumerate(comprehensive_report['recommendations'][:5], 1):
            report_content += f"""
{i}. {rec['title']} (Priority: {rec['priority']})
   Category: {rec['category']}
   Description: {rec['description']}
"""
        
        report_content += f"""

COMPLIANCE STATUS
=================
Overall Compliance Score: {comprehensive_report['compliance_and_risk_management']['overall_compliance_score']:.1%}
Compliance Rating: {comprehensive_report['compliance_and_risk_management']['compliance_rating']}

DISCLAIMERS
===========
"""
        
        for disclaimer in comprehensive_report['appendices']['disclaimers']:
            report_content += f"• {disclaimer}\n"
        
        return report_content


def generate_client_report(price_data: pd.DataFrame,
                          backtest_results: Dict[str, Any],
                          optimization_results: List[Dict[str, Any]],
                          portfolio_weights: Dict[str, float],
                          client_name: str = "Institutional Client",
                          output_directory: str = "./reports/") -> Dict[str, str]:
    """
    Generate complete client report package with all deliverables.
    
    Returns:
        Dictionary mapping deliverable names to file paths
    """
    
    report_generator = InstitutionalReportGenerator(price_data)
    
    # Generate comprehensive report
    comprehensive_report = report_generator.generate_comprehensive_report(
        backtest_results, optimization_results, portfolio_weights, client_name
    )
    
    # Generate executive dashboard
    executive_dashboard = report_generator.generate_executive_dashboard(comprehensive_report)
    
    # Create output directory
    import os
    os.makedirs(output_directory, exist_ok=True)
    
    deliverables = {}
    
    try:
        # 1. Comprehensive Report (JSON)
        json_path = f"{output_directory}/comprehensive_report.json"
        with open(json_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        deliverables['comprehensive_report_json'] = json_path
        
        # 2. Executive Dashboard (JSON) 
        dashboard_path = f"{output_directory}/executive_dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(executive_dashboard, f, indent=2, default=str)
        deliverables['executive_dashboard'] = dashboard_path
        
        # 3. Text Report
        text_path = f"{output_directory}/institutional_report.txt"
        text_report = report_generator._format_report_for_export(comprehensive_report)
        with open(text_path, 'w') as f:
            f.write(text_report)
        deliverables['text_report'] = text_path
        
        # 4. Generate visualizations
        chart_files = create_institutional_report_visualizations(
            backtest_results, optimization_results, output_directory
        )
        deliverables.update(chart_files)
        
        print(f"Generated {len(deliverables)} report deliverables for {client_name}")
        
    except Exception as e:
        print(f"Error generating client reports: {e}")
    
    return deliverables