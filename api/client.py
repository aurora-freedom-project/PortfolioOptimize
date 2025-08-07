# api/client.py
import requests
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings

class PortfolioOptimizationClient:
    """
    Python client for the Portfolio Optimization API.
    
    Provides a convenient interface for hedge funds, asset managers, and institutional
    investors to access portfolio optimization, backtesting, and analytics services.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the Portfolio Optimization API
            api_key: API key for authentication (if required)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health and get system information."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_available_methods(self) -> Dict[str, Any]:
        """Get list of available optimization methods."""
        response = self.session.get(f"{self.base_url}/methods")
        response.raise_for_status()
        return response.json()
    
    def get_usage_examples(self) -> Dict[str, Any]:
        """Get API usage examples."""
        response = self.session.get(f"{self.base_url}/examples")
        response.raise_for_status()
        return response.json()
    
    def optimize_portfolio(self, 
                          price_data: Union[pd.DataFrame, Dict[str, List[float]]],
                          current_allocations: Dict[str, float],
                          method: str = "mean_variance",
                          risk_free_rate: float = 0.02,
                          constraints: Optional[Dict[str, Dict[str, float]]] = None,
                          **method_kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method.
        
        Args:
            price_data: Historical price data (DataFrame with datetime index or dict)
            current_allocations: Current portfolio allocations {ticker: weight}
            method: Optimization method
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            constraints: Weight constraints {ticker: {'min': float, 'max': float}}
            **method_kwargs: Additional method-specific parameters
        
        Returns:
            Optimization results including optimal weights and metrics
        """
        
        # Convert DataFrame to API format if needed
        if isinstance(price_data, pd.DataFrame):
            price_dict = {col: price_data[col].tolist() for col in price_data.columns}
            dates = [d.strftime('%Y-%m-%d') for d in price_data.index]
        else:
            price_dict = price_data
            if 'dates' not in method_kwargs:
                raise ValueError("Dates must be provided when using price_data as dict")
            dates = method_kwargs.pop('dates')
        
        # Convert allocations to API format
        allocations_list = [{"ticker": ticker, "weight": weight} 
                           for ticker, weight in current_allocations.items()]
        
        # Prepare request
        request_data = {
            "price_data": price_dict,
            "dates": dates,
            "current_allocations": allocations_list,
            "method": method,
            "risk_free_rate": risk_free_rate
        }
        
        if constraints:
            request_data["constraints"] = constraints
        
        # Add method-specific parameters
        request_data.update(method_kwargs)
        
        # Make request
        response = self.session.post(f"{self.base_url}/optimize", json=request_data)
        response.raise_for_status()
        return response.json()
    
    def optimize_black_litterman(self,
                                price_data: Union[pd.DataFrame, Dict[str, List[float]]],
                                current_allocations: Dict[str, float],
                                views: Dict[str, float],
                                view_confidences: Dict[str, float],
                                risk_free_rate: float = 0.02,
                                dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Black-Litterman optimization with investor views.
        
        Args:
            price_data: Historical price data
            current_allocations: Current portfolio allocations
            views: Expected return views {ticker: expected_return}
            view_confidences: Confidence in views {ticker: confidence_level}
            risk_free_rate: Risk-free rate
            dates: Date strings (required if price_data is dict)
        
        Returns:
            Black-Litterman optimization results
        """
        
        # Convert DataFrame to API format if needed
        if isinstance(price_data, pd.DataFrame):
            price_dict = {col: price_data[col].tolist() for col in price_data.columns}
            dates = [d.strftime('%Y-%m-%d') for d in price_data.index]
        else:
            price_dict = price_data
            if dates is None:
                raise ValueError("Dates must be provided when using price_data as dict")
        
        allocations_list = [{"ticker": ticker, "weight": weight} 
                           for ticker, weight in current_allocations.items()]
        
        request_data = {
            "price_data": price_dict,
            "dates": dates,
            "current_allocations": allocations_list,
            "method": "black_litterman",
            "risk_free_rate": risk_free_rate,
            "views": views,
            "view_confidences": view_confidences
        }
        
        response = self.session.post(f"{self.base_url}/optimize/black-litterman", json=request_data)
        response.raise_for_status()
        return response.json()
    
    def run_backtest(self,
                    price_data: Union[pd.DataFrame, Dict[str, List[float]]],
                    initial_allocations: Dict[str, float],
                    strategies: Dict[str, Dict[str, Any]],
                    rebalance_frequency: str = 'M',
                    lookback_window: int = 252,
                    dates: Optional[List[str]] = None,
                    poll_interval: int = 5,
                    max_wait_time: int = 300) -> Dict[str, Any]:
        """
        Run comprehensive backtest of multiple strategies.
        
        Args:
            price_data: Historical price data
            initial_allocations: Starting portfolio allocations
            strategies: Strategy configurations {name: {method: str, **params}}
            rebalance_frequency: Rebalancing frequency ('D', 'W', 'M', 'Q')
            lookback_window: Lookback window in days
            dates: Date strings (required if price_data is dict)
            poll_interval: Polling interval for job status (seconds)
            max_wait_time: Maximum wait time for completion (seconds)
        
        Returns:
            Comprehensive backtest results
        """
        
        # Convert DataFrame to API format if needed
        if isinstance(price_data, pd.DataFrame):
            price_dict = {col: price_data[col].tolist() for col in price_data.columns}
            dates = [d.strftime('%Y-%m-%d') for d in price_data.index]
        else:
            price_dict = price_data
            if dates is None:
                raise ValueError("Dates must be provided when using price_data as dict")
        
        allocations_list = [{"ticker": ticker, "weight": weight} 
                           for ticker, weight in initial_allocations.items()]
        
        request_data = {
            "price_data": price_dict,
            "dates": dates,
            "initial_allocations": allocations_list,
            "strategies": strategies,
            "rebalance_frequency": rebalance_frequency,
            "lookback_window": lookback_window
        }
        
        # Start backtest job
        response = self.session.post(f"{self.base_url}/backtest", json=request_data)
        response.raise_for_status()
        job_info = response.json()
        job_id = job_info["job_id"]
        
        print(f"Backtest started with job ID: {job_id}")
        
        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            status_response.raise_for_status()
            status = status_response.json()
            
            if status["status"] == "COMPLETED":
                # Get results
                result_response = self.session.get(f"{self.base_url}/jobs/{job_id}/result")
                result_response.raise_for_status()
                return result_response.json()
            
            elif status["status"] == "FAILED":
                raise RuntimeError(f"Backtest failed: {status.get('error', 'Unknown error')}")
            
            print(f"Backtest running... ({status['status']})")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Backtest did not complete within {max_wait_time} seconds")
    
    def generate_analytics(self,
                          price_data: Union[pd.DataFrame, Dict[str, List[float]]],
                          portfolio_weights: Dict[str, float],
                          benchmark_data: Optional[Union[pd.DataFrame, Dict[str, List[float]]]] = None,
                          risk_free_rate: float = 0.02,
                          dates: Optional[List[str]] = None,
                          benchmark_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio analytics.
        
        Args:
            price_data: Historical price data
            portfolio_weights: Portfolio weights {ticker: weight}
            benchmark_data: Benchmark price data (optional)
            risk_free_rate: Risk-free rate
            dates: Date strings (required if price_data is dict)
            benchmark_dates: Benchmark date strings
        
        Returns:
            Comprehensive analytics results
        """
        
        # Convert DataFrame to API format if needed
        if isinstance(price_data, pd.DataFrame):
            price_dict = {col: price_data[col].tolist() for col in price_data.columns}
            dates = [d.strftime('%Y-%m-%d') for d in price_data.index]
        else:
            price_dict = price_data
            if dates is None:
                raise ValueError("Dates must be provided when using price_data as dict")
        
        weights_list = [{"ticker": ticker, "weight": weight} 
                       for ticker, weight in portfolio_weights.items()]
        
        request_data = {
            "price_data": price_dict,
            "dates": dates,
            "portfolio_weights": weights_list,
            "risk_free_rate": risk_free_rate
        }
        
        # Add benchmark data if provided
        if benchmark_data is not None:
            if isinstance(benchmark_data, pd.DataFrame):
                benchmark_dict = {col: benchmark_data[col].tolist() for col in benchmark_data.columns}
                benchmark_dates = [d.strftime('%Y-%m-%d') for d in benchmark_data.index]
            else:
                benchmark_dict = benchmark_data
                if benchmark_dates is None:
                    raise ValueError("Benchmark dates must be provided when using benchmark_data as dict")
            
            request_data["benchmark_data"] = benchmark_dict
            request_data["benchmark_dates"] = benchmark_dates
        
        response = self.session.post(f"{self.base_url}/analytics", json=request_data)
        response.raise_for_status()
        return response.json()
    
    def generate_institutional_report(self,
                                    price_data: Union[pd.DataFrame, Dict[str, List[float]]],
                                    portfolio_weights: Dict[str, float],
                                    client_name: str = "Institutional Client",
                                    report_type: str = "QUARTERLY",
                                    benchmark_data: Optional[Union[pd.DataFrame, Dict[str, List[float]]]] = None,
                                    backtest_config: Optional[Dict[str, Any]] = None,
                                    dates: Optional[List[str]] = None,
                                    poll_interval: int = 10,
                                    max_wait_time: int = 600) -> Dict[str, Any]:
        """
        Generate comprehensive institutional report.
        
        Args:
            price_data: Historical price data
            portfolio_weights: Portfolio weights
            client_name: Client name for report
            report_type: Report type ('MONTHLY', 'QUARTERLY', 'ANNUAL')
            benchmark_data: Benchmark data (optional)
            backtest_config: Backtest configuration (optional)
            dates: Date strings (required if price_data is dict)
            poll_interval: Polling interval for job status (seconds)
            max_wait_time: Maximum wait time for completion (seconds)
        
        Returns:
            Report generation results with file paths
        """
        
        # Prepare analytics request
        if isinstance(price_data, pd.DataFrame):
            price_dict = {col: price_data[col].tolist() for col in price_data.columns}
            dates = [d.strftime('%Y-%m-%d') for d in price_data.index]
        else:
            price_dict = price_data
            if dates is None:
                raise ValueError("Dates must be provided when using price_data as dict")
        
        weights_list = [{"ticker": ticker, "weight": weight} 
                       for ticker, weight in portfolio_weights.items()]
        
        analytics_request = {
            "price_data": price_dict,
            "dates": dates,
            "portfolio_weights": weights_list,
            "risk_free_rate": 0.02
        }
        
        if benchmark_data is not None:
            if isinstance(benchmark_data, pd.DataFrame):
                benchmark_dict = {col: benchmark_data[col].tolist() for col in benchmark_data.columns}
                benchmark_dates = [d.strftime('%Y-%m-%d') for d in benchmark_data.index]
            else:
                benchmark_dict = benchmark_data
                benchmark_dates = dates  # Assume same dates if not provided
            
            analytics_request["benchmark_data"] = benchmark_dict
            analytics_request["benchmark_dates"] = benchmark_dates
        
        request_data = {
            "analytics_data": analytics_request,
            "client_name": client_name,
            "report_type": report_type
        }
        
        if backtest_config:
            request_data["backtest_data"] = backtest_config
        
        # Start report generation job
        response = self.session.post(f"{self.base_url}/reports/generate", json=request_data)
        response.raise_for_status()
        job_info = response.json()
        job_id = job_info["job_id"]
        
        print(f"Report generation started with job ID: {job_id}")
        
        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            status_response.raise_for_status()
            status = status_response.json()
            
            if status["status"] == "COMPLETED":
                # Get results
                result_response = self.session.get(f"{self.base_url}/jobs/{job_id}/result")
                result_response.raise_for_status()
                return result_response.json()
            
            elif status["status"] == "FAILED":
                raise RuntimeError(f"Report generation failed: {status.get('error', 'Unknown error')}")
            
            print(f"Report generation running... ({status['status']})")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Report generation did not complete within {max_wait_time} seconds")
    
    def download_report_file(self, job_id: str, file_type: str, save_path: str) -> str:
        """
        Download a specific report file.
        
        Args:
            job_id: Job ID from report generation
            file_type: Type of file to download
            save_path: Local path to save the file
        
        Returns:
            Path where file was saved
        """
        
        response = self.session.get(f"{self.base_url}/jobs/{job_id}/download/{file_type}")
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {file_type} to {save_path}")
        return save_path
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a background job."""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get result of a completed job."""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()

# Convenience functions for common workflows
def quick_optimize(price_data: pd.DataFrame, 
                   current_weights: Dict[str, float],
                   method: str = "mean_variance",
                   api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Quick portfolio optimization with minimal setup.
    
    Args:
        price_data: DataFrame with price data (datetime index, ticker columns)
        current_weights: Current portfolio weights {ticker: weight}
        method: Optimization method
        api_url: API base URL
    
    Returns:
        Optimization results
    """
    client = PortfolioOptimizationClient(api_url)
    return client.optimize_portfolio(price_data, current_weights, method)

def quick_backtest(price_data: pd.DataFrame,
                   initial_weights: Dict[str, float],
                   methods: List[str] = ["mean_variance", "hrp", "risk_parity"],
                   api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Quick backtesting of multiple strategies.
    
    Args:
        price_data: DataFrame with price data
        initial_weights: Initial portfolio weights
        methods: List of optimization methods to test
        api_url: API base URL
    
    Returns:
        Backtest results
    """
    client = PortfolioOptimizationClient(api_url)
    
    strategies = {method: {"method": method} for method in methods}
    
    return client.run_backtest(
        price_data, initial_weights, strategies,
        rebalance_frequency='M', lookback_window=252
    )

def compare_optimization_methods(price_data: pd.DataFrame,
                                current_weights: Dict[str, float],
                                methods: List[str] = None,
                                api_url: str = "http://localhost:8000") -> pd.DataFrame:
    """
    Compare multiple optimization methods side by side.
    
    Args:
        price_data: DataFrame with price data
        current_weights: Current portfolio weights
        methods: List of methods to compare (None for all available)
        api_url: API base URL
    
    Returns:
        DataFrame comparing methods and their key metrics
    """
    client = PortfolioOptimizationClient(api_url)
    
    if methods is None:
        available_methods = client.get_available_methods()
        methods = list(available_methods['methods'].keys())
    
    results = []
    
    for method in methods:
        try:
            result = client.optimize_portfolio(price_data, current_weights, method)
            optimal_metrics = result['optimal_portfolio']['metrics']
            
            results.append({
                'method': method,
                'expected_return': optimal_metrics.get('expected_return', 0),
                'standard_deviation': optimal_metrics.get('standard_deviation', 0),
                'sharpe_ratio': optimal_metrics.get('sharpe_ratio', 0),
                'max_drawdown': optimal_metrics.get('max_drawdown', 0)
            })
            
        except Exception as e:
            print(f"Error optimizing with {method}: {e}")
            continue
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Example usage of the client
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'AAPL': 150 + np.cumsum(np.random.randn(100) * 0.02),
        'GOOGL': 2800 + np.cumsum(np.random.randn(100) * 0.03),
        'MSFT': 300 + np.cumsum(np.random.randn(100) * 0.025),
    }, index=dates)
    
    current_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
    
    # Initialize client
    client = PortfolioOptimizationClient("http://localhost:8000")
    
    # Check API health
    try:
        health = client.health_check()
        print("API Health:", health['status'])
        
        # Run optimization
        result = client.optimize_portfolio(
            sample_data, current_weights, method="mean_variance"
        )
        
        print("Optimization successful!")
        print("Optimal weights:", result['optimal_portfolio']['weights'])
        print("Sharpe ratio:", result['optimal_portfolio']['metrics']['sharpe_ratio'])
        
    except requests.exceptions.ConnectionError:
        print("API server not running. Start the server with: uvicorn api.endpoints:app --reload")