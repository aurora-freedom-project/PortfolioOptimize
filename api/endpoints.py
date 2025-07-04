# api/endpoints.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import os
import tempfile
import asyncio
from datetime import datetime, timedelta
import uuid

# Import core modules
from core.optimization.mean_variance import optimize_mean_variance
from core.optimization.black_litterman import optimize_black_litterman
from core.optimization.hrp import optimize_hrp
from core.optimization.advanced_optimization import run_advanced_optimization
from core.backtesting import run_comprehensive_backtest, PortfolioBacktester
from core.analytics import generate_comprehensive_analytics_report
from core.reporting import generate_client_report
from shell.display.advanced_visualization import create_institutional_report_visualizations

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="Professional portfolio optimization and analytics API for institutional investors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class AssetAllocation(BaseModel):
    ticker: str
    weight: float = Field(..., ge=0, le=1, description="Weight between 0 and 1")

class OptimizationRequest(BaseModel):
    price_data: Dict[str, List[float]] = Field(..., description="Historical price data by ticker")
    dates: List[str] = Field(..., description="Date strings in YYYY-MM-DD format")
    current_allocations: List[AssetAllocation] = Field(..., description="Current portfolio allocations")
    method: str = Field(..., description="Optimization method")
    risk_free_rate: float = Field(0.02, ge=0, le=0.1, description="Risk-free rate")
    constraints: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Weight constraints by ticker")
    
    @validator('method')
    def validate_method(cls, v):
        allowed_methods = ['mean_variance', 'black_litterman', 'hrp', 'max_sharpe_l2', 
                          'min_cvar', 'semivariance', 'risk_parity', 'market_neutral', 'cla']
        if v not in allowed_methods:
            raise ValueError(f"Method must be one of {allowed_methods}")
        return v

class BlackLittermanRequest(OptimizationRequest):
    views: Dict[str, float] = Field({}, description="Investor views on expected returns")
    view_confidences: Dict[str, float] = Field({}, description="Confidence levels for views")

class BacktestRequest(BaseModel):
    price_data: Dict[str, List[float]]
    dates: List[str]
    initial_allocations: List[AssetAllocation]
    strategies: Dict[str, Dict[str, Any]] = Field(..., description="Strategy configurations")
    rebalance_frequency: str = Field('M', description="Rebalancing frequency: D, W, M, Q")
    lookback_window: int = Field(252, ge=60, le=1000, description="Lookback window in days")

class AnalyticsRequest(BaseModel):
    price_data: Dict[str, List[float]]
    dates: List[str]
    portfolio_weights: List[AssetAllocation]
    benchmark_data: Optional[Dict[str, List[float]]] = None
    benchmark_dates: Optional[List[str]] = None
    risk_free_rate: float = Field(0.02, ge=0, le=0.1)

class ReportRequest(BaseModel):
    analytics_data: AnalyticsRequest
    backtest_data: Optional[BacktestRequest] = None
    client_name: str = Field("Institutional Client", description="Client name for report")
    report_type: str = Field("QUARTERLY", description="Report type: MONTHLY, QUARTERLY, ANNUAL")

# Response models
class OptimizationResponse(BaseModel):
    provided_portfolio: Dict[str, Any]
    optimal_portfolio: Dict[str, Any]
    method: str
    method_info: Optional[Dict[str, Any]] = None
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_free_rate: float
    efficient_frontier_portfolios: Optional[List[Dict[str, Any]]] = None

class BacktestResponse(BaseModel):
    strategy_comparison: Dict[str, Any]
    rolling_analysis: Optional[Dict[str, Any]]
    backtest_config: Dict[str, Any]
    best_strategy: Optional[str]

class AnalyticsResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    risk_attribution: Dict[str, Any]
    performance_attribution: Dict[str, Any]
    rolling_analysis: Dict[str, Any]
    stress_test_results: Dict[str, Any]
    analysis_date: str
    analytics_version: str

# Helper functions
def convert_to_dataframe(price_data: Dict[str, List[float]], dates: List[str]) -> pd.DataFrame:
    """Convert API price data to pandas DataFrame."""
    date_index = pd.to_datetime(dates)
    df = pd.DataFrame(price_data, index=date_index)
    return df

def convert_allocations_to_dict(allocations: List[AssetAllocation]) -> Dict[str, float]:
    """Convert allocation list to dictionary."""
    return {alloc.ticker: alloc.weight for alloc in allocations}

def validate_data_consistency(price_data: Dict[str, List[float]], dates: List[str]):
    """Validate that all price series have the same length as dates."""
    expected_length = len(dates)
    for ticker, prices in price_data.items():
        if len(prices) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=f"Price data for {ticker} has {len(prices)} points, expected {expected_length}"
            )

# Job tracking for long-running operations
class JobTracker:
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, job_type: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            'id': job_id,
            'type': job_type,
            'status': 'RUNNING',
            'created_at': datetime.now(),
            'result': None,
            'error': None
        }
        return job_id
    
    def update_job(self, job_id: str, status: str, result: Any = None, error: str = None):
        if job_id in self.jobs:
            self.jobs[job_id].update({
                'status': status,
                'result': result,
                'error': error,
                'updated_at': datetime.now()
            })
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

job_tracker = JobTracker()

# API Endpoints

@app.get("/", summary="API Health Check")
async def root():
    """API health check endpoint."""
    return {
        "message": "Portfolio Optimization API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """Detailed health check with system information."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "fastapi": "Available"
        },
        "endpoints": {
            "optimization": "/optimize",
            "backtesting": "/backtest", 
            "analytics": "/analytics",
            "reporting": "/reports"
        }
    }

@app.post("/optimize", response_model=OptimizationResponse, summary="Portfolio Optimization")
async def optimize_portfolio(request: OptimizationRequest):
    """
    Optimize portfolio using specified method.
    
    Supports multiple optimization methods:
    - mean_variance: Classic Markowitz optimization
    - black_litterman: Black-Litterman model
    - hrp: Hierarchical Risk Parity
    - max_sharpe_l2: Max Sharpe with L2 regularization
    - min_cvar: Minimize Conditional Value at Risk
    - semivariance: Semivariance optimization
    - risk_parity: Risk parity optimization
    - market_neutral: Market neutral strategies
    - cla: Critical Line Algorithm
    """
    try:
        # Validate data consistency
        validate_data_consistency(request.price_data, request.dates)
        
        # Convert to DataFrame
        price_df = convert_to_dataframe(request.price_data, request.dates)
        allocations_dict = convert_allocations_to_dict(request.current_allocations)
        tickers = list(price_df.columns)
        
        # Convert constraints format if provided
        constraints = None
        if request.constraints:
            constraints = {ticker: (bounds['min'], bounds['max']) 
                         for ticker, bounds in request.constraints.items()}
        
        # Run optimization based on method
        if request.method == 'mean_variance':
            result = optimize_mean_variance(price_df, tickers, allocations_dict, request.risk_free_rate)
        
        elif request.method == 'black_litterman':
            # For regular optimization request, use empty views
            result = optimize_black_litterman(
                price_df, tickers, allocations_dict, {}, {}, request.risk_free_rate
            )
        
        elif request.method == 'hrp':
            result = optimize_hrp(price_df, tickers, allocations_dict)
        
        elif request.method in ['max_sharpe_l2', 'min_cvar', 'semivariance', 'risk_parity', 'market_neutral', 'cla']:
            result = run_advanced_optimization(
                price_df, tickers, allocations_dict, request.method,
                constraints=constraints, risk_free_rate=request.risk_free_rate
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown optimization method: {request.method}")
        
        return OptimizationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/optimize/black-litterman", response_model=OptimizationResponse, summary="Black-Litterman Optimization")
async def optimize_black_litterman_with_views(request: BlackLittermanRequest):
    """
    Black-Litterman optimization with investor views.
    
    Allows specification of:
    - Expected return views for specific assets
    - Confidence levels for each view
    """
    try:
        validate_data_consistency(request.price_data, request.dates)
        
        price_df = convert_to_dataframe(request.price_data, request.dates)
        allocations_dict = convert_allocations_to_dict(request.current_allocations)
        tickers = list(price_df.columns)
        
        result = optimize_black_litterman(
            price_df, tickers, allocations_dict, 
            request.views, request.view_confidences, request.risk_free_rate
        )
        
        return OptimizationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Black-Litterman optimization failed: {str(e)}")

@app.post("/backtest", summary="Portfolio Backtesting")
async def backtest_strategies(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Comprehensive backtesting of multiple portfolio strategies.
    
    Returns job ID for long-running backtests. Use /jobs/{job_id} to check status.
    """
    try:
        validate_data_consistency(request.price_data, request.dates)
        
        # Create background job for long-running backtest
        job_id = job_tracker.create_job("BACKTEST")
        
        async def run_backtest_job():
            try:
                price_df = convert_to_dataframe(request.price_data, request.dates)
                initial_allocations = convert_allocations_to_dict(request.initial_allocations)
                
                result = run_comprehensive_backtest(
                    price_df, initial_allocations, request.strategies,
                    request.rebalance_frequency, request.lookback_window
                )
                
                job_tracker.update_job(job_id, "COMPLETED", result)
                
            except Exception as e:
                job_tracker.update_job(job_id, "FAILED", error=str(e))
        
        background_tasks.add_task(run_backtest_job)
        
        return {
            "job_id": job_id,
            "status": "RUNNING",
            "message": "Backtest started. Use /jobs/{job_id} to check status."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest initialization failed: {str(e)}")

@app.post("/analytics", response_model=AnalyticsResponse, summary="Portfolio Analytics")
async def generate_analytics(request: AnalyticsRequest):
    """
    Generate comprehensive portfolio analytics and risk metrics.
    
    Includes:
    - Performance metrics (Sharpe, Sortino, Calmar ratios)
    - Risk attribution analysis
    - Stress testing results
    - Rolling performance analysis
    """
    try:
        validate_data_consistency(request.price_data, request.dates)
        
        price_df = convert_to_dataframe(request.price_data, request.dates)
        portfolio_weights = convert_allocations_to_dict(request.portfolio_weights)
        
        benchmark_df = None
        if request.benchmark_data and request.benchmark_dates:
            validate_data_consistency(request.benchmark_data, request.benchmark_dates)
            benchmark_df = convert_to_dataframe(request.benchmark_data, request.benchmark_dates)
        
        result = generate_comprehensive_analytics_report(
            price_df, portfolio_weights, benchmark_df, request.risk_free_rate
        )
        
        return AnalyticsResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")

@app.post("/reports/generate", summary="Generate Institutional Report")
async def generate_institutional_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """
    Generate comprehensive institutional report with visualizations.
    
    Creates:
    - Executive summary
    - Detailed analytics
    - Risk analysis
    - Strategy recommendations
    - Professional visualizations
    """
    try:
        job_id = job_tracker.create_job("REPORT_GENERATION")
        
        async def generate_report_job():
            try:
                # Generate analytics
                price_df = convert_to_dataframe(request.analytics_data.price_data, request.analytics_data.dates)
                portfolio_weights = convert_allocations_to_dict(request.analytics_data.portfolio_weights)
                
                benchmark_df = None
                if request.analytics_data.benchmark_data and request.analytics_data.benchmark_dates:
                    benchmark_df = convert_to_dataframe(
                        request.analytics_data.benchmark_data, 
                        request.analytics_data.benchmark_dates
                    )
                
                analytics_result = generate_comprehensive_analytics_report(
                    price_df, portfolio_weights, benchmark_df, request.analytics_data.risk_free_rate
                )
                
                # Generate backtest if provided
                backtest_result = {}
                optimization_results = []
                
                if request.backtest_data:
                    backtest_price_df = convert_to_dataframe(
                        request.backtest_data.price_data, 
                        request.backtest_data.dates
                    )
                    initial_allocations = convert_allocations_to_dict(request.backtest_data.initial_allocations)
                    
                    backtest_result = run_comprehensive_backtest(
                        backtest_price_df, initial_allocations, request.backtest_data.strategies,
                        request.backtest_data.rebalance_frequency, request.backtest_data.lookback_window
                    )
                
                # Generate report files
                temp_dir = tempfile.mkdtemp()
                
                deliverables = generate_client_report(
                    price_df, backtest_result, optimization_results, 
                    portfolio_weights, request.client_name, temp_dir
                )
                
                job_tracker.update_job(job_id, "COMPLETED", {
                    "analytics": analytics_result,
                    "backtest": backtest_result,
                    "deliverables": deliverables,
                    "temp_directory": temp_dir
                })
                
            except Exception as e:
                job_tracker.update_job(job_id, "FAILED", error=str(e))
        
        background_tasks.add_task(generate_report_job)
        
        return {
            "job_id": job_id,
            "status": "RUNNING",
            "message": "Report generation started. Use /jobs/{job_id} to check status."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/jobs/{job_id}", summary="Check Job Status")
async def get_job_status(job_id: str):
    """
    Check the status of a background job (backtest or report generation).
    """
    job = job_tracker.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Don't return the full result in status check to avoid large responses
    response = {
        "job_id": job["id"],
        "type": job["type"],
        "status": job["status"],
        "created_at": job["created_at"].isoformat()
    }
    
    if job["status"] == "FAILED":
        response["error"] = job["error"]
    elif job["status"] == "COMPLETED":
        response["message"] = "Job completed successfully. Use /jobs/{job_id}/result to get results."
        if "updated_at" in job:
            response["completed_at"] = job["updated_at"].isoformat()
    
    return response

@app.get("/jobs/{job_id}/result", summary="Get Job Result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed job.
    """
    job = job_tracker.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] == "RUNNING":
        raise HTTPException(status_code=202, detail="Job still running")
    
    if job["status"] == "FAILED":
        raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")
    
    return job["result"]

@app.get("/jobs/{job_id}/download/{file_type}", summary="Download Report Files")
async def download_report_file(job_id: str, file_type: str):
    """
    Download generated report files.
    
    Available file types:
    - comprehensive_report_json
    - executive_dashboard
    - text_report
    - comprehensive_dashboard
    - efficient_frontier
    - interactive_dashboard
    """
    job = job_tracker.get_job(job_id)
    
    if not job or job["status"] != "COMPLETED":
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    
    if job["type"] != "REPORT_GENERATION":
        raise HTTPException(status_code=400, detail="Job is not a report generation job")
    
    deliverables = job["result"].get("deliverables", {})
    
    if file_type not in deliverables:
        available_types = list(deliverables.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"File type '{file_type}' not found. Available: {available_types}"
        )
    
    file_path = deliverables[file_type]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine media type based on file extension
    media_type = "application/octet-stream"
    if file_path.endswith('.json'):
        media_type = "application/json"
    elif file_path.endswith('.txt'):
        media_type = "text/plain"
    elif file_path.endswith('.html'):
        media_type = "text/html"
    elif file_path.endswith('.png'):
        media_type = "image/png"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=os.path.basename(file_path)
    )

@app.get("/methods", summary="Available Optimization Methods")
async def get_optimization_methods():
    """
    Get list of available optimization methods with descriptions.
    """
    methods = {
        "mean_variance": {
            "name": "Mean-Variance Optimization",
            "description": "Classic Markowitz optimization for efficient frontier",
            "parameters": ["risk_free_rate"],
            "constraints_supported": True
        },
        "black_litterman": {
            "name": "Black-Litterman Model",
            "description": "Bayesian approach incorporating investor views",
            "parameters": ["risk_free_rate", "views", "view_confidences"],
            "constraints_supported": False
        },
        "hrp": {
            "name": "Hierarchical Risk Parity",
            "description": "Machine learning-based diversification approach",
            "parameters": [],
            "constraints_supported": False
        },
        "max_sharpe_l2": {
            "name": "L2 Regularized Max Sharpe",
            "description": "Maximum Sharpe ratio with L2 regularization for diversification",
            "parameters": ["risk_free_rate", "gamma"],
            "constraints_supported": True
        },
        "min_cvar": {
            "name": "Minimum CVaR",
            "description": "Minimize Conditional Value at Risk (tail risk focus)",
            "parameters": ["confidence_level"],
            "constraints_supported": True
        },
        "semivariance": {
            "name": "Semivariance Optimization",
            "description": "Downside risk optimization using semivariance",
            "parameters": ["risk_free_rate"],
            "constraints_supported": True
        },
        "risk_parity": {
            "name": "Risk Parity",
            "description": "Equal risk contribution portfolio",
            "parameters": ["rp_method"],
            "constraints_supported": False
        },
        "market_neutral": {
            "name": "Market Neutral",
            "description": "Market neutral long-short portfolio",
            "parameters": ["target_volatility", "long_short_ratio"],
            "constraints_supported": False
        },
        "cla": {
            "name": "Critical Line Algorithm",
            "description": "Exact solution for mean-variance optimization",
            "parameters": ["target_return"],
            "constraints_supported": False
        }
    }
    
    return {
        "methods": methods,
        "total_methods": len(methods),
        "categories": {
            "classical": ["mean_variance", "black_litterman"],
            "alternative": ["hrp", "risk_parity"],
            "advanced": ["max_sharpe_l2", "min_cvar", "semivariance", "market_neutral", "cla"]
        }
    }

@app.get("/examples", summary="API Usage Examples")
async def get_usage_examples():
    """
    Get example requests for different API endpoints.
    """
    examples = {
        "optimization_request": {
            "price_data": {
                "AAPL": [150.0, 152.0, 148.0, 155.0],
                "GOOGL": [2800.0, 2820.0, 2790.0, 2850.0],
                "MSFT": [300.0, 305.0, 298.0, 310.0]
            },
            "dates": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "current_allocations": [
                {"ticker": "AAPL", "weight": 0.4},
                {"ticker": "GOOGL", "weight": 0.3},
                {"ticker": "MSFT", "weight": 0.3}
            ],
            "method": "mean_variance",
            "risk_free_rate": 0.02
        },
        "backtest_request": {
            "price_data": "# Same as optimization_request",
            "dates": "# Same as optimization_request", 
            "initial_allocations": "# Same as optimization_request",
            "strategies": {
                "mean_variance": {"method": "mean_variance"},
                "risk_parity": {"method": "risk_parity", "rp_method": "equal_marginal_contrib"}
            },
            "rebalance_frequency": "M",
            "lookback_window": 126
        },
        "analytics_request": {
            "price_data": "# Same as optimization_request",
            "dates": "# Same as optimization_request",
            "portfolio_weights": "# Same as current_allocations",
            "risk_free_rate": 0.02
        }
    }
    
    return {
        "examples": examples,
        "endpoints": {
            "optimization": "POST /optimize",
            "black_litterman": "POST /optimize/black-litterman", 
            "backtesting": "POST /backtest",
            "analytics": "POST /analytics",
            "reports": "POST /reports/generate"
        }
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)