# core/models.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

class OptimizationMethod:
    """Available optimization methods."""
    MEAN_VARIANCE = "MEAN_VARIANCE"
    BLACK_LITTERMAN = "BLACK_LITTERMAN"
    HIERARCHICAL_RISK_PARITY = "HIERARCHICAL_RISK_PARITY"

class PortfolioModel(BaseModel):
    """Portfolio optimization input model."""
    
    tickers: List[str] = Field(
        description="List of stock tickers in the portfolio")
    
    allocations: Dict[str, float] = Field(
        default_factory=dict,
        description="Weight allocation for each ticker (total = 1)")
    
    constraints: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Minimum and maximum weight limits for each ticker")
    
    start_date: Union[int, str] = Field(
        description="Analysis start date (YYYY-MM-DD, ISO string, or timestamp)")
    
    end_date: Union[int, str] = Field(
        description="Analysis end date (YYYY-MM-DD, ISO string, or timestamp)")

    risk_free_rate: float = Field(
        default=0.02,
        description="Risk-free rate of return (e.g., 0.02 for 2%)")
    
    optimization_method: str = Field(
        default=OptimizationMethod.MEAN_VARIANCE,
        description="Optimization method to use")
    
    investor_views: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Investor views for Black-Litterman model")
    
    # Field validators
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Check ticker list is valid."""
        if not v:
            raise ValueError("Ticker list cannot be empty")
        
        if len(v) != len(set(v)):
            raise ValueError("Ticker list contains duplicate values")
        
        return v
    
    @field_validator('allocations')
    @classmethod
    def validate_allocations(cls, v, info):
        """Validate allocation weights."""
        tickers = info.data.get('tickers', [])
        
        for ticker in v.keys():
            if ticker not in tickers:
                raise ValueError(f"Ticker '{ticker}' in allocations is not in the tickers list")
        
        if v and abs(sum(v.values()) - 1.0) > 0.0001:  
            raise ValueError(f"Total allocation weight ({sum(v.values())}) must be 1")
        
        for ticker, allocation in v.items():
            if allocation < 0 or allocation > 1:
                raise ValueError(f"Weight for '{ticker}' ({allocation}) must be >= 0 and <= 1")
        
        return v
    
    @model_validator(mode='after')
    def validate_dates_order(self):
        """Validate date order and minimum time period."""
        # Convert dates to datetime objects
        start_date = self._parse_date(self.start_date)
        end_date = self._parse_date(self.end_date)
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        # Determine minimum years based on optimization method
        if self.optimization_method == OptimizationMethod.MEAN_VARIANCE:
            min_required_years = 3
            method_name = "Mean-Variance"
        elif self.optimization_method == OptimizationMethod.BLACK_LITTERMAN:
            min_required_years = 5
            method_name = "Black-Litterman"
        else:
            min_required_years = 3  # Default is 3 years
            method_name = "Hierarchical Risk Parity"
        
        # Calculate minimum required end date
        min_end_date = start_date.replace(year=start_date.year + min_required_years)
        
        # Check if end date meets minimum requirement
        if end_date < min_end_date:
            days_short = (min_end_date - end_date).days
            raise ValueError(
                f"Time period for {method_name} optimization must be at least {min_required_years} years. "
                f"Currently missing {days_short} days."
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_constraints(self):
        """Validate constraints against allocations."""
        for ticker, allocation in self.allocations.items():
            if ticker in self.constraints:
                min_val, max_val = self.constraints[ticker]
                if allocation < min_val or allocation > max_val:
                    raise ValueError(
                        f"Allocation for '{ticker}' ({allocation}) must be within "
                        f"constraints range [{min_val}, {max_val}]"
                    )
        return self
    
    def _parse_date(self, date_value):
        """Parse date from various formats to datetime."""
        if isinstance(date_value, datetime):
            return date_value
            
        if isinstance(date_value, str):
            if date_value.isdigit():
                # Unix timestamp
                return datetime.fromtimestamp(int(date_value))
            
            # Try ISO format
            try:
                return datetime.fromisoformat(date_value)
            except ValueError:
                # Try YYYY-MM-DD format
                try:
                    return datetime.strptime(date_value, "%Y-%m-%d")
                except ValueError as e:
                    raise ValueError(f"Invalid date format: {e}")
        
        if isinstance(date_value, int):
            return datetime.fromtimestamp(date_value)
            
        raise ValueError(f"Unsupported date format: {type(date_value)}")