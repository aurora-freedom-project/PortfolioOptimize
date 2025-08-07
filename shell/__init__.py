# shell/__init__.py
from .app import run_portfolio_optimization
from .cli import (
    parse_args,
    parse_tickers,
    parse_allocations,
    parse_constraints,
    parse_investor_views,
    display_portfolio_results
)

__all__ = [
    'run_portfolio_optimization',
    'parse_args',
    'parse_tickers',
    'parse_allocations',
    'parse_constraints', 
    'parse_investor_views',
    'display_portfolio_results'
]