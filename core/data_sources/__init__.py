# core/data_sources/__init__.py
"""
Data Sources Module
==================

Comprehensive data integration supporting multiple sources:
- Offline CSV/Excel files
- Real-time market data feeds (Yahoo Finance, Alpha Vantage, Bloomberg, etc.)
- Database storage and retrieval (PostgreSQL, SQLite, MongoDB)
- Financial data APIs (yfinance, quandl, FRED, etc.)
- Custom data connectors

Designed for both offline analysis and real-time trading systems.
"""

from .file_data import FileDataSource

# Optional imports for market feeds (only if dependencies are available)
try:
    from .market_feeds import (
        YahooFinanceDataSource,
        AlphaVantageDataSource,
        QuandlDataSource,
        FREDDataSource
    )
    MARKET_FEEDS_AVAILABLE = True
except ImportError:
    MARKET_FEEDS_AVAILABLE = False

# Optional imports for database (only if dependencies are available)
try:
    from .database import DatabaseDataSource
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Optional imports for data manager (only if dependencies are available)
try:
    from .data_manager import DataManager, DataSourceType
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False

# Build __all__ list based on available imports
__all__ = ['FileDataSource']

if MARKET_FEEDS_AVAILABLE:
    __all__.extend([
        'YahooFinanceDataSource', 
        'AlphaVantageDataSource',
        'QuandlDataSource',
        'FREDDataSource'
    ])

if DATABASE_AVAILABLE:
    __all__.append('DatabaseDataSource')

if DATA_MANAGER_AVAILABLE:
    __all__.extend(['DataManager', 'DataSourceType'])