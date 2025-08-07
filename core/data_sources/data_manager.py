# core/data_sources/data_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import warnings
import asyncio
import threading
import time
import os

from .file_data import FileDataSource
from .market_feeds import create_market_data_source, BaseMarketDataSource
from .database import DatabaseDataSource

class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    FILE = "file"
    YAHOO = "yahoo"
    ALPHAVANTAGE = "alphavantage"
    QUANDL = "quandl"
    FRED = "fred"
    CRYPTO = "crypto"
    DATABASE = "database"
    REALTIME = "realtime"

class DataManager:
    """
    Unified data manager supporting both offline and real-time data sources.
    
    Features:
    - Multiple data source integration
    - Automatic fallback between sources
    - Data caching and persistence
    - Real-time data streaming
    - Data quality monitoring
    - Automatic data updates
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 cache_directory: str = "./data_cache/",
                 database_url: str = None):
        """
        Initialize data manager.
        
        Args:
            config: Configuration for data sources
            cache_directory: Directory for caching data
            database_url: Database URL for persistence
        """
        
        self.config = config or {}
        self.cache_directory = cache_directory
        self.data_sources = {}
        self.cache = {}
        self.last_update_times = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_directory, exist_ok=True)
        
        # Initialize data sources
        self._initialize_data_sources()
        
        # Initialize database if URL provided
        self.database = None
        if database_url:
            try:
                self.database = DatabaseDataSource(database_url)
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")
        
        # Real-time streaming
        self.streaming_active = False
        self.streaming_thread = None
        self.streaming_callbacks = {}
    
    def _initialize_data_sources(self):
        """Initialize configured data sources."""
        
        # File data source (always available)
        self.data_sources['file'] = FileDataSource(self.cache_directory)
        
        # Market data sources
        for source_type in ['yahoo', 'alphavantage', 'quandl', 'fred', 'crypto']:
            if source_type in self.config:
                try:
                    source_config = self.config[source_type]
                    self.data_sources[source_type] = create_market_data_source(
                        source_type, **source_config
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize {source_type}: {e}")
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      source_priority: List[str] = None,
                      fallback: bool = True,
                      cache: bool = True) -> pd.DataFrame:
        """
        Get price data with intelligent source selection and fallback.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source_priority: Ordered list of preferred sources
            fallback: Whether to try alternative sources on failure
            cache: Whether to use/update cache
        
        Returns:
            DataFrame with price data
        """
        
        # Default source priority
        if source_priority is None:
            source_priority = ['database', 'file', 'yahoo', 'alphavantage']
        
        # Create cache key
        cache_key = f"prices_{'-'.join(sorted(tickers))}_{start_date}_{end_date}"
        
        # Check cache first
        if cache and cache_key in self.cache:
            cache_data = self.cache[cache_key]
            if self._is_cache_fresh(cache_key, max_age_hours=1):
                return cache_data.copy()
        
        price_data = pd.DataFrame()
        successful_source = None
        
        # Try sources in priority order
        for source_name in source_priority:
            if source_name not in self.data_sources and source_name != 'database':
                continue
            
            try:
                if source_name == 'database' and self.database:
                    price_data = self.database.get_price_data(tickers, start_date, end_date)
                elif source_name in self.data_sources:
                    source = self.data_sources[source_name]
                    
                    if isinstance(source, FileDataSource):
                        # For file source, try to find appropriate file
                        price_data = self._get_file_data(source, tickers, start_date, end_date)
                    else:
                        # Market data source
                        price_data = source.get_price_data(tickers, start_date, end_date)
                
                # Check if we got valid data
                if not price_data.empty and len(price_data.columns) > 0:
                    successful_source = source_name
                    break
                    
            except Exception as e:
                print(f"Error fetching data from {source_name}: {e}")
                if not fallback:
                    raise
                continue
        
        # If no data found, try to download and cache
        if price_data.empty and 'yahoo' in self.data_sources:
            try:
                print(f"Downloading fresh data for {tickers}")
                yahoo_source = self.data_sources['yahoo']
                price_data = yahoo_source.get_price_data(tickers, start_date, end_date)
                
                # Store in database if available
                if not price_data.empty and self.database:
                    self.database.store_price_data(price_data, source='yahoo')
                
                # Save to file cache
                if not price_data.empty:
                    cache_file = os.path.join(
                        self.cache_directory, 
                        f"cached_prices_{datetime.now().strftime('%Y%m%d')}.csv"
                    )
                    price_data.to_csv(cache_file)
                
                successful_source = 'yahoo'
                
            except Exception as e:
                print(f"Error downloading fresh data: {e}")
                if not fallback:
                    raise
        
        # Cache the result
        if cache and not price_data.empty:
            self.cache[cache_key] = price_data.copy()
            self.last_update_times[cache_key] = datetime.now()
        
        # Filter to requested tickers and ensure they exist
        if not price_data.empty:
            available_tickers = [t for t in tickers if t in price_data.columns]
            if available_tickers:
                price_data = price_data[available_tickers]
            
            # Add missing tickers as NaN columns
            for ticker in tickers:
                if ticker not in price_data.columns:
                    price_data[ticker] = np.nan
            
            print(f"Retrieved data from {successful_source} for {len(available_tickers)}/{len(tickers)} tickers")
        
        return price_data
    
    def _get_file_data(self, 
                      file_source: FileDataSource, 
                      tickers: List[str],
                      start_date: str = None,
                      end_date: str = None) -> pd.DataFrame:
        """Try to get data from available files."""
        
        # Look for appropriate files
        available_files = file_source.get_available_files('.csv')
        
        # Try common file names
        common_names = [
            'merged_stock_prices.csv',
            'stock_prices.csv',
            'price_data.csv',
            'market_data.csv'
        ]
        
        for filename in common_names:
            if filename in available_files:
                try:
                    return file_source.load_price_data(
                        filename, tickers=tickers, 
                        start_date=start_date, end_date=end_date
                    )
                except Exception:
                    continue
        
        # Try first available CSV file
        for filename in available_files:
            if filename.endswith('.csv'):
                try:
                    return file_source.load_price_data(
                        filename, tickers=tickers,
                        start_date=start_date, end_date=end_date
                    )
                except Exception:
                    continue
        
        return pd.DataFrame()
    
    def get_real_time_prices(self, 
                           tickers: List[str],
                           source: str = 'yahoo') -> Dict[str, float]:
        """
        Get real-time price quotes.
        
        Args:
            tickers: List of ticker symbols
            source: Data source to use
        
        Returns:
            Dictionary mapping tickers to current prices
        """
        
        if source not in self.data_sources:
            raise ValueError(f"Source {source} not available")
        
        data_source = self.data_sources[source]
        
        if not hasattr(data_source, 'get_real_time_price'):
            raise ValueError(f"Source {source} does not support real-time data")
        
        try:
            prices = data_source.get_real_time_price(tickers)
            
            # Store in database if available
            if prices and self.database:
                # Convert to DataFrame for storage
                current_time = datetime.now()
                price_df = pd.DataFrame({
                    ticker: [price] for ticker, price in prices.items()
                }, index=[current_time])
                
                self.database.store_price_data(price_df, source=f'{source}_realtime')
            
            return prices
            
        except Exception as e:
            raise ValueError(f"Error fetching real-time prices: {e}")
    
    def start_real_time_streaming(self, 
                                tickers: List[str],
                                callback_func: callable,
                                source: str = 'yahoo',
                                interval_seconds: int = 60):
        """
        Start real-time price streaming.
        
        Args:
            tickers: List of tickers to stream
            callback_func: Function to call with new price data
            source: Data source for streaming
            interval_seconds: Update interval in seconds
        """
        
        if self.streaming_active:
            print("Streaming already active")
            return
        
        self.streaming_active = True
        self.streaming_callbacks[source] = callback_func
        
        def streaming_worker():
            """Background worker for price streaming."""
            while self.streaming_active:
                try:
                    prices = self.get_real_time_prices(tickers, source)
                    if prices:
                        callback_func(prices)
                except Exception as e:
                    print(f"Streaming error: {e}")
                
                time.sleep(interval_seconds)
        
        self.streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
        self.streaming_thread.start()
        
        print(f"Started real-time streaming for {len(tickers)} tickers")
    
    def stop_real_time_streaming(self):
        """Stop real-time price streaming."""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
        print("Stopped real-time streaming")
    
    def update_database_cache(self, 
                            tickers: List[str],
                            source: str = 'yahoo',
                            days_back: int = 30) -> int:
        """
        Update database cache with recent data.
        
        Args:
            tickers: List of tickers to update
            source: Data source to use
            days_back: Number of days to fetch
        
        Returns:
            Number of records updated
        """
        
        if not self.database:
            raise ValueError("Database not configured")
        
        if source not in self.data_sources:
            raise ValueError(f"Source {source} not available")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Update database
        records_updated = self.database.update_data_from_source(
            tickers, 
            self.data_sources[source],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Clear cache to force fresh data
        self.clear_cache()
        
        return records_updated
    
    def get_available_tickers(self, source: str = None) -> List[str]:
        """
        Get list of available tickers from data sources.
        
        Args:
            source: Specific source to check (all sources if None)
        
        Returns:
            List of available ticker symbols
        """
        
        tickers = set()
        
        # Check database
        if self.database:
            try:
                market_info = self.database.get_market_info()
                if not market_info.empty:
                    tickers.update(market_info['ticker'].tolist())
            except:
                pass
        
        # Check file sources
        if source is None or source == 'file':
            file_source = self.data_sources.get('file')
            if file_source:
                try:
                    # Preview available files to extract tickers
                    csv_files = file_source.get_available_files('.csv')
                    for file_path in csv_files[:3]:  # Check first 3 files
                        preview = file_source.preview_file(file_path)
                        if 'potential_ticker_columns' in preview:
                            tickers.update(preview['potential_ticker_columns'])
                except:
                    pass
        
        return sorted(list(tickers))
    
    def get_data_quality_report(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Generate data quality report for specified tickers.
        
        Args:
            tickers: List of tickers to analyze
        
        Returns:
            Data quality report
        """
        
        report = {
            'tickers_analyzed': tickers,
            'analysis_date': datetime.now().isoformat(),
            'source_coverage': {},
            'data_gaps': {},
            'recommendations': []
        }
        
        # Check each source
        for source_name, source in self.data_sources.items():
            try:
                if isinstance(source, FileDataSource):
                    # Check file availability
                    files = source.get_available_files('.csv')
                    report['source_coverage'][source_name] = {
                        'available': len(files) > 0,
                        'file_count': len(files)
                    }
                else:
                    # Check market data source
                    test_data = source.get_price_data(tickers[:1], 
                        start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
                    report['source_coverage'][source_name] = {
                        'available': not test_data.empty,
                        'responsive': True
                    }
                    
            except Exception as e:
                report['source_coverage'][source_name] = {
                    'available': False,
                    'error': str(e)
                }
        
        # Check database coverage
        if self.database:
            try:
                coverage = self.database.get_data_coverage(tickers)
                report['database_coverage'] = coverage.to_dict('records')
            except:
                report['database_coverage'] = []
        
        # Generate recommendations
        available_sources = [s for s, info in report['source_coverage'].items() 
                           if info.get('available', False)]
        
        if len(available_sources) == 0:
            report['recommendations'].append("No data sources available - configure at least one source")
        elif len(available_sources) == 1:
            report['recommendations'].append("Consider adding additional data sources for redundancy")
        
        if not self.database:
            report['recommendations'].append("Configure database for better performance and caching")
        
        return report
    
    def export_data(self, 
                   tickers: List[str],
                   output_file: str,
                   start_date: str = None,
                   end_date: str = None,
                   format: str = 'csv') -> str:
        """
        Export data to file.
        
        Args:
            tickers: List of tickers to export
            output_file: Output file path
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            format: Output format ('csv', 'excel', 'parquet')
        
        Returns:
            Path of exported file
        """
        
        # Get data
        data = self.get_price_data(tickers, start_date, end_date)
        
        if data.empty:
            raise ValueError("No data available for export")
        
        # Use file source to save
        file_source = self.data_sources['file']
        return file_source.save_data(data, output_file, format)
    
    def _is_cache_fresh(self, cache_key: str, max_age_hours: float = 1.0) -> bool:
        """Check if cached data is still fresh."""
        
        if cache_key not in self.last_update_times:
            return False
        
        last_update = self.last_update_times[cache_key]
        age_hours = (datetime.now() - last_update).total_seconds() / 3600
        
        return age_hours < max_age_hours
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.last_update_times.clear()
        
        # Also clear data source caches
        for source in self.data_sources.values():
            if hasattr(source, 'clear_cache'):
                source.clear_cache()
    
    def configure_source(self, source_type: str, **config):
        """
        Configure or reconfigure a data source.
        
        Args:
            source_type: Type of source ('yahoo', 'alphavantage', etc.)
            **config: Configuration parameters
        """
        
        try:
            if source_type == 'database':
                database_url = config.get('database_url')
                if database_url:
                    self.database = DatabaseDataSource(database_url)
            else:
                self.data_sources[source_type] = create_market_data_source(
                    source_type, **config
                )
            
            # Update configuration
            self.config[source_type] = config
            
            print(f"Configured {source_type} data source")
            
        except Exception as e:
            raise ValueError(f"Error configuring {source_type}: {e}")
    
    def close(self):
        """Clean up resources."""
        
        # Stop streaming
        if self.streaming_active:
            self.stop_real_time_streaming()
        
        # Close database connection
        if self.database:
            self.database.close()
        
        # Clear caches
        self.clear_cache()

# Factory function
def create_data_manager(config_file: str = None, **kwargs) -> DataManager:
    """
    Create data manager from configuration file or parameters.
    
    Args:
        config_file: Path to configuration file (JSON/YAML)
        **kwargs: Direct configuration parameters
    
    Returns:
        Configured DataManager instance
    """
    
    config = {}
    
    # Load from file if provided
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    # Override with direct parameters
    config.update(kwargs)
    
    return DataManager(config)

# Convenience functions
def quick_get_data(tickers: List[str], 
                  days: int = 365,
                  source: str = 'yahoo') -> pd.DataFrame:
    """
    Quick function to get recent price data.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history
        source: Preferred data source
    
    Returns:
        DataFrame with price data
    """
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    manager = DataManager({source: {}})
    
    return manager.get_price_data(
        tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        source_priority=[source]
    )

def setup_real_time_monitoring(tickers: List[str], 
                              callback: callable,
                              interval: int = 60) -> DataManager:
    """
    Quick setup for real-time price monitoring.
    
    Args:
        tickers: List of tickers to monitor
        callback: Function to call with price updates
        interval: Update interval in seconds
    
    Returns:
        DataManager instance with streaming active
    """
    
    manager = DataManager({'yahoo': {}})
    manager.start_real_time_streaming(tickers, callback, interval_seconds=interval)
    
    return manager