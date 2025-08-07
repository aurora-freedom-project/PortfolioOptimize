# core/data_sources/market_feeds.py
import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

class BaseMarketDataSource(ABC):
    """Base class for market data sources."""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        """
        Initialize market data source.
        
        Args:
            api_key: API key for authenticated access
            rate_limit: Minimum seconds between API calls
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_call_time = 0
        self.cache = {}
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last_call)
        
        self.last_call_time = time.time()
    
    @abstractmethod
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """Get historical price data for tickers."""
        pass
    
    @abstractmethod
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Get current real-time prices for tickers."""
        pass

class YahooFinanceDataSource(BaseMarketDataSource):
    """
    Yahoo Finance data source using yfinance library.
    Free tier with good coverage of global markets.
    """
    
    def __init__(self, rate_limit: float = 0.5):
        """Initialize Yahoo Finance data source."""
        super().__init__(rate_limit=rate_limit)
        
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance library required. Install with: pip install yfinance")
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """
        Get historical price data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo', '1h', '5m', etc.)
        
        Returns:
            DataFrame with adjusted close prices
        """
        
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Create cache key
        cache_key = f"yahoo_{'-'.join(tickers)}_{start_date}_{end_date}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        self._rate_limit_wait()
        
        try:
            # Download data for multiple tickers
            if len(tickers) == 1:
                ticker = self.yf.Ticker(tickers[0])
                data = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if data.empty:
                    raise ValueError(f"No data found for ticker {tickers[0]}")
                
                # Use adjusted close price
                price_data = pd.DataFrame({tickers[0]: data['Close']})
                
            else:
                # Multiple tickers
                data = self.yf.download(tickers, start=start_date, end=end_date, 
                                      interval=interval, group_by='ticker', 
                                      auto_adjust=True, prepost=True, threads=True)
                
                if data.empty:
                    raise ValueError(f"No data found for tickers {tickers}")
                
                # Extract close prices
                if len(tickers) == 1:
                    price_data = pd.DataFrame({tickers[0]: data['Close']})
                else:
                    price_data = pd.DataFrame()
                    for ticker in tickers:
                        if ticker in data.columns.levels[0]:
                            price_data[ticker] = data[ticker]['Close']
                        else:
                            print(f"Warning: No data found for {ticker}")
            
            # Clean data
            price_data = price_data.dropna(how='all')
            price_data = price_data.fillna(method='ffill').fillna(method='bfill')
            
            # Cache result
            self.cache[cache_key] = price_data.copy()
            
            return price_data
            
        except Exception as e:
            raise ValueError(f"Error fetching data from Yahoo Finance: {e}")
    
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Get current real-time prices from Yahoo Finance."""
        
        self._rate_limit_wait()
        
        prices = {}
        
        try:
            for ticker in tickers:
                ticker_obj = self.yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Try different price fields
                current_price = (info.get('currentPrice') or 
                               info.get('regularMarketPrice') or
                               info.get('previousClose'))
                
                if current_price:
                    prices[ticker] = float(current_price)
                else:
                    # Fallback to recent close
                    recent_data = ticker_obj.history(period='1d', interval='1m')
                    if not recent_data.empty:
                        prices[ticker] = float(recent_data['Close'].iloc[-1])
            
            return prices
            
        except Exception as e:
            raise ValueError(f"Error fetching real-time prices: {e}")
    
    def get_market_info(self, ticker: str) -> Dict[str, Any]:
        """Get detailed market information for a ticker."""
        
        try:
            ticker_obj = self.yf.Ticker(ticker)
            info = ticker_obj.info
            
            return {
                'symbol': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume')
            }
            
        except Exception as e:
            return {'symbol': ticker, 'error': str(e)}

class AlphaVantageDataSource(BaseMarketDataSource):
    """
    Alpha Vantage data source for premium market data.
    Requires API key but provides high-quality data.
    """
    
    def __init__(self, api_key: str, rate_limit: float = 12.0):
        """
        Initialize Alpha Vantage data source.
        
        Args:
            api_key: Alpha Vantage API key
            rate_limit: Seconds between calls (free tier: 5 calls/minute)
        """
        super().__init__(api_key=api_key, rate_limit=rate_limit)
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """Get historical price data from Alpha Vantage."""
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")
        
        # Map intervals
        av_intervals = {
            '1d': 'TIME_SERIES_DAILY_ADJUSTED',
            '1wk': 'TIME_SERIES_WEEKLY_ADJUSTED',
            '1mo': 'TIME_SERIES_MONTHLY_ADJUSTED'
        }
        
        function = av_intervals.get(interval, 'TIME_SERIES_DAILY_ADJUSTED')
        
        price_data = pd.DataFrame()
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                params = {
                    'function': function,
                    'symbol': ticker,
                    'apikey': self.api_key,
                    'outputsize': 'full'
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract time series data
                if 'Time Series (Daily)' in data:
                    ts_data = data['Time Series (Daily)']
                elif 'Weekly Adjusted Time Series' in data:
                    ts_data = data['Weekly Adjusted Time Series']
                elif 'Monthly Adjusted Time Series' in data:
                    ts_data = data['Monthly Adjusted Time Series']
                else:
                    print(f"Warning: No time series data found for {ticker}")
                    continue
                
                # Convert to DataFrame
                ticker_df = pd.DataFrame.from_dict(ts_data, orient='index')
                ticker_df.index = pd.to_datetime(ticker_df.index)
                ticker_df = ticker_df.sort_index()
                
                # Use adjusted close
                price_data[ticker] = ticker_df['5. adjusted close'].astype(float)
                
            except Exception as e:
                print(f"Error fetching {ticker} from Alpha Vantage: {e}")
                continue
        
        # Filter date range
        if start_date:
            price_data = price_data[price_data.index >= pd.to_datetime(start_date)]
        if end_date:
            price_data = price_data[price_data.index <= pd.to_datetime(end_date)]
        
        return price_data
    
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Get real-time prices from Alpha Vantage."""
        
        prices = {}
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    prices[ticker] = float(quote['05. price'])
                
            except Exception as e:
                print(f"Error fetching real-time price for {ticker}: {e}")
                continue
        
        return prices

class QuandlDataSource(BaseMarketDataSource):
    """
    Quandl data source for economic and financial data.
    Good for alternative datasets and economic indicators.
    """
    
    def __init__(self, api_key: str, rate_limit: float = 1.0):
        """Initialize Quandl data source."""
        super().__init__(api_key=api_key, rate_limit=rate_limit)
        
        try:
            import quandl
            quandl.ApiConfig.api_key = api_key
            self.quandl = quandl
        except ImportError:
            raise ImportError("quandl library required. Install with: pip install quandl")
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """Get data from Quandl."""
        
        price_data = pd.DataFrame()
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                # Quandl uses different database/dataset format
                # Example: "WIKI/AAPL" for US equities
                if '/' not in ticker:
                    # Default to WIKI database for US stocks
                    quandl_code = f"WIKI/{ticker}"
                else:
                    quandl_code = ticker
                
                data = self.quandl.get(quandl_code, 
                                     start_date=start_date, 
                                     end_date=end_date)
                
                # Use adjusted close if available, otherwise close
                if 'Adj. Close' in data.columns:
                    price_data[ticker] = data['Adj. Close']
                elif 'Close' in data.columns:
                    price_data[ticker] = data['Close']
                else:
                    # Use first numeric column
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price_data[ticker] = data[numeric_cols[0]]
                
            except Exception as e:
                print(f"Error fetching {ticker} from Quandl: {e}")
                continue
        
        return price_data
    
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Quandl typically doesn't provide real-time data."""
        raise NotImplementedError("Quandl does not provide real-time price data")

class FREDDataSource(BaseMarketDataSource):
    """
    Federal Reserve Economic Data (FRED) source.
    Excellent for macroeconomic indicators and interest rates.
    """
    
    def __init__(self, api_key: str, rate_limit: float = 1.0):
        """Initialize FRED data source."""
        super().__init__(api_key=api_key, rate_limit=rate_limit)
        
        try:
            import fredapi
            self.fred = fredapi.Fred(api_key=api_key)
        except ImportError:
            raise ImportError("fredapi library required. Install with: pip install fredapi")
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """Get economic data from FRED."""
        
        price_data = pd.DataFrame()
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                data = self.fred.get_series(ticker, 
                                          start=start_date, 
                                          end=end_date)
                
                price_data[ticker] = data
                
            except Exception as e:
                print(f"Error fetching {ticker} from FRED: {e}")
                continue
        
        return price_data
    
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Get latest values from FRED."""
        
        prices = {}
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                # Get most recent observation
                data = self.fred.get_series(ticker, limit=1)
                if not data.empty:
                    prices[ticker] = float(data.iloc[-1])
                
            except Exception as e:
                print(f"Error fetching latest value for {ticker}: {e}")
                continue
        
        return prices
    
    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """Search for FRED data series."""
        
        try:
            return self.fred.search(search_text, limit=limit)
        except Exception as e:
            print(f"Error searching FRED: {e}")
            return pd.DataFrame()

class CryptoDataSource(BaseMarketDataSource):
    """
    Cryptocurrency data source using CoinGecko API.
    Free tier with good coverage of crypto markets.
    """
    
    def __init__(self, rate_limit: float = 1.0):
        """Initialize crypto data source."""
        super().__init__(rate_limit=rate_limit)
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """Get historical crypto prices from CoinGecko."""
        
        price_data = pd.DataFrame()
        
        # Convert date range to days
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            days = (end - start).days
        else:
            days = 365  # Default to 1 year
        
        for ticker in tickers:
            self._rate_limit_wait()
            
            try:
                # Convert ticker to CoinGecko ID (simplified mapping)
                coin_id = self._ticker_to_coingecko_id(ticker)
                
                url = f"{self.base_url}/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily' if interval == '1d' else 'hourly'
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract prices
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                
                price_data[ticker] = df['price']
                
            except Exception as e:
                print(f"Error fetching {ticker} from CoinGecko: {e}")
                continue
        
        return price_data
    
    def get_real_time_price(self, tickers: List[str]) -> Dict[str, float]:
        """Get current crypto prices from CoinGecko."""
        
        prices = {}
        
        try:
            # Convert tickers to CoinGecko IDs
            coin_ids = [self._ticker_to_coingecko_id(ticker) for ticker in tickers]
            
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': 'usd'
            }
            
            self._rate_limit_wait()
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for ticker, coin_id in zip(tickers, coin_ids):
                if coin_id in data and 'usd' in data[coin_id]:
                    prices[ticker] = data[coin_id]['usd']
            
        except Exception as e:
            print(f"Error fetching crypto prices: {e}")
        
        return prices
    
    def _ticker_to_coingecko_id(self, ticker: str) -> str:
        """Convert ticker symbol to CoinGecko ID."""
        
        # Common mappings
        mappings = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'XRP': 'ripple',
            'DOGE': 'dogecoin',
            'MATIC': 'matic-network'
        }
        
        return mappings.get(ticker.upper(), ticker.lower())

# Factory function to create data sources
def create_market_data_source(source_type: str, **kwargs) -> BaseMarketDataSource:
    """
    Factory function to create market data sources.
    
    Args:
        source_type: Type of data source ('yahoo', 'alphavantage', 'quandl', 'fred', 'crypto')
        **kwargs: Additional arguments for the data source
    
    Returns:
        Market data source instance
    """
    
    sources = {
        'yahoo': YahooFinanceDataSource,
        'alphavantage': AlphaVantageDataSource,
        'quandl': QuandlDataSource,
        'fred': FREDDataSource,
        'crypto': CryptoDataSource
    }
    
    if source_type.lower() not in sources:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(sources.keys())}")
    
    return sources[source_type.lower()](**kwargs)

# Convenience function for multi-source data fetching
def fetch_multi_source_data(tickers: List[str],
                           sources: Dict[str, Dict[str, Any]],
                           start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
    """
    Fetch data from multiple sources and combine.
    
    Args:
        tickers: List of ticker symbols
        sources: Dict mapping source names to their configurations
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Combined DataFrame with data from all sources
    """
    
    combined_data = pd.DataFrame()
    
    for source_name, config in sources.items():
        try:
            source = create_market_data_source(source_name, **config)
            data = source.get_price_data(tickers, start_date, end_date)
            
            # Add source suffix to columns
            data.columns = [f"{col}_{source_name}" for col in data.columns]
            
            if combined_data.empty:
                combined_data = data
            else:
                combined_data = combined_data.join(data, how='outer')
                
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            continue
    
    return combined_data