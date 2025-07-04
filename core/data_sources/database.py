# core/data_sources/database.py
import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
import os
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import warnings

Base = declarative_base()

class PriceData(Base):
    """SQLAlchemy model for price data."""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    adjusted_close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MarketData(Base):
    """SQLAlchemy model for market metadata."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    currency = Column(String(10), default='USD')
    exchange = Column(String(50))
    country = Column(String(50))
    data_source = Column(String(50))
    last_updated = Column(DateTime, default=datetime.utcnow)

class PortfolioHistory(Base):
    """SQLAlchemy model for portfolio optimization history."""
    __tablename__ = 'portfolio_history'
    
    id = Column(Integer, primary_key=True)
    portfolio_name = Column(String(100), nullable=False)
    optimization_date = Column(DateTime, nullable=False)
    method = Column(String(50), nullable=False)
    tickers = Column(Text)  # JSON string of ticker list
    weights = Column(Text)  # JSON string of weights
    metrics = Column(Text)  # JSON string of performance metrics
    parameters = Column(Text)  # JSON string of optimization parameters
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseDataSource:
    """
    Database data source supporting multiple database backends.
    
    Supports:
    - SQLite (file-based, good for development)
    - PostgreSQL (production-grade)
    - MySQL (alternative production option)
    - In-memory SQLite (for testing)
    """
    
    def __init__(self, 
                 database_url: str = "sqlite:///portfolio_data.db",
                 echo: bool = False):
        """
        Initialize database connection.
        
        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements (debug mode)
        
        Examples:
            SQLite: "sqlite:///portfolio_data.db"
            PostgreSQL: "postgresql://user:pass@localhost/portfolio_db"
            MySQL: "mysql+pymysql://user:pass@localhost/portfolio_db"
            In-memory: "sqlite:///:memory:"
        """
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"Warning: Could not create tables: {e}")
    
    def store_price_data(self, 
                        data: pd.DataFrame, 
                        source: str = 'manual',
                        overwrite: bool = False) -> int:
        """
        Store price data in database.
        
        Args:
            data: DataFrame with datetime index and ticker columns
            source: Data source identifier
            overwrite: Whether to overwrite existing data
        
        Returns:
            Number of records stored
        """
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")
        
        session = self.Session()
        records_stored = 0
        
        try:
            for date in data.index:
                for ticker in data.columns:
                    price = data.loc[date, ticker]
                    
                    if pd.isna(price):
                        continue
                    
                    # Check if record exists
                    existing = session.query(PriceData).filter(
                        PriceData.ticker == ticker,
                        PriceData.date == date
                    ).first()
                    
                    if existing and not overwrite:
                        continue
                    
                    if existing and overwrite:
                        existing.close_price = float(price)
                        existing.adjusted_close = float(price)
                        existing.updated_at = datetime.utcnow()
                    else:
                        # Create new record
                        price_record = PriceData(
                            ticker=ticker,
                            date=date,
                            close_price=float(price),
                            adjusted_close=float(price)
                        )
                        session.add(price_record)
                    
                    records_stored += 1
            
            session.commit()
            print(f"Stored {records_stored} price records")
            
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error storing price data: {e}")
        
        finally:
            session.close()
        
        return records_stored
    
    def get_price_data(self, 
                      tickers: List[str],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      price_type: str = 'adjusted_close') -> pd.DataFrame:
        """
        Retrieve price data from database.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            price_type: Type of price ('close_price', 'adjusted_close', 'open_price', etc.)
        
        Returns:
            DataFrame with datetime index and ticker columns
        """
        
        session = self.Session()
        
        try:
            # Build query
            query = session.query(PriceData.ticker, PriceData.date, 
                                getattr(PriceData, price_type)).filter(
                PriceData.ticker.in_(tickers)
            )
            
            if start_date:
                query = query.filter(PriceData.date >= pd.to_datetime(start_date))
            if end_date:
                query = query.filter(PriceData.date <= pd.to_datetime(end_date))
            
            # Execute query and convert to DataFrame
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results, columns=['ticker', 'date', 'price'])
            
            # Pivot to get tickers as columns
            price_data = df.pivot(index='date', columns='ticker', values='price')
            price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data.sort_index()
            
            # Ensure all requested tickers are present
            for ticker in tickers:
                if ticker not in price_data.columns:
                    price_data[ticker] = np.nan
            
            price_data = price_data[tickers]  # Maintain order
            
            return price_data
            
        except Exception as e:
            raise ValueError(f"Error retrieving price data: {e}")
        
        finally:
            session.close()
    
    def store_market_info(self, market_info: Dict[str, Dict[str, Any]]) -> int:
        """
        Store market information for tickers.
        
        Args:
            market_info: Dict mapping ticker to market data
        
        Returns:
            Number of records stored
        """
        
        session = self.Session()
        records_stored = 0
        
        try:
            for ticker, info in market_info.items():
                # Check if record exists
                existing = session.query(MarketData).filter(
                    MarketData.ticker == ticker
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in info.items():
                        if hasattr(existing, key) and value is not None:
                            setattr(existing, key, value)
                    existing.last_updated = datetime.utcnow()
                else:
                    # Create new record
                    market_record = MarketData(ticker=ticker, **info)
                    session.add(market_record)
                
                records_stored += 1
            
            session.commit()
            print(f"Stored market info for {records_stored} tickers")
            
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error storing market info: {e}")
        
        finally:
            session.close()
        
        return records_stored
    
    def get_market_info(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Retrieve market information from database.
        
        Args:
            tickers: List of ticker symbols (all if None)
        
        Returns:
            DataFrame with market information
        """
        
        session = self.Session()
        
        try:
            query = session.query(MarketData)
            
            if tickers:
                query = query.filter(MarketData.ticker.in_(tickers))
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in results:
                row = {
                    'ticker': record.ticker,
                    'name': record.name,
                    'sector': record.sector,
                    'industry': record.industry,
                    'market_cap': record.market_cap,
                    'pe_ratio': record.pe_ratio,
                    'dividend_yield': record.dividend_yield,
                    'beta': record.beta,
                    'currency': record.currency,
                    'exchange': record.exchange,
                    'country': record.country,
                    'data_source': record.data_source,
                    'last_updated': record.last_updated
                }
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            raise ValueError(f"Error retrieving market info: {e}")
        
        finally:
            session.close()
    
    def store_portfolio_optimization(self,
                                   portfolio_name: str,
                                   method: str,
                                   tickers: List[str],
                                   weights: Dict[str, float],
                                   metrics: Dict[str, Any],
                                   parameters: Dict[str, Any] = None) -> int:
        """
        Store portfolio optimization results.
        
        Args:
            portfolio_name: Name of the portfolio
            method: Optimization method used
            tickers: List of tickers in portfolio
            weights: Portfolio weights
            metrics: Performance metrics
            parameters: Optimization parameters
        
        Returns:
            Record ID
        """
        
        session = self.Session()
        
        try:
            portfolio_record = PortfolioHistory(
                portfolio_name=portfolio_name,
                optimization_date=datetime.utcnow(),
                method=method,
                tickers=json.dumps(tickers),
                weights=json.dumps(weights),
                metrics=json.dumps(metrics, default=str),
                parameters=json.dumps(parameters or {}, default=str)
            )
            
            session.add(portfolio_record)
            session.commit()
            
            record_id = portfolio_record.id
            print(f"Stored portfolio optimization with ID: {record_id}")
            
            return record_id
            
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error storing portfolio optimization: {e}")
        
        finally:
            session.close()
    
    def get_portfolio_history(self, 
                            portfolio_name: str = None,
                            method: str = None,
                            limit: int = 100) -> pd.DataFrame:
        """
        Retrieve portfolio optimization history.
        
        Args:
            portfolio_name: Filter by portfolio name
            method: Filter by optimization method
            limit: Maximum number of records
        
        Returns:
            DataFrame with portfolio history
        """
        
        session = self.Session()
        
        try:
            query = session.query(PortfolioHistory)
            
            if portfolio_name:
                query = query.filter(PortfolioHistory.portfolio_name == portfolio_name)
            if method:
                query = query.filter(PortfolioHistory.method == method)
            
            query = query.order_by(PortfolioHistory.optimization_date.desc())
            query = query.limit(limit)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in results:
                row = {
                    'id': record.id,
                    'portfolio_name': record.portfolio_name,
                    'optimization_date': record.optimization_date,
                    'method': record.method,
                    'tickers': json.loads(record.tickers),
                    'weights': json.loads(record.weights),
                    'metrics': json.loads(record.metrics),
                    'parameters': json.loads(record.parameters),
                    'created_at': record.created_at
                }
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            raise ValueError(f"Error retrieving portfolio history: {e}")
        
        finally:
            session.close()
    
    def update_data_from_source(self, 
                               tickers: List[str],
                               data_source,
                               start_date: str = None,
                               end_date: str = None) -> int:
        """
        Update database with data from external source.
        
        Args:
            tickers: List of tickers to update
            data_source: Market data source instance
            start_date: Start date for update
            end_date: End date for update
        
        Returns:
            Number of records updated
        """
        
        try:
            # Fetch data from source
            price_data = data_source.get_price_data(tickers, start_date, end_date)
            
            # Store in database
            records_stored = self.store_price_data(
                price_data, 
                source=type(data_source).__name__,
                overwrite=True
            )
            
            # Update market info if available
            if hasattr(data_source, 'get_market_info'):
                market_info = {}
                for ticker in tickers:
                    try:
                        info = data_source.get_market_info(ticker)
                        market_info[ticker] = info
                    except:
                        continue
                
                if market_info:
                    self.store_market_info(market_info)
            
            return records_stored
            
        except Exception as e:
            raise ValueError(f"Error updating data from source: {e}")
    
    def get_data_coverage(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Get data coverage information for tickers.
        
        Args:
            tickers: List of tickers to check (all if None)
        
        Returns:
            DataFrame with coverage information
        """
        
        session = self.Session()
        
        try:
            # Get data coverage using raw SQL for efficiency
            if tickers:
                ticker_filter = "WHERE ticker IN ({})".format(
                    ','.join(f"'{t}'" for t in tickers)
                )
            else:
                ticker_filter = ""
            
            query = text(f"""
                SELECT 
                    ticker,
                    COUNT(*) as record_count,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    COUNT(DISTINCT DATE(date)) as unique_dates
                FROM price_data 
                {ticker_filter}
                GROUP BY ticker
                ORDER BY ticker
            """)
            
            result = session.execute(query)
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            coverage_data = pd.DataFrame(rows, columns=[
                'ticker', 'record_count', 'start_date', 'end_date', 'unique_dates'
            ])
            
            # Convert dates
            coverage_data['start_date'] = pd.to_datetime(coverage_data['start_date'])
            coverage_data['end_date'] = pd.to_datetime(coverage_data['end_date'])
            
            # Calculate coverage metrics
            coverage_data['date_range_days'] = (
                coverage_data['end_date'] - coverage_data['start_date']
            ).dt.days
            
            coverage_data['coverage_ratio'] = (
                coverage_data['unique_dates'] / (coverage_data['date_range_days'] + 1)
            ).round(3)
            
            return coverage_data
            
        except Exception as e:
            raise ValueError(f"Error getting data coverage: {e}")
        
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clean up old data to manage database size.
        
        Args:
            days_to_keep: Number of days to retain
        
        Returns:
            Number of records deleted
        """
        
        session = self.Session()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old price data
            deleted_count = session.query(PriceData).filter(
                PriceData.date < cutoff_date
            ).delete()
            
            session.commit()
            print(f"Deleted {deleted_count} old price records")
            
            return deleted_count
            
        except Exception as e:
            session.rollback()
            raise ValueError(f"Error cleaning up data: {e}")
        
        finally:
            session.close()
    
    def backup_database(self, backup_path: str) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
        
        Returns:
            Path of backup file
        """
        
        try:
            if self.database_url.startswith('sqlite'):
                # For SQLite, copy the file
                import shutil
                db_path = self.database_url.replace('sqlite:///', '')
                
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_path)
                    print(f"Database backed up to: {backup_path}")
                    return backup_path
                else:
                    raise ValueError(f"Database file not found: {db_path}")
            
            else:
                # For other databases, use pg_dump or mysqldump
                raise NotImplementedError("Backup for non-SQLite databases not implemented")
                
        except Exception as e:
            raise ValueError(f"Error creating backup: {e}")
    
    def execute_raw_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """
        Execute raw SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            DataFrame with query results
        """
        
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            raise ValueError(f"Error executing query: {e}")
    
    def close(self):
        """Close database connection."""
        self.engine.dispose()

# Convenience functions
def create_database_connection(database_url: str = None) -> DatabaseDataSource:
    """
    Create database connection with environment variable support.
    
    Args:
        database_url: Database URL (uses environment variable if None)
    
    Returns:
        DatabaseDataSource instance
    """
    
    if database_url is None:
        database_url = os.getenv('PORTFOLIO_DB_URL', 'sqlite:///portfolio_data.db')
    
    return DatabaseDataSource(database_url)

def migrate_csv_to_database(csv_file: str, 
                          database_url: str = None,
                          tickers: List[str] = None) -> int:
    """
    Migrate CSV data to database.
    
    Args:
        csv_file: Path to CSV file
        database_url: Database URL
        tickers: List of tickers to migrate
    
    Returns:
        Number of records migrated
    """
    
    from .file_data import FileDataSource
    
    # Load CSV data
    file_source = FileDataSource()
    data = file_source.load_price_data(csv_file, tickers=tickers)
    
    # Store in database
    db_source = create_database_connection(database_url)
    records_stored = db_source.store_price_data(data, source='csv_migration')
    
    return records_stored