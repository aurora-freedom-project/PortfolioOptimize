# core/data_sources/file_data.py
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

class FileDataSource:
    """
    File-based data source supporting multiple formats for offline analysis.
    
    Supports:
    - CSV files (most common)
    - Excel files (.xlsx, .xls)
    - Parquet files (high performance)
    - JSON files
    - HDF5 files (large datasets)
    - Pickle files (Python objects)
    """
    
    def __init__(self, base_path: str = "./data/"):
        """
        Initialize file data source.
        
        Args:
            base_path: Base directory for data files
        """
        self.base_path = base_path
        self.cache = {}  # Simple in-memory cache
        
        # Ensure data directory exists
        os.makedirs(base_path, exist_ok=True)
    
    def load_price_data(self, 
                       file_path: str,
                       date_column: str = None,
                       price_columns: List[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load price data from file with flexible formatting options.
        
        Args:
            file_path: Path to data file (relative to base_path or absolute)
            date_column: Name of date column (auto-detect if None)
            price_columns: List of price column names (auto-detect if None)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            tickers: List of tickers to include (all if None)
        
        Returns:
            DataFrame with datetime index and ticker columns
        """
        
        # Resolve file path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.base_path, file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Check cache
        cache_key = f"{file_path}_{start_date}_{end_date}_{tickers}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        # Load based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = self._load_csv(file_path, date_column)
            elif file_ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path, date_column)
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            elif file_ext in ['.h5', '.hdf5']:
                df = pd.read_hdf(file_path, key='data')
            elif file_ext == '.pkl':
                df = pd.read_pickle(file_path)
            else:
                # Default to CSV
                df = self._load_csv(file_path, date_column)
                
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")
        
        # Process the dataframe
        df = self._process_dataframe(df, date_column, price_columns, 
                                   start_date, end_date, tickers)
        
        # Cache result
        self.cache[cache_key] = df.copy()
        
        return df
    
    def _load_csv(self, file_path: str, date_column: str = None) -> pd.DataFrame:
        """Load CSV file with intelligent date parsing."""
        
        # Try different common separators and encodings
        separators = [',', ';', '\t']
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for sep in separators:
            for encoding in encodings:
                try:
                    # First, peek at the file to understand structure
                    sample = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=5)
                    
                    if len(sample.columns) > 1:  # Valid separator found
                        # Load full file with proper date parsing
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding, index_col=0, parse_dates=True)
                        
                        # Auto-detect date column if not specified and not already used as index
                        if date_column is None and not isinstance(df.index, pd.DatetimeIndex):
                            date_column = self._detect_date_column(df)
                        
                        return df
                        
                except Exception:
                    continue
        
        # Fallback to pandas default with date parsing
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def _load_excel(self, file_path: str, date_column: str = None) -> pd.DataFrame:
        """Load Excel file with sheet detection."""
        
        try:
            # Try to load first sheet
            excel_file = pd.ExcelFile(file_path)
            
            # Use first sheet by default
            sheet_name = excel_file.sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {e}")
    
    def _detect_date_column(self, df: pd.DataFrame) -> str:
        """Auto-detect date column in dataframe."""
        
        date_column_names = ['date', 'Date', 'DATE', 'datetime', 'timestamp', 
                           'time', 'Time', 'index', 'Index']
        
        # Check for common date column names
        for col in date_column_names:
            if col in df.columns:
                return col
        
        # Check for datetime-like data in first few columns
        for col in df.columns[:3]:
            try:
                pd.to_datetime(df[col].head(10))
                return col
            except:
                continue
        
        # Check if index looks like dates
        try:
            pd.to_datetime(df.index)
            return None  # Use index as date
        except:
            pass
        
        # Default to first column
        return df.columns[0]
    
    def _process_dataframe(self, 
                          df: pd.DataFrame,
                          date_column: str = None,
                          price_columns: List[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Process and clean the dataframe."""
        
        # Set date index
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("Could not parse date information from data")
        
        # Ensure index is properly sorted
        df = df.sort_index()
        
        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # Select price columns
        if price_columns:
            df = df[price_columns]
        elif tickers:
            # Filter to requested tickers
            available_tickers = [col for col in tickers if col in df.columns]
            if not available_tickers:
                raise ValueError(f"None of the requested tickers {tickers} found in data")
            df = df[available_tickers]
        
        # Clean data
        df = self._clean_price_data(df)
        
        return df
    
    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data by handling missing values and outliers."""
        
        # Remove non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        # Handle missing values
        # Forward fill first, then backward fill
        df = df.ffill().bfill()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove columns with too many missing values (>50%)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        
        # Basic outlier detection (optional)
        # Remove extreme outliers (>5 standard deviations)
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                outlier_mask = np.abs(df[col] - mean) > 5 * std
                df.loc[outlier_mask, col] = np.nan
        
        # Final forward fill
        df = df.ffill()
        
        return df
    
    def save_data(self, 
                  df: pd.DataFrame, 
                  file_path: str, 
                  format: str = 'csv',
                  **kwargs) -> str:
        """
        Save dataframe to file in specified format.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('csv', 'excel', 'parquet', 'json', 'hdf5', 'pickle')
            **kwargs: Additional arguments for pandas save methods
        
        Returns:
            Full path of saved file
        """
        
        # Resolve file path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.base_path, file_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, **kwargs)
            elif format.lower() == 'excel':
                df.to_excel(file_path, **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, **kwargs)
            elif format.lower() == 'json':
                df.to_json(file_path, **kwargs)
            elif format.lower() == 'hdf5':
                df.to_hdf(file_path, key='data', **kwargs)
            elif format.lower() == 'pickle':
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"Data saved to: {file_path}")
            return file_path
            
        except Exception as e:
            raise ValueError(f"Error saving file: {e}")
    
    def get_available_files(self, extension: str = None) -> List[str]:
        """
        Get list of available data files in base directory.
        
        Args:
            extension: Filter by file extension (e.g., '.csv')
        
        Returns:
            List of available file paths
        """
        
        files = []
        
        for root, dirs, filenames in os.walk(self.base_path):
            for filename in filenames:
                if extension is None or filename.lower().endswith(extension.lower()):
                    rel_path = os.path.relpath(os.path.join(root, filename), self.base_path)
                    files.append(rel_path)
        
        return sorted(files)
    
    def preview_file(self, file_path: str, rows: int = 10) -> Dict[str, Any]:
        """
        Preview file structure and data.
        
        Args:
            file_path: Path to file
            rows: Number of rows to preview
        
        Returns:
            Dictionary with file information and preview
        """
        
        # Resolve file path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.base_path, file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load preview
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=rows)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, nrows=rows)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path, nrows=rows)
            
            # File info
            file_info = {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'columns': list(df.columns),
                'column_count': len(df.columns),
                'preview_rows': len(df),
                'dtypes': df.dtypes.to_dict(),
                'preview_data': df.to_dict('records'),
                'potential_date_columns': [col for col in df.columns 
                                         if any(date_word in col.lower() 
                                               for date_word in ['date', 'time', 'timestamp'])],
                'potential_ticker_columns': [col for col in df.columns 
                                           if col.isupper() and len(col) <= 10]
            }
            
            return file_info
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
    
    def convert_file_format(self, 
                           input_path: str, 
                           output_path: str, 
                           output_format: str) -> str:
        """
        Convert file from one format to another.
        
        Args:
            input_path: Source file path
            output_path: Output file path
            output_format: Target format ('csv', 'excel', 'parquet', etc.)
        
        Returns:
            Path of converted file
        """
        
        # Load data
        df = self.load_price_data(input_path)
        
        # Save in new format
        return self.save_data(df, output_path, output_format)
    
    def merge_files(self, 
                   file_paths: List[str], 
                   output_path: str,
                   how: str = 'outer',
                   on_column: str = None) -> str:
        """
        Merge multiple data files into one.
        
        Args:
            file_paths: List of file paths to merge
            output_path: Output file path
            how: Merge method ('outer', 'inner', 'left', 'right')
            on_column: Column to merge on (uses index if None)
        
        Returns:
            Path of merged file
        """
        
        dataframes = []
        
        for file_path in file_paths:
            try:
                df = self.load_price_data(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid dataframes to merge")
        
        # Merge dataframes
        if on_column:
            # Merge on specific column
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = pd.merge(merged_df, df, on=on_column, how=how)
        else:
            # Merge on index (for time series data)
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = merged_df.join(df, how=how, rsuffix='_dup')
        
        # Remove duplicate columns
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Save merged data
        return self.save_data(merged_df, output_path)
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()


# Convenience functions for common operations
def load_csv_data(file_path: str, 
                  tickers: List[str] = None,
                  start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
    """
    Quick function to load CSV data.
    
    Args:
        file_path: Path to CSV file (absolute or relative)
        tickers: List of ticker symbols to include
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with price data
    """
    
    # Handle absolute vs relative paths
    if os.path.isabs(file_path):
        # For absolute paths, use parent directory as base_path
        base_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_source = FileDataSource(base_path)
        return file_source.load_price_data(
            file_path=file_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
    else:
        # For relative paths, use default base_path
        file_source = FileDataSource()
        return file_source.load_price_data(
            file_path=file_path,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )

def preview_data_file(file_path: str) -> Dict[str, Any]:
    """
    Quick function to preview a data file.
    
    Args:
        file_path: Path to data file
    
    Returns:
        File preview information
    """
    
    file_source = FileDataSource()
    return file_source.preview_file(file_path)

def convert_data_file(input_path: str, 
                      output_path: str, 
                      output_format: str = 'csv') -> str:
    """
    Quick function to convert data file format.
    
    Args:
        input_path: Source file path
        output_path: Output file path  
        output_format: Target format
    
    Returns:
        Path of converted file
    """
    
    file_source = FileDataSource()
    return file_source.convert_file_format(input_path, output_path, output_format)