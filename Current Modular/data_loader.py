"""
Data Loader - Utilities for loading and preprocessing market data.

This module provides functions and classes for loading and preprocessing
market data from various sources and formats.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and preprocesses market data from various sources.
    
    This class provides methods to load data from CSV files, databases,
    and other sources, and preprocess it for backtesting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data = None
        self.data_info = {}
        logger.info("DataLoader initialized")
    
    def load_csv(self, file_path: str, date_column: str = 'date',
                date_format: Optional[str] = None,
                dropna: bool = True, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            date_column: Name of the date column
            date_format: Format string for parsing dates
            dropna: Whether to drop rows with NA values
            **kwargs: Additional keyword arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path, **kwargs)
            
            # Process the date column
            if date_column in df.columns:
                if date_format:
                    df[date_column] = pd.to_datetime(df[date_column], format=date_format)
                else:
                    df[date_column] = pd.to_datetime(df[date_column])
                
                # Set the date column as index if requested
                if self.config.get('set_date_index', False):
                    df.set_index(date_column, inplace=True)
                    df.index.name = date_column
            
            # Drop NA values if requested
            if dropna:
                df = df.dropna()
                
            # Standardize column names
            if self.config.get('standardize_columns', False):
                df = self._standardize_columns(df)
            
            # Sort by date if requested
            if self.config.get('sort_by_date', True) and date_column in df.columns:
                df = df.sort_values(date_column)
                
            # Store the data
            self.data = df
            
            # Store data info
            self.data_info = {
                'source': file_path,
                'rows': len(df),
                'columns': list(df.columns),
                'start_date': df[date_column].min() if date_column in df.columns else None,
                'end_date': df[date_column].max() if date_column in df.columns else None
            }
            
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def filter_by_date(self, start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None,
                      date_column: str = 'date') -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            date_column: Name of the date column
            
        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            logger.error("No data loaded. Call load_csv or another load method first.")
            return pd.DataFrame()
        
        if date_column not in self.data.columns:
            logger.error(f"Date column '{date_column}' not found in data")
            return self.data
        
        # Make a copy of the data
        df = self.data.copy()
        
        # Filter by start date
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df = df[df[date_column] >= start_date]
        
        # Filter by end date
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df = df[df[date_column] <= end_date]
        
        logger.info(f"Filtered data from {start_date} to {end_date}, resulting in {len(df)} rows")
        return df
    
    def filter_trading_hours(self, df: Optional[pd.DataFrame] = None,
                           hour_column: str = 'hour',
                           minute_column: str = 'minute',
                           date_column: str = 'date',
                           trading_hours: Optional[Dict] = None) -> pd.DataFrame:
        """
        Filter data by trading hours.
        
        Args:
            df: DataFrame to filter or None to use self.data
            hour_column: Name of the hour column
            minute_column: Name of the minute column
            date_column: Name of the date column
            trading_hours: Dictionary with trading hour settings
            
        Returns:
            Filtered DataFrame
        """
        if df is None:
            if self.data is None:
                logger.error("No data loaded. Call load_csv or another load method first.")
                return pd.DataFrame()
            df = self.data.copy()
        
        # If date_column exists but hour/minute columns don't, extract them
        if date_column in df.columns and (hour_column not in df.columns or minute_column not in df.columns):
            df[hour_column] = df[date_column].dt.hour
            df[minute_column] = df[date_column].dt.minute
        
        # Get trading hours from config if not provided
        if trading_hours is None:
            trading_hours = self.config.get('trading_hours', {
                'morning_start': {'hour': 9, 'minute': 30},
                'morning_end': {'hour': 16, 'minute': 0}
            })
        
        # Extract trading hours
        morning_start_hour = trading_hours.get('morning_start', {}).get('hour', 9)
        morning_start_minute = trading_hours.get('morning_start', {}).get('minute', 30)
        morning_end_hour = trading_hours.get('morning_end', {}).get('hour', 16)
        morning_end_minute = trading_hours.get('morning_end', {}).get('minute', 0)
        
        # Create trading hours mask
        mask = (
            # Morning session: 9:30 AM to 4:00 PM
            ((df[hour_column] == morning_start_hour) & (df[minute_column] >= morning_start_minute)) |
            ((df[hour_column] > morning_start_hour) & (df[hour_column] < morning_end_hour)) |
            ((df[hour_column] == morning_end_hour) & (df[minute_column] <= morning_end_minute))
        )
        
        # If afternoon session is defined, include it
        if 'afternoon_start' in trading_hours:
            afternoon_start_hour = trading_hours.get('afternoon_start', {}).get('hour', 0)
            afternoon_start_minute = trading_hours.get('afternoon_start', {}).get('minute', 0)
            afternoon_end_hour = trading_hours.get('afternoon_end', {}).get('hour', 0)
            afternoon_end_minute = trading_hours.get('afternoon_end', {}).get('minute', 0)
            
            # Add afternoon session to mask
            afternoon_mask = (
                ((df[hour_column] == afternoon_start_hour) & (df[minute_column] >= afternoon_start_minute)) |
                ((df[hour_column] > afternoon_start_hour) & (df[hour_column] < afternoon_end_hour)) |
                ((df[hour_column] == afternoon_end_hour) & (df[minute_column] <= afternoon_end_minute))
            )
            
            mask = mask | afternoon_mask
        
        # Apply the mask
        filtered_df = df[mask].copy()
        
        logger.info(f"Filtered trading hours, resulting in {len(filtered_df)} rows")
        return filtered_df
    
    def calculate_indicators(self, df: Optional[pd.DataFrame] = None,
                           indicators_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame to process or None to use self.data
            indicators_config: Dictionary with indicator settings
            
        Returns:
            DataFrame with calculated indicators
        """
        if df is None:
            if self.data is None:
                logger.error("No data loaded. Call load_csv or another load method first.")
                return pd.DataFrame()
            df = self.data.copy()
        
        # Get indicators config from config if not provided
        if indicators_config is None:
            indicators_config = self.config.get('indicators', {})
        
        # Calculate moving averages
        if 'moving_averages' in indicators_config:
            for ma_config in indicators_config['moving_averages']:
                window = ma_config.get('window', 20)
                column = ma_config.get('column', 'close')
                name = ma_config.get('name', f'MA_{window}')
                
                if column in df.columns:
                    df[name] = df[column].rolling(window=window).mean()
        
        # Calculate Bollinger Bands
        if 'bollinger_bands' in indicators_config:
            for bb_config in indicators_config['bollinger_bands']:
                window = bb_config.get('window', 20)
                column = bb_config.get('column', 'close')
                std_dev = bb_config.get('std_dev', 2)
                
                if column in df.columns:
                    # Calculate middle band (SMA)
                    df[f'middle_band'] = df[column].rolling(window=window).mean()
                    
                    # Calculate standard deviation
                    df[f'std_dev'] = df[column].rolling(window=window).std()
                    
                    # Calculate upper and lower bands
                    df[f'upper_band'] = df[f'middle_band'] + std_dev * df[f'std_dev']
                    df[f'lower_band'] = df[f'middle_band'] - std_dev * df[f'std_dev']
        
        # Calculate RSI
        if 'rsi' in indicators_config:
            for rsi_config in indicators_config['rsi']:
                window = rsi_config.get('window', 14)
                column = rsi_config.get('column', 'close')
                
                if column in df.columns:
                    df[f'RSI'] = self._calculate_rsi(df[column], window)
        
        # Calculate ATR
        if 'atr' in indicators_config:
            for atr_config in indicators_config['atr']:
                window = atr_config.get('window', 14)
                
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    df[f'ATR'] = self._calculate_atr(df, window)
        
        # Calculate additional indicators as needed...
        
        logger.info(f"Calculated indicators based on configuration")
        return df
    
    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            series: Price series
            window: RSI window
            
        Returns:
            Series with RSI values
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / (loss + 1e-10)  # Add small constant to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with high, low, close columns
            window: ATR window
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with standardized column names
        """
        # Define column mapping based on common variants
        column_mapping = {
            'Date': 'date',
            'Timestamp': 'date',
            'Time': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'AdjClose': 'adj_close',
            'Volume': 'volume'
        }
        
        # Apply mapping to columns
        renamed_columns = {}
        for col in df.columns:
            if col in column_mapping:
                renamed_columns[col] = column_mapping[col]
        
        if renamed_columns:
            df = df.rename(columns=renamed_columns)
            logger.info(f"Standardized column names: {renamed_columns}")
        
        return df
    
    def get_data_info(self) -> Dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        return self.data_info

# Create utility functions for easy access
def load_from_csv(file_path: str, date_column: str = 'date', config: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        date_column: Name of the date column
        config: Configuration dictionary
        **kwargs: Additional keyword arguments for pd.read_csv
        
    Returns:
        DataFrame with loaded data
    """
    loader = DataLoader(config)
    return loader.load_csv(file_path, date_column, **kwargs)
