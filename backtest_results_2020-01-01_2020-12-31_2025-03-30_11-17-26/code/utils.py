"""
utils.py - Common utility functions for the backtester

This module provides shared utility functions for directory setup,
data loading, and technical indicator calculations.
"""

import pandas as pd
import numpy as np
import os
import shutil
import logging
from datetime import datetime, timedelta, date, time

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(start_date, end_date, base_name="backtest_results"):
    """
    Set up output directories with timestamp and return file paths.
    
    Args:
        start_date: Start date of the backtest (YYYY-MM-DD)
        end_date: End date of the backtest (YYYY-MM-DD)
        base_name: Prefix for the directory name
        
    Returns:
        Tuple of (output_dir, file_paths)
    """
    # Create date range string for directory naming
    date_range_str = f"{start_date}_{end_date}"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'{base_name}_{date_range_str}_{current_time}'

    # Create the main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create subdirectories
    subdirs = [
        os.path.join(output_dir, 'code'),
        os.path.join(output_dir, 'executed_trade_plots'),
        os.path.join(output_dir, 'ml'),
        os.path.join(output_dir, 'quarterly_analysis'),
    ]
    
    for directory in subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Set up file paths
    file_paths = {
        'trade_log': os.path.join(output_dir, f'trade_log_mes_5min_{date_range_str}.csv'),
        'portfolio_value': os.path.join(output_dir, f'portfolio_value_mes_5min_{date_range_str}.csv'),
        'regime_log': os.path.join(output_dir, f'market_regime_log_{date_range_str}.csv'),
        'market_type_log': os.path.join(output_dir, f'market_type_log_{date_range_str}.csv'),
        'summary': os.path.join(output_dir, f'summary_{date_range_str}_{current_time}.txt'),
        'ml_trade_log': os.path.join(output_dir, f'ml_trade_log_mes_5min_{date_range_str}.csv'),
        'ml_predictions': os.path.join(output_dir, 'ml', f'ml_predictions_{date_range_str}.csv'),
        'potential_trades': os.path.join(output_dir, 'ml', f'potential_trades_{date_range_str}.csv'),
    }
    
    # Return both the output directory and file paths
    return output_dir, file_paths

def copy_project_files(output_dir):
    """
    Copy all relevant Python files to the code subdirectory.
    
    Args:
        output_dir: Main output directory
    """
    code_dir = os.path.join(output_dir, 'code')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)
    
    # Copy all Python files in the current directory
    for file in os.listdir(current_dir):
        if file.endswith('.py'):
            file_path = os.path.join(current_dir, file)
            dest_path = os.path.join(code_dir, file)
            try:
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied {file} to code directory")
            except Exception as e:
                logger.warning(f"Could not copy {file}: {e}")

def load_and_process_data(file_path, start_date, end_date, filter_hours=True):
    """
    Load, prepare, and filter CSV price data.
    
    Args:
        file_path: Path to the CSV file
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        filter_hours: Whether to filter to trading hours
        
    Returns:
        DataFrame with processed price data
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file '{file_path}' not found.")
        return None
    
    try:
        # Try to read with standard headers first
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
        
        # If no standard headers, assume OHLCV format
        if 'date' not in df.columns:
            col_names = ['date', 'open', 'high', 'low', 'close', 'volume']
            # Add names for any additional columns
            for i in range(len(col_names), len(df.columns)):
                col_names.append(f'column_{i}')
            # Assign column names
            df.columns = col_names[:len(df.columns)]
    
    except Exception as e:
        # Fallback method
        df = pd.read_csv(file_path, header=None)
        col_names = ['date', 'open', 'high', 'low', 'close', 'volume']
        for i in range(len(col_names), len(df.columns)):
            col_names.append(f'column_{i}')
        df.columns = col_names[:len(df.columns)]
    
    # Convert date column to datetime
    try:
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle timezone if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert('US/Eastern')
        
    except Exception as e:
        logger.error(f"Date conversion failed: {e}")
        return None
    
    # Filter by date range
    try:
        start_date_dt = pd.Timestamp(start_date)
        end_date_dt = pd.Timestamp(end_date)
        
        df = df[(df['date'].dt.date >= start_date_dt.date()) & 
                (df['date'].dt.date <= end_date_dt.date())]
    except Exception as e:
        logger.error(f"Error in date filtering: {e}")
        return None
    
    # Add time components for filtering
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    # Filter to trading hours if requested
    if filter_hours:
        df = filter_trading_hours(df)
    
    # Sort by date
    df.sort_values('date', inplace=True)
    
    logger.info(f"Loaded {len(df)} rows from {start_date} to {end_date}")
    return df.reset_index(drop=True)

def filter_trading_hours(df):
    """Filter data to only include regular trading hours (9:30 AM to 4:00 PM ET)"""
    trading_hours_mask = (
        ((df['hour'] == 9) & (df['minute'] >= 30)) |  # 9:30 AM and later
        ((df['hour'] > 9) & (df['hour'] < 16)) |      # 10 AM to 3:59 PM
        ((df['hour'] == 16) & (df['minute'] == 0))    # 4:00 PM exactly
    )
    return df[trading_hours_mask].reset_index(drop=True)

def find_closest_weekday(target_date, target_weekday):
    """
    Find the date of the closest specified weekday to a target date.
    
    Args:
        target_date: The reference date
        target_weekday: Integer representing weekday (0=Monday, 6=Sunday)
        
    Returns:
        Date of the closest matching weekday
    """
    # Calculate days difference
    diff = (target_date.weekday() - target_weekday) % 7
    
    # If diff > 3, it's closer to go forward to next occurrence
    if diff > 3:
        # Go to next occurrence
        return target_date + timedelta(days=7-diff)
    else:
        # Go back to previous occurrence
        return target_date - timedelta(days=diff)

def calculate_season_dates(year, season_config):
    """
    Calculate start and end dates for all seasons in a given year.
    
    Args:
        year: The year to calculate season dates for
        season_config: The season configuration dictionary
        
    Returns:
        Dictionary of season date ranges {season_name: (start_date, end_date)}
    """
    seasons = {}
    
    for name, config in season_config.items():
        # Get reference dates
        start_month, start_day = config['start_reference']
        end_month, end_day = config['end_reference']
        
        # Create reference dates
        start_ref = date(year, start_month, start_day)
        end_ref = date(year, end_month, end_day)
        
        # Find closest matching weekdays
        start_date = find_closest_weekday(start_ref, config['start_day'])
        end_date = find_closest_weekday(end_ref, config['end_day'])
        
        # Store in seasons dictionary
        seasons[name] = (start_date, end_date)
    
    return seasons

def is_in_season(current_date, season_dates):
    """
    Check if a date falls within any defined season.
    
    Args:
        current_date: The date to check
        season_dates: Dictionary of season date ranges
        
    Returns:
        Tuple of (in_season, season_name)
    """
    for season, date_ranges in season_dates.items():
        for start_date, end_date in date_ranges:
            if start_date <= current_date <= end_date:
                return True, season
    
    return False, None

def calculate_indicators(df, config):
    """
    Calculate all technical indicators used in the strategy.
    
    Args:
        df: DataFrame with price data
        config: Configuration dictionary with indicator settings
        
    Returns:
        DataFrame with added technical indicators
    """
    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(config['strategy']['bb_window']).mean()
    df['std_dev'] = df['close'].rolling(config['strategy']['bb_window']).std()
    df['upper_band'] = df['middle_band'] + 2 * df['std_dev']
    df['lower_band'] = df['middle_band'] - 2 * df['std_dev']
    
    # RSI
    df['RSI'] = calculate_rsi(df, periods=config['strategy']['rsi_window'])
    
    # ATR calculation
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    df['ATR'] = df['true_range'].rolling(window=config['strategy']['rsi_window']).mean()
    
    # Volume indicator
    df['avg_volume'] = df['volume'].rolling(window=config['strategy']['bb_window']).mean()
    
    # Market regime indicators
    df['ADX'] = calculate_adx(df, window=config['regime']['adx_window'])
    df['MA'] = df['close'].rolling(window=config['regime']['ma_window']).mean()
    df['MA_slope'] = calculate_ma_slope(df, 
                                       ma_window=config['regime']['ma_window'], 
                                       slope_window=config['regime']['ma_slope_window'])
    
    # Add volatility regime
    df['volatility_regime'], df['atr_ratio'] = calculate_volatility_regime(df, 
                                                                         window=config['regime']['volatility_window'], 
                                                                         threshold=config['regime']['volatility_threshold'])
    
    # Add columns for regime scoring that will be filled during backtest
    df['regime_score'] = np.nan
    df['favorable_regime'] = np.nan
    df['position_size_adj'] = np.nan
    
    # Calculate warm-up period for NaN values
    warm_up = max(config['strategy']['bb_window'], 
                 config['regime']['ma_window'], 
                 config['regime']['adx_window'] * 2) + config['regime']['ma_slope_window']
    
    return df.iloc[warm_up:].reset_index(drop=True)

def calculate_rsi(data, periods=14):
    """Calculate RSI indicator using vectorized operations"""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    # Add small constant to avoid division by zero
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(data, window=14):
    """Calculate ADX indicator using vectorized operations"""
    # Calculate true range first
    data = data.copy()
    data['tr0'] = abs(data['high'] - data['low'])
    data['tr1'] = abs(data['high'] - data['close'].shift(1))
    data['tr2'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    data['atr'] = data['tr'].rolling(window).mean()
    
    # Calculate +DI and -DI
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']
    
    data['plus_dm'] = np.where(
        (data['up_move'] > data['down_move']) & (data['up_move'] > 0),
        data['up_move'],
        0
    )
    
    data['minus_dm'] = np.where(
        (data['down_move'] > data['up_move']) & (data['down_move'] > 0),
        data['down_move'],
        0
    )
    
    # Calculate smoothed values
    data['plus_di'] = 100 * data['plus_dm'].rolling(window).mean() / data['atr']
    data['minus_di'] = 100 * data['minus_dm'].rolling(window).mean() / data['atr']
    
    # Calculate directional movement index
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'] + 1e-10)
    
    # Calculate ADX
    data['adx'] = data['dx'].rolling(window).mean()
    
    return data['adx']

def calculate_ma_slope(data, ma_window=50, slope_window=10):
    """Calculate moving average slope as percentage change"""
    ma = data['close'].rolling(ma_window).mean()
    slope_pct = (ma / ma.shift(slope_window) - 1) * 100
    return slope_pct

def calculate_volatility_regime(data, window=20, threshold=1.8):
    """Determine market volatility regime (low, normal, high)"""
    # Use pre-calculated ATR if available, otherwise calculate it
    if 'ATR' not in data.columns:
        # Calculate ATR
        data['high_low'] = data['high'] - data['low']
        data['high_prev_close'] = abs(data['high'] - data['close'].shift(1))
        data['low_prev_close'] = abs(data['low'] - data['close'].shift(1))
        data['true_range'] = data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        data['ATR'] = data['true_range'].rolling(window=window).mean()
    
    # Calculate average ATR over the volatility window
    avg_atr = data['ATR'].rolling(window=window).mean()
    
    # Calculate ATR relative to its average
    atr_ratio = data['ATR'] / avg_atr
    
    # Determine volatility regime
    volatility_regime = np.where(
        atr_ratio > threshold, 2,         # High volatility
        np.where(
            atr_ratio < 0.6, 0,           # Low volatility
            1                             # Normal volatility
        )
    )
    
    return pd.Series(volatility_regime, index=data.index), atr_ratio

def define_explicit_seasons(years, season_config):
    """
    Define seasons explicitly with fixed dates instead of using weekday adjustment.

    Args:
        years: List of years to define seasons for
        season_config: Configuration dictionary with season definitions

    Returns:
        Dictionary of season date ranges
    """
    seasons = {}

    for year in years:
        for name, config in season_config.items():
            start_month, start_day = config['start_reference']
            end_month, end_day = config['end_reference']

            # Create dates without weekday adjustment
            start_date = date(year, start_month, start_day)
            end_date = date(year, end_month, end_day)

            # Ensure start date is a weekday (Mon-Fri)
            while start_date.weekday() > 4:  # 5=Sat, 6=Sun
                start_date += timedelta(days=1)  # Move to next day

            # Ensure end date is a weekday (Mon-Fri)
            while end_date.weekday() > 4:  # 5=Sat, 6=Sun
                end_date -= timedelta(days=1)  # Move to previous day

            # Add to seasons dictionary
            if name not in seasons:
                seasons[name] = []
            seasons[name].append((start_date, end_date))

    return seasons

def initialize_random_seeds(seed=None):
    """
    Initialize all random number generators consistently.
    """
    from config import config

    # Use provided seed or get from config
    random_seed = seed if seed is not None else config['global']['random_seed']

    # Set all seeds
    import random
    import numpy as np

    # Python's random
    random.seed(random_seed)

    # NumPy
    np.random.seed(random_seed)

    # Try to set sklearn seed (for HMM)
    try:
        from sklearn.utils import check_random_state
        _ = check_random_state(random_seed)
    except ImportError:
        pass

    # If using PyTorch or TensorFlow, set their seeds too
    try:
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(random_seed)
    except ImportError:
        pass

    logger.info(f"All random seeds initialized to: {random_seed}")
    return random_seed