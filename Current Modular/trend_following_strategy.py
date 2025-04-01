"""
Trend Following Strategy - Implementation of a trend following strategy.

This module provides a concrete implementation of the BaseStrategy for 
trading based on trend following principles using moving averages and momentum.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union

# Import base classes
from strategies.base_strategy import BaseStrategy
from market_regimes.regime_detector import RegimeDetector
from exits.exit_strategy import ExitStrategy
from risk.position_sizer import PositionSizer

# Configure logging
logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    A trend following strategy using moving averages and momentum indicators.
    
    This strategy looks for strong trends with price momentum in the direction
    of the trend, entering trades with the expectation of price continuing in the
    established trend direction.
    """
    
    def __init__(self, 
                 name: str = "TrendFollowing", 
                 config: Optional[Dict] = None,
                 regime_detector: Optional[RegimeDetector] = None,
                 exit_strategy: Optional[ExitStrategy] = None,
                 position_sizer: Optional[PositionSizer] = None):
        """
        Initialize the trend following strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
            regime_detector: Market regime detector
            exit_strategy: Exit strategy
            position_sizer: Position sizing strategy
        """
        super().__init__(name, config)
        
        # Components
        self.regime_detector = regime_detector
        self.exit_strategy = exit_strategy
        self.position_sizer = position_sizer
        
        # Strategy parameters (with defaults)
        self.ma_slope_threshold = config.get('ma_slope_threshold', 0.1)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.rsi_min = config.get('rsi_min', 40)  # Min RSI for longs
        self.rsi_max = config.get('rsi_max', 60)  # Max RSI for shorts
        self.volume_multiplier = config.get('volume_multiplier', 1.2)
        self.enable_short = config.get('enable_short', True)
        self.enable_long = config.get('enable_long', True)
        
        # Tracking metrics
        self.metrics = {
            'long_signals': 0,
            'short_signals': 0,
            'regime_filtered': 0,
            'slope_filtered': 0,
            'adx_filtered': 0
        }
        
        logger.info(f"TrendFollowingStrategy initialized with MA slope threshold: {self.ma_slope_threshold}, ADX threshold: {self.adx_threshold}")
    
    def _initialize_strategy(self) -> None:
        """Initialize strategy-specific components."""
        # Initialize components if provided
        if self.regime_detector:
            self.regime_detector.initialize(self.data)
            logger.info(f"Regime detector {self.regime_detector.name} initialized")
        
        if self.exit_strategy:
            self.exit_strategy.initialize(self.data)
            logger.info(f"Exit strategy {self.exit_strategy.name} initialized")
        
        if self.position_sizer:
            self.position_sizer.initialize(self.data)
            logger.info(f"Position sizer {self.position_sizer.name} initialized")
    
    def check_entry(self, bar: pd.Series, bar_index: int) -> Dict:
        """
        Check for entry signals based on trend following principles.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            
        Returns:
            Dictionary with entry signal information
        """
        # Skip if not enough data
        if bar_index < 1:
            return {'entry': False}
        
        # Get previous bar for entry decisions
        prev_bar = self.data.iloc[bar_index - 1]
        
        # Initialize result
        result = {'entry': False}
        
        # Update market regime if available
        current_regime = None
        if self.regime_detector:
            regime_info = self.regime_detector.update(bar, bar_index)
            current_regime = regime_info['regime']
        
        # Check if we're in a trend-following regime, if detector is available
        if current_regime and current_regime != 'trend_following' and self.regime_detector:
            # Only take trades in trend-following regime
            if 'confidence' in regime_info and regime_info['confidence'] > 0.7:
                # Skip if we're confident this is not a trend-following regime
                self.metrics['regime_filtered'] += 1
                return {'entry': False, 'regime_filtered': True, 'regime': current_regime}
        
        # Check ADX for trend strength
        adx_value = prev_bar.get('ADX', 0)
        if adx_value < self.adx_threshold:
            self.metrics['adx_filtered'] += 1
            return {'entry': False, 'adx_filtered': True, 'adx': adx_value}
        
        # Check MA slope for trend direction
        ma_slope = prev_bar.get('MA_slope', 0)
        
        # Long entry conditions
        long_signal = False
        if self.enable_long:
            long_signal = (
                # Trend direction is up
                ma_slope > self.ma_slope_threshold and
                # Price is above moving average
                prev_bar['close'] > prev_bar['MA'] and
                # RSI shows momentum but not overbought
                prev_bar['RSI'] > self.rsi_min and prev_bar['RSI'] < 70 and
                # Confirm with volume
                prev_bar['volume'] > prev_bar['avg_volume'] * self.volume_multiplier
            )
            
            if long_signal:
                self.metrics['long_signals'] += 1
        
        # Short entry conditions
        short_signal = False
        if self.enable_short:
            short_signal = (
                # Trend direction is down
                ma_slope < -self.ma_slope_threshold and
                # Price is below moving average
                prev_bar['close'] < prev_bar['MA'] and
                # RSI shows downward momentum but not oversold
                prev_bar['RSI'] < self.rsi_max and prev_bar['RSI'] > 30 and
                # Confirm with volume
                prev_bar['volume'] > prev_bar['avg_volume'] * self.volume_multiplier
            )
            
            if short_signal:
                self.metrics['short_signals'] += 1
        
        # Determine final signal
        if long_signal:
            # Calculate position size
            size = 1.0  # Default size
            if self.position_sizer:
                account_value = self.engine.current_capital
                size = self.position_sizer.calculate_position_size(
                    bar, bar_index, 'long', bar['open'], account_value,
                    risk_params={'atr': prev_bar.get('ATR', 0)},
                    market_regime=current_regime
                )
            
            # Return entry signal
            return {
                'entry': True,
                'type': 'long',
                'price': bar['open'],  # Enter at the open of the current bar
                'size': size,
                'reason': 'uptrend_momentum',
                'adx': adx_value,
                'ma_slope': ma_slope,
                'rsi': prev_bar['RSI'],
                'volume_ratio': prev_bar['volume'] / prev_bar['avg_volume'] if 'avg_volume' in prev_bar else 0,
                'regime': current_regime
            }
        
        elif short_signal:
            # Calculate position size
            size = 1.0  # Default size
            if self.position_sizer:
                account_value = self.engine.current_capital
                size = self.position_sizer.calculate_position_size(
                    bar, bar_index, 'short', bar['open'], account_value,
                    risk_params={'atr': prev_bar.get('ATR', 0)},
                    market_regime=current_regime
                )
            
            # Return entry signal
            return {
                'entry': True,
                'type': 'short',
                'price': bar['open'],  # Enter at the open of the current bar
                'size': size,
                'reason': 'downtrend_momentum',
                'adx': adx_value,
                'ma_slope': ma_slope,
                'rsi': prev_bar['RSI'],
                'volume_ratio': prev_bar['volume'] / prev_bar['avg_volume'] if 'avg_volume' in prev_bar else 0,
                'regime': current_regime
            }
        
        # No entry signal
        return {'entry': False}
    
    def check_exit(self, bar: pd.Series, bar_index: int, position: float, 
                  entry_price: float, entry_time: Any) -> Dict:
        """
        Check for exit signals.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size (positive for long, negative for short)
            entry_price: Entry price of the current position
            entry_time: Entry time of the current position
            
        Returns:
            Dictionary with exit signal information
        """
        # Use exit strategy if available
        if self.exit_strategy:
            # Find entry bar index
            entry_bar = entry_time if isinstance(entry_time, int) else self.data.index.get_loc(entry_time)
            
            # Get exit signal from exit strategy
            exit_signal = self.exit_strategy.check_exit(
                bar, bar_index, position, entry_price, entry_bar,
                atr=bar.get('ATR', 0),
                regime=self.regime_detector.get_current_regime() if self.regime_detector else 'trend_following'
            )
            
            return exit_signal
        
        # Simple default exit strategy: exit when trend reverses
        is_long = position > 0
        ma_slope = bar.get('MA_slope', 0)
        
        if is_long and ma_slope < -self.ma_slope_threshold:
            return {
                'exit': True,
                'price': bar['close'],
                'reason': 'trend_reversal'
            }
        
        elif not is_long and ma_slope > self.ma_slope_threshold:
            return {
                'exit': True,
                'price': bar['close'],
                'reason': 'trend_reversal'
            }
        
        # No exit signal
        return {'exit': False}
    
    def on_entry(self, bar: pd.Series, bar_index: int, position: float, entry_price: float) -> None:
        """
        Handle entry events.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Position size (positive for long, negative for short)
            entry_price: Entry price
        """
        # Log entry
        position_type = 'long' if position > 0 else 'short'
        logger.info(f"Entered {position_type} position of size {abs(position)} at {entry_price}")
    
    def on_exit(self, bar: pd.Series, bar_index: int, trade: Dict) -> None:
        """
        Handle exit events.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            trade: Dictionary with trade information
        """
        # Log exit
        logger.info(f"Exited {trade['type']} position at {trade['exit_price']}, " +
                   f"profit: {trade['profit']}, reason: {trade['exit_reason']}")
