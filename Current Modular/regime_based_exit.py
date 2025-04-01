"""
Regime-Based Exit Strategy - Implementation of an exit strategy that adapts to market regimes.

This module provides a concrete implementation of the ExitStrategy that adapts its
exit parameters based on the current market regime (trend-following, mean-reverting, 
or neutral).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union

from exits.exit_strategy import ExitStrategy
from market_regimes.regime_detector import RegimeDetector

# Configure logging
logger = logging.getLogger(__name__)

class RegimeBasedExitStrategy(ExitStrategy):
    """
    An exit strategy that adapts to market regimes.
    
    This strategy adjusts stop-loss, take-profit, and other exit parameters
    based on the current market regime (trend-following, mean-reverting, or neutral).
    """
    
    def __init__(self, 
                 name: str = "RegimeBasedExit", 
                 config: Optional[Dict] = None,
                 regime_detector: Optional[RegimeDetector] = None):
        """
        Initialize the regime-based exit strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
            regime_detector: Market regime detector
        """
        super().__init__(name, config)
        
        # Store regime detector
        self.regime_detector = regime_detector
        
        # Default exit parameters - will be overridden by regime-specific parameters
        self.enable_trailing_stop = config.get('enable_trailing_stop', True)
        self.trailing_stop_multiplier = config.get('trailing_stop_multiplier', 2.0)
        self.dynamic_target_enable = config.get('dynamic_target_enable', True)
        self.dynamic_target_multiplier = config.get('dynamic_target_multiplier', 1.0)
        self.max_bars_held = config.get('max_bars_held', 12)
        
        # Tracking variables
        self.highest_price_since_entry = {}  # Key: entry_bar, Value: highest price
        self.lowest_price_since_entry = {}   # Key: entry_bar, Value: lowest price
        self.current_stop_loss = {}          # Key: entry_bar, Value: current stop price
        self.current_profit_target = {}      # Key: entry_bar, Value: current target price
        
        # Metrics
        self.metrics = {
            'trailing_stop_exits': 0,
            'profit_target_exits': 0,
            'time_exits': 0,
            'regime_changes': 0
        }
        
        # Load regime-specific parameters
        self.regime_params = self._load_regime_parameters()
        
        logger.info(f"RegimeBasedExitStrategy initialized")
    
    def _load_regime_parameters(self) -> Dict:
        """
        Load regime-specific exit parameters from config.
        
        Returns:
            Dictionary with regime-specific parameters
        """
        # Default parameters for each regime
        regime_params = {
            'trend_following': {
                'enable_trailing_stop': True,
                'trailing_stop_multiplier': 1.0,  # Tighter trailing for trends
                'dynamic_target_multiplier': 2.0,  # More aggressive targets
                'max_bars_held': 24  # Allow trends to develop longer
            },
            'mean_reverting': {
                'enable_trailing_stop': True,
                'trailing_stop_multiplier': 2.0,  # Wider trailing for mean reversion
                'dynamic_target_multiplier': 1.0,  # Target at mean
                'max_bars_held': 6  # Shorter hold time for mean reversion
            },
            'neutral': {
                'enable_trailing_stop': True,
                'trailing_stop_multiplier': 1.5,  # Middle ground
                'dynamic_target_multiplier': 1.2,  # Moderate targets
                'max_bars_held': 12  # Standard hold time
            }
        }
        
        # Override with any config parameters
        for regime in regime_params:
            if regime in self.config:
                regime_params[regime].update(self.config[regime])
        
        return regime_params
    
    def _initialize_exit_strategy(self) -> None:
        """Initialize strategy-specific components."""
        # Initialize regime detector if provided
        if self.regime_detector:
            self.regime_detector.initialize(self.data)
            logger.info(f"Regime detector {self.regime_detector.name} initialized")
    
    def _get_regime_parameters(self, current_regime: str) -> Dict:
        """
        Get exit parameters for the current regime.
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Dictionary with regime-specific parameters
        """
        # Default to neutral if regime not recognized
        if current_regime not in self.regime_params:
            current_regime = 'neutral'
            
        return self.regime_params[current_regime]
    
    def check_exit(self, bar: pd.Series, bar_index: int, position: float,
                  entry_price: float, entry_bar: int, **kwargs) -> Dict:
        """
        Check for exit signals based on the current market regime.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size (positive for long, negative for short)
            entry_price: Entry price of the current position
            entry_bar: Bar index when the position was entered
            **kwargs: Additional keyword arguments:
                - atr: ATR value at entry
                - regime: Override the current regime
                
        Returns:
            Dictionary with exit signal information
        """
        # Get current ATR if provided or use a default value
        atr = kwargs.get('atr', bar.get('ATR', 0))
        if atr == 0 and 'ATR' in bar:
            atr = bar['ATR']
        
        # Get current regime
        current_regime = kwargs.get('regime')
        if current_regime is None and self.regime_detector:
            regime_info = self.regime_detector.update(bar, bar_index)
            current_regime = regime_info['regime']
        else:
            current_regime = current_regime or 'neutral'
        
        # Get parameters for the current regime
        params = self._get_regime_parameters(current_regime)
        enable_trailing_stop = params.get('enable_trailing_stop', self.enable_trailing_stop)
        trailing_stop_multiplier = params.get('trailing_stop_multiplier', self.trailing_stop_multiplier)
        dynamic_target_enable = params.get('dynamic_target_enable', self.dynamic_target_enable)
        dynamic_target_multiplier = params.get('dynamic_target_multiplier', self.dynamic_target_multiplier)
        max_bars_held = params.get('max_bars_held', self.max_bars_held)
        
        # Initialize tracking variables for this position if new
        if entry_bar not in self.highest_price_since_entry:
            self.highest_price_since_entry[entry_bar] = entry_price
            self.lowest_price_since_entry[entry_bar] = entry_price
            
            # Initial stop loss - basic ATR-based
            stop_distance = atr * trailing_stop_multiplier
            if position > 0:  # Long position
                self.current_stop_loss[entry_bar] = entry_price - stop_distance
            else:  # Short position
                self.current_stop_loss[entry_bar] = entry_price + stop_distance
            
            # Initial profit target - based on ATR or middle band
            if dynamic_target_enable:
                target_distance = atr * dynamic_target_multiplier
                if position > 0:  # Long position
                    self.current_profit_target[entry_bar] = entry_price + target_distance
                else:  # Short position
                    self.current_profit_target[entry_bar] = entry_price - target_distance
            else:
                # Default to middle band as target
                if 'middle_band' in bar:
                    self.current_profit_target[entry_bar] = bar['middle_band']
                else:
                    # Fallback to ATR-based target
                    target_distance = atr * 1.0  # Default multiplier
                    if position > 0:  # Long position
                        self.current_profit_target[entry_bar] = entry_price + target_distance
                    else:  # Short position
                        self.current_profit_target[entry_bar] = entry_price - target_distance
        
        # Update highest/lowest prices since entry
        self.highest_price_since_entry[entry_bar] = max(self.highest_price_since_entry[entry_bar], bar['high'])
        self.lowest_price_since_entry[entry_bar] = min(self.lowest_price_since_entry[entry_bar], bar['low'])
        
        # Update trailing stop if enabled
        if enable_trailing_stop:
            is_long = position > 0
            
            if is_long:
                # For long positions, trail price when it rises
                highest_price = self.highest_price_since_entry[entry_bar]
                trail_distance = atr * trailing_stop_multiplier
                new_stop = highest_price - trail_distance
                
                # Only move stop loss up, never down
                if new_stop > self.current_stop_loss[entry_bar]:
                    self.current_stop_loss[entry_bar] = new_stop
            else:
                # For short positions, trail price when it falls
                lowest_price = self.lowest_price_since_entry[entry_bar]
                trail_distance = atr * trailing_stop_multiplier
                new_stop = lowest_price + trail_distance
                
                # Only move stop loss down, never up
                if new_stop < self.current_stop_loss[entry_bar]:
                    self.current_stop_loss[entry_bar] = new_stop
        
        # Check for exits
        
        # 1. Stop loss hit
        if position > 0 and bar['low'] <= self.current_stop_loss[entry_bar]:
            # Long position hit stop loss
            self.metrics['trailing_stop_exits'] += 1
            return {
                'exit': True,
                'price': self.current_stop_loss[entry_bar],
                'reason': 'trailing_stop',
                'regime': current_regime
            }
            
        elif position < 0 and bar['high'] >= self.current_stop_loss[entry_bar]:
            # Short position hit stop loss
            self.metrics['trailing_stop_exits'] += 1
            return {
                'exit': True,
                'price': self.current_stop_loss[entry_bar],
                'reason': 'trailing_stop',
                'regime': current_regime
            }
        
        # 2. Profit target hit
        if position > 0 and bar['high'] >= self.current_profit_target[entry_bar]:
            # Long position hit profit target
            self.metrics['profit_target_exits'] += 1
            return {
                'exit': True,
                'price': self.current_profit_target[entry_bar],
                'reason': 'profit_target',
                'regime': current_regime
            }
            
        elif position < 0 and bar['low'] <= self.current_profit_target[entry_bar]:
            # Short position hit profit target
            self.metrics['profit_target_exits'] += 1
            return {
                'exit': True,
                'price': self.current_profit_target[entry_bar],
                'reason': 'profit_target',
                'regime': current_regime
            }
        
        # 3. Maximum hold time reached
        bars_held = bar_index - entry_bar
        if bars_held >= max_bars_held:
            self.metrics['time_exits'] += 1
            return {
                'exit': True,
                'price': bar['close'],
                'reason': 'max_time',
                'regime': current_regime
            }
        
        # 4. Regime change exit (optional)
        if self.config.get('exit_on_regime_change', False) and 'prev_regime' in kwargs:
            prev_regime = kwargs['prev_regime']
            if prev_regime != current_regime:
                self.metrics['regime_changes'] += 1
                return {
                    'exit': True,
                    'price': bar['close'],
                    'reason': 'regime_change',
                    'old_regime': prev_regime,
                    'new_regime': current_regime
                }
        
        # No exit signal
        return {
            'exit': False,
            'highest_price': self.highest_price_since_entry[entry_bar],
            'lowest_price': self.lowest_price_since_entry[entry_bar],
            'current_stop': self.current_stop_loss[entry_bar],
            'current_target': self.current_profit_target[entry_bar]
        }
    
    def on_bar_update(self, bar: pd.Series, bar_index: int, position: float,
                    entry_price: float, entry_bar: int, **kwargs) -> None:
        """
        Update strategy state on each bar.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size
            entry_price: Entry price
            entry_bar: Bar index when the position was entered
            **kwargs: Additional keyword arguments
        """
        # Continue updating highest/lowest prices
        if entry_bar in self.highest_price_since_entry:
            self.highest_price_since_entry[entry_bar] = max(self.highest_price_since_entry[entry_bar], bar['high'])
            self.lowest_price_since_entry[entry_bar] = min(self.lowest_price_since_entry[entry_bar], bar['low'])
    
    def on_exit(self, bar: pd.Series, bar_index: int, position: float,
               entry_price: float, entry_bar: int, exit_price: float,
               exit_reason: str, **kwargs) -> None:
        """
        Handle exit events.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Position size that was exited
            entry_price: Entry price of the position
            entry_bar: Bar index when the position was entered
            exit_price: Exit price
            exit_reason: Reason for the exit
            **kwargs: Additional keyword arguments
        """
        # Clean up tracking variables
        if entry_bar in self.highest_price_since_entry:
            del self.highest_price_since_entry[entry_bar]
        if entry_bar in self.lowest_price_since_entry:
            del self.lowest_price_since_entry[entry_bar]
        if entry_bar in self.current_stop_loss:
            del self.current_stop_loss[entry_bar]
        if entry_bar in self.current_profit_target:
            del self.current_profit_target[entry_bar]
        
        # Log exit
        position_type = 'long' if position > 0 else 'short'
        logger.info(f"Exited {position_type} position from bar {entry_bar} at price {exit_price}, reason: {exit_reason}")
    
    def calculate_price_capture(self, trade: Dict) -> float:
        """
        Calculate the price capture efficiency for a trade.
        
        Args:
            trade: Dictionary with trade information
            
        Returns:
            Price capture percentage (0-100)
        """
        # Get trade details
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        is_long = trade['type'] == 'long'
        
        if 'highest_price_reached' not in trade or 'lowest_price_reached' not in trade:
            return 0.0
        
        highest_price = trade['highest_price_reached']
        lowest_price = trade['lowest_price_reached']
        
        # Ensure values are present
        if highest_price is None or lowest_price is None:
            return 0.0
        
        # Calculate price range and capture
        if is_long:
            # For long trades, measure from entry to highest
            potential_profit = highest_price - entry_price
            actual_profit = exit_price - entry_price
        else:
            # For short trades, measure from entry to lowest
            potential_profit = entry_price - lowest_price
            actual_profit = entry_price - exit_price
        
        # Calculate percentage
        if potential_profit > 0:
            capture_pct = (actual_profit / potential_profit) * 100
            return min(max(0, capture_pct), 100)  # Clip to 0-100 range
        
        return 0.0
