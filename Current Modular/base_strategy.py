"""
Base Strategy Interface - Defines the contract for trading strategies.

This module provides the BaseStrategy abstract class which all trading strategies
should inherit from to ensure compatibility with the backtesting engine.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement
    to be compatible with the backtesting engine. It provides methods for
    checking entry and exit conditions, and handling events during the backtest.
    """
    
    def __init__(self, name: str = "BaseStrategy", config: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.initialized = False
        self.metrics = {}
        logger.info(f"Strategy {name} initialized")
    
    def initialize(self, data: pd.DataFrame, engine: Any) -> None:
        """
        Initialize the strategy with data and engine reference.
        
        This method is called by the engine before the backtest starts.
        
        Args:
            data: Full dataset for the backtest
            engine: Reference to the backtesting engine
        """
        self.data = data
        self.engine = engine
        self.initialized = True
        self._initialize_strategy()
        logger.info(f"Strategy {self.name} initialized with {len(data)} bars of data")
    
    @abstractmethod
    def _initialize_strategy(self) -> None:
        """
        Perform strategy-specific initialization.
        
        This method should be implemented by subclasses to perform any
        strategy-specific initialization.
        """
        pass
    
    @abstractmethod
    def check_entry(self, bar: pd.Series, bar_index: int) -> Dict:
        """
        Check for entry signals.
        
        This method is called by the engine for each bar when no position is open.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            
        Returns:
            Dictionary with entry signal information:
            {
                'entry': bool,  # Whether to enter a position
                'type': str,    # 'long' or 'short'
                'price': float, # Entry price
                'size': float,  # Position size
                'reason': str,  # Entry reason
                # ... strategy-specific fields
            }
        """
        pass
    
    @abstractmethod
    def check_exit(self, bar: pd.Series, bar_index: int, position: float, 
                  entry_price: float, entry_time: Any) -> Dict:
        """
        Check for exit signals.
        
        This method is called by the engine for each bar when a position is open.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size (positive for long, negative for short)
            entry_price: Entry price of the current position
            entry_time: Entry time of the current position
            
        Returns:
            Dictionary with exit signal information:
            {
                'exit': bool,    # Whether to exit the position
                'price': float,  # Exit price
                'reason': str,   # Exit reason
                # ... strategy-specific fields
            }
        """
        pass
    
    def on_entry(self, bar: pd.Series, bar_index: int, position: float, entry_price: float) -> None:
        """
        Handle entry events.
        
        This method is called by the engine after an entry is executed.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Position size (positive for long, negative for short)
            entry_price: Entry price
        """
        pass
    
    def on_exit(self, bar: pd.Series, bar_index: int, trade: Dict) -> None:
        """
        Handle exit events.
        
        This method is called by the engine after an exit is executed.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            trade: Dictionary with trade information
        """
        pass
    
    def get_metrics(self) -> Dict:
        """
        Get strategy-specific metrics.
        
        This method is called by the engine after the backtest is complete.
        
        Returns:
            Dictionary with strategy-specific metrics
        """
        return self.metrics
