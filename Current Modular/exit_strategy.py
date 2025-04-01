"""
Exit Strategy - Base class for trade exit strategies.

This module provides the ExitStrategy abstract base class which defines
the interface for all trade exit strategy implementations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class ExitStrategy(ABC):
    """
    Abstract base class for trade exit strategies.
    
    This class defines the interface for all exit strategy implementations.
    Exit strategies might include fixed stops, trailing stops, time-based exits, etc.
    """
    
    def __init__(self, name: str = "ExitStrategy", config: Optional[Dict] = None):
        """
        Initialize the exit strategy.
        
        Args:
            name: Name of the strategy
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.metrics = {}
        logger.info(f"ExitStrategy {name} initialized")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with data.
        
        Args:
            data: Full dataset for the backtest
        """
        self.data = data
        self._initialize_exit_strategy()
        logger.info(f"ExitStrategy {self.name} initialized with {len(data)} bars of data")
    
    @abstractmethod
    def _initialize_exit_strategy(self) -> None:
        """
        Perform strategy-specific initialization.
        
        This method should be implemented by subclasses to perform any
        strategy-specific initialization.
        """
        pass
    
    @abstractmethod
    def check_exit(self, bar: pd.Series, bar_index: int, position: float,
                  entry_price: float, entry_bar: int, **kwargs) -> Dict:
        """
        Check for exit signals.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size (positive for long, negative for short)
            entry_price: Entry price of the current position
            entry_bar: Bar index when the position was entered
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with exit signal information:
            {
                'exit': bool,    # Whether to exit the position
                'price': float,  # Exit price
                'reason': str,   # Exit reason (e.g., 'stop_loss', 'take_profit', 'time_exit')
                # ... strategy-specific fields
            }
        """
        pass
    
    def on_bar_update(self, bar: pd.Series, bar_index: int, position: float,
                    entry_price: float, entry_bar: int, **kwargs) -> None:
        """
        Update strategy state on each bar.
        
        This method is called for each bar even when no exit occurs.
        It can be used to update internal tracking variables.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            position: Current position size
            entry_price: Entry price
            entry_bar: Bar index when the position was entered
            **kwargs: Additional keyword arguments
        """
        pass
    
    def on_exit(self, bar: pd.Series, bar_index: int, position: float,
               entry_price: float, entry_bar: int, exit_price: float,
               exit_reason: str, **kwargs) -> None:
        """
        Handle exit events.
        
        This method is called after an exit is executed.
        
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
        pass
    
    def get_metrics(self) -> Dict:
        """
        Get strategy-specific metrics.
        
        Returns:
            Dictionary with strategy-specific metrics
        """
        return self.metrics
