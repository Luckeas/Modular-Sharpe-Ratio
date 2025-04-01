"""
Position Sizer - Base class for position sizing strategies.

This module provides the PositionSizer abstract base class which defines
the interface for all position sizing implementations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    
    This class defines the interface for all position sizing implementations.
    Position sizing methods might include fixed size, percent risk, ATR-based, etc.
    """
    
    def __init__(self, name: str = "PositionSizer", config: Optional[Dict] = None):
        """
        Initialize the position sizer.
        
        Args:
            name: Name of the position sizer
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.metrics = {}
        logger.info(f"PositionSizer {name} initialized")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the position sizer with data.
        
        Args:
            data: Full dataset for the backtest
        """
        self.data = data
        self._initialize_position_sizer()
        logger.info(f"PositionSizer {self.name} initialized with {len(data)} bars of data")
    
    @abstractmethod
    def _initialize_position_sizer(self) -> None:
        """
        Perform position sizer-specific initialization.
        
        This method should be implemented by subclasses to perform any
        position sizer-specific initialization.
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, 
                              bar: pd.Series, 
                              bar_index: int, 
                              signal_type: str, 
                              entry_price: float,
                              account_value: float,
                              risk_params: Optional[Dict] = None,
                              market_regime: Optional[str] = None,
                              **kwargs) -> float:
        """
        Calculate position size based on the given parameters.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            signal_type: Type of signal ('long' or 'short')
            entry_price: Entry price
            account_value: Current account value
            risk_params: Risk parameters
            market_regime: Current market regime
            **kwargs: Additional keyword arguments
            
        Returns:
            Position size (number of units/contracts/shares)
        """
        pass
    
    def adjust_position_size(self, 
                           base_size: float, 
                           market_regime: Optional[str] = None,
                           regime_score: Optional[float] = None,
                           **kwargs) -> float:
        """
        Adjust position size based on market regime or other factors.
        
        Args:
            base_size: Base position size
            market_regime: Current market regime
            regime_score: Score/confidence of the current market regime (0-100)
            **kwargs: Additional keyword arguments
            
        Returns:
            Adjusted position size
        """
        # Default implementation - no adjustment
        return base_size
    
    def get_metrics(self) -> Dict:
        """
        Get position sizer-specific metrics.
        
        Returns:
            Dictionary with position sizer-specific metrics
        """
        return self.metrics
