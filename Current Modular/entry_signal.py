"""
Entry Signal - Base class for trade entry signals.

This module provides the EntrySignal abstract base class which defines
the interface for all trade entry signal implementations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class EntrySignal(ABC):
    """
    Abstract base class for trade entry signals.
    
    This class defines the interface for all entry signal implementations.
    Entry signals might include indicator-based signals, pattern recognition, etc.
    """
    
    def __init__(self, name: str = "EntrySignal", config: Optional[Dict] = None):
        """
        Initialize the entry signal.
        
        Args:
            name: Name of the signal
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.metrics = {
            'signal_count': 0,
            'long_signals': 0,
            'short_signals': 0
        }
        logger.info(f"EntrySignal {name} initialized")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the signal with data.
        
        Args:
            data: Full dataset for the backtest
        """
        self.data = data
        self._initialize_entry_signal()
        logger.info(f"EntrySignal {self.name} initialized with {len(data)} bars of data")
    
    @abstractmethod
    def _initialize_entry_signal(self) -> None:
        """
        Perform signal-specific initialization.
        
        This method should be implemented by subclasses to perform any
        signal-specific initialization.
        """
        pass
    
    @abstractmethod
    def check_entry(self, bar: pd.Series, bar_index: int, **kwargs) -> Dict:
        """
        Check for entry signals.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with entry signal information:
            {
                'entry': bool,  # Whether an entry signal was detected
                'type': str,    # 'long' or 'short'
                'price': float, # Entry price
                'size': float,  # Suggested position size
                'reason': str,  # Entry reason (e.g., 'rsi_oversold', 'breakout')
                # ... signal-specific fields
            }
        """
        pass
    
    def on_bar_update(self, bar: pd.Series, bar_index: int, **kwargs) -> None:
        """
        Update signal state on each bar.
        
        This method is called for each bar even when no signal is generated.
        It can be used to update internal tracking variables.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            **kwargs: Additional keyword arguments
        """
        pass
    
    def filter_signal(self, signal: Dict, market_regime: Optional[str] = None, 
                     risk_context: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Apply filters to raw entry signals based on market regime and risk context.
        
        Args:
            signal: Raw entry signal
            market_regime: Current market regime
            risk_context: Current risk context
            **kwargs: Additional keyword arguments
            
        Returns:
            Filtered entry signal
        """
        # Skip filtering if no signal
        if not signal['entry']:
            return signal
        
        # Default implementation - just track signal counts
        self.metrics['signal_count'] += 1
        if signal['type'] == 'long':
            self.metrics['long_signals'] += 1
        elif signal['type'] == 'short':
            self.metrics['short_signals'] += 1
        
        return signal
    
    def get_metrics(self) -> Dict:
        """
        Get signal-specific metrics.
        
        Returns:
            Dictionary with signal-specific metrics
        """
        return self.metrics
