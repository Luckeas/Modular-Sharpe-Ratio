"""
Regime Detector - Base class for market regime detection.

This module provides the RegimeDetector abstract base class which defines
the interface for all market regime detection implementations.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class RegimeDetector(ABC):
    """
    Abstract base class for market regime detection.
    
    This class defines the interface for all market regime detection implementations.
    Market regimes might include trend-following, mean-reverting, neutral, etc.
    """
    
    def __init__(self, name: str = "RegimeDetector", config: Optional[Dict] = None):
        """
        Initialize the regime detector.
        
        Args:
            name: Name of the detector
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.regime_history = []
        self.last_update = None
        self.current_regime = None
        self.initialized = False
        logger.info(f"RegimeDetector {name} initialized")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the detector with data.
        
        Args:
            data: Full dataset for analysis
        """
        self.data = data
        self.initialized = True
        self._initialize_detector()
        logger.info(f"RegimeDetector {self.name} initialized with {len(data)} bars of data")
    
    @abstractmethod
    def _initialize_detector(self) -> None:
        """
        Perform detector-specific initialization.
        
        This method should be implemented by subclasses to perform any
        detector-specific initialization.
        """
        pass
    
    @abstractmethod
    def detect_regime(self, bar: pd.Series, bar_index: int) -> Dict:
        """
        Detect the current market regime.
        
        This method should analyze the current market conditions and
        determine the appropriate regime.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            
        Returns:
            Dictionary with regime information:
            {
                'regime': str,       # Market regime (e.g., 'trend_following', 'mean_reverting', 'neutral')
                'confidence': float, # Confidence in the regime detection (0-1)
                'metrics': Dict,     # Regime-specific metrics
                # ... detector-specific fields
            }
        """
        pass
    
    def update(self, bar: pd.Series, bar_index: int, force: bool = False) -> Dict:
        """
        Update the regime detection.
        
        This method is called by clients to get the current regime.
        It respects the update frequency and returns cached results when appropriate.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            force: Force an update regardless of the update frequency
            
        Returns:
            Dictionary with regime information
        """
        current_time = bar.name if hasattr(bar, 'name') else bar_index
        
        # Check if update is needed
        update_needed = force or self.current_regime is None
        
        if not update_needed and self.last_update is not None:
            # Check if enough time has passed since the last update
            if isinstance(self.last_update, datetime) and isinstance(current_time, datetime):
                update_frequency = self.config.get('update_frequency_days', 1)
                days_elapsed = (current_time - self.last_update).days
                update_needed = days_elapsed >= update_frequency
            elif isinstance(self.last_update, int) and isinstance(current_time, int):
                update_frequency = self.config.get('update_frequency_bars', 20)
                bars_elapsed = current_time - self.last_update
                update_needed = bars_elapsed >= update_frequency
        
        # Perform update if needed
        if update_needed:
            regime_info = self.detect_regime(bar, bar_index)
            self.current_regime = regime_info.get('regime', 'neutral')
            self.last_update = current_time
            
            # Record to history
            history_entry = {
                'time': current_time,
                'regime': self.current_regime,
                'confidence': regime_info.get('confidence', 0.0),
                'metrics': regime_info.get('metrics', {})
            }
            self.regime_history.append(history_entry)
            
            logger.info(f"Updated regime to {self.current_regime} with confidence {regime_info.get('confidence', 0.0):.2f}")
            
            return regime_info
        else:
            # Return cached results
            return {
                'regime': self.current_regime,
                'confidence': self.regime_history[-1]['confidence'] if self.regime_history else 0.0,
                'metrics': self.regime_history[-1]['metrics'] if self.regime_history else {},
                'cached': True
            }
    
    def get_regime_history(self) -> List[Dict]:
        """
        Get the history of regime changes.
        
        Returns:
            List of dictionaries with regime history
        """
        return self.regime_history
    
    def get_current_regime(self) -> str:
        """
        Get the current market regime.
        
        Returns:
            Current regime or 'neutral' if not initialized
        """
        return self.current_regime or 'neutral'
    
    def get_regime_parameters(self, regime: Optional[str] = None) -> Dict:
        """
        Get parameters specific to a market regime.
        
        Args:
            regime: Market regime or None to use current regime
            
        Returns:
            Dictionary with regime-specific parameters
        """
        if regime is None:
            regime = self.get_current_regime()
            
        regime_params = self.config.get('regime_parameters', {}).get(regime, {})
        
        # Include common parameters
        common_params = self.config.get('common_parameters', {})
        params = {**common_params, **regime_params}
        
        # Include the regime itself
        params['regime'] = regime
        
        return params
