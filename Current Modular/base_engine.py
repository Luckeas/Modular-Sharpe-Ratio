"""
Base Backtesting Engine - The core component of the backtesting framework.

This module provides the BaseEngine class which serves as the foundation for all
backtesting operations. It handles the main event loop, position tracking, and
results collection.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class BaseEngine:
    """
    The core backtesting engine that drives the simulation.
    
    This class handles the main backtesting loop, processes market data,
    manages positions, and collects results. It is designed to be strategy-agnostic
    and can be used with any compatible strategy implementation.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.0,
                 data_handler=None,
                 portfolio_manager=None,
                 event_manager=None,
                 risk_manager=None,
                 reporter=None):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital for the backtest
            transaction_cost: Cost per transaction (e.g., commission)
            data_handler: Component that provides market data
            portfolio_manager: Component that tracks positions and portfolio value
            event_manager: Component that manages events during backtest
            risk_manager: Component that handles risk management
            reporter: Component that generates performance reports
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Components
        self.data_handler = data_handler
        self.portfolio_manager = portfolio_manager
        self.event_manager = event_manager
        self.risk_manager = risk_manager
        self.reporter = reporter
        
        # Tracking variables
        self.current_capital = initial_capital
        self.portfolio_values = []
        self.trades = []
        self.current_position = 0
        self.current_position_value = 0
        self.current_position_entry_price = 0
        self.current_position_entry_time = None
        
        # Strategy and state
        self.strategy = None
        self.is_running = False
        self.current_bar = 0
        self.warmup_period = 0
        
        logger.info(f"BaseEngine initialized with {initial_capital:.2f} initial capital")
    
    def set_strategy(self, strategy):
        """
        Set the trading strategy to use for backtesting.
        
        Args:
            strategy: A strategy object that implements the required interface
        """
        self.strategy = strategy
        logger.info(f"Strategy set to {strategy.__class__.__name__}")
    
    def run(self, data: pd.DataFrame, warmup_bars: int = 0) -> Dict:
        """
        Run the backtest on the provided data.
        
        Args:
            data: DataFrame with price data and indicators
            warmup_bars: Number of bars to skip at the beginning (for indicator warmup)
            
        Returns:
            Dictionary with backtest results
        """
        if self.strategy is None:
            raise ValueError("No strategy has been set. Call set_strategy() before running.")
        
        self.is_running = True
        self.warmup_period = warmup_bars
        
        # Initialize tracking variables
        self.current_capital = self.initial_capital
        self.portfolio_values = []
        self.trades = []
        self.current_position = 0
        self.current_bar = 0
        
        # Prepare for backtest
        self.strategy.initialize(data, self)
        
        if self.data_handler:
            self.data_handler.initialize(data)
        
        if self.portfolio_manager:
            self.portfolio_manager.initialize(self.initial_capital)
        
        # Main backtest loop
        for i in range(len(data)):
            self.current_bar = i
            bar = data.iloc[i]
            
            # Skip bars during warmup period
            if i < warmup_bars:
                self.portfolio_values.append(self.current_capital)
                continue
            
            # Process bar
            self._process_bar(bar, i, data)
            
            # Track portfolio value
            self.portfolio_values.append(self._calculate_portfolio_value(bar))
        
        # Finalize backtest
        self.is_running = False
        
        # Generate results
        results = self._generate_results(data)
        
        # Log summary
        self._log_summary(results)
        
        return results
    
    def _process_bar(self, bar, bar_index, data):
        """
        Process a single bar of data.
        
        Args:
            bar: Current bar data (pandas Series)
            bar_index: Index of the current bar
            data: Full dataset
        """
        # Check for exit signals if we have an open position
        if self.current_position != 0:
            exit_signal = self.strategy.check_exit(bar, bar_index, self.current_position,
                                                 self.current_position_entry_price,
                                                 self.current_position_entry_time)
            
            if exit_signal['exit']:
                self._execute_exit(bar, bar_index, exit_signal)
        
        # Check for entry signals if we don't have an open position
        if self.current_position == 0:
            entry_signal = self.strategy.check_entry(bar, bar_index)
            
            if entry_signal['entry']:
                self._execute_entry(bar, bar_index, entry_signal)
    
    def _execute_entry(self, bar, bar_index, entry_signal):
        """
        Execute an entry order based on the entry signal.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            entry_signal: Dictionary with entry signal details
        """
        # Get entry details
        entry_price = entry_signal.get('price', bar['open'])
        entry_type = entry_signal.get('type', 'long')
        position_size = entry_signal.get('size', 1)
        
        # Adjust position direction based on type
        position = position_size if entry_type == 'long' else -position_size
        
        # Apply transaction costs
        entry_cost = abs(position) * self.transaction_cost
        self.current_capital -= entry_cost
        
        # Update position tracking
        self.current_position = position
        self.current_position_entry_price = entry_price
        self.current_position_entry_time = bar.name if hasattr(bar, 'name') else bar_index
        
        # Log the entry
        logger.info(f"Entered {entry_type} position of {abs(position)} units at {entry_price:.2f}")
        
        # Notify the strategy
        self.strategy.on_entry(bar, bar_index, position, entry_price)
    
    def _execute_exit(self, bar, bar_index, exit_signal):
        """
        Execute an exit order based on the exit signal.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            exit_signal: Dictionary with exit signal details
        """
        # Get exit details
        exit_price = exit_signal.get('price', bar['close'])
        exit_reason = exit_signal.get('reason', 'signal')
        
        # Calculate profit/loss
        profit = (exit_price - self.current_position_entry_price) * self.current_position
        
        # Apply transaction costs
        exit_cost = abs(self.current_position) * self.transaction_cost
        self.current_capital -= exit_cost
        
        # Update capital
        self.current_capital += profit
        
        # Record trade
        trade = {
            'entry_time': self.current_position_entry_time,
            'exit_time': bar.name if hasattr(bar, 'name') else bar_index,
            'entry_price': self.current_position_entry_price,
            'exit_price': exit_price,
            'position': self.current_position,
            'profit': profit,
            'type': 'long' if self.current_position > 0 else 'short',
            'exit_reason': exit_reason,
            'bars_held': bar_index - (self.current_position_entry_time if isinstance(self.current_position_entry_time, int) else self.current_position_entry_time.value),
            'transaction_costs': exit_cost + abs(self.current_position) * self.transaction_cost
        }
        
        self.trades.append(trade)
        
        # Log the exit
        logger.info(f"Exited {trade['type']} position at {exit_price:.2f}, profit: {profit:.2f}, reason: {exit_reason}")
        
        # Reset position tracking
        self.current_position = 0
        self.current_position_entry_price = 0
        self.current_position_entry_time = None
        
        # Notify the strategy
        self.strategy.on_exit(bar, bar_index, trade)
    
    def _calculate_portfolio_value(self, bar):
        """
        Calculate the current portfolio value.
        
        Args:
            bar: Current bar data
            
        Returns:
            Current portfolio value
        """
        # If no position, portfolio value is just the cash
        if self.current_position == 0:
            return self.current_capital
        
        # Otherwise, add unrealized P&L
        close_price = bar['close']
        unrealized_pnl = (close_price - self.current_position_entry_price) * self.current_position
        return self.current_capital + unrealized_pnl
    
    def _generate_results(self, data):
        """
        Generate comprehensive backtest results.
        
        Args:
            data: The full dataset used for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        # Basic backtest results
        results = {
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_values[-1],
            'total_return': (self.portfolio_values[-1] / self.initial_capital - 1) * 100,
            'number_of_trades': len(self.trades)
        }
        
        # Calculate additional metrics
        if len(self.trades) > 0:
            # Win rate
            profitable_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
            results['win_rate'] = profitable_trades / len(self.trades) * 100
            
            # Average profit per trade
            results['avg_profit'] = sum(trade['profit'] for trade in self.trades) / len(self.trades)
            
            # Maximum drawdown
            portfolio_series = pd.Series(self.portfolio_values)
            drawdown = (portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax() * 100
            results['max_drawdown'] = drawdown.max()
            
            # Sharpe ratio (simple implementation)
            returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                results['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                results['sharpe_ratio'] = 0
        
        # Include strategy-specific metrics if available
        if hasattr(self.strategy, 'get_metrics'):
            strategy_metrics = self.strategy.get_metrics()
            if strategy_metrics:
                results['strategy_metrics'] = strategy_metrics
        
        return results
    
    def _log_summary(self, results):
        """
        Log a summary of the backtest results.
        
        Args:
            results: Dictionary with backtest results
        """
        logger.info("=" * 50)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Initial Capital: {results['initial_capital']:.2f}")
        logger.info(f"Final Capital: {results['final_capital']:.2f}")
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Number of Trades: {results['number_of_trades']}")
        
        if 'win_rate' in results:
            logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        if 'avg_profit' in results:
            logger.info(f"Average Profit per Trade: {results['avg_profit']:.2f}")
        if 'max_drawdown' in results:
            logger.info(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        if 'sharpe_ratio' in results:
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
