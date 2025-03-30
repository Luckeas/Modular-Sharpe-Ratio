"""
trade_analysis.py - Trade analysis and performance reporting

This module provides functions for analyzing trading performance
and generating reports.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_performance(trades, portfolio_series, initial_capital):
    """
    Analyze backtest performance and generate metrics.

    Args:
        trades: List of executed trades
        portfolio_series: Series of portfolio values indexed by date
        initial_capital: Initial account capital

    Returns:
        Dictionary of performance metrics
    """
    # Basic metrics
    results = {
        'initial_capital': initial_capital,
        'final_value': portfolio_series.iloc[-1] if len(portfolio_series) > 0 else initial_capital,
        'number_of_trades': len(trades)
    }

    # Calculate profit/loss
    results['profit_loss'] = results['final_value'] - initial_capital
    results['total_return_pct'] = (results['profit_loss'] / initial_capital) * 100

    # Win rate and average profit
    if trades:
        results['win_rate'] = sum(1 for t in trades if t['profit'] > 0) / len(trades) * 100
        results['avg_profit'] = np.mean([t['profit'] for t in trades])
        results['total_profit'] = sum(t['profit'] for t in trades if t['profit'] > 0)
        results['total_loss'] = sum(t['profit'] for t in trades if t['profit'] <= 0)

        # Profit factor
        if results['total_loss'] != 0:
            results['profit_factor'] = abs(results['total_profit'] / results['total_loss'])
        else:
            results['profit_factor'] = float('inf')

        # Enhanced time analysis metrics
        # Average bars held for all trades
        results['avg_bars_held'] = np.mean([t['bars_held'] for t in trades])

        # Separate winning and losing trades
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]

        if winning_trades:
            results['winning_bars_held'] = np.mean([t['bars_held'] for t in winning_trades])
            results['winning_trades_count'] = len(winning_trades)
        else:
            results['winning_bars_held'] = 0
            results['winning_trades_count'] = 0

        if losing_trades:
            results['losing_bars_held'] = np.mean([t['bars_held'] for t in losing_trades])
            results['losing_trades_count'] = len(losing_trades)
        else:
            results['losing_bars_held'] = 0
            results['losing_trades_count'] = 0

        # Add overnight analysis
        overnight_trades = []
        intraday_trades = []

        for trade in trades:
            entry_day = trade['entry_time'].date()
            exit_day = trade['exit_time'].date()

            if entry_day == exit_day:
                intraday_trades.append(trade)
            else:
                overnight_trades.append(trade)

        results['overnight_trades_count'] = len(overnight_trades)
        results['overnight_trades_pct'] = len(overnight_trades) / len(trades) * 100 if len(trades) > 0 else 0
        results['intraday_trades_count'] = len(intraday_trades)
        results['intraday_trades_pct'] = len(intraday_trades) / len(trades) * 100 if len(trades) > 0 else 0

        # Calculate performance by trade duration type
        if overnight_trades:
            results['overnight_win_rate'] = sum(1 for t in overnight_trades if t['profit'] > 0) / len(
                overnight_trades) * 100
            results['overnight_avg_profit'] = np.mean([t['profit'] for t in overnight_trades])
            results['overnight_avg_bars'] = np.mean([t['bars_held'] for t in overnight_trades])
        else:
            results['overnight_win_rate'] = 0
            results['overnight_avg_profit'] = 0
            results['overnight_avg_bars'] = 0

        if intraday_trades:
            results['intraday_win_rate'] = sum(1 for t in intraday_trades if t['profit'] > 0) / len(
                intraday_trades) * 100
            results['intraday_avg_profit'] = np.mean([t['profit'] for t in intraday_trades])
            results['intraday_avg_bars'] = np.mean([t['bars_held'] for t in intraday_trades])
        else:
            results['intraday_win_rate'] = 0
            results['intraday_avg_profit'] = 0
            results['intraday_avg_bars'] = 0
    else:
        results['win_rate'] = 0
        results['avg_profit'] = 0
        results['profit_factor'] = 0
        results['total_profit'] = 0
        results['total_loss'] = 0

        # Default values for time analysis
        results['avg_bars_held'] = 0
        results['winning_bars_held'] = 0
        results['losing_bars_held'] = 0
        results['overnight_trades_count'] = 0
        results['overnight_trades_pct'] = 0
        results['intraday_trades_count'] = 0
        results['intraday_trades_pct'] = 0
        results['overnight_win_rate'] = 0
        results['overnight_avg_profit'] = 0
        results['overnight_avg_bars'] = 0
        results['intraday_win_rate'] = 0
        results['intraday_avg_profit'] = 0
        results['intraday_avg_bars'] = 0

    # Calculate Sharpe ratio and drawdowns if enough data
    if len(portfolio_series) > 1:
        daily_returns = portfolio_series.pct_change().dropna()
        results['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(
            252) if daily_returns.std() != 0 else 0

        # Calculate drawdown
        drawdown = ((portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax()) * 100
        results['max_drawdown_pct'] = drawdown.max()

        # Peak and low values
        results['peak_value'] = portfolio_series.max()
        results['lowest_value'] = portfolio_series.min()
    else:
        results['sharpe_ratio'] = 0
        results['max_drawdown_pct'] = 0
        results['peak_value'] = initial_capital
        results['lowest_value'] = initial_capital

    return results

def analyze_by_regime(trades, regime_score_bins):
    """
    Analyze trade performance by regime score ranges.
    
    Args:
        trades: List of executed trades
        regime_score_bins: Dictionary of regime score distributions
        
    Returns:
        Dictionary of performance metrics by regime
    """
    if not trades:
        return {}
    
    # Group trades by regime score ranges
    score_ranges = {
        '40-60': [], '61-80': [], '81-100': []
    }
    
    for trade in trades:
        score = trade['regime_score']
        if score >= 81:
            score_ranges['81-100'].append(trade)
        elif score >= 61:
            score_ranges['61-80'].append(trade)
        else:  # score >= 40 (minimum to take a trade)
            score_ranges['40-60'].append(trade)
    
    # Calculate metrics for each range
    results = {}
    for score_range, group_trades in score_ranges.items():
        if group_trades:
            results[score_range] = {
                'count': len(group_trades),
                'win_rate': sum(1 for t in group_trades if t['profit'] > 0) / len(group_trades) * 100,
                'avg_profit': np.mean([t['profit'] for t in group_trades]),
                'total_profit': sum(t['profit'] for t in group_trades)
            }
    
    return results

def analyze_trades_by_market_type(trades):
    """
    Analyze trade performance by market type.
    
    Args:
        trades: List of executed trades
        
    Returns:
        Dictionary of performance metrics by market type
    """
    if not trades or 'market_type' not in trades[0]:
        return {}
    
    # Group trades by market type
    market_type_trades = {}
    for trade in trades:
        market_type = trade['market_type']
        if market_type not in market_type_trades:
            market_type_trades[market_type] = []
        market_type_trades[market_type].append(trade)
    
    # Calculate metrics for each market type
    results = {}
    for market_type, mt_trades in market_type_trades.items():
        if mt_trades:
            results[market_type] = {
                'count': len(mt_trades),
                'win_rate': sum(1 for t in mt_trades if t['profit'] > 0) / len(mt_trades) * 100,
                'avg_profit': np.mean([t['profit'] for t in mt_trades]),
                'total_profit': sum(t['profit'] for t in mt_trades)
            }
    
    return results

def analyze_by_season(trades):
    """
    Analyze trade performance by season.
    
    Args:
        trades: List of executed trades
        
    Returns:
        Dictionary of performance metrics by season
    """
    if not trades or 'season' not in trades[0]:
        return {}
    
    # Group trades by season
    season_trades = {}
    for trade in trades:
        season = trade.get('season', 'Unknown')
        if season not in season_trades:
            season_trades[season] = []
        season_trades[season].append(trade)
    
    # Calculate metrics for each season
    results = {}
    for season, st_trades in season_trades.items():
        if st_trades:
            total_profit = sum(t['profit'] for t in st_trades)
            win_trades = sum(1 for t in st_trades if t['profit'] > 0)
            loss_trades = sum(1 for t in st_trades if t['profit'] <= 0)
            
            # Calculate profit factor
            profit_sum = sum(t['profit'] for t in st_trades if t['profit'] > 0)
            loss_sum = abs(sum(t['profit'] for t in st_trades if t['profit'] <= 0))
            profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
            
            results[season] = {
                'count': len(st_trades),
                'win_count': win_trades,
                'loss_count': loss_trades,
                'win_rate': win_trades / len(st_trades) * 100 if len(st_trades) > 0 else 0,
                'avg_profit': total_profit / len(st_trades) if len(st_trades) > 0 else 0,
                'total_profit': total_profit,
                'profit_factor': profit_factor,
                'best_trade': max(t['profit'] for t in st_trades) if st_trades else 0,
                'worst_trade': min(t['profit'] for t in st_trades) if st_trades else 0
            }
    
    return results

def analyze_exit_reasons(trades):
    """
    Analyze trade performance by exit reason.
    
    Args:
        trades: List of executed trades
        
    Returns:
        Tuple of (exit_reasons_count, profit_by_exit_reason)
    """
    if not trades:
        return {}, {}
    
    # Count exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = 0
        exit_reasons[reason] += 1
    
    # Calculate profit by exit reason
    profit_by_exit = {}
    for reason in exit_reasons.keys():
        reason_trades = [t for t in trades if t['exit_reason'] == reason]
        profit_by_exit[reason] = {
            'avg_profit': np.mean([t['profit'] for t in reason_trades]),
            'win_rate': sum(1 for t in reason_trades if t['profit'] > 0) / len(reason_trades) * 100,
            'count': len(reason_trades)
        }
    
    return exit_reasons, profit_by_exit

def analyze_exit_strategies(trades):
    """
    Analyze performance by exit strategy configuration.
    
    Args:
        trades: List of executed trades
        
    Returns:
        Dictionary of performance metrics by exit strategy
    """
    if not trades:
        return {}
    
    # Group trades by strategy configuration
    trailing_enabled = [t for t in trades if t.get('used_trailing_stop', False)]
    trailing_disabled = [t for t in trades if not t.get('used_trailing_stop', False)]
    
    dynamic_enabled = [t for t in trades if t.get('used_dynamic_target', False)]
    dynamic_disabled = [t for t in trades if not t.get('used_dynamic_target', False)]
    
    # Different exit reasons
    exit_types = {}
    for t in trades:
        reason = t['exit_reason']
        if reason not in exit_types:
            exit_types[reason] = []
        exit_types[reason].append(t)
    
    # Calculate metrics for each group
    results = {}
    
    # Add trailing stop metrics
    if trailing_enabled:
        results['trailing_stop_enabled'] = {
            'count': len(trailing_enabled),
            'win_rate': sum(1 for t in trailing_enabled if t['profit'] > 0) / len(trailing_enabled) * 100,
            'avg_profit': sum(t['profit'] for t in trailing_enabled) / len(trailing_enabled),
            'total_profit': sum(t['profit'] for t in trailing_enabled)
        }
    
    if trailing_disabled:
        results['trailing_stop_disabled'] = {
            'count': len(trailing_disabled),
            'win_rate': sum(1 for t in trailing_disabled if t['profit'] > 0) / len(trailing_disabled) * 100,
            'avg_profit': sum(t['profit'] for t in trailing_disabled) / len(trailing_disabled),
            'total_profit': sum(t['profit'] for t in trailing_disabled)
        }
    
    # Add dynamic target metrics
    if dynamic_enabled:
        results['dynamic_target_enabled'] = {
            'count': len(dynamic_enabled),
            'win_rate': sum(1 for t in dynamic_enabled if t['profit'] > 0) / len(dynamic_enabled) * 100,
            'avg_profit': sum(t['profit'] for t in dynamic_enabled) / len(dynamic_enabled),
            'total_profit': sum(t['profit'] for t in dynamic_enabled)
        }
    
    if dynamic_disabled:
        results['dynamic_target_disabled'] = {
            'count': len(dynamic_disabled),
            'win_rate': sum(1 for t in dynamic_disabled if t['profit'] > 0) / len(dynamic_disabled) * 100,
            'avg_profit': sum(t['profit'] for t in dynamic_disabled) / len(dynamic_disabled),
            'total_profit': sum(t['profit'] for t in dynamic_disabled)
        }
    
    # Add exit reason metrics
    for reason, reason_trades in exit_types.items():
        results[f'exit_{reason}'] = {
            'count': len(reason_trades),
            'win_rate': sum(1 for t in reason_trades if t['profit'] > 0) / len(reason_trades) * 100,
            'avg_profit': sum(t['profit'] for t in reason_trades) / len(reason_trades),
            'total_profit': sum(t['profit'] for t in reason_trades)
        }
    
    # Calculate price capture efficiency for long and short trades
    if 'highest_price_reached' in trades[0] and 'lowest_price_reached' in trades[0]:
        long_trades = [t for t in trades if t['type'] == 'long']
        short_trades = [t for t in trades if t['type'] == 'short']
        
        # For long trades
        potential_captures = []
        for t in long_trades:
            if t.get('highest_price_reached') and t['highest_price_reached'] > t['entry_price']:
                potential_move = t['highest_price_reached'] - t['entry_price']
                actual_move = t['exit_price'] - t['entry_price']
                capture_pct = (actual_move / potential_move) * 100 if potential_move > 0 else 0
                potential_captures.append(capture_pct)
        
        # For short trades
        for t in short_trades:
            if t.get('lowest_price_reached') and t['lowest_price_reached'] < t['entry_price']:
                potential_move = t['entry_price'] - t['lowest_price_reached']
                actual_move = t['entry_price'] - t['exit_price']
                capture_pct = (actual_move / potential_move) * 100 if potential_move > 0 else 0
                potential_captures.append(capture_pct)
        
        if potential_captures:
            avg_capture = sum(potential_captures) / len(potential_captures)
            
            # Compare trailing stop vs fixed stop capture
            trailing_captures = []
            fixed_captures = []
            
            for t in trades:
                if 'highest_price_reached' not in t or 'lowest_price_reached' not in t:
                    continue
                    
                if t['type'] == 'long' and t['highest_price_reached'] > t['entry_price']:
                    potential_move = t['highest_price_reached'] - t['entry_price']
                    actual_move = t['exit_price'] - t['entry_price']
                    capture_pct = (actual_move / potential_move) * 100 if potential_move > 0 else 0
                    
                    if t.get('used_trailing_stop', False):
                        trailing_captures.append(capture_pct)
                    else:
                        fixed_captures.append(capture_pct)
                
                elif t['type'] == 'short' and t['lowest_price_reached'] < t['entry_price']:
                    potential_move = t['entry_price'] - t['lowest_price_reached']
                    actual_move = t['entry_price'] - t['exit_price']
                    capture_pct = (actual_move / potential_move) * 100 if potential_move > 0 else 0
                    
                    if t.get('used_trailing_stop', False):
                        trailing_captures.append(capture_pct)
                    else:
                        fixed_captures.append(capture_pct)
            
            avg_trailing = sum(trailing_captures) / len(trailing_captures) if trailing_captures else 0
            avg_fixed = sum(fixed_captures) / len(fixed_captures) if fixed_captures else 0
            
            results['price_capture_efficiency'] = {
                'avg_capture_pct': avg_capture,
                'trailing_capture_pct': avg_trailing,
                'fixed_capture_pct': avg_fixed,
                'trailing_advantage': avg_trailing - avg_fixed if trailing_captures and fixed_captures else 0
            }
    
    return results


def create_summary_report(results, trades_by_regime, trades_by_market,
                          exit_reasons, profit_by_exit, regime_score_bins,
                          season_metrics, exit_strategy_metrics, output_path):
    """
    Create a text summary report of backtest results.

    Args:
        results: Dictionary of performance metrics
        trades_by_regime: Analysis of trades by regime
        trades_by_market: Analysis of trades by market type
        exit_reasons: Dictionary of exit reason counts
        profit_by_exit: Analysis of profit by exit reason
        regime_score_bins: Dictionary of regime score distributions
        season_metrics: Analysis of trades by season
        exit_strategy_metrics: Analysis of exit strategy performance
        output_path: Path to save the report
    """
    try:
        with open(output_path, 'w') as summary_file:
            # Write header
            summary_file.write(f"======================================================\n")
            summary_file.write(f"             BACKTEST SUMMARY REPORT                  \n")
            summary_file.write(f"            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                    \n")
            summary_file.write(f"======================================================\n\n")

            # Performance results
            summary_file.write(f"PERFORMANCE RESULTS\n")
            summary_file.write(f"------------------------------------------------------\n")
            summary_file.write(f"Initial Capital: ${results['initial_capital']:.2f}\n")
            summary_file.write(f"Final Portfolio Value: ${results['final_value']:.2f}\n")
            summary_file.write(f"Net Profit/Loss: ${results['profit_loss']:.2f}\n")
            summary_file.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
            summary_file.write(f"Number of Trades: {results['number_of_trades']}\n")
            summary_file.write(f"Win Rate: {results['win_rate']:.2f}%\n")
            summary_file.write(f"Average Trade P/L: ${results['avg_profit']:.2f}\n")
            summary_file.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
            summary_file.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            summary_file.write(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%\n\n")

            # Add enhanced trade duration analysis section
            summary_file.write(f"TRADE DURATION ANALYSIS\n")
            summary_file.write(f"------------------------------------------------------\n")
            summary_file.write(f"Average Bars Held: {results.get('avg_bars_held', 0):.2f} (5-minute bars)\n")
            summary_file.write(f"Winning Trades Bars Held: {results.get('winning_bars_held', 0):.2f}\n")
            summary_file.write(f"Losing Trades Bars Held: {results.get('losing_bars_held', 0):.2f}\n\n")

            summary_file.write(
                f"Intraday Trades: {results.get('intraday_trades_count', 0)} ({results.get('intraday_trades_pct', 0):.2f}%)\n")
            summary_file.write(f"  Win Rate: {results.get('intraday_win_rate', 0):.2f}%\n")
            summary_file.write(f"  Avg Profit: ${results.get('intraday_avg_profit', 0):.2f}\n")
            summary_file.write(f"  Avg Bars Held: {results.get('intraday_avg_bars', 0):.2f}\n\n")

            summary_file.write(
                f"Overnight Trades: {results.get('overnight_trades_count', 0)} ({results.get('overnight_trades_pct', 0):.2f}%)\n")
            summary_file.write(f"  Win Rate: {results.get('overnight_win_rate', 0):.2f}%\n")
            summary_file.write(f"  Avg Profit: ${results.get('overnight_avg_profit', 0):.2f}\n")
            summary_file.write(f"  Avg Bars Held: {results.get('overnight_avg_bars', 0):.2f}\n\n")

            # Season analysis if available
            if season_metrics:
                summary_file.write(f"PERFORMANCE BY SEASON\n")
                summary_file.write(f"------------------------------------------------------\n")
                for season, metrics in season_metrics.items():
                    summary_file.write(f"{season}:\n")
                    summary_file.write(f"  Trades: {metrics['trade_count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n")
                    summary_file.write(f"  Average profit per trade: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Profit factor: {metrics['profit_factor']:.2f}\n\n")

            # Regime score distribution
            if regime_score_bins:
                summary_file.write(f"MARKET REGIME STATISTICS\n")
                summary_file.write(f"------------------------------------------------------\n")
                summary_file.write(f"Regime score distribution:\n")
                total_bars = sum(regime_score_bins.values())
                for bin_name, count in regime_score_bins.items():
                    percent = count / total_bars * 100 if total_bars > 0 else 0
                    summary_file.write(f"  {bin_name}: {count} bars ({percent:.1f}%)\n")
                summary_file.write("\n")

            # Performance by regime score
            if trades_by_regime:
                summary_file.write(f"PERFORMANCE BY REGIME SCORE\n")
                summary_file.write(f"------------------------------------------------------\n")
                for score_range, metrics in trades_by_regime.items():
                    summary_file.write(f"Score range {score_range}:\n")
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

            # Performance by market type
            if trades_by_market:
                summary_file.write(f"PERFORMANCE BY MARKET TYPE\n")
                summary_file.write(f"------------------------------------------------------\n")
                for market_type, metrics in trades_by_market.items():
                    summary_file.write(f"{market_type.upper()} Market:\n")
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

            # Exit reason analysis
            if exit_reasons:
                summary_file.write(f"EXIT REASON DISTRIBUTION\n")
                summary_file.write(f"------------------------------------------------------\n")
                total_trades = sum(exit_reasons.values())
                for reason, count in exit_reasons.items():
                    summary_file.write(f"{reason}: {count} trades ({count / total_trades * 100:.1f}%)\n")

                summary_file.write(f"\nAVERAGE PROFIT BY EXIT REASON\n")
                summary_file.write(f"------------------------------------------------------\n")
                for reason, metrics in profit_by_exit.items():
                    summary_file.write(f"{reason}:\n")
                    summary_file.write(f"  Count: {metrics['count']}\n")
                    summary_file.write(f"  Avg profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n\n")

            # Exit Strategy Analysis
            if exit_strategy_metrics:
                summary_file.write(f"EXIT STRATEGY PERFORMANCE\n")
                summary_file.write(f"------------------------------------------------------\n")

                # Compare trailing stop performance
                if 'trailing_stop_enabled' in exit_strategy_metrics:
                    summary_file.write(f"Trailing Stop Enabled:\n")
                    metrics = exit_strategy_metrics['trailing_stop_enabled']
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

                if 'trailing_stop_disabled' in exit_strategy_metrics:
                    summary_file.write(f"Standard Stop Loss:\n")
                    metrics = exit_strategy_metrics['trailing_stop_disabled']
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

                # Compare dynamic target performance
                if 'dynamic_target_enabled' in exit_strategy_metrics:
                    summary_file.write(f"Dynamic ATR-Based Targets:\n")
                    metrics = exit_strategy_metrics['dynamic_target_enabled']
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

                if 'dynamic_target_disabled' in exit_strategy_metrics:
                    summary_file.write(f"Fixed Bollinger Band Targets:\n")
                    metrics = exit_strategy_metrics['dynamic_target_disabled']
                    summary_file.write(f"  Trades: {metrics['count']}\n")
                    summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                    summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                    summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

                # Specific exit type analysis
                for key, metrics in exit_strategy_metrics.items():
                    if key.startswith('exit_'):
                        exit_type = key.replace('exit_', '')
                        summary_file.write(f"Exit Type - {exit_type.replace('_', ' ').title()}:\n")
                        summary_file.write(f"  Trades: {metrics['count']}\n")
                        summary_file.write(f"  Win rate: {metrics['win_rate']:.2f}%\n")
                        summary_file.write(f"  Average profit: ${metrics['avg_profit']:.2f}\n")
                        summary_file.write(f"  Total profit: ${metrics['total_profit']:.2f}\n\n")

                # Price capture efficiency
                if 'price_capture_efficiency' in exit_strategy_metrics:
                    summary_file.write(f"Price Capture Efficiency:\n")
                    metrics = exit_strategy_metrics['price_capture_efficiency']
                    summary_file.write(f"  Average potential movement captured: {metrics['avg_capture_pct']:.2f}%\n")
                    summary_file.write(
                        f"  Trailing stop vs. fixed stop advantage: {metrics['trailing_advantage']:.2f}%\n\n")

        logger.info(f"Summary report saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error creating summary file: {e}")

def analyze_quarterly_performance(trades, portfolio_series, initial_capital):
    """
    Analyze strategy performance on a quarterly basis.
    
    Args:
        trades: List of executed trades
        portfolio_series: Series of portfolio values indexed by date
        initial_capital: Initial account capital
        
    Returns:
        DataFrame with quarterly performance metrics
    """
    # Ensure we have trades to analyze
    if not trades or len(portfolio_series) == 0:
        return pd.DataFrame()
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Ensure datetime format for entry_time and exit_time
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Extract quarter information
    trades_df['entry_year'] = trades_df['entry_time'].dt.year
    trades_df['entry_quarter'] = trades_df['entry_time'].dt.quarter
    trades_df['year_quarter'] = trades_df['entry_year'].astype(str) + '-Q' + trades_df['entry_quarter'].astype(str)
    
    # Get unique year-quarters
    quarters = sorted(trades_df['year_quarter'].unique())
    
    # Calculate quarterly portfolio changes
    portfolio_df = portfolio_series.reset_index()
    portfolio_df.columns = ['date', 'value']
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df['year'] = portfolio_df['date'].dt.year
    portfolio_df['quarter'] = portfolio_df['date'].dt.quarter
    portfolio_df['year_quarter'] = portfolio_df['year'].astype(str) + '-Q' + portfolio_df['quarter'].astype(str)
    
    # Analyze each quarter
    quarterly_results = []
    
    for quarter in quarters:
        # Trades for this quarter
        quarter_trades = trades_df[trades_df['year_quarter'] == quarter]
        
        # Portfolio series for this quarter
        quarter_portfolio = portfolio_df[portfolio_df['year_quarter'] == quarter]
        
        if len(quarter_trades) == 0 or len(quarter_portfolio) == 0:
            continue
            
        # Starting and ending portfolio values
        start_value = quarter_portfolio['value'].iloc[0]
        end_value = quarter_portfolio['value'].iloc[-1]
        
        # Calculate metrics
        total_trades = len(quarter_trades)
        profitable_trades = sum(quarter_trades['profit'] > 0)
        losing_trades = sum(quarter_trades['profit'] <= 0)
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = quarter_trades['profit'].sum()
        avg_profit = quarter_trades['profit'].mean() if total_trades > 0 else 0
        avg_winning_trade = quarter_trades[quarter_trades['profit'] > 0]['profit'].mean() if profitable_trades > 0 else 0
        avg_losing_trade = quarter_trades[quarter_trades['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = quarter_trades[quarter_trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(quarter_trades[quarter_trades['profit'] <= 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate quarterly return
        quarter_return = (end_value - start_value) / start_value * 100
        
        # Calculate drawdown
        if len(quarter_portfolio) > 1:
            quarter_series = pd.Series(quarter_portfolio['value'].values, index=quarter_portfolio['date'])
            drawdown = ((quarter_series.cummax() - quarter_series) / quarter_series.cummax()) * 100
            max_drawdown = drawdown.max()
        else:
            max_drawdown = 0
        
        # Combine metrics for this quarter
        quarter_result = {
            'year_quarter': quarter,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'quarter_return': quarter_return,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'start_value': start_value,
            'end_value': end_value
        }
        
        quarterly_results.append(quarter_result)
    
    # Convert to DataFrame
    quarterly_df = pd.DataFrame(quarterly_results)
    
    return quarterly_df

def compare_strategies(orig_results, ml_results, orig_trades, ml_trades):

    """
    Compare the results of the original and ML-enhanced strategies.
    
    Args:
        orig_results: Original strategy performance results
        ml_results: ML-enhanced strategy performance results
        orig_trades: List of original strategy trades
        ml_trades: List of ML-enhanced strategy trades
        
    Returns:
        Dictionary of comparison results
    """
    comparison = {
        'original': {
            'total_trades': len(orig_trades),
            'win_rate': orig_results['win_rate'],
            'profit_loss': orig_results['profit_loss'],
            'total_return_pct': orig_results['total_return_pct'],
            'avg_profit': orig_results['avg_profit'],
            'profit_factor': orig_results['profit_factor'],
            'sharpe_ratio': orig_results['sharpe_ratio'],
            'max_drawdown_pct': orig_results['max_drawdown_pct']
        },
        'ml_enhanced': {
            'total_trades': len(ml_trades),
            'win_rate': ml_results['win_rate'],
            'profit_loss': ml_results['profit_loss'],
            'total_return_pct': ml_results['total_return_pct'],
            'avg_profit': ml_results['avg_profit'],
            'profit_factor': ml_results['profit_factor'],
            'sharpe_ratio': ml_results['sharpe_ratio'],
            'max_drawdown_pct': ml_results['max_drawdown_pct']
        },
        'difference': {
            'total_trades': len(ml_trades) - len(orig_trades),
            'win_rate': ml_results['win_rate'] - orig_results['win_rate'],
            'profit_loss': ml_results['profit_loss'] - orig_results['profit_loss'],
            'total_return_pct': ml_results['total_return_pct'] - orig_results['total_return_pct'],
            'avg_profit': ml_results['avg_profit'] - orig_results['avg_profit'],
            'profit_factor': ml_results['profit_factor'] - orig_results['profit_factor'],
            'sharpe_ratio': ml_results['sharpe_ratio'] - orig_results['sharpe_ratio'],
            'max_drawdown_pct': ml_results['max_drawdown_pct'] - orig_results['max_drawdown_pct']
        },
        'percent_change': {
            'total_trades': (len(ml_trades) - len(orig_trades)) / len(orig_trades) * 100 if len(orig_trades) > 0 else float('inf'),
            'win_rate': (ml_results['win_rate'] - orig_results['win_rate']) / orig_results['win_rate'] * 100 if orig_results['win_rate'] > 0 else float('inf'),
            'profit_loss': (ml_results['profit_loss'] - orig_results['profit_loss']) / abs(orig_results['profit_loss']) * 100 if orig_results['profit_loss'] != 0 else float('inf'),
            'avg_profit': (ml_results['avg_profit'] - orig_results['avg_profit']) / abs(orig_results['avg_profit']) * 100 if orig_results['avg_profit'] != 0 else float('inf')
        }
    }
    
    # Calculate trade overlap
    orig_entry_times = set(t['entry_time'] for t in orig_trades)
    ml_entry_times = set(t['entry_time'] for t in ml_trades)
    
    common_trades = orig_entry_times.intersection(ml_entry_times)
    
    comparison['trade_overlap'] = {
        'common_trades': len(common_trades),
        'orig_only': len(orig_entry_times - ml_entry_times),
        'ml_only': len(ml_entry_times - orig_entry_times),
        'overlap_pct': len(common_trades) / len(orig_entry_times) * 100 if len(orig_entry_times) > 0 else 0
    }
    
    return comparison