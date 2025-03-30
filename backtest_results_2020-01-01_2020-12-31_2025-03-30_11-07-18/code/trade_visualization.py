"""
trade_visualization.py - Trade visualization functions

This module provides simplified visualization functions for trades and performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import logging
from datetime import datetime, timedelta

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_trade(df, trade_data, save_dir='executed_trade_plots',
                    rsi_oversold=30, rsi_overbought=70):
    """
    Create enhanced visualization of an executed trade with entry, exit, and key indicators.

    Args:
        df: DataFrame with price and indicator data
        trade_data: Dictionary with trade details
        save_dir: Directory to save visualization
        rsi_oversold: RSI oversold threshold
        rsi_overbought: RSI overbought threshold

    Returns:
        Path to the saved visualization file
    """
    # Create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract trade details
    entry_time = trade_data['entry_time']
    exit_time = trade_data['exit_time']
    entry_price = trade_data['entry_price']
    exit_price = trade_data['exit_price']
    stop_loss = trade_data['stop_loss']
    profit_target = trade_data['profit_target']
    position_type = trade_data['type']
    profit = trade_data['profit']
    rsi = trade_data.get('rsi', 50)
    atr = trade_data.get('atr', 0)
    volume = trade_data.get('volume', 0)
    avg_volume = trade_data.get('avg_volume', 0)
    exit_reason = trade_data.get('exit_reason', 'unknown')
    regime_score = trade_data.get('regime_score', 0)
    favorable_regime = trade_data.get('favorable_regime', False)

    # NEW: Extract additional parameters for enhanced visualization
    market_type = trade_data.get('market_type', 'neutral')
    hmm_confidence = trade_data.get('hmm_confidence', None)
    bars_held = trade_data.get('bars_held', 0)
    used_trailing_stop = trade_data.get('used_trailing_stop', False)
    highest_price_reached = trade_data.get('highest_price_reached', None)
    lowest_price_reached = trade_data.get('lowest_price_reached', None)

    # Get appropriate data window for plot
    plot_data = get_plot_window(df, entry_time, exit_time)

    # Find indices for entry and exit bars
    entry_idx, exit_idx = find_entry_exit_indices(plot_data, entry_time, exit_time)

    # Setup figure with more panels
    fig_height = 14  # Increased height for additional panels
    fig = plt.figure(figsize=(12, fig_height))

    # Create panel layout with additional panels for market regime
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])

    # Create Price Panel with enhanced annotations
    ax1 = plt.subplot(gs[0])
    create_price_panel(ax1, plot_data, entry_idx, exit_idx, entry_price, exit_price,
                       stop_loss, profit_target, position_type, exit_reason,
                       highest_price_reached, lowest_price_reached, used_trailing_stop,
                       market_type)

    # Create RSI Panel with market type-specific thresholds
    ax_rsi = plt.subplot(gs[1], sharex=ax1)
    create_rsi_panel(ax_rsi, plot_data, entry_idx,
                     rsi_oversold if market_type != 'trend_following' else 40,
                     rsi_overbought if market_type != 'trend_following' else 60)

    # Create Volume Panel
    ax_vol = plt.subplot(gs[2], sharex=ax1)
    volume_condition_met = volume > avg_volume * 1.5 if market_type == 'mean_reverting' else volume > avg_volume * 1.2
    create_volume_panel(ax_vol, plot_data, entry_idx, volume, avg_volume, volume_condition_met, market_type)

    # NEW: Create Market Regime Panel
    ax_regime = plt.subplot(gs[3], sharex=ax1)
    create_regime_panel(ax_regime, plot_data, entry_idx, exit_idx, regime_score, market_type, hmm_confidence)

    # Format x-axis labels for time
    format_time_axis(plt, plot_data)

    # Add enhanced trade information box
    add_trade_info_box(ax1, trade_data, volume_condition_met)

    # Set title
    position_type_str = position_type.upper()
    result_str = "PROFIT" if profit > 0 else "LOSS"
    regime_str = "FAVORABLE REGIME" if favorable_regime else "UNFAVORABLE REGIME"
    market_type_str = market_type.upper()

    title = f"{position_type_str} Trade ({result_str}) - {market_type_str} - {regime_str} - Score: {regime_score:.1f}"
    ax1.set_title(title)

    plt.tight_layout()

    # Save the visualization
    file_name = f"{save_dir}/{position_type}_{market_type}_{entry_time.strftime('%Y-%m-%d_%H-%M')}_{result_str.lower()}_{exit_reason}.png"
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close()

    return file_name


def create_price_panel(ax, plot_data, entry_idx, exit_idx, entry_price, exit_price,
                       stop_loss, profit_target, position_type, exit_reason,
                       highest_price_reached=None, lowest_price_reached=None,
                       used_trailing_stop=False, market_type='neutral'):
    """Create enhanced price panel with additional trade condition markers"""
    # Draw candlesticks
    width = 0.8
    for i, row in plot_data.iterrows():
        # Determine candle color
        if row['close'] >= row['open']:
            color = 'green'
            bottom = row['open']
            height = max(row['close'] - row['open'], 0.01)
        else:
            color = 'red'
            bottom = row['close']
            height = max(row['open'] - row['close'], 0.01)

        # Draw body and wick
        ax.add_patch(plt.Rectangle((i, bottom), width, height, color=color))
        ax.plot([i + width / 2, i + width / 2], [row['low'], row['high']], color='black', linewidth=1.5)

    # Add grid
    ax.grid(True, axis='y', alpha=0.3)

    # Highlight entry and exit bars
    ax.axvspan(entry_idx - 0.5, entry_idx + 0.5, color='lightgreen', alpha=0.4, label='Entry Bar')
    ax.axvspan(exit_idx - 0.5, exit_idx + 0.5, color='lightcoral', alpha=0.3, label='Exit Bar')

    # Mark entry price
    ax.plot(entry_idx, entry_price, 'g^', markersize=10, label=f'Entry: {entry_price}')

    # Mark exit price with appropriate marker by reason
    if exit_reason == 'profit_target':
        ax.plot(exit_idx, exit_price, 'b*', markersize=10, label=f'Exit: {exit_price} (Target)')
    elif exit_reason == 'stop_loss':
        ax.plot(exit_idx, exit_price, 'rx', markersize=10, label=f'Exit: {exit_price} (Stop)')
    elif exit_reason == 'trailing_stop':
        ax.plot(exit_idx, exit_price, 'm+', markersize=10, label=f'Exit: {exit_price} (Trailing)')
    else:
        ax.plot(exit_idx, exit_price, 'ko', markersize=10, label=f'Exit: {exit_price} (Time)')

    # Draw stop loss and profit target lines
    ax.axhline(y=stop_loss, color='red', linestyle='--', linewidth=1.5, label=f'Stop Loss: {stop_loss}')
    ax.axhline(y=profit_target, color='blue', linestyle='--', linewidth=1.5, label=f'Target: {profit_target}')

    # Plot Bollinger Bands if available
    if all(col in plot_data.columns for col in ['middle_band', 'upper_band', 'lower_band']):
        ax.plot(plot_data.index, plot_data['middle_band'], color='purple', linewidth=1.5, label='BB Middle')
        ax.plot(plot_data.index, plot_data['upper_band'], color='purple', linestyle='--', linewidth=1.5,
                label='BB Upper')
        ax.plot(plot_data.index, plot_data['lower_band'], color='purple', linestyle='--', linewidth=1.5,
                label='BB Lower')

    # NEW: Plot MA if available
    if 'MA' in plot_data.columns:
        ax.plot(plot_data.index, plot_data['MA'], color='blue', linewidth=1.5, label='MA')

    # NEW: Show highest/lowest price reached with dotted lines (if available)
    if position_type == 'long' and highest_price_reached is not None:
        ax.axhline(y=highest_price_reached, color='green', linestyle=':', linewidth=1.0,
                   label=f'Highest: {highest_price_reached}')
    elif position_type == 'short' and lowest_price_reached is not None:
        ax.axhline(y=lowest_price_reached, color='red', linestyle=':', linewidth=1.0,
                   label=f'Lowest: {lowest_price_reached}')

    # NEW: Draw trailing stop activation thresholds based on market type
    if position_type == 'long':
        threshold_multiplier = 1.0075 if market_type == 'trend_following' else (
            1.01 if market_type == 'mean_reverting' else 1.009)
        trailing_threshold = entry_price * threshold_multiplier
        ax.axhline(y=trailing_threshold, color='magenta', linestyle='-.', linewidth=1.0, alpha=0.5,
                   label=f'Trail Activation: {trailing_threshold:.2f}')
    else:  # Short position
        threshold_multiplier = 0.9925 if market_type == 'trend_following' else (
            0.99 if market_type == 'mean_reverting' else 0.991)
        trailing_threshold = entry_price * threshold_multiplier
        ax.axhline(y=trailing_threshold, color='magenta', linestyle='-.', linewidth=1.0, alpha=0.5,
                   label=f'Trail Activation: {trailing_threshold:.2f}')

    ax.set_ylabel('Price')
    ax.legend(loc='upper right', fontsize=8)


def create_rsi_panel(ax, plot_data, entry_idx, rsi_oversold, rsi_overbought):
    """Create the RSI indicator panel with market type-specific thresholds"""
    if 'RSI' not in plot_data.columns:
        ax.text(0.5, 0.5, 'RSI data not available', horizontalalignment='center')
        return

    ax.plot(plot_data.index, plot_data['RSI'], color='blue', linewidth=1.5, label='RSI')
    ax.axhline(y=rsi_oversold, color='green', linestyle='--', linewidth=1, label=f'RSI {rsi_oversold}')
    ax.axhline(y=rsi_overbought, color='red', linestyle='--', linewidth=1, label=f'RSI {rsi_overbought}')
    ax.axhline(y=50, color='gray', linestyle='-', linewidth=0.8, label='RSI 50')

    # Mark entry point
    ax.axvline(x=entry_idx, color='green', linestyle='-', alpha=0.5)

    # NEW: Mark RSI at entry
    if entry_idx < len(plot_data):
        entry_rsi = plot_data.iloc[entry_idx]['RSI']
        ax.plot(entry_idx, entry_rsi, 'go', markersize=8)
        ax.annotate(f'RSI: {entry_rsi:.1f}',
                    xy=(entry_idx, entry_rsi),
                    xytext=(entry_idx + 0.5, entry_rsi + 5),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    fontsize=8)

    ax.set_ylim(0, 100)
    ax.set_ylabel('RSI')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def create_volume_panel(ax, plot_data, entry_idx, entry_volume, avg_volume, volume_condition_met, market_type):
    """Create enhanced volume panel with market type-specific thresholds"""
    # Draw volume bars with colors matching candles
    volume_colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in plot_data.iterrows()]
    ax.bar(range(len(plot_data)), plot_data['volume'], color=volume_colors, width=0.8)

    # Draw market type-specific volume threshold line
    threshold_mult = 1.5 if market_type == 'mean_reverting' else 1.2
    if 'avg_volume' in plot_data.columns:
        # Use dynamic average volume if available
        volume_threshold = plot_data['avg_volume'] * threshold_mult
        ax.plot(plot_data.index, volume_threshold, color='blue', linestyle='--', linewidth=1.5,
                label=f'{threshold_mult}x Avg Volume ({market_type})')
    else:
        # Use fixed threshold based on entry bar average
        ax.axhline(y=avg_volume * threshold_mult, color='blue', linestyle='--', linewidth=1.5,
                   label=f'{threshold_mult}x Avg Volume ({market_type})')

    # Highlight entry bar volume
    if 0 <= entry_idx < len(plot_data):
        entry_vol = plot_data.iloc[entry_idx]['volume'] if entry_idx < len(plot_data) else entry_volume

        # Enhanced annotation with text box
        offset = max(plot_data['volume']) * 0.05  # 5% of max for positioning
        condition_status = "✓ VOLUME CONDITION MET" if volume_condition_met else "✗ VOLUME CONDITION NOT MET"
        text_color = 'green' if volume_condition_met else 'red'
        bbox_props = dict(boxstyle="round,pad=0.5", fc='yellow', ec='black', alpha=0.7)

        ax.annotate(condition_status,
                    xy=(entry_idx, entry_vol),
                    xytext=(entry_idx, entry_vol + offset),
                    fontsize=9, color=text_color, weight='bold',
                    ha='center', bbox=bbox_props)

    ax.set_ylabel('Volume')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def create_regime_panel(ax, plot_data, entry_idx, exit_idx, regime_score, market_type, hmm_confidence=None):
    """NEW: Create a panel showing market regime information"""
    # Create regime score plot if available
    if 'regime_score' in plot_data.columns:
        ax.plot(plot_data.index, plot_data['regime_score'], color='purple', linewidth=1.5, label='Regime Score')

        # Add market type-specific threshold lines
        if market_type == 'trend_following':
            threshold = 70
            color = 'green'
        elif market_type == 'mean_reverting':
            threshold = 40
            color = 'blue'
        else:  # neutral
            threshold = 50
            color = 'gray'

        ax.axhline(y=threshold, color=color, linestyle='--', linewidth=1.5,
                   label=f'{market_type.capitalize()} Threshold ({threshold})')

    # Mark entry and exit areas
    ax.axvline(x=entry_idx, color='green', linestyle='-', alpha=0.5, label='Entry')
    ax.axvline(x=exit_idx, color='red', linestyle='-', alpha=0.5, label='Exit')

    # Add market type annotation
    if entry_idx < len(plot_data) and 'regime_score' in plot_data.columns:
        entry_score = plot_data.iloc[entry_idx]['regime_score'] if 'regime_score' in plot_data.columns else regime_score
        text_y = 80  # Position in upper part of plot

        # Confidence string
        conf_str = f", Conf: {hmm_confidence:.2f}" if hmm_confidence is not None else ""

        ax.text(entry_idx, text_y, f"{market_type.upper()}{conf_str}",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
                ha='center', fontsize=10, weight='bold')

    # Set appropriate y-limit
    ax.set_ylim(0, 100)
    ax.set_ylabel('Regime')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def add_trade_info_box(ax, trade_data, volume_condition_met):
    """Add enhanced text box with trade information including market type specifics"""
    # Basic trade info
    position_type = trade_data['type']
    profit = trade_data['profit']
    rsi = trade_data.get('rsi', 0)
    atr = trade_data.get('atr', 0)
    volume = trade_data.get('volume', 0)
    avg_volume = trade_data.get('avg_volume', 0)
    exit_reason = trade_data.get('exit_reason', 'unknown')
    regime_score = trade_data.get('regime_score', 0)
    market_type = trade_data.get('market_type', 'neutral')
    bars_held = trade_data.get('bars_held', 0)
    ma_slope = trade_data.get('ma_slope', 0)

    # NEW: Entry criteria info
    if position_type == 'long':
        if market_type == 'mean_reverting':
            entry_criteria = "LONG MEAN REVERSION"
            entry_checks = [
                f"Below Lower BB: {'✓' if trade_data.get('below_lower_bb', True) else '✗'}",
                f"RSI < 35: {'✓' if rsi < 35 else '✗'}",
                f"Volume > 1.5x Avg: {'✓' if volume_condition_met else '✗'}"
            ]
        elif market_type == 'trend_following':
            entry_criteria = "LONG TREND FOLLOWING"
            entry_checks = [
                f"Above MA: {'✓' if trade_data.get('above_ma', True) else '✗'}",
                f"50 < RSI < 70: {'✓' if 50 < rsi < 70 else '✗'}",
                f"MA Slope > 0.1: {'✓' if ma_slope > 0.1 else '✗'}",
                f"Volume > 1.2x Avg: {'✓' if volume_condition_met else '✗'}"
            ]
        else:  # neutral
            entry_criteria = "LONG NEUTRAL"
            entry_checks = [
                "Mixed Criteria (See Trade Log)"
            ]
    else:  # short
        if market_type == 'mean_reverting':
            entry_criteria = "SHORT MEAN REVERSION"
            entry_checks = [
                f"Above Upper BB: {'✓' if trade_data.get('above_upper_bb', True) else '✗'}",
                f"RSI > 65: {'✓' if rsi > 65 else '✗'}",
                f"Volume > 1.5x Avg: {'✓' if volume_condition_met else '✗'}"
            ]
        elif market_type == 'trend_following':
            entry_criteria = "SHORT TREND FOLLOWING"
            entry_checks = [
                f"Below MA: {'✓' if trade_data.get('below_ma', True) else '✗'}",
                f"30 < RSI < 50: {'✓' if 30 < rsi < 50 else '✗'}",
                f"MA Slope < -0.1: {'✓' if ma_slope < -0.1 else '✗'}",
                f"Volume > 1.2x Avg: {'✓' if volume_condition_met else '✗'}"
            ]
        else:  # neutral
            entry_criteria = "SHORT NEUTRAL"
            entry_checks = [
                "Mixed Criteria (See Trade Log)"
            ]

    # Calculate volume ratio
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0

    # Build text string
    textstr = '\n'.join((
        f"Market Type: {market_type.upper()}",
        f"Regime Score: {regime_score:.1f}",
        f"Entry: {entry_criteria}",
        '\n'.join(entry_checks),
        f"",
        f"P&L: ${profit:.2f}",
        f"Type: {position_type.upper()}",
        f"RSI: {rsi:.2f}",
        f"MA Slope: {ma_slope:.3f}",
        f"ATR: {atr:.2f}",
        f"Vol Ratio: {volume_ratio:.2f}x",
        f"Bars Held: {bars_held}",
        f"Exit: {exit_reason.replace('_', ' ').title()}"
    ))

    # Set text box properties
    box_color = 'lightgreen' if profit > 0 else 'lightsalmon'
    props = dict(boxstyle='round', facecolor=box_color, alpha=0.5)

    # Place text box
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

def get_plot_window(df, entry_time, exit_time):
    """Get plot window based on trading day boundaries"""
    # Find entry and exit times
    entry_mask = df['date'] == entry_time
    exit_mask = df['date'] == exit_time
    
    if not entry_mask.any():
        entry_idx = (df['date'] - entry_time).abs().idxmin()
        entry_mask = df.index == entry_idx
    
    if not exit_mask.any():
        exit_idx = (df['date'] - exit_time).abs().idxmin()
        exit_mask = df.index == exit_idx
    
    # Calculate time window
    entry_date = entry_time.date()
    exit_date = exit_time.date()
    
    # Handle same-day vs multi-day trades
    if entry_date == exit_date:
        # Same day - show entire trading day
        day_mask = df['date'].dt.date == entry_date
        plot_data = df[day_mask].copy().reset_index(drop=True)
    else:
        # Multi-day - show entry and exit days if <= 3 days
        days_between = (exit_date - entry_date).days
        
        if days_between <= 3:
            # Show all days in between
            mask = (df['date'] >= entry_time) & (df['date'] <= exit_time)
            plot_data = df[mask].copy().reset_index(drop=True)
        else:
            # Just show entry and exit days
            entry_day = df[df['date'].dt.date == entry_date]
            exit_day = df[df['date'].dt.date == exit_date]
            plot_data = pd.concat([entry_day, exit_day]).reset_index(drop=True)
    
    # If window too small, extend it
    if len(plot_data) < 20:
        # Look back/forward more bars
        entry_idx = df[df['date'] == entry_time].index[0] if entry_mask.any() else -1
        exit_idx = df[df['date'] == exit_time].index[0] if exit_mask.any() else -1
        
        if entry_idx >= 0 and exit_idx >= 0:
            start_idx = max(0, entry_idx - 10)
            end_idx = min(len(df) - 1, exit_idx + 10)
            plot_data = df.iloc[start_idx:end_idx+1].copy().reset_index(drop=True)
    
    return plot_data

def find_entry_exit_indices(plot_data, entry_time, exit_time):
    """Find the indices of entry and exit bars in the plot data window"""
    entry_mask = plot_data['date'] == entry_time
    exit_mask = plot_data['date'] == exit_time
    
    if not entry_mask.any():
        # Find closest time
        entry_idx = (plot_data['date'] - entry_time).abs().idxmin()
    else:
        entry_idx = plot_data[entry_mask].index[0]
    
    if not exit_mask.any():
        # Find closest time
        exit_idx = (plot_data['date'] - exit_time).abs().idxmin()
    else:
        exit_idx = plot_data[exit_mask].index[0]
    
    return entry_idx, exit_idx

def generate_season_performance_charts(trades, season_metrics, output_dir, date_range_str):
    """
    Generate charts comparing performance across seasons.
    
    Args:
        trades: List of all executed trades
        season_metrics: Dictionary with performance metrics by season
        output_dir: Directory to save the charts
        date_range_str: String with date range for file naming
    """
    if not season_metrics:
        logger.warning("No season metrics available for visualization")
        return
    
    # 1. Win rate by season
    plt.figure(figsize=(12, 6))
    seasons = list(season_metrics.keys())
    win_rates = [season_metrics[s]['win_rate'] for s in seasons]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#95a5a6'] # Blue, Green, Red, Yellow, Gray
    color_map = {season: color for season, color in zip(seasons, colors[:len(seasons)])}
    
    plt.bar(seasons, win_rates, color=[color_map[s] for s in seasons])
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.3)
    plt.title('Win Rate by Season')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, max(100, max(win_rates) * 1.1))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"season_win_rates_{date_range_str}.png"), dpi=150)
    plt.close()
    
    # 2. Average profit by season
    plt.figure(figsize=(12, 6))
    avg_profits = [season_metrics[s]['avg_profit'] for s in seasons]
    
    bars = plt.bar(seasons, avg_profits, color=[color_map[s] for s in seasons])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Average Profit by Season')
    plt.ylabel('Average Profit ($)')
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height,
                f'${height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"season_avg_profits_{date_range_str}.png"), dpi=150)
    plt.close()
    
    # 3. Total profit by season
    plt.figure(figsize=(12, 6))
    total_profits = [season_metrics[s]['total_profit'] for s in seasons]
    
    bars = plt.bar(seasons, total_profits, color=[color_map[s] for s in seasons])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Total Profit by Season')
    plt.ylabel('Total Profit ($)')
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height,
                f'${height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"season_total_profits_{date_range_str}.png"), dpi=150)
    plt.close()
    
    # 4. Trade count by season
    plt.figure(figsize=(12, 6))
    trade_counts = [season_metrics[s]['trade_count'] for s in seasons]
    
    bars = plt.bar(seasons, trade_counts, color=[color_map[s] for s in seasons])
    plt.title('Trade Count by Season')
    plt.ylabel('Number of Trades')
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"season_trade_counts_{date_range_str}.png"), dpi=150)
    plt.close()
    
    # 5. Profit distribution by season
    plt.figure(figsize=(14, 8))
    
    # Group trades by season
    trades_by_season = {}
    for trade in trades:
        season = trade.get('season', 'Unknown')
        if season not in trades_by_season:
            trades_by_season[season] = []
        trades_by_season[season].append(trade)
    
    # Plot profit distributions
    for i, season in enumerate(seasons):
        if season in trades_by_season and trades_by_season[season]:
            season_profits = [t['profit'] for t in trades_by_season[season]]
            
            # Use kernel density estimate
            plt.subplot(len(seasons), 1, i+1)
            plt.hist(season_profits, bins=20, alpha=0.6, color=color_map[season])
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.title(f'{season} Profit Distribution')
            plt.grid(alpha=0.3)
            
            if i == len(seasons) - 1:
                plt.xlabel('Profit ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"season_profit_distribution_{date_range_str}.png"), dpi=150)
    plt.close()

def format_time_axis(plt, plot_data):
    """Format the x-axis with appropriate time labels"""
    # Enhanced X-axis labels with date changes
    x_labels = []
    prev_date = None
    
    for i, row in plot_data.iterrows():
        current_date = row['date'].date()
        current_time = row['date'].time()
        
        # Add date labels at date changes and key hours
        if prev_date is None or current_date != prev_date:
            x_labels.append(row['date'].strftime('%m-%d\n%H:%M'))
            prev_date = current_date
        elif current_time.hour in [9, 12, 16] and current_time.minute == 0:
            # Mark key hours
            x_labels.append(row['date'].strftime('%H:%M'))
        elif i % 6 == 0:  # Every 6th label (approx. 30 minutes)
            x_labels.append(row['date'].strftime('%H:%M'))
        else:
            x_labels.append('')
    
    plt.xticks(range(len(plot_data)), x_labels, rotation=45, fontsize=8)

def generate_performance_charts(portfolio_series, trades, output_dir, date_range_str):
    """
    Generate performance charts and save to output directory.
    
    Args:
        portfolio_series: Series of portfolio values indexed by date
        trades: List of executed trades
        output_dir: Directory to save the charts
        date_range_str: String with date range for file naming
    """
    # 1. Portfolio value chart
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series.index, portfolio_series, label='Portfolio Value')
    plt.title(f'Portfolio Value - {date_range_str}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"portfolio_value_{date_range_str}.png"))
    plt.close()
    
    # 2. Equity curve with drawdowns
    if len(portfolio_series) > 1:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot()
        ax.plot(portfolio_series.index, portfolio_series, label='Portfolio Value')
        
        # Add drawdown shading
        drawdown = (portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax() * 100
        ax.fill_between(portfolio_series.index, 0, drawdown, alpha=0.3, color='red', label='Drawdown %')
        
        # Add drawdown percentage on right y-axis
        ax2 = ax.twinx()
        ax2.plot(portfolio_series.index, drawdown, 'r--', alpha=0.5)
        ax2.set_ylabel('Drawdown %')
        ax2.set_ylim(0, max(drawdown) * 1.5 if len(drawdown) > 0 and max(drawdown) > 0 else 10)
        
        plt.title(f'Portfolio Value with Drawdowns - {date_range_str}')
        plt.xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        plt.savefig(os.path.join(output_dir, f"equity_curve_with_drawdowns_{date_range_str}.png"))
        plt.close()
    
    # 3. Monthly returns
    if len(portfolio_series) > 20:
        # Resample to monthly returns
        monthly_returns = portfolio_series.resample('M').last().pct_change() * 100
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            plt.figure(figsize=(12, 6))
            colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
            plt.bar(monthly_returns.index.strftime('%Y-%m'), monthly_returns, color=colors)
            plt.title(f'Monthly Returns - {date_range_str}')
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.grid(True, axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"monthly_returns_{date_range_str}.png"))
            plt.close()

def generate_quarterly_analysis_charts(quarterly_df, output_dir, date_range_str):
    """
    Generate visualizations of quarterly performance and save to quarterly_analysis directory.

    Args:
        quarterly_df: DataFrame with quarterly performance metrics
        output_dir: Main output directory
        date_range_str: String with date range for file naming
    """
    if quarterly_df.empty:
        logger.warning("No quarterly data available for visualization")
        return

    # Create quarterly_analysis subdirectory path
    quarterly_dir = os.path.join(output_dir, 'quarterly_analysis')
    if not os.path.exists(quarterly_dir):
        os.makedirs(quarterly_dir)

    # Save quarterly data to CSV
    csv_path = os.path.join(quarterly_dir, f"quarterly_performance_{date_range_str}.csv")
    quarterly_df.to_csv(csv_path, index=False)
    logger.info(f"Quarterly analysis saved to {csv_path}")

    # Generate quarterly performance charts
    plt.figure(figsize=(14, 8))

    # 1. Quarterly Returns
    plt.subplot(2, 2, 1)
    bars = plt.bar(quarterly_df['year_quarter'], quarterly_df['quarter_return'])
    plt.title('Quarterly Returns')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height - 1.5,
                     f'{height:.1f}%', ha='center', va='top')

    plt.grid(axis='y', alpha=0.3)

    # 2. Win Rate by Quarter
    plt.subplot(2, 2, 2)
    bars = plt.bar(quarterly_df['year_quarter'], quarterly_df['win_rate'])
    plt.title('Win Rate by Quarter')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)

    # 3. Trade Count by Quarter
    plt.subplot(2, 2, 3)
    bars = plt.bar(quarterly_df['year_quarter'], quarterly_df['total_trades'])
    plt.title('Trade Count by Quarter')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.grid(axis='y', alpha=0.3)

    # 4. Average Profit per Trade
    plt.subplot(2, 2, 4)
    bars = plt.bar(quarterly_df['year_quarter'], quarterly_df['avg_profit'])
    plt.title('Average Profit per Trade')
    plt.ylabel('Average Profit ($)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'${height:.2f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height - 1.5,
                     f'${height:.2f}', ha='center', va='top')

    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(quarterly_dir, f"quarterly_performance_{date_range_str}.png"), dpi=150)
    plt.close()

    # Create quarterly metrics summary file
    summary_path = os.path.join(quarterly_dir, f"quarterly_summary_{date_range_str}.txt")
    with open(summary_path, 'w') as f:
        f.write("QUARTERLY PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"{'Quarter':<10} {'Return':<10} {'Win Rate':<10} {'Trades':<10} {'Avg Profit':<15} {'Profit Factor':<15}\n")
        f.write("-" * 60 + "\n")

        for _, row in quarterly_df.iterrows():
            f.write(f"{row['year_quarter']:<10} ")
            f.write(f"{row['quarter_return']:.2f}%{' ':<5} ")
            f.write(f"{row['win_rate']:.2f}%{' ':<5} ")
            f.write(f"{int(row['total_trades']):<10} ")
            f.write(f"${row['avg_profit']:.2f}{' ':<10} ")

            if row['profit_factor'] == float('inf'):
                f.write(f"∞{' ':<14}\n")
            else:
                f.write(f"{row['profit_factor']:.2f}{' ':<10}\n")

        f.write("\n\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    logger.info(f"Quarterly summary saved to {summary_path}")

def generate_comparison_plots(orig_portfolio, ml_portfolio, orig_trades, ml_trades, plots_dir):
    """
    Generate plots comparing the original and ML-enhanced strategies.
    
    Args:
        orig_portfolio: Original strategy portfolio value series
        ml_portfolio: ML-enhanced strategy portfolio value series
        orig_trades: List of original strategy trades
        ml_trades: List of ML-enhanced strategy trades
        plots_dir: Directory to save plots
    """
    # 1. Equity curves comparison
    plt.figure(figsize=(12, 6))
    plt.plot(orig_portfolio.index, orig_portfolio.values, label='Original Strategy')
    plt.plot(ml_portfolio.index, ml_portfolio.values, label='ML-Enhanced Strategy')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'equity_curve_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Drawdown comparison
    plt.figure(figsize=(12, 6))
    
    # Calculate drawdowns
    orig_dd = (orig_portfolio.cummax() - orig_portfolio) / orig_portfolio.cummax() * 100
    ml_dd = (ml_portfolio.cummax() - ml_portfolio) / ml_portfolio.cummax() * 100
    
    plt.plot(orig_dd.index, orig_dd.values, label='Original Strategy', color='red', alpha=0.7)
    plt.plot(ml_dd.index, ml_dd.values, label='ML-Enhanced Strategy', color='blue', alpha=0.7)
    plt.title('Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'drawdown_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Trade P&L distribution
    plt.figure(figsize=(10, 6))
    
    orig_profits = [t['profit'] for t in orig_trades]
    ml_profits = [t['profit'] for t in ml_trades]
    
    plt.hist(orig_profits, bins=30, alpha=0.5, label='Original Strategy')
    plt.hist(ml_profits, bins=30, alpha=0.5, label='ML-Enhanced Strategy')
    plt.title('Trade P&L Distribution')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'pnl_distribution.png'), dpi=150)
    plt.close()
    
    # 4. Monthly win rate comparison (if enough data)
    if len(orig_trades) > 20 and len(ml_trades) > 20:
        # Create monthly win rate dataframes
        orig_df = pd.DataFrame([{'date': t['entry_time'], 'profit': t['profit']} for t in orig_trades])
        ml_df = pd.DataFrame([{'date': t['entry_time'], 'profit': t['profit']} for t in ml_trades])
        
        # Extract month
        orig_df['month'] = orig_df['date'].dt.strftime('%Y-%m')
        ml_df['month'] = ml_df['date'].dt.strftime('%Y-%m')
        
        # Calculate win rate by month
        orig_monthly = orig_df.groupby('month').apply(lambda x: (x['profit'] > 0).mean() * 100)
        ml_monthly = ml_df.groupby('month').apply(lambda x: (x['profit'] > 0).mean() * 100)
        
        # Merge series
        monthly_win_rates = pd.DataFrame({
            'Original': orig_monthly,
            'ML-Enhanced': ml_monthly
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        monthly_win_rates.plot(kind='bar', ax=plt.gca())
        plt.title('Monthly Win Rate Comparison')
        plt.xlabel('Month')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'monthly_win_rate.png'), dpi=150)
        plt.close()