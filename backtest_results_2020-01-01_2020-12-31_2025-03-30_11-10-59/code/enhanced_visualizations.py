"""
enhanced_visualizations.py - Improved visualizations for regime-based backtesting

This module provides enhanced visualizations that show market regimes alongside
price data and performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_regime_backtest(df, trades, portfolio_values, regime_log, market_type_log, 
                             output_dir, date_range_str=None):
    """
    Generate enhanced visualizations for regime-based backtesting.
    
    Args:
        df: DataFrame with price and indicator data
        trades: List of executed trades
        portfolio_values: List of portfolio values
        regime_log: List of regime information
        market_type_log: List of market type information
        output_dir: Directory to save visualizations
        date_range_str: String with date range for file naming
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create date range string if not provided
    if date_range_str is None:
        date_range_str = f"{df['date'].min().strftime('%Y%m%d')}_{df['date'].max().strftime('%Y%m%d')}"
    
    # Convert log data to DataFrames
    regime_df = pd.DataFrame(regime_log)
    market_type_df = pd.DataFrame(market_type_log)
    portfolio_df = pd.DataFrame({'date': df['date'][:len(portfolio_values)], 'value': portfolio_values})
    
    # Convert trades to DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame()
    
    # 1. Create main visualization: Price chart with regimes, trades, and portfolio value
    create_master_visualization(df, trades_df, regime_df, market_type_df, portfolio_df, output_dir, date_range_str)
    
    # 2. Create regime transition analysis
    create_regime_transition_analysis(regime_df, market_type_df, output_dir, date_range_str)
    
    # 3. Create trade performance by regime
    if not trades_df.empty:
        create_trade_by_regime_analysis(trades_df, output_dir, date_range_str)
    
    # 4. Create regime prediction accuracy (if ML was used)
    if 'predicted_regime' in market_type_df.columns:
        create_prediction_accuracy_analysis(market_type_df, output_dir, date_range_str)
    
    logger.info(f"Enhanced visualizations saved to {output_dir}")

def create_master_visualization(df, trades_df, regime_df, market_type_df, portfolio_df, output_dir, date_range_str):
    """
    Create a comprehensive master visualization showing price, regimes, and trades.
    
    Args:
        df: DataFrame with price and indicator data
        trades_df: DataFrame with trade information
        regime_df: DataFrame with regime information
        market_type_df: DataFrame with market type information
        portfolio_df: DataFrame with portfolio values
        output_dir: Directory to save visualization
        date_range_str: String with date range for file naming
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    # 1. Price chart with regimes
    ax1 = fig.add_subplot(gs[0])
    
    # Plot price
    ax1.plot(df['date'], df['close'], color='black', linewidth=1.5, label='Price')
    
    # Add moving average
    if 'MA' in df.columns:
        ax1.plot(df['date'], df['MA'], color='blue', linewidth=1, alpha=0.7, label='Moving Average')
    
    # Add Bollinger Bands
    if all(col in df.columns for col in ['upper_band', 'lower_band', 'middle_band']):
        ax1.plot(df['date'], df['upper_band'], 'g--', linewidth=0.8, alpha=0.5, label='Upper BB')
        ax1.plot(df['date'], df['middle_band'], 'g-', linewidth=0.8, alpha=0.5, label='Middle BB')
        ax1.plot(df['date'], df['lower_band'], 'g--', linewidth=0.8, alpha=0.5, label='Lower BB')
    
    # Highlight market regimes
    if not market_type_df.empty:
        # Add rectangle for each regime period
        for i in range(len(market_type_df)):
            # Get start and end dates for this regime
            start_date = market_type_df['date'].iloc[i]
            if i < len(market_type_df) - 1:
                end_date = market_type_df['date'].iloc[i+1]
            else:
                end_date = df['date'].max()
            
            # Get regime and determine color
            regime = market_type_df['market_type'].iloc[i]
            if regime == 'trend_following':
                color = 'lightgreen'
                alpha = 0.3
                label = 'Trend Following' if i == 0 else None
            elif regime == 'mean_reverting':
                color = 'lightblue'
                alpha = 0.3
                label = 'Mean Reverting' if i == 0 else None
            elif regime == 'neutral':
                color = 'lightyellow'
                alpha = 0.3
                label = 'Neutral' if i == 0 else None
            elif regime == 'bullish_low_variance':
                color = 'green'
                alpha = 0.2
                label = 'Bullish Low Var' if i == 0 else None
            elif regime == 'bullish_high_variance':
                color = 'yellow'
                alpha = 0.2
                label = 'Bullish High Var' if i == 0 else None
            elif regime == 'bearish_high_variance':
                color = 'red'
                alpha = 0.2
                label = 'Bearish High Var' if i == 0 else None
            elif regime == 'bearish_low_variance':
                color = 'orange'
                alpha = 0.2
                label = 'Bearish Low Var' if i == 0 else None
            else:
                color = 'lightgray'
                alpha = 0.2
                label = regime if i == 0 else None
            
            # Add the rectangle
            ax1.axvspan(start_date, end_date, color=color, alpha=alpha, label=label)
    
    # Add trades to chart
    if not trades_df.empty:
        # Add entry points
        for _, trade in trades_df.iterrows():
            if trade['type'] == 'long':
                marker = '^'  # up triangle for long
                color = 'green'
            else:
                marker = 'v'  # down triangle for short
                color = 'red'
            
            # Plot entry
            ax1.scatter(trade['entry_time'], trade['entry_price'], 
                      marker=marker, s=100, color=color, zorder=5)
            
            # Plot exit
            ax1.scatter(trade['exit_time'], trade['exit_price'], 
                      marker='o', s=80, color='blue', zorder=5)
            
            # Add connecting line
            ax1.plot([trade['entry_time'], trade['exit_time']], 
                    [trade['entry_price'], trade['exit_price']], 
                    color='blue', alpha=0.5, linestyle='--', linewidth=1)
    
    ax1.set_title('Price with Market Regimes and Trades')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 2. Regime score plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    if 'regime_score' in regime_df.columns:
        ax2.plot(regime_df['date'], regime_df['regime_score'], color='purple', linewidth=1.5)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Regime Score')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
    
    # 3. RSI plot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    if 'RSI' in df.columns:
        ax3.plot(df['date'], df['RSI'], color='blue', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
    
    # 4. Portfolio value
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    if not portfolio_df.empty:
        ax4.plot(portfolio_df['date'], portfolio_df['value'], color='green', linewidth=1.5)
        ax4.set_ylabel('Portfolio Value')
        ax4.grid(True, alpha=0.3)
    
    # Set common x-axis properties
    ax4.set_xlabel('Date')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'master_visualization_{date_range_str}.png'), dpi=150)
    plt.close()

def create_regime_transition_analysis(regime_df, market_type_df, output_dir, date_range_str):
    """
    Create visualizations analyzing regime transitions.
    
    Args:
        regime_df: DataFrame with regime information
        market_type_df: DataFrame with market type information
        output_dir: Directory to save visualizations
        date_range_str: String with date range for file naming
    """
    if market_type_df.empty:
        return
    
    # Calculate regime durations
    durations = []
    
    for i in range(len(market_type_df) - 1):
        start_date = market_type_df['date'].iloc[i]
        end_date = market_type_df['date'].iloc[i+1]
        duration = (end_date - start_date).days
        
        durations.append({
            'regime': market_type_df['market_type'].iloc[i],
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration
        })
    
    # Handle the last regime
    if len(market_type_df) > 0:
        start_date = market_type_df['date'].iloc[-1]
        end_date = regime_df['date'].max() if not regime_df.empty else start_date
        duration = (end_date - start_date).days
        
        durations.append({
            'regime': market_type_df['market_type'].iloc[-1],
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration
        })
    
    # Convert to DataFrame
    durations_df = pd.DataFrame(durations)
    
    if durations_df.empty:
        return
    
    # 1. Create regime duration barplot
    plt.figure(figsize=(12, 6))
    
    # Group by regime and calculate average duration
    regime_durations = durations_df.groupby('regime')['duration_days'].mean().reset_index()
    
    # Plot
    colors = {
        'trend_following': 'lightgreen',
        'mean_reverting': 'lightblue',
        'neutral': 'lightyellow',
        'bullish_low_variance': 'green',
        'bullish_high_variance': 'yellow',
        'bearish_high_variance': 'red',
        'bearish_low_variance': 'orange'
    }
    
    bar_colors = [colors.get(regime, 'gray') for regime in regime_durations['regime']]
    
    plt.bar(regime_durations['regime'], regime_durations['duration_days'], color=bar_colors)
    plt.title('Average Duration by Market Regime')
    plt.xlabel('Regime')
    plt.ylabel('Average Duration (Days)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(regime_durations['duration_days']):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'regime_durations_{date_range_str}.png'), dpi=150)
    plt.close()
    
    # 2. Create regime transition heatmap
    if len(durations_df) > 1:
        # Create transition pairs
        transitions = []
        
        for i in range(len(durations_df) - 1):
            from_regime = durations_df['regime'].iloc[i]
            to_regime = durations_df['regime'].iloc[i+1]
            
            transitions.append({
                'from_regime': from_regime,
                'to_regime': to_regime
            })
        
        # Create transition count matrix
        if transitions:
            # Convert to DataFrame
            transitions_df = pd.DataFrame(transitions)
            
            # Get unique regimes
            all_regimes = sorted(set(durations_df['regime']))
            
            # Create transition matrix
            transition_matrix = pd.DataFrame(0, index=all_regimes, columns=all_regimes)
            
            # Fill matrix with counts
            for _, row in transitions_df.iterrows():
                transition_matrix.loc[row['from_regime'], row['to_regime']] += 1
            
            # Normalize to get probabilities (by row)
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix.div(row_sums, axis=0).fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_probs, annot=True, fmt='.2f', cmap='Blues')
            plt.title('Regime Transition Probabilities')
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'regime_transitions_{date_range_str}.png'), dpi=150)
            plt.close()

def create_trade_by_regime_analysis(trades_df, output_dir, date_range_str):
    """
    Create visualizations analyzing trade performance by regime.
    
    Args:
        trades_df: DataFrame with trade information
        output_dir: Directory to save visualizations
        date_range_str: String with date range for file naming
    """
    if trades_df.empty or 'market_type' not in trades_df.columns:
        return
    
    # Calculate performance metrics by regime
    regime_stats = trades_df.groupby('market_type').agg({
        'profit': ['count', 'mean', 'sum'],
        'type': 'count'
    }).reset_index()
    
    # Calculate win rate by regime
    win_rates = []
    for regime in trades_df['market_type'].unique():
        regime_trades = trades_df[trades_df['market_type'] == regime]
        
        if len(regime_trades) > 0:
            win_rate = sum(regime_trades['profit'] > 0) / len(regime_trades) * 100
            win_rates.append({
                'market_type': regime,
                'win_rate': win_rate,
                'n_trades': len(regime_trades),
                'n_winning': sum(regime_trades['profit'] > 0),
                'n_losing': sum(regime_trades['profit'] <= 0),
                'avg_profit': regime_trades['profit'].mean(),
                'total_profit': regime_trades['profit'].sum()
            })
    
    # Convert to DataFrame
    win_rates_df = pd.DataFrame(win_rates)
    
    if win_rates_df.empty:
        return
    
    # Create win rate barplot
    plt.figure(figsize=(12, 6))
    
    # Define colors based on regime
    colors = {
        'trend_following': 'lightgreen',
        'mean_reverting': 'lightblue',
        'neutral': 'lightyellow',
        'bullish_low_variance': 'green',
        'bullish_high_variance': 'yellow',
        'bearish_high_variance': 'red',
        'bearish_low_variance': 'orange'
    }
    
    # Create bar chart of win rates
    bar_colors = [colors.get(regime, 'gray') for regime in win_rates_df['market_type']]
    
    plt.bar(win_rates_df['market_type'], win_rates_df['win_rate'], color=bar_colors)
    plt.title('Win Rate by Market Regime')
    plt.xlabel('Regime')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(win_rates_df['win_rate']):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'win_rate_by_regime_{date_range_str}.png'), dpi=150)
    plt.close()
    
    # Create average profit barplot
    plt.figure(figsize=(12, 6))
    
    plt.bar(win_rates_df['market_type'], win_rates_df['avg_profit'], color=bar_colors)
    plt.title('Average Profit by Market Regime')
    plt.xlabel('Regime')
    plt.ylabel('Average Profit ($)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(win_rates_df['avg_profit']):
        plt.text(i, v + (0.1 if v >= 0 else -0.1), f"${v:.2f}", ha='center', va='bottom' if v >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_profit_by_regime_{date_range_str}.png'), dpi=150)
    plt.close()
    
    # Create trade count pie chart
    plt.figure(figsize=(10, 8))
    
    # Plot
    plt.pie(win_rates_df['n_trades'], labels=win_rates_df['market_type'], autopct='%1.1f%%',
           colors=bar_colors, explode=[0.05] * len(win_rates_df), shadow=True)
    plt.title('Distribution of Trades by Market Regime')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trade_distribution_by_regime_{date_range_str}.png'), dpi=150)
    plt.close()
    
    # Create stacked bar of winning vs losing trades
    plt.figure(figsize=(12, 6))
    
    # Create stacked bar
    width = 0.6
    bottoms = np.zeros(len(win_rates_df))
    
    # Plot winning trades
    plt.bar(win_rates_df['market_type'], win_rates_df['n_winning'], width, color='green', alpha=0.7, label='Winning Trades')
    
    # Plot losing trades
    plt.bar(win_rates_df['market_type'], win_rates_df['n_losing'], width, bottom=win_rates_df['n_winning'],
           color='red', alpha=0.7, label='Losing Trades')
    
    plt.title('Winning vs Losing Trades by Market Regime')
    plt.xlabel('Regime')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (win, lose) in enumerate(zip(win_rates_df['n_winning'], win_rates_df['n_losing'])):
        plt.text(i, win/2, str(int(win)), ha='center', color='white')
        plt.text(i, win + lose/2, str(int(lose)), ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'win_lose_by_regime_{date_range_str}.png'), dpi=150)
    plt.close()

def create_prediction_accuracy_analysis(market_type_df, output_dir, date_range_str):
    """
    Create visualizations analyzing ML prediction accuracy.
    
    Args:
        market_type_df: DataFrame with market type and prediction information
        output_dir: Directory to save visualizations
        date_range_str: String with date range for file naming
    """
    # Check if we have prediction data
    if market_type_df.empty or 'predicted_regime' not in market_type_df.columns:
        return
    
    # Get unique regimes
    all_regimes = sorted(set(market_type_df['market_type'].unique()).union(
                           set(market_type_df['predicted_regime'].unique())))
    
    # Create confusion matrix
    confusion = pd.DataFrame(0, index=all_regimes, columns=all_regimes)
    
    # Calculate prediction lag matrix
    prediction_horizon = market_type_df.get('prediction_horizon', 5)  # Default to 5 days
    
    # For each actual regime change, look forward to see if it was predicted
    for i in range(len(market_type_df) - 1):
        actual_regime = market_type_df['market_type'].iloc[i]
        
        # Look back prediction_horizon days to see if this change was predicted
        for j in range(max(0, i - prediction_horizon), i):
            predicted_regime = market_type_df['predicted_regime'].iloc[j]
            confusion.loc[actual_regime, predicted_regime] += 1
    
    # Normalize to get accuracy by actual regime
    row_sums = confusion.sum(axis=1)
    accuracy_matrix = confusion.div(row_sums, axis=0).fillna(0)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Prediction Accuracy ({prediction_horizon}-Day Horizon)')
    plt.xlabel('Predicted Regime')
    plt.ylabel('Actual Regime')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_accuracy_{date_range_str}.png'), dpi=150)
    plt.close()
    
    # Calculate overall accuracy
    if 'correct_prediction' in market_type_df.columns:
        overall_accuracy = market_type_df['correct_prediction'].mean() * 100
        
        # Create rolling accuracy plot
        plt.figure(figsize=(12, 6))
        rolling_accuracy = market_type_df['correct_prediction'].rolling(20).mean() * 100
        plt.plot(market_type_df['date'], rolling_accuracy, color='blue', linewidth=1.5)
        plt.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7,
                   label=f'Overall: {overall_accuracy:.1f}%')
        plt.title('20-Day Rolling Prediction Accuracy')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rolling_accuracy_{date_range_str}.png'), dpi=150)
        plt.close()
    
    # Create ML confidence analysis if available
    if 'ml_confidence' in market_type_df.columns:
        # Create confidence histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(market_type_df['ml_confidence'], bins=20, kde=True)
        plt.title('Distribution of ML Prediction Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confidence_distribution_{date_range_str}.png'), dpi=150)
        plt.close()
        
        # Create confidence vs accuracy plot if we have accuracy data
        if 'correct_prediction' in market_type_df.columns:
            # Bin confidence and calculate accuracy
            market_type_df['confidence_bin'] = pd.cut(market_type_df['ml_confidence'], bins=10)
            accuracy_by_confidence = market_type_df.groupby('confidence_bin')['correct_prediction'].mean() * 100
            
            plt.figure(figsize=(10, 6))
            accuracy_by_confidence.plot(kind='bar', color='blue')
            plt.title('Prediction Accuracy by Confidence Level')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'accuracy_by_confidence_{date_range_str}.png'), dpi=150)
            plt.close()
