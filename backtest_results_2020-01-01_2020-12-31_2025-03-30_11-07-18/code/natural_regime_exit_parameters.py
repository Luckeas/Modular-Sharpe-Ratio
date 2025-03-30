"""
natural_regime_exit_parameters.py - Test exit parameters on naturally occurring market regimes

This script tests different exit parameter combinations on trades that occur in their natural
market regimes, rather than forcing the entire dataset into a single regime type.
"""

import pandas as pd
import numpy as np
import os
import logging
import copy
import json
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob

# Import utilities
from utils import setup_directories, load_and_process_data, calculate_indicators

# Import standard backtester components
from unified_backtester import config, run_backtest
# Keep the import for check_entry_signal from backtester_common
from backtester_common import check_entry_signal
# Import analysis functions
from trade_analysis import analyze_performance, analyze_exit_strategies

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_exit_parameters_with_natural_regimes(df, output_dir, param_combinations, use_ml=False,
                                              market_type_window=None, market_type_update_frequency=None):
    """
    Test different exit parameters respecting the natural market type detection.

    Args:
        df: DataFrame with price data and indicators
        output_dir: Directory to save results
        param_combinations: List of dictionaries with parameter combinations to test
        use_ml: Whether to use ML-enhanced backtester
        market_type_window: Days to look back for market type detection (overrides config)
        market_type_update_frequency: Days between market type checks (overrides config)

    Returns:
        Dictionary with best parameters for each market type
    """
    # Reset market type detection state at the start of each test
    reset_market_type_detection()
    logger.info("Starting fresh market regime detection")

    # Log HMM configuration
    if 'hmm_detector' in config:
        logger.info(f"HMM will use {config['hmm_detector']['min_samples']} minimum samples for training")
    
    # Create dictionary to store results by market type
    results_by_market_type = {'mean_reverting': [], 'trend_following': [], 'neutral': []}
    original_config = copy.deepcopy(config)
    
    # Store original config values
    original_market_type_window = config['market_type']['window']
    original_market_type_update_frequency = config['market_type']['update_frequency']
    
    # Override with provided values if specified
    if market_type_window is not None:
        config['market_type']['window'] = market_type_window
        logger.info(f"Overriding market type window to {market_type_window} days")
    
    if market_type_update_frequency is not None:
        config['market_type']['update_frequency'] = market_type_update_frequency
        logger.info(f"Overriding market type update frequency to {market_type_update_frequency} days")
    
    logger.info(f"Using unified backtester with ML {'enabled' if use_ml else 'disabled'}")
    
    # First, run a reference backtest to identify when market type changes occur
    logger.info("Running reference backtest to identify market type periods...")
    ref_file_paths = {
        'trade_log': os.path.join(output_dir, 'reference_trade_log.csv'),
        'portfolio_value': os.path.join(output_dir, 'reference_portfolio_value.csv'),
        'regime_log': os.path.join(output_dir, 'reference_regime_log.csv'),
        'market_type_log': os.path.join(output_dir, 'reference_market_type_log.csv'),
        'summary': os.path.join(output_dir, 'reference_summary.txt')
    }

    reference_trades, _, _, _, market_type_log, _, _, _ = run_backtest(
        df.copy(), visualize_trades=False, file_paths=ref_file_paths, use_ml=use_ml
    )
    
    # Extract market type periods
    market_periods = []
    current_type = None
    start_date = None
    
    # Create a detailed market shift log file
    market_shift_log_path = os.path.join(output_dir, 'market_regime_shifts.log')
    with open(market_shift_log_path, 'w') as shift_log:
        shift_log.write("MARKET REGIME SHIFT LOG\n")
        shift_log.write("=" * 80 + "\n")
        shift_log.write("Date                    | Previous Regime   | New Regime       | Duration    | Notes\n")
        shift_log.write("-" * 80 + "\n")
        
        for entry in market_type_log:
            if current_type is None:
                # Initial market type
                current_type = entry['market_type']
                start_date = entry['date']
                formatted_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
                shift_log.write(f"{formatted_date} | {'INITIAL':16} | {current_type:16} | {'N/A':11} | Initial market type detection\n")
                logger.info(f"MARKET REGIME INITIALIZED: {current_type} at {formatted_date}")
            elif entry['market_type'] != current_type:
                # Market type has changed - record the shift
                end_date = entry['date']
                duration = end_date - start_date
                
                # Add to periods list
                market_periods.append({
                    'market_type': current_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': duration
                })
                
                # Format for logging
                formatted_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
                duration_str = f"{duration.days}d {duration.seconds//3600}h"
                
                # Log the change
                shift_message = f"MARKET REGIME SHIFT: {current_type} → {entry['market_type']} at {formatted_date} (lasted {duration_str})"
                logger.info(shift_message)
                
                # Write to the shift log file
                shift_log.write(f"{formatted_date} | {current_type:16} | {entry['market_type']:16} | {duration_str:11} | ")
                
                # Add regime metrics if available
                metrics = entry.get('metrics', {})
                if metrics:
                    metrics_str = f"ADX: {metrics.get('trend_strength', 0):.1f}, "
                    metrics_str += f"Vol Ratio: {metrics.get('volatility_ratio', 0):.2f}, "
                    metrics_str += f"Score: {metrics.get('market_type_score', 0):.1f}"
                    shift_log.write(metrics_str)
                
                shift_log.write("\n")
                
                # Start new period
                current_type = entry['market_type']
                start_date = end_date
    
        # Add the final period
        if current_type is not None and start_date is not None:
            end_date = df['date'].max()
            duration = end_date - start_date
            
            market_periods.append({
                'market_type': current_type,
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration
            })
            
            # Log the final period
            formatted_end = end_date.strftime("%Y-%m-%d %H:%M:%S")
            duration_str = f"{duration.days}d {duration.seconds//3600}h"
            shift_log.write(f"{formatted_end} | {current_type:16} | {'END OF DATA':16} | {duration_str:11} | Final period\n")
    
    # Create a market periods summary file with visualizations
    with open(os.path.join(output_dir, 'market_periods_summary.txt'), 'w') as summary_file:
        summary_file.write("MARKET PERIODS SUMMARY\n")
        summary_file.write("=" * 80 + "\n\n")
        
        # Add statistics about market periods
        regime_counts = {'mean_reverting': 0, 'trend_following': 0, 'neutral': 0}
        regime_durations = {'mean_reverting': [], 'trend_following': [], 'neutral': []}
        
        for period in market_periods:
            regime = period['market_type']
            regime_counts[regime] += 1
            regime_durations[regime].append(period['duration'])
        
        summary_file.write("REGIME OCCURRENCE STATISTICS:\n")
        summary_file.write("-" * 80 + "\n")
        for regime, count in regime_counts.items():
            if count > 0:
                avg_duration = sum((d.total_seconds() for d in regime_durations[regime]), 0) / count / 86400  # Convert to days
                summary_file.write(f"{regime.upper():16}: {count} occurrences, Average duration: {avg_duration:.2f} days\n")
            else:
                summary_file.write(f"{regime.upper():16}: {count} occurrences\n")
        
        summary_file.write("\n")
        summary_file.write("DETAILED MARKET PERIODS:\n")
        summary_file.write("-" * 80 + "\n")
        summary_file.write(f"{'Period':8} | {'Market Type':16} | {'Start Date':20} | {'End Date':20} | {'Duration':10}\n")
        summary_file.write("-" * 80 + "\n")
        
        for i, period in enumerate(market_periods):
            duration_str = f"{period['duration'].days}d {period['duration'].seconds//3600}h"
            start_str = period['start_date'].strftime("%Y-%m-%d %H:%M:%S")
            end_str = period['end_date'].strftime("%Y-%m-%d %H:%M:%S")
            
            summary_file.write(f"{i+1:8} | {period['market_type']:16} | {start_str:20} | {end_str:20} | {duration_str:10}\n")
    
    # Display identified market periods to console
    logger.info(f"Identified {len(market_periods)} market periods:")
    for i, period in enumerate(market_periods):
        duration_str = f"{period['duration'].days}d {period['duration'].seconds//3600}h"
        logger.info(f"Period {i+1}: {period['market_type']} from {period['start_date']} to {period['end_date']} (Duration: {duration_str})")
    
    # Also map reference trades to their market type for baseline comparison
    ref_trades_by_market = {'mean_reverting': [], 'trend_following': [], 'neutral': []}
    for trade in reference_trades:
        entry_time = trade['entry_time']
        for period in market_periods:
            if period['start_date'] <= entry_time <= period['end_date']:
                ref_trades_by_market[period['market_type']].append(trade)
                break
    
    # Log reference trade counts by market type
    logger.info("Reference trade counts by market type:")
    for market_type, trades in ref_trades_by_market.items():
        logger.info(f"  {market_type}: {len(trades)} trades")
    
    # Test each parameter combination
    for params in param_combinations:
        param_id = f"TS{'_on' if params['enable_trailing_stop'] else '_off'}_TSM{params['trailing_stop_multiplier']}_DT{'_on' if params['dynamic_target_enable'] else '_off'}_DTM{params['dynamic_target_multiplier']}"
        logger.info(f"Testing parameters: {param_id}")
        
        # Create parameter-specific output directory
        param_dir = os.path.join(output_dir, param_id)
        os.makedirs(param_dir, exist_ok=True)
        
        # Update config with test parameters
        config['risk']['enable_trailing_stop'] = params['enable_trailing_stop']
        config['risk']['trailing_stop_atr_multiplier'] = params['trailing_stop_multiplier']
        config['risk']['dynamic_target_enable'] = params['dynamic_target_enable']
        config['risk']['dynamic_target_atr_multiplier'] = params['dynamic_target_multiplier']
        
        # Run full backtest with these parameters
        file_paths = {
            'trade_log': os.path.join(param_dir, 'trade_log.csv'),
            'portfolio_value': os.path.join(param_dir, 'portfolio_value.csv'),
            'regime_log': os.path.join(param_dir, 'regime_log.csv'),
            'market_type_log': os.path.join(param_dir, 'market_type_log.csv'),
            'summary': os.path.join(param_dir, 'summary.txt')
        }

        trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
            df.copy(), visualize_trades=False, file_paths=file_paths, use_ml=use_ml
        )
        
        # Skip if no trades were executed
        if not trades:
            logger.warning(f"No trades executed with {param_id}. Skipping.")
            continue
        
        # Group trades by market type at entry time
        trades_by_market = {'mean_reverting': [], 'trend_following': [], 'neutral': []}
        for trade in trades:
            entry_time = trade['entry_time']
            for period in market_periods:
                if period['start_date'] <= entry_time <= period['end_date']:
                    market_type = period['market_type']
                    trades_by_market[market_type].append(trade)
                    break
        
        # Analyze performance by market type
        for market_type, market_trades in trades_by_market.items():
            if market_trades:
                # Calculate key metrics
                win_rate = sum(1 for t in market_trades if t['profit'] > 0) / len(market_trades) * 100
                avg_profit = sum(t['profit'] for t in market_trades) / len(market_trades)
                total_profit = sum(t['profit'] for t in market_trades)
                
                # Calculate drawdown if portfolio values are available
                max_drawdown = 0
                sharpe_ratio = 0
                
                # Calculate additional metrics if possible
                if len(portfolio_values) > 0:
                    # This is an approximation since we can't easily extract market-specific portfolio values
                    portfolio_df = pd.DataFrame({'date': df['date'], 'value': portfolio_values})
                    portfolio_series = portfolio_df.set_index('date')['value']
                    
                    # Calculate drawdown
                    drawdown = ((portfolio_series.cummax() - portfolio_series) / portfolio_series.cummax()) * 100
                    max_drawdown = drawdown.max()
                    
                    # Calculate Sharpe ratio (simple approximation)
                    daily_returns = portfolio_series.pct_change().dropna()
                    if len(daily_returns) > 0 and daily_returns.std() > 0:
                        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                
                # Store result with parameter information
                result = {
                    **params,
                    'total_trades': len(market_trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'profit_loss': total_profit,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                }
                
                # Add exit strategy metrics if available
                exit_metrics = analyze_exit_strategies(market_trades)
                if 'price_capture_efficiency' in exit_metrics:
                    result['price_capture_pct'] = exit_metrics['price_capture_efficiency']['avg_capture_pct']
                    result['trailing_advantage'] = exit_metrics['price_capture_efficiency'].get('trailing_advantage', 0)
                
                results_by_market_type[market_type].append(result)
                
                logger.info(f"Results for {market_type} with {param_id}: "
                           f"Trades: {len(market_trades)}, "
                           f"Win Rate: {win_rate:.2f}%, "
                           f"Profit: ${total_profit:.2f}")
    
    # Convert to DataFrames and find best parameters for each market type
    market_results_dfs = {}
    best_params = {}
    
    for market_type, results in results_by_market_type.items():
        if results:
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            # Sort by profit
            results_df = results_df.sort_values('profit_loss', ascending=False)
            # Save results
            results_df.to_csv(os.path.join(output_dir, f'{market_type}_parameter_results.csv'), index=False)
            # Store best parameters
            best_params[market_type] = results_df.iloc[0].to_dict()
            # Store DataFrame for return
            market_results_dfs[market_type] = results_df
            
            # Create a detailed summary for this market type
            with open(os.path.join(output_dir, f'{market_type}_summary.txt'), 'w') as f:
                f.write(f"PARAMETER TEST RESULTS FOR {market_type.upper()} MARKET TYPE\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total parameter combinations tested: {len(results)}\n")
                f.write(f"Reference trades in this market type: {len(ref_trades_by_market[market_type])}\n\n")
                
                f.write("TOP 5 PARAMETER COMBINATIONS:\n")
                f.write("-" * 80 + "\n")
                
                for i, row in results_df.head(5).iterrows():
                    f.write(f"Rank {i+1}:\n")
                    f.write(f"  Trailing Stop: {'Enabled' if row['enable_trailing_stop'] else 'Disabled'}\n")
                    f.write(f"  Trailing Stop Multiplier: {row['trailing_stop_multiplier']}\n")
                    f.write(f"  Dynamic Target: {'Enabled' if row['dynamic_target_enable'] else 'Disabled'}\n")
                    f.write(f"  Dynamic Target Multiplier: {row['dynamic_target_multiplier']}\n")
                    f.write(f"  Profit/Loss: ${row['profit_loss']:.2f}\n")
                    f.write(f"  Win Rate: {row['win_rate']:.2f}%\n")
                    f.write(f"  Total Trades: {row['total_trades']}\n")
                    f.write(f"  Average Profit/Trade: ${row['avg_profit']:.2f}\n")
                    if 'max_drawdown' in row and not pd.isna(row['max_drawdown']):
                        f.write(f"  Max Drawdown: {row['max_drawdown']:.2f}%\n")
                    if 'price_capture_pct' in row and not pd.isna(row['price_capture_pct']):
                        f.write(f"  Price Capture: {row['price_capture_pct']:.2f}%\n")
                    f.write("\n")
    
    # Save the best parameters for all market types
    best_params_file = os.path.join(output_dir, 'best_exit_parameters_by_market.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters for each market type saved to {best_params_file}")
    
    # Restore original config
    config.update(original_config)
    
    # Also restore the specific market type parameters we may have overridden
    config['market_type']['window'] = original_market_type_window
    config['market_type']['update_frequency'] = original_market_type_update_frequency
    
    return best_params, market_results_dfs

def calculate_composite_score(df):
    """
    Calculate a composite score that balances profit and risk metrics.
    
    The score combines the following factors:
    - Profit/Loss (higher is better)
    - Max Drawdown (lower is better)
    - Sharpe Ratio (higher is better)
    - Win Rate (higher is better)
    - Number of trades (more is generally better for statistical significance)
    
    Returns a DataFrame with the composite score.
    """
    # Create a copy to avoid modifying the original
    df_scored = df.copy()
    
    # Ensure all required columns are available
    required_cols = ['profit_loss', 'win_rate', 'total_trades']
    optional_cols = ['max_drawdown', 'sharpe_ratio']
    
    for col in required_cols:
        if col not in df_scored.columns:
            print(f"Warning: Required column '{col}' missing. Cannot calculate composite score.")
            return df
    
    # Define column weights (adjust these to emphasize different aspects)
    weights = {
        'profit_loss': 0.35,          # 35% weight on profitability
        'win_rate': 0.20,             # 20% weight on consistency
        'max_drawdown': 0.25,         # 25% weight on drawdown risk
        'sharpe_ratio': 0.15,         # 15% weight on risk-adjusted returns  
        'total_trades': 0.05          # 5% weight on sample size
    }
    
    # Adjust available weights if some columns are missing
    available_weights = {k: weights[k] for k in weights if k in df_scored.columns}
    weight_sum = sum(available_weights.values())
    normalized_weights = {k: v/weight_sum for k, v in available_weights.items()}
    
    # Normalize each metric to a 0-1 scale
    for col in available_weights.keys():
        if col == 'max_drawdown':
            # For drawdown, lower is better, so invert the score
            if df_scored[col].max() != df_scored[col].min():
                df_scored[f'{col}_score'] = 1 - ((df_scored[col] - df_scored[col].min()) / 
                                             (df_scored[col].max() - df_scored[col].min()))
            else:
                df_scored[f'{col}_score'] = 1.0  # All values are the same
        else:
            # For other metrics, higher is better
            if df_scored[col].max() != df_scored[col].min():
                df_scored[f'{col}_score'] = (df_scored[col] - df_scored[col].min()) / (df_scored[col].max() - df_scored[col].min())
            else:
                df_scored[f'{col}_score'] = 1.0  # All values are the same
    
    # Calculate the composite score
    df_scored['composite_score'] = sum(df_scored[f'{col}_score'] * normalized_weights[col] for col in available_weights.keys())
    
    # Add ranking based on composite score
    df_scored['rank'] = df_scored['composite_score'].rank(ascending=False, method='min').astype(int)
    
    # Sort by composite score (descending)
    df_scored = df_scored.sort_values('composite_score', ascending=False)
    
    return df_scored

def reset_market_type_detection():
    """Reset all stateful variables in the detect_market_type function"""
    # Import detect_market_type first
    from refactored_backtester import detect_market_type
    
    # Clear the function attributes that store state
    if hasattr(detect_market_type, 'historical_scores'):
        del detect_market_type.historical_scores
    if hasattr(detect_market_type, 'score_dates'):
        del detect_market_type.score_dates
    if hasattr(detect_market_type, 'market_type_counts'):
        del detect_market_type.market_type_counts
    logger.info("Market type detection state has been reset")

def display_top_results(results, top_n=10, use_composite=True):
    """Display the top N results for each market type"""
    for market_type, df in results.items():
        if df.empty:
            continue
            
        print(f"\n{'='*30} TOP {top_n} FOR {market_type.upper()} {'='*30}")
        
        # Calculate composite score if requested
        if use_composite:
            df_sorted = calculate_composite_score(df)
            ranking_method = "Composite Score (Balanced Profit/Risk)"
        else:
            # Traditional ranking by profit only
            df_sorted = df.sort_values('profit_loss', ascending=False).reset_index(drop=True)
            ranking_method = "Profit/Loss Only"
        
        print(f"Ranking Method: {ranking_method}\n")
        
        # Format the top results
        for i, row in df_sorted.head(top_n).iterrows():
            rank_display = f"Rank {row['rank']}:" if 'rank' in row else f"Rank {i+1}:"
            print(f"\n{rank_display}")
            print(f"  Trailing Stop: {'Enabled' if row['enable_trailing_stop'] else 'Disabled'}")
            print(f"  Trailing Stop Multiplier: {row['trailing_stop_multiplier']}")
            print(f"  Dynamic Target: {'Enabled' if row['dynamic_target_enable'] else 'Disabled'}")
            print(f"  Dynamic Target Multiplier: {row['dynamic_target_multiplier']}")
            print(f"  Profit/Loss: ${row['profit_loss']:.2f}")
            print(f"  Win Rate: {row['win_rate']:.2f}%")
            print(f"  Total Trades: {row['total_trades']}")
            print(f"  Average Profit/Trade: ${row['avg_profit']:.2f}")
            
            if 'max_drawdown' in row and not pd.isna(row['max_drawdown']):
                print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
            if 'sharpe_ratio' in row and not pd.isna(row['sharpe_ratio']):
                print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
            if 'composite_score' in row and not pd.isna(row['composite_score']):
                print(f"  Composite Score: {row['composite_score']:.4f}")
            if 'price_capture_pct' in row and not pd.isna(row['price_capture_pct']):
                print(f"  Price Capture: {row['price_capture_pct']:.2f}%")
            if 'trailing_advantage' in row and not pd.isna(row['trailing_advantage']):
                print(f"  Trailing Advantage: {row['trailing_advantage']:.2f}%")

def generate_radar_charts(results, output_dir):
    """Generate radar charts to visualize parameter performance across multiple metrics"""
    for market_type, df in results.items():
        if df.empty:
            continue
        
        # Create output directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'radar_charts')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Calculate composite scores
        df_scored = calculate_composite_score(df)
        
        # Get top 5 parameter combinations
        top_params = df_scored.head(5)
        
        # Metrics to include in radar chart (must be normalized 0-1)
        metrics = ['profit_loss_score', 'win_rate_score', 'total_trades_score']
        if 'max_drawdown_score' in df_scored.columns:
            metrics.append('max_drawdown_score')
        if 'sharpe_ratio_score' in df_scored.columns:
            metrics.append('sharpe_ratio_score')
        
        # Number of metrics
        N = len(metrics)
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add metric labels
        metric_labels = [m.replace('_score', '').replace('_', ' ').title() for m in metrics]
        metric_labels += [metric_labels[0]]  # Close the loop
        plt.xticks(angles, metric_labels, size=12)
        
        # Plot each parameter combination
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (_, row) in enumerate(top_params.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=f"Rank {i+1} (Score: {row['composite_score']:.3f})", color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add parameter details to legend
        legend_elements = []
        for i, (_, row) in enumerate(top_params.iterrows()):
            ts_status = "TS:ON" if row['enable_trailing_stop'] else "TS:OFF"
            ts_mult = f"TSM:{row['trailing_stop_multiplier']}"
            dt_status = "DT:ON" if row['dynamic_target_enable'] else "DT:OFF"
            dt_mult = f"DTM:{row['dynamic_target_multiplier']}"
            param_str = f"Rank {i+1}: {ts_status}, {ts_mult}, {dt_status}, {dt_mult}"
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=param_str))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
        
        plt.grid(True)
        plt.title(f"Top 5 Parameter Combinations - {market_type.upper()} Market", size=15)
        
        # Save the chart
        filename = f"{market_type}_radar_chart.png"
        plt.savefig(os.path.join(plots_dir, filename), bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Generated radar chart for {market_type}")

def update_hmm_configuration(min_samples=None):
    """
    Update the HMM configuration with custom parameters.

    Args:
        min_samples: Minimum samples required for HMM training
    """
    if 'hmm_detector' not in config:
        logger.warning("HMM detector configuration not found in config")
        return

    if min_samples is not None:
        config['hmm_detector']['min_samples'] = min_samples
        logger.info(f"Updated HMM min_samples to {min_samples}")

def create_best_params_combined(results, output_dir, use_composite=True):
    """Create a combined JSON with the best parameters for each market type"""
    best_params = {}
    
    for market_type, df in results.items():
        if df.empty:
            continue
        
        # Get the best parameter combination
        if use_composite:
            df_scored = calculate_composite_score(df)
            best_row = df_scored.iloc[0]  # Already sorted by composite score
        else:
            best_row = df.sort_values('profit_loss', ascending=False).iloc[0]
        
        best_params[market_type] = {
            'enable_trailing_stop': bool(best_row['enable_trailing_stop']),
            'trailing_stop_multiplier': float(best_row['trailing_stop_multiplier']),
            'dynamic_target_enable': bool(best_row['dynamic_target_enable']),
            'dynamic_target_multiplier': float(best_row['dynamic_target_multiplier']),
            'ranking_method': 'composite_score' if use_composite else 'profit_loss'
        }
        
        # Add metrics for reference
        metrics = {}
        for metric in ['profit_loss', 'win_rate', 'avg_profit', 'total_trades']:
            if metric in best_row:
                metrics[metric] = float(best_row[metric])
        
        for risk_metric in ['max_drawdown', 'sharpe_ratio', 'composite_score']:
            if risk_metric in best_row:
                metrics[risk_metric] = float(best_row[risk_metric])
                
        best_params[market_type]['metrics'] = metrics
    
    # Save to JSON file
    method_str = "balanced" if use_composite else "profit_only"
    best_params_file = os.path.join(output_dir, f'best_exit_parameters_by_market_{method_str}.json')
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"Combined best parameters saved to: {best_params_file}")
    return best_params

def generate_parameter_impact_plots(results, output_dir, use_composite=True):
    """Generate plots showing the impact of each parameter on performance"""
    for market_type, df in results.items():
        if df.empty:
            continue
        
        # Create output directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'parameter_impact')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Calculate composite score if using balanced approach
        if use_composite:
            df = calculate_composite_score(df)
            metric_column = 'composite_score'
            metric_label = 'Composite Score'
        else:
            metric_column = 'profit_loss'
            metric_label = 'Profit ($)'
        
        # 1. Trailing Stop (Enabled vs Disabled)
        plt.figure(figsize=(12, 6))
        ts_df = df.groupby('enable_trailing_stop')[metric_column].mean().reset_index()
        
        plt.bar(['Disabled', 'Enabled'], ts_df[metric_column], color=['#ff9999', '#66b3ff'])
        plt.title(f"Impact of Trailing Stop Setting on {metric_label} - {market_type.upper()} Market")
        plt.xlabel('Trailing Stop')
        plt.ylabel(metric_label)
        
        for i, row in ts_df.iterrows():
            value = f"${row[metric_column]:.0f}" if metric_column == 'profit_loss' else f"{row[metric_column]:.3f}"
            plt.text(i, row[metric_column], value, ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        filename = f"{market_type}_trailing_stop_impact_{metric_column}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Dynamic Target (Enabled vs Disabled)
        plt.figure(figsize=(12, 6))
        dt_df = df.groupby('dynamic_target_enable')[metric_column].mean().reset_index()
        
        plt.bar(['Disabled', 'Enabled'], dt_df[metric_column], color=['#ff9999', '#66b3ff'])
        plt.title(f"Impact of Dynamic Target Setting on {metric_label} - {market_type.upper()} Market")
        plt.xlabel('Dynamic Target')
        plt.ylabel(metric_label)
        
        for i, row in dt_df.iterrows():
            value = f"${row[metric_column]:.0f}" if metric_column == 'profit_loss' else f"{row[metric_column]:.3f}"
            plt.text(i, row[metric_column], value, ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        filename = f"{market_type}_dynamic_target_impact_{metric_column}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Trailing Stop Multiplier
        plt.figure(figsize=(12, 6))
        tsm_df = df.groupby('trailing_stop_multiplier')[metric_column].mean().reset_index()
        
        plt.bar(tsm_df['trailing_stop_multiplier'].astype(str), tsm_df[metric_column], color='#66b3ff')
        plt.title(f"Impact of Trailing Stop Multiplier on {metric_label} - {market_type.upper()} Market")
        plt.xlabel('Trailing Stop Multiplier')
        plt.ylabel(metric_label)
        
        for i, row in tsm_df.iterrows():
            value = f"${row[metric_column]:.0f}" if metric_column == 'profit_loss' else f"{row[metric_column]:.3f}"
            plt.text(i, row[metric_column], value, ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        filename = f"{market_type}_ts_multiplier_impact_{metric_column}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Dynamic Target Multiplier
        plt.figure(figsize=(12, 6))
        dtm_df = df.groupby('dynamic_target_multiplier')[metric_column].mean().reset_index()
        
        plt.bar(dtm_df['dynamic_target_multiplier'].astype(str), dtm_df[metric_column], color='#66b3ff')
        plt.title(f"Impact of Dynamic Target Multiplier on {metric_label} - {market_type.upper()} Market")
        plt.xlabel('Dynamic Target Multiplier')
        plt.ylabel(metric_label)
        
        for i, row in dtm_df.iterrows():
            value = f"${row[metric_column]:.0f}" if metric_column == 'profit_loss' else f"{row[metric_column]:.3f}"
            plt.text(i, row[metric_column], value, ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        filename = f"{market_type}_dt_multiplier_impact_{metric_column}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated parameter impact plots for {market_type} using {metric_label}")

def create_comparison_table(results, output_dir):
    """Create comparison table between profit-only and balanced rankings"""
    comparison_data = []
    
    for market_type, df in results.items():
        if df.empty:
            continue
        
        # Get top parameters using both methods
        df_scored = calculate_composite_score(df)
        profit_best = df.sort_values('profit_loss', ascending=False).iloc[0]
        balanced_best = df_scored.iloc[0]
        
        # Create comparison entry
        entry = {
            'market_type': market_type,
            'profit_only_ts': 'Enabled' if profit_best['enable_trailing_stop'] else 'Disabled',
            'profit_only_ts_mult': profit_best['trailing_stop_multiplier'],
            'profit_only_dt': 'Enabled' if profit_best['dynamic_target_enable'] else 'Disabled',
            'profit_only_dt_mult': profit_best['dynamic_target_multiplier'],
            'profit_only_profit': profit_best['profit_loss'],
            'profit_only_win_rate': profit_best['win_rate'],
            
            'balanced_ts': 'Enabled' if balanced_best['enable_trailing_stop'] else 'Disabled',
            'balanced_ts_mult': balanced_best['trailing_stop_multiplier'],
            'balanced_dt': 'Enabled' if balanced_best['dynamic_target_enable'] else 'Disabled',
            'balanced_dt_mult': balanced_best['dynamic_target_multiplier'],
            'balanced_profit': balanced_best['profit_loss'],
            'balanced_win_rate': balanced_best['win_rate'],
        }
        
        # Add risk metrics if available
        for metric in ['max_drawdown', 'sharpe_ratio']:
            if metric in profit_best and metric in balanced_best:
                entry[f'profit_only_{metric}'] = profit_best[metric]
                entry[f'balanced_{metric}'] = balanced_best[metric]
        
        if 'composite_score' in balanced_best:
            entry['balanced_score'] = balanced_best['composite_score']
        
        comparison_data.append(entry)
    
    # Convert to DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(output_dir, 'ranking_method_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Ranking method comparison saved to: {csv_path}")
        
        # Also create an HTML version for better viewing
        html_path = os.path.join(output_dir, 'ranking_method_comparison.html')
        comparison_df.to_html(html_path, index=False)
        print(f"HTML comparison table saved to: {html_path}")
    
    return comparison_data

def analyze_test_results(market_results, output_dir, top_n=5):
    """
    Perform comprehensive analysis on test results
    
    Args:
        market_results: Dictionary with market type as keys and result DataFrames as values
        output_dir: Directory to save analysis results
        top_n: Number of top results to display for each market type
    """
    logger.info("Beginning comprehensive analysis of test results")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Display top results using composite score (balanced approach)
    print("\n=== RESULTS RANKED BY COMPOSITE SCORE (BALANCED PROFIT/RISK) ===")
    display_top_results(market_results, top_n=top_n, use_composite=True)
    
    # Also display top results using traditional profit-only ranking for comparison
    print("\n=== RESULTS RANKED BY PROFIT ONLY (FOR COMPARISON) ===")
    display_top_results(market_results, top_n=top_n, use_composite=False)
    
    # Generate parameter impact plots
    generate_parameter_impact_plots(market_results, plots_dir, use_composite=True)
    
    # Try to generate radar charts (which show multiple metrics at once)
    try:
        generate_radar_charts(market_results, plots_dir)
    except Exception as e:
        logger.error(f"Could not generate radar charts: {e}")
    
    # Create best parameter files using both methods
    best_params_balanced = create_best_params_combined(market_results, output_dir, use_composite=True)
    best_params_profit = create_best_params_combined(market_results, output_dir, use_composite=False)
    
    # Create comparison table between the two ranking methods
    comparison = create_comparison_table(market_results, output_dir)
    
    print("\n=== COMPARISON OF RANKING METHODS ===")
    print("PROFIT-ONLY RANKING vs BALANCED PROFIT/RISK RANKING")
    
    for entry in comparison:
        market = entry['market_type'].upper()
        print(f"\n{market} MARKET:")
        
        print("PROFIT-ONLY BEST PARAMETERS:")
        print(f"  Trailing Stop: {entry['profit_only_ts']}")
        print(f"  Trailing Stop Multiplier: {entry['profit_only_ts_mult']}")
        print(f"  Dynamic Target: {entry['profit_only_dt']}")
        print(f"  Dynamic Target Multiplier: {entry['profit_only_dt_mult']}")
        print(f"  Profit: ${entry['profit_only_profit']:.2f}")
        print(f"  Win Rate: {entry['profit_only_win_rate']:.2f}%")
        if 'profit_only_max_drawdown' in entry:
            print(f"  Max Drawdown: {entry['profit_only_max_drawdown']:.2f}%")
        
        print("\nBALANCED RANKING BEST PARAMETERS:")
        print(f"  Trailing Stop: {entry['balanced_ts']}")
        print(f"  Trailing Stop Multiplier: {entry['balanced_ts_mult']}")
        print(f"  Dynamic Target: {entry['balanced_dt']}")
        print(f"  Dynamic Target Multiplier: {entry['balanced_dt_mult']}")
        print(f"  Profit: ${entry['balanced_profit']:.2f}")
        print(f"  Win Rate: {entry['balanced_win_rate']:.2f}%")
        if 'balanced_max_drawdown' in entry:
            print(f"  Max Drawdown: {entry['balanced_max_drawdown']:.2f}%")
        if 'balanced_score' in entry:
            print(f"  Composite Score: {entry['balanced_score']:.4f}")
    
    logger.info("Analysis complete. Check the output directory for detailed results.")


# Main execution
if __name__ == "__main__":
    # Configure test parameters
    test_config = {
        'data_file': config['data']['file_path'],
        'start_date': config['data']['start_date'],
        'end_date': config['data']['end_date'],
        'use_ml': False,  # Set to True to test with ML-enhanced backtester
        'test_name': f"natural_regime_exit_param_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'top_n_results': 5,  # Number of top results to display for each market type

        # Additional parameters for regime detection
        'market_type_update_frequency': 5,  # Days between market type checks
        'market_type_window': 20,  # Days to look back for detection
        'log_market_shifts': True,  # Whether to log market shifts in detail
        'hmm_min_samples': 100  # NEW: Minimum samples for HMM training
    }

    # Set up output directory
    output_dir = test_config['test_name']
    os.makedirs(output_dir, exist_ok=True)

    # Update HMM configuration with specified parameters
    if 'hmm_min_samples' in test_config:
        update_hmm_configuration(min_samples=test_config['hmm_min_samples'])

    # Load and prepare data
    df_5min = load_and_process_data(
        test_config['data_file'],
        test_config['start_date'],
        test_config['end_date']
    )
    
    if df_5min is None or len(df_5min) == 0:
        logger.error("No data available after loading. Exiting.")
        exit(1)
    
    # Calculate all indicators
    df_5min = calculate_indicators(df_5min, config)
    
    # Define parameter combinations to test
    param_combinations = []
    
    # Test with trailing stop on/off
    for ts_enabled in [True, False]:
        # Test different trailing stop multipliers
        for ts_mult in [1.0, 1.5, 2.0]:
            # Test with dynamic target on/off
            for dt_enabled in [True, False]:
                # Test different dynamic target multipliers
                for dt_mult in [0.5, 1.0, 1.5]:
                    param_combinations.append({
                        'enable_trailing_stop': ts_enabled,
                        'trailing_stop_multiplier': ts_mult,
                        'dynamic_target_enable': dt_enabled,
                        'dynamic_target_multiplier': dt_mult
                    })
    
    # Run the parameter tests with natural market regimes
    logger.info(f"Starting parameter testing with {len(param_combinations)} combinations across natural market regimes")
    
    # Create a visual representation of market regimes for the report
    fig, ax = plt.figure(figsize=(14, 6)), plt.gca()
    ax.set_title("Market Regime Visualization for Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    
    # Run the parameter tests with specified market regime parameters
    best_params, market_results = test_exit_parameters_with_natural_regimes(
        df_5min, 
        output_dir, 
        param_combinations,
        use_ml=test_config['use_ml'],
        market_type_window=test_config.get('market_type_window'),
        market_type_update_frequency=test_config.get('market_type_update_frequency')
    )
    
    # Print best parameters for each market type based on profit only
    print("\nBest Exit Parameters by Market Type (Profit Only):")
    for market_type, params in best_params.items():
        if params:
            print(f"\n{market_type.upper()}:")
            print(f"  Trailing Stop: {'Enabled' if params['enable_trailing_stop'] else 'Disabled'}")
            print(f"  Trailing Stop Multiplier: {params['trailing_stop_multiplier']}")
            print(f"  Dynamic Target: {'Enabled' if params['dynamic_target_enable'] else 'Disabled'}")
            print(f"  Dynamic Target Multiplier: {params['dynamic_target_multiplier']}")
            if 'profit_loss' in params:
                print(f"  Profit: ${params['profit_loss']:.2f}")
            if 'win_rate' in params:
                print(f"  Win Rate: {params['win_rate']:.2f}%")
    
    logger.info(f"Parameter testing complete. Results saved to {output_dir}")
    
    # Perform comprehensive analysis with balanced metrics
    analyze_test_results(market_results, output_dir, top_n=test_config['top_n_results'])