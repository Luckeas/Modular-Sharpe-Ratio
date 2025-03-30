"""
analyze_profits.py - Script to analyze profitable trades and strategy success patterns

This script identifies patterns in profitable runs by analyzing trade data
from multiple backtest runs, focusing on the most successful scenarios.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from scipy import stats

def load_profitable_runs(base_dir, top_percentile=25):
    """
    Load trade logs from top-performing runs.
    
    Args:
        base_dir: Directory containing robustness test results
        top_percentile: Percentile threshold for top runs (e.g., 25 = top 25%)
        
    Returns:
        DataFrame with combined trade data from the most profitable runs
    """
    # Find all seed directories
    seed_dirs = glob.glob(os.path.join(base_dir, "seed_*"))
    
    # Get results CSV to identify top runs
    results_df = pd.read_csv(os.path.join(base_dir, "robustness_results.csv"))
    
    # Calculate profit threshold for top percentile
    profit_threshold = results_df['profit_loss'].quantile(1 - top_percentile/100)
    
    # Filter to top runs
    top_seeds = results_df[results_df['profit_loss'] >= profit_threshold]['seed'].tolist()
    seed_dirs = [d for d in seed_dirs if int(d.split('_')[-1]) in top_seeds]
    
    print(f"Analyzing {len(seed_dirs)} top-performing runs (top {top_percentile}% with profit >= ${profit_threshold:.2f})")
    
    # Load all trade logs
    all_trades = []
    
    for seed_dir in seed_dirs:
        seed = int(seed_dir.split('_')[-1])
        trade_log_path = os.path.join(seed_dir, 'trade_log.csv')
        
        if os.path.exists(trade_log_path):
            trades_df = pd.read_csv(trade_log_path)
            trades_df['seed'] = seed
            
            # Add information from results
            seed_results = results_df[results_df['seed'] == seed]
            if not seed_results.empty:
                trades_df['run_profit_loss'] = seed_results['profit_loss'].values[0]
                trades_df['run_win_rate'] = seed_results['win_rate'].values[0]
                trades_df['run_sharpe'] = seed_results['sharpe_ratio'].values[0]
                trades_df['run_max_dd'] = seed_results['max_drawdown_pct'].values[0]
            
            all_trades.append(trades_df)
    
    if not all_trades:
        print("No trade logs found.")
        return None
    
    # Combine all trades
    combined_df = pd.concat(all_trades, ignore_index=True)
    
    # Convert date columns
    for date_col in ['entry_time', 'exit_time']:
        combined_df[date_col] = pd.to_datetime(combined_df[date_col])
    
    # Add derived metrics
    combined_df['profit_pct'] = combined_df['profit'] / combined_df['entry_account_value'] * 100
    combined_df['holding_time'] = (combined_df['exit_time'] - combined_df['entry_time']).dt.total_seconds() / 3600
    combined_df['trade_result'] = np.where(combined_df['profit'] > 0, 'win', 'loss')
    
    return combined_df

def analyze_profitable_patterns(trades_df):
    """
    Analyze characteristics of trades in profitable runs.
    
    Args:
        trades_df: DataFrame with trade data from profitable runs
        
    Returns:
        Dictionary with analysis results
    """
    # Split into winning and losing trades
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] <= 0]
    
    # Percentage of winning vs losing trades
    win_rate = len(winning_trades) / len(trades_df) * 100
    
    # Average profit/loss
    avg_win = winning_trades['profit'].mean()
    avg_loss = losing_trades['profit'].mean()
    
    # Risk-Reward ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Analyze by market type
    market_type_analysis = {}
    for market_type in trades_df['market_type'].unique():
        mt_trades = trades_df[trades_df['market_type'] == market_type]
        mt_win_rate = len(mt_trades[mt_trades['profit'] > 0]) / len(mt_trades) * 100
        
        market_type_analysis[market_type] = {
            'count': len(mt_trades),
            'win_rate': mt_win_rate,
            'avg_profit': mt_trades['profit'].mean(),
            'avg_win': mt_trades[mt_trades['profit'] > 0]['profit'].mean() if len(mt_trades[mt_trades['profit'] > 0]) > 0 else 0,
            'avg_loss': mt_trades[mt_trades['profit'] <= 0]['profit'].mean() if len(mt_trades[mt_trades['profit'] <= 0]) > 0 else 0,
            'pct_of_trades': len(mt_trades) / len(trades_df) * 100
        }
    
    # Analyze by exit reason
    exit_reason_analysis = {}
    for reason in trades_df['exit_reason'].unique():
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        reason_win_rate = len(reason_trades[reason_trades['profit'] > 0]) / len(reason_trades) * 100
        
        exit_reason_analysis[reason] = {
            'count': len(reason_trades),
            'win_rate': reason_win_rate,
            'avg_profit': reason_trades['profit'].mean(),
            'pct_of_trades': len(reason_trades) / len(trades_df) * 100
        }
    
    # Analyze by regime score
    regime_bins = [0, 40, 60, 80, 100]
    regime_labels = ['0-40', '40-60', '60-80', '80-100']
    trades_df['regime_bin'] = pd.cut(trades_df['regime_score'], bins=regime_bins, labels=regime_labels)
    
    regime_analysis = {}
    for regime_bin in trades_df['regime_bin'].unique():
        if pd.isna(regime_bin):
            continue
            
        regime_trades = trades_df[trades_df['regime_bin'] == regime_bin]
        regime_win_rate = len(regime_trades[regime_trades['profit'] > 0]) / len(regime_trades) * 100
        
        regime_analysis[regime_bin] = {
            'count': len(regime_trades),
            'win_rate': regime_win_rate,
            'avg_profit': regime_trades['profit'].mean(),
            'pct_of_trades': len(regime_trades) / len(trades_df) * 100
        }
    
    # Analyze trade durations
    duration_analysis = {
        'avg_holding_time': trades_df['holding_time'].mean(),
        'winning_holding_time': winning_trades['holding_time'].mean(),
        'losing_holding_time': losing_trades['holding_time'].mean()
    }
    
    # Analyze position sizes
    position_analysis = {
        'avg_position_size': trades_df['num_contracts'].mean(),
        'winning_position_size': winning_trades['num_contracts'].mean(),
        'losing_position_size': losing_trades['num_contracts'].mean()
    }
    
    # Return combined analysis
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward': risk_reward,
        'market_type_analysis': market_type_analysis,
        'exit_reason_analysis': exit_reason_analysis,
        'regime_analysis': regime_analysis,
        'duration_analysis': duration_analysis,
        'position_analysis': position_analysis
    }

def identify_profit_streaks(trades_df):
    """
    Identify periods of consecutive profitable trades.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        DataFrame with profit streak characteristics
    """
    # Group by seed to analyze each run
    profit_streaks = []
    
    for seed in trades_df['seed'].unique():
        seed_trades = trades_df[trades_df['seed'] == seed].sort_values('entry_time')
        
        # Calculate running P&L
        seed_trades['cumulative_pnl'] = seed_trades['profit'].cumsum()
        
        # Find periods of continuously increasing P&L
        in_streak = False
        streak_start_idx = None
        
        for i in range(1, len(seed_trades)):
            # If this trade is profitable
            if seed_trades.iloc[i]['profit'] > 0:
                # Start a new streak if we're not in one
                if not in_streak:
                    in_streak = True
                    streak_start_idx = i
            else:
                # End streak if we have one
                if in_streak:
                    streak_end_idx = i - 1
                    streak_trades = seed_trades.iloc[streak_start_idx:streak_end_idx + 1]
                    
                    # Only track streaks of 3+ trades
                    if len(streak_trades) >= 3:
                        profit_streaks.append({
                            'seed': seed,
                            'start_date': streak_trades['entry_time'].min(),
                            'end_date': streak_trades['exit_time'].max(),
                            'duration_days': (streak_trades['exit_time'].max() - streak_trades['entry_time'].min()).total_seconds() / (24*3600),
                            'num_trades': len(streak_trades),
                            'total_profit': streak_trades['profit'].sum(),
                            'avg_profit_per_trade': streak_trades['profit'].mean(),
                            'regime_scores': streak_trades['regime_score'].mean(),
                            'market_types': streak_trades['market_type'].value_counts().to_dict(),
                            'exit_reasons': streak_trades['exit_reason'].value_counts().to_dict()
                        })
                    
                    in_streak = False
        
        # Handle streak that goes to the end
        if in_streak:
            streak_end_idx = len(seed_trades) - 1
            streak_trades = seed_trades.iloc[streak_start_idx:streak_end_idx + 1]
            
            # Only track streaks of 3+ trades
            if len(streak_trades) >= 3:
                profit_streaks.append({
                    'seed': seed,
                    'start_date': streak_trades['entry_time'].min(),
                    'end_date': streak_trades['exit_time'].max(),
                    'duration_days': (streak_trades['exit_time'].max() - streak_trades['entry_time'].min()).total_seconds() / (24*3600),
                    'num_trades': len(streak_trades),
                    'total_profit': streak_trades['profit'].sum(),
                    'avg_profit_per_trade': streak_trades['profit'].mean(),
                    'regime_scores': streak_trades['regime_score'].mean(),
                    'market_types': streak_trades['market_type'].value_counts().to_dict(),
                    'exit_reasons': streak_trades['exit_reason'].value_counts().to_dict()
                })
    
    if not profit_streaks:
        return pd.DataFrame()
        
    # Convert to DataFrame and sort by total profit (best first)
    profit_streaks_df = pd.DataFrame(profit_streaks)
    profit_streaks_df = profit_streaks_df.sort_values('total_profit', ascending=False)
    
    return profit_streaks_df

def plot_success_patterns(trades_df, output_dir):
    """
    Create visualizations showing profitable trade patterns.
    
    Args:
        trades_df: DataFrame with trade data
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Profit distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(trades_df['profit'], bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Profits in Top-Performing Runs')
    plt.xlabel('Profit/Loss ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Profit by market type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='market_type', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Market Type in Top-Performing Runs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_by_market_type.png'), dpi=150)
    plt.close()
    
    # 3. Profit by exit reason
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='exit_reason', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Exit Reason in Top-Performing Runs')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_by_exit_reason.png'), dpi=150)
    plt.close()
    
    # 4. Profit by regime score range
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='regime_bin', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Regime Score in Top-Performing Runs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_by_regime.png'), dpi=150)
    plt.close()
    
    # 5. Correlation between metrics
    metrics = ['profit', 'regime_score', 'bars_held', 'atr', 'num_contracts', 'position_size_adj']
    available_metrics = [m for m in metrics if m in trades_df.columns]
    
    if len(available_metrics) >= 2:
        corr_matrix = trades_df[available_metrics].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Between Trade Metrics in Top-Performing Runs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=150)
        plt.close()
    
    # 6. Trade holding time analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(data=trades_df, x='holding_time', hue='trade_result', multiple='dodge', bins=20)
    plt.title('Trade Holding Time Distribution by Outcome')
    plt.xlabel('Holding Time (hours)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'holding_time_distribution.png'), dpi=150)
    plt.close()
    
    # 7. Scatter plot of regime score vs profit
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=trades_df, x='regime_score', y='profit', hue='market_type', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Regime Score vs Profit by Market Type')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'regime_score_vs_profit.png'), dpi=150)
    plt.close()
    
    # 8. Market type distribution in profitable runs
    market_counts = trades_df['market_type'].value_counts()
    plt.figure(figsize=(10, 6))
    market_counts.plot(kind='bar')
    plt.title('Market Type Distribution in Top-Performing Runs')
    plt.ylabel('Number of Trades')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(market_counts):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'market_type_distribution.png'), dpi=150)
    plt.close()

def compare_with_average_runs(top_trades_df, base_dir, output_dir):
    """
    Compare top-performing runs with average runs.
    
    Args:
        top_trades_df: DataFrame with trades from top runs
        base_dir: Directory with all robustness test results
        output_dir: Directory to save comparison results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results from all runs
    results_df = pd.read_csv(os.path.join(base_dir, "robustness_results.csv"))
    
    # Identify median runs for comparison
    top_seeds = top_trades_df['seed'].unique()
    median_profit = results_df['profit_loss'].median()
    
    # Find seeds with profit close to median
    profit_diff = abs(results_df['profit_loss'] - median_profit)
    median_seeds = results_df.loc[profit_diff.nsmallest(min(5, len(profit_diff))).index, 'seed'].tolist()
    
    # Load trades from median runs
    median_trades = []
    for seed_dir in glob.glob(os.path.join(base_dir, "seed_*")):
        seed = int(seed_dir.split('_')[-1])
        if seed in median_seeds:
            trade_log_path = os.path.join(seed_dir, 'trade_log.csv')
            if os.path.exists(trade_log_path):
                trades_df = pd.read_csv(trade_log_path)
                trades_df['seed'] = seed
                median_trades.append(trades_df)
    
    if not median_trades:
        print("No median trade logs found for comparison.")
        return
    
    # Combine median trades
    median_trades_df = pd.concat(median_trades, ignore_index=True)
    
    # Convert date columns
    for date_col in ['entry_time', 'exit_time']:
        median_trades_df[date_col] = pd.to_datetime(median_trades_df[date_col])
    
    # Add derived metrics
    median_trades_df['profit_pct'] = median_trades_df['profit'] / median_trades_df['entry_account_value'] * 100
    median_trades_df['holding_time'] = (median_trades_df['exit_time'] - median_trades_df['entry_time']).dt.total_seconds() / 3600
    median_trades_df['trade_result'] = np.where(median_trades_df['profit'] > 0, 'win', 'loss')
    
    # Add group labels
    top_trades_df['group'] = 'top'
    median_trades_df['group'] = 'median'
    
    # Combine for comparison
    combined_df = pd.concat([top_trades_df, median_trades_df], ignore_index=True)
    
    # Compare key metrics
    metrics_to_compare = [
        'profit', 'holding_time', 'regime_score', 
        'num_contracts', 'atr', 'rsi'
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        if metric in combined_df.columns and i < len(axes):
            sns.boxplot(x='group', y=metric, data=combined_df, ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].grid(True, alpha=0.3)
            
            # Perform t-test
            top_vals = top_trades_df[metric].dropna()
            median_vals = median_trades_df[metric].dropna()
            
            if len(top_vals) > 0 and len(median_vals) > 0:
                t_stat, p_val = stats.ttest_ind(top_vals, median_vals, equal_var=False)
                sig_text = f"p={p_val:.4f}" + (" *" if p_val < 0.05 else "")
                axes[i].annotate(sig_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                                ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_vs_median_comparison.png'), dpi=150)
    plt.close()
    
    # Compare market type distribution
    market_type_comp = pd.DataFrame({
        'top': top_trades_df['market_type'].value_counts(normalize=True) * 100,
        'median': median_trades_df['market_type'].value_counts(normalize=True) * 100
    }).fillna(0)
    
    plt.figure(figsize=(10, 6))
    market_type_comp.plot(kind='bar')
    plt.title('Market Type Distribution: Top vs Median Runs')
    plt.ylabel('Percentage of Trades')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'market_type_comparison.png'), dpi=150)
    plt.close()
    
    # Compare exit reason distribution
    exit_reason_comp = pd.DataFrame({
        'top': top_trades_df['exit_reason'].value_counts(normalize=True) * 100,
        'median': median_trades_df['exit_reason'].value_counts(normalize=True) * 100
    }).fillna(0)
    
    plt.figure(figsize=(12, 6))
    exit_reason_comp.plot(kind='bar')
    plt.title('Exit Reason Distribution: Top vs Median Runs')
    plt.ylabel('Percentage of Trades')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exit_reason_comparison.png'), dpi=150)
    plt.close()
    
    # Save comparison data
    comparison_summary = {
        'top_win_rate': (top_trades_df['profit'] > 0).mean() * 100,
        'median_win_rate': (median_trades_df['profit'] > 0).mean() * 100,
        'top_avg_profit': top_trades_df['profit'].mean(),
        'median_avg_profit': median_trades_df['profit'].mean(),
        'top_avg_position': top_trades_df['num_contracts'].mean() if 'num_contracts' in top_trades_df.columns else 0,
        'median_avg_position': median_trades_df['num_contracts'].mean() if 'num_contracts' in median_trades_df.columns else 0
    }
    
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(comparison_summary, f, indent=4)
    
    return comparison_summary

def create_summary_report(analysis_results, profit_streaks, comparison_results, output_path):
    """
    Create a text summary report of profitability analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        profit_streaks: DataFrame with best profit streaks
        comparison_results: Dictionary with comparison metrics
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("TOP PERFORMER ANALYSIS SUMMARY REPORT\n")
        f.write("===================================\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Win Rate: {analysis_results['win_rate']:.2f}%\n")
        f.write(f"Average Win: ${analysis_results['avg_win']:.2f}\n")
        f.write(f"Average Loss: ${analysis_results['avg_loss']:.2f}\n")
        f.write(f"Risk-Reward Ratio: {analysis_results['risk_reward']:.2f}\n\n")
        
        # Trade holding time
        duration = analysis_results['duration_analysis']
        f.write(f"Average Holding Time: {duration['avg_holding_time']:.2f} hours\n")
        f.write(f"Winning Trade Hold Time: {duration['winning_holding_time']:.2f} hours\n")
        f.write(f"Losing Trade Hold Time: {duration['losing_holding_time']:.2f} hours\n\n")
        
        # Position sizing
        position = analysis_results['position_analysis']
        f.write(f"Average Position Size: {position['avg_position_size']:.2f} contracts\n")
        f.write(f"Winning Position Size: {position['winning_position_size']:.2f} contracts\n")
        f.write(f"Losing Position Size: {position['losing_position_size']:.2f} contracts\n\n")
        
        f.write("MARKET TYPE ANALYSIS\n")
        f.write("-------------------\n")
        for market_type, metrics in analysis_results['market_type_analysis'].items():
            f.write(f"{market_type.upper()}:\n")
            f.write(f"  Count: {metrics['count']} trades ({metrics['pct_of_trades']:.2f}% of total)\n")
            f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
            f.write(f"  Avg Profit: ${metrics['avg_profit']:.2f}\n")
            f.write(f"  Avg Win: ${metrics['avg_win']:.2f}\n")
            f.write(f"  Avg Loss: ${metrics['avg_loss']:.2f}\n\n")
        
        f.write("EXIT REASON ANALYSIS\n")
        f.write("-------------------\n")
        for reason, metrics in analysis_results['exit_reason_analysis'].items():
            f.write(f"{reason.replace('_', ' ').title()}:\n")
            f.write(f"  Count: {metrics['count']} trades ({metrics['pct_of_trades']:.2f}% of total)\n")
            f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
            f.write(f"  Avg Profit: ${metrics['avg_profit']:.2f}\n\n")
        
        f.write("REGIME SCORE ANALYSIS\n")
        f.write("--------------------\n")
        for regime, metrics in analysis_results['regime_analysis'].items():
            if pd.isna(regime):
                continue
            f.write(f"Regime {regime}:\n")
            f.write(f"  Count: {metrics['count']} trades ({metrics['pct_of_trades']:.2f}% of total)\n")
            f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
            f.write(f"  Avg Profit: ${metrics['avg_profit']:.2f}\n\n")
        
        f.write("BEST PROFIT STREAKS\n")
        f.write("-----------------\n")
        if not profit_streaks.empty:
            for i, streak in profit_streaks.head(5).iterrows():
                f.write(f"Seed {streak['seed']}:\n")
                f.write(f"  Period: {streak['start_date'].strftime('%Y-%m-%d')} to {streak['end_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"  Duration: {streak['duration_days']:.1f} days\n")
                f.write(f"  Number of Trades: {streak['num_trades']}\n")
                f.write(f"  Total Profit: ${streak['total_profit']:.2f}\n")
                f.write(f"  Average Profit Per Trade: ${streak['avg_profit_per_trade']:.2f}\n")
                f.write(f"  Average Regime Score: {streak['regime_scores']:.2f}\n")
                f.write(f"  Market Types: {streak['market_types']}\n")
                f.write(f"  Exit Reasons: {streak['exit_reasons']}\n\n")
        else:
            f.write("No significant profit streaks identified.\n\n")
        
        f.write("COMPARISON WITH AVERAGE RUNS\n")
        f.write("---------------------------\n")
        if comparison_results:
            f.write(f"Win Rate: {comparison_results['top_win_rate']:.2f}% vs {comparison_results['median_win_rate']:.2f}% (median runs)\n")
            f.write(f"Avg Profit: ${comparison_results['top_avg_profit']:.2f} vs ${comparison_results['median_avg_profit']:.2f} (median runs)\n")
            f.write(f"Avg Position Size: {comparison_results['top_avg_position']:.2f} vs {comparison_results['median_avg_position']:.2f} (median runs)\n\n")
            
            win_rate_diff = comparison_results['top_win_rate'] - comparison_results['median_win_rate']
            profit_diff = comparison_results['top_avg_profit'] - comparison_results['median_avg_profit']
            
            f.write(f"Key differences:\n")
            f.write(f"  Win Rate: +{win_rate_diff:.2f}% in top runs\n")
            f.write(f"  Avg Profit: +${profit_diff:.2f} in top runs\n")
        else:
            f.write("Comparison data not available.\n\n")
        
        f.write("SUCCESS FACTORS\n")
        f.write("--------------\n")
        
        # Identify success factors
        success_factors = []
        
        # Check market type distribution
        for market_type, metrics in analysis_results['market_type_analysis'].items():
            if metrics['win_rate'] > 60:
                success_factors.append(f"High win rate in {market_type} market type ({metrics['win_rate']:.2f}%)")
        
        # Check regime scores
        for regime, metrics in analysis_results['regime_analysis'].items():
            if pd.isna(regime):
                continue
            if metrics['win_rate'] > 60:
                success_factors.append(f"High win rate in regime {regime} ({metrics['win_rate']:.2f}%)")
        
        # Check exit reasons
        for reason, metrics in analysis_results['exit_reason_analysis'].items():
            if reason in ['profit_target', 'trailing_stop'] and metrics['pct_of_trades'] > 40:
                success_factors.append(f"High percentage of {reason} exits ({metrics['pct_of_trades']:.2f}%)")
        
        # Check holding times
        if duration['winning_holding_time'] < duration['losing_holding_time']:
            success_factors.append(f"Shorter holding times for winning trades ({duration['winning_holding_time']:.2f} hours vs {duration['losing_holding_time']:.2f} hours)")
        
        # Generate success factor list
        if success_factors:
            f.write("Based on the analysis, the following success factors were identified:\n")
            for factor in success_factors:
                f.write(f"- {factor}\n")
            
            f.write("\nRecommended strategy enhancements:\n")
            
            # Market type recommendations
            market_recommendations = False
            for market_type, metrics in analysis_results['market_type_analysis'].items():
                if metrics['win_rate'] > 60:
                    market_recommendations = True
                    f.write(f"- For {market_type} markets: Increase position sizes when conditions are favorable\n")
            
            if not market_recommendations:
                f.write("- No clearly superior market type was identified\n")
            
            # Exit strategy recommendations
            for reason, metrics in analysis_results['exit_reason_analysis'].items():
                if reason == 'profit_target' and metrics['win_rate'] > 60:
                    f.write("- Profit targets: Consider slightly more aggressive targets\n")
                if reason == 'trailing_stop' and metrics['win_rate'] > 60:
                    f.write("- Trailing stops: Current settings work well, consider tightening slightly\n")
                if reason == 'stop_loss' and metrics['win_rate'] < 40:
                    f.write("- Stop losses: Consider widening stops to reduce false exits\n")
            
            # Regime recommendations
            high_regime_scores = []
            for regime, metrics in analysis_results['regime_analysis'].items():
                if pd.isna(regime):
                    continue
                if metrics['win_rate'] > 60:
                    high_regime_scores.append(regime)
            
            if high_regime_scores:
                regime_str = ", ".join(high_regime_scores)
                f.write(f"- Focus on high regime scores: {regime_str} produced the best results\n")
            
            # Position sizing recommendations
            if position['winning_position_size'] > position['losing_position_size']:
                f.write("- Consider more aggressive position sizing in optimal conditions\n")
            
        else:
            f.write("No clear success factors identified. The strategy shows consistent performance across different conditions.\n")
        
        f.write("\n\nAnalysis generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def main():
    """Main execution function for profitability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze profitable runs and identify success patterns")
    parser.add_argument('--input', type=str, required=True, help='Directory with robustness test results')
    parser.add_argument('--output', type=str, help='Directory to save analysis results')
    parser.add_argument('--percentile', type=int, default=25, help='Top percentile to analyze (default: 25)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"profit_analysis_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load trade data from top runs
    print(f"Loading trade data from top {args.percentile}% runs in {args.input}...")
    trades_df = load_profitable_runs(args.input, top_percentile=args.percentile)
    
    if trades_df is None:
        print("No trade data found. Exiting.")
        return
    
    print(f"Loaded {len(trades_df)} trades from {len(trades_df['seed'].unique())} top-performing runs")
    
    # Save combined trade data for external analysis
    trades_df.to_csv(os.path.join(output_dir, 'top_trades.csv'), index=False)
    
    # Analyze profitable patterns
    print("Analyzing profitable trade patterns...")
    analysis_results = analyze_profitable_patterns(trades_df)
    
    # Identify profit streaks
    print("Identifying best profit streaks...")
    profit_streaks = identify_profit_streaks(trades_df)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_success_patterns(trades_df, os.path.join(output_dir, 'visualizations'))
    
    # Compare with average runs
    print("Comparing with average runs...")
    comparison_results = compare_with_average_runs(trades_df, args.input, os.path.join(output_dir, 'comparisons'))
    
    # Create summary report
    report_path = os.path.join(output_dir, 'profit_analysis_report.txt')
    print(f"Creating summary report: {report_path}")
    create_summary_report(analysis_results, profit_streaks, comparison_results, report_path)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
