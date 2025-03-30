"""
analyze_losses.py - Script to analyze losing trades and strategy failure modes

This script helps identify patterns in losing trades by analyzing trade data 
from multiple backtest runs, focusing on worst-performing scenarios.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

def load_trade_logs(base_dir, only_negative=False):
    """
    Load trade logs from multiple robustness test runs.
    
    Args:
        base_dir: Directory containing robustness test results
        only_negative: If True, only load runs with negative P&L
        
    Returns:
        DataFrame with combined trade data
    """
    # Find all seed directories
    seed_dirs = glob.glob(os.path.join(base_dir, "seed_*"))
    
    # Get results CSV to identify losing runs
    results_df = pd.read_csv(os.path.join(base_dir, "robustness_results.csv"))
    
    if only_negative:
        # Filter to negative P&L runs
        negative_seeds = results_df[results_df['profit_loss'] < 0]['seed'].tolist()
        seed_dirs = [d for d in seed_dirs if int(d.split('_')[-1]) in negative_seeds]
        print(f"Analyzing {len(seed_dirs)} runs with negative P&L")
    
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

def analyze_losing_trades(trades_df):
    """
    Analyze characteristics of losing trades compared to winning trades.
    
    Args:
        trades_df: DataFrame with trade data
        
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
            'avg_loss': mt_trades[mt_trades['profit'] <= 0]['profit'].mean() if len(mt_trades[mt_trades['profit'] <= 0]) > 0 else 0
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
        regime_trades = trades_df[trades_df['regime_bin'] == regime_bin]
        regime_win_rate = len(regime_trades[regime_trades['profit'] > 0]) / len(regime_trades) * 100
        
        regime_analysis[regime_bin] = {
            'count': len(regime_trades),
            'win_rate': regime_win_rate,
            'avg_profit': regime_trades['profit'].mean(),
            'pct_of_trades': len(regime_trades) / len(trades_df) * 100
        }
    
    # Return combined analysis
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward': risk_reward,
        'market_type_analysis': market_type_analysis,
        'exit_reason_analysis': exit_reason_analysis,
        'regime_analysis': regime_analysis
    }

def identify_worst_case_scenarios(trades_df):
    """
    Identify market conditions that lead to the worst losses.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        DataFrame with worst loss streaks and their characteristics
    """
    # Group by seed to analyze each run
    loss_streaks = []
    
    for seed in trades_df['seed'].unique():
        seed_trades = trades_df[trades_df['seed'] == seed].sort_values('entry_time')
        
        # Calculate running P&L and drawdown
        seed_trades['cumulative_pnl'] = seed_trades['profit'].cumsum()
        seed_trades['drawdown'] = seed_trades['cumulative_pnl'] - seed_trades['cumulative_pnl'].cummax()
        
        # Find worst drawdown
        worst_dd_idx = seed_trades['drawdown'].idxmin()
        if pd.isna(worst_dd_idx):
            continue
            
        worst_dd = seed_trades.loc[worst_dd_idx, 'drawdown']
        
        # Find the start of this drawdown period
        peak_idx = seed_trades[seed_trades['entry_time'] <= seed_trades.loc[worst_dd_idx, 'entry_time']]['cumulative_pnl'].idxmax()
        
        if pd.isna(peak_idx):
            continue
            
        # Extract the drawdown period
        dd_period = seed_trades.loc[peak_idx:worst_dd_idx]
        
        if len(dd_period) < 2:
            continue
        
        # Analyze this period
        loss_streaks.append({
            'seed': seed,
            'start_date': dd_period['entry_time'].min(),
            'end_date': dd_period['exit_time'].max(),
            'duration_days': (dd_period['exit_time'].max() - dd_period['entry_time'].min()).total_seconds() / (24*3600),
            'num_trades': len(dd_period),
            'total_loss': worst_dd,
            'avg_loss_per_trade': worst_dd / len(dd_period),
            'regime_scores': dd_period['regime_score'].mean(),
            'market_types': dd_period['market_type'].value_counts().to_dict(),
            'exit_reasons': dd_period['exit_reason'].value_counts().to_dict()
        })
    
    if not loss_streaks:
        return pd.DataFrame()
        
    # Convert to DataFrame and sort by total loss (worst first)
    loss_streaks_df = pd.DataFrame(loss_streaks)
    loss_streaks_df = loss_streaks_df.sort_values('total_loss')
    
    return loss_streaks_df

def plot_profit_distribution(trades_df, output_dir):
    """
    Create visualizations showing the distribution of profits and losses.
    
    Args:
        trades_df: DataFrame with trade data
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Overall profit distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(trades_df['profit'], bins=50, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Trade Profits/Losses')
    plt.xlabel('Profit/Loss ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_distribution.png'), dpi=150)
    plt.close()
    
    # 2. Profit by market type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='market_type', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Market Type')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_by_market_type.png'), dpi=150)
    plt.close()
    
    # 3. Profit by exit reason
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='exit_reason', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Exit Reason')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_by_exit_reason.png'), dpi=150)
    plt.close()
    
    # 4. Profit by regime score range
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='regime_bin', y='profit', data=trades_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Profit/Loss by Regime Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profit_by_regime.png'), dpi=150)
    plt.close()
    
    # 5. Correlation between metrics
    metrics = ['profit', 'regime_score', 'bars_held', 'atr', 'volume', 'ADX', 'RSI']
    available_metrics = [m for m in metrics if m in trades_df.columns]
    
    if len(available_metrics) >= 2:
        corr_matrix = trades_df[available_metrics].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Between Trade Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=150)
        plt.close()

def compare_winning_losing_runs(trades_df, output_dir):
    """
    Compare characteristics between winning and losing backtest runs.
    
    Args:
        trades_df: DataFrame with trade data from multiple runs
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group by seed to get run-level metrics
    run_metrics = []
    
    for seed in trades_df['seed'].unique():
        seed_trades = trades_df[trades_df['seed'] == seed]
        
        if 'run_profit_loss' not in seed_trades.columns:
            continue
            
        # Get run-level metrics (already in the dataframe)
        run_metrics.append({
            'seed': seed,
            'profit_loss': seed_trades['run_profit_loss'].iloc[0],
            'win_rate': seed_trades['run_win_rate'].iloc[0],
            'sharpe_ratio': seed_trades['run_sharpe'].iloc[0],
            'max_drawdown': seed_trades['run_max_dd'].iloc[0],
            'num_trades': len(seed_trades),
            # Add more detailed metrics
            'avg_profit': seed_trades['profit'].mean(),
            'median_profit': seed_trades['profit'].median(),
            'profit_std': seed_trades['profit'].std(),
            'largest_win': seed_trades['profit'].max(),
            'largest_loss': seed_trades['profit'].min(),
            'pct_trend_following': (seed_trades['market_type'] == 'trend_following').mean() * 100,
            'pct_mean_reverting': (seed_trades['market_type'] == 'mean_reverting').mean() * 100,
            'pct_profit_target': (seed_trades['exit_reason'] == 'profit_target').mean() * 100,
            'pct_stop_loss': (seed_trades['exit_reason'] == 'stop_loss').mean() * 100,
            'avg_regime_score': seed_trades['regime_score'].mean()
        })
    
    run_metrics_df = pd.DataFrame(run_metrics)
    
    # Categorize runs
    run_metrics_df['run_category'] = np.where(run_metrics_df['profit_loss'] > 0, 'profitable', 'unprofitable')
    
    # Save to CSV
    run_metrics_df.to_csv(os.path.join(output_dir, 'run_metrics.csv'), index=False)
    
    # Create visualizations comparing run categories
    for metric in ['win_rate', 'avg_profit', 'num_trades', 'pct_trend_following', 
                   'pct_profit_target', 'pct_stop_loss', 'avg_regime_score']:
        if metric in run_metrics_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='run_category', y=metric, data=run_metrics_df)
            plt.title(f'{metric} by Run Outcome')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{metric}_by_outcome.png'), dpi=150)
            plt.close()
    
    # Calculate t-test for key metrics to find significant differences
    from scipy import stats
    
    profitable = run_metrics_df[run_metrics_df['run_category'] == 'profitable']
    unprofitable = run_metrics_df[run_metrics_df['run_category'] == 'unprofitable']
    
    significant_differences = []
    
    for metric in run_metrics_df.columns:
        if metric in ['seed', 'profit_loss', 'run_category']:
            continue
            
        if metric in profitable.columns and metric in unprofitable.columns:
            try:
                t_stat, p_value = stats.ttest_ind(
                    profitable[metric].dropna(), 
                    unprofitable[metric].dropna(), 
                    equal_var=False
                )
                
                if p_value < 0.05:
                    significant_differences.append({
                        'metric': metric,
                        'profitable_mean': profitable[metric].mean(),
                        'unprofitable_mean': unprofitable[metric].mean(),
                        'difference': profitable[metric].mean() - unprofitable[metric].mean(),
                        'p_value': p_value
                    })
            except:
                pass
    
    # Save significant differences
    if significant_differences:
        sig_df = pd.DataFrame(significant_differences)
        sig_df.to_csv(os.path.join(output_dir, 'significant_differences.csv'), index=False)
        
        # Create a summary visualization
        if len(sig_df) > 0:
            plt.figure(figsize=(12, 6))
            metrics = sig_df['metric'].values
            diffs = sig_df['difference'].values
            
            colors = ['g' if x > 0 else 'r' for x in diffs]
            plt.bar(metrics, diffs, color=colors)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title('Significant Differences Between Profitable and Unprofitable Runs')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'significant_differences.png'), dpi=150)
            plt.close()

def create_summary_report(analysis_results, loss_streaks, output_path):
    """
    Create a text summary report of loss analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
        loss_streaks: DataFrame with worst loss streaks
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("LOSS ANALYSIS SUMMARY REPORT\n")
        f.write("===========================\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Win Rate: {analysis_results['win_rate']:.2f}%\n")
        f.write(f"Average Win: ${analysis_results['avg_win']:.2f}\n")
        f.write(f"Average Loss: ${analysis_results['avg_loss']:.2f}\n")
        f.write(f"Risk-Reward Ratio: {analysis_results['risk_reward']:.2f}\n\n")
        
        f.write("MARKET TYPE ANALYSIS\n")
        f.write("-------------------\n")
        for market_type, metrics in analysis_results['market_type_analysis'].items():
            f.write(f"{market_type.upper()}:\n")
            f.write(f"  Count: {metrics['count']} trades\n")
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
        
        f.write("WORST LOSS STREAKS\n")
        f.write("----------------\n")
        if not loss_streaks.empty:
            for i, streak in loss_streaks.head(5).iterrows():
                f.write(f"Seed {streak['seed']}:\n")
                f.write(f"  Period: {streak['start_date'].strftime('%Y-%m-%d')} to {streak['end_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"  Duration: {streak['duration_days']:.1f} days\n")
                f.write(f"  Number of Trades: {streak['num_trades']}\n")
                f.write(f"  Total Loss: ${streak['total_loss']:.2f}\n")
                f.write(f"  Average Loss Per Trade: ${streak['avg_loss_per_trade']:.2f}\n")
                f.write(f"  Average Regime Score: {streak['regime_scores']:.2f}\n")
                f.write(f"  Market Types: {streak['market_types']}\n")
                f.write(f"  Exit Reasons: {streak['exit_reasons']}\n\n")
        else:
            f.write("No significant loss streaks identified.\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("--------------\n")
        
        # Identify issues and make recommendations
        issues = []
        
        # Check win rate across market types
        for market_type, metrics in analysis_results['market_type_analysis'].items():
            if metrics['win_rate'] < 45:
                issues.append(f"Low win rate in {market_type} market type ({metrics['win_rate']:.2f}%)")
        
        # Check exit reasons
        for reason, metrics in analysis_results['exit_reason_analysis'].items():
            if reason in ['stop_loss', 'time_exit'] and metrics['pct_of_trades'] > 30:
                issues.append(f"High percentage of {reason} exits ({metrics['pct_of_trades']:.2f}%)")
                
            if reason in ['stop_loss'] and metrics['win_rate'] < 30:
                issues.append(f"Very low win rate for {reason} exits ({metrics['win_rate']:.2f}%)")
        
        # Check regime scores
        low_regime_win_rates = False
        for regime, metrics in analysis_results['regime_analysis'].items():
            if pd.isna(regime):
                continue
            if metrics['win_rate'] < 45:
                low_regime_win_rates = True
                issues.append(f"Low win rate in regime {regime} ({metrics['win_rate']:.2f}%)")
        
        # Generate recommendations based on identified issues
        if issues:
            f.write("Based on the analysis, the following issues were identified:\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            
            f.write("\nRecommended improvements:\n")
            
            # Market type recommendations
            market_recommendations = False
            for market_type, metrics in analysis_results['market_type_analysis'].items():
                if metrics['win_rate'] < 45:
                    market_recommendations = True
                    f.write(f"- For {market_type} markets: Consider increasing filter stringency or avoiding trading\n")
            
            if not market_recommendations:
                f.write("- Market type detection appears effective; maintain current approach\n")
            
            # Exit strategy recommendations
            if any(metrics['pct_of_trades'] > 30 for reason, metrics in analysis_results['exit_reason_analysis'].items() if reason == 'stop_loss'):
                f.write("- Adjust stop loss placement: Consider wider stops or volatility-based positioning\n")
                
            if any(metrics['pct_of_trades'] > 30 for reason, metrics in analysis_results['exit_reason_analysis'].items() if reason == 'time_exit'):
                f.write("- Review time-based exits: Consider extending holding periods or implementing trailing stops\n")
            
            # Regime recommendations
            if low_regime_win_rates:
                f.write("- Regime filters need refinement: Consider more strict filters for lower regime scores\n")
            
            # Position sizing recommendations
            f.write("- Implement more conservative position sizing during detected high-risk periods\n")
            
        else:
            f.write("No major issues identified. The strategy appears to have a good balance of risk and reward.\n")
            f.write("Continue monitoring performance and consider minor parameter adjustments as market conditions evolve.\n")
        
        f.write("\n\nAnalysis generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def main():
    """Main execution function for loss analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trading losses and identify failure patterns")
    parser.add_argument('--input', type=str, required=True, help='Directory with robustness test results')
    parser.add_argument('--output', type=str, help='Directory to save analysis results')
    parser.add_argument('--negative-only', action='store_true', help='Only analyze runs with negative P&L')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"loss_analysis_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load trade data
    print(f"Loading trade data from {args.input}...")
    trades_df = load_trade_logs(args.input, only_negative=args.negative_only)
    
    if trades_df is None:
        print("No trade data found. Exiting.")
        return
    
    print(f"Loaded {len(trades_df)} trades from {len(trades_df['seed'].unique())} different runs")
    
    # Save combined trade data for external analysis
    trades_df.to_csv(os.path.join(output_dir, 'all_trades.csv'), index=False)
    
    # Analyze losing trades
    print("Analyzing trade performance...")
    analysis_results = analyze_losing_trades(trades_df)
    
    # Identify worst loss streaks
    print("Identifying worst losing streaks...")
    loss_streaks = identify_worst_case_scenarios(trades_df)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_profit_distribution(trades_df, os.path.join(output_dir, 'visualizations'))
    
    # Compare profitable vs unprofitable runs
    print("Comparing winning and losing runs...")
    compare_winning_losing_runs(trades_df, os.path.join(output_dir, 'comparisons'))
    
    # Create summary report
    report_path = os.path.join(output_dir, 'loss_analysis_report.txt')
    print(f"Creating summary report: {report_path}")
    create_summary_report(analysis_results, loss_streaks, report_path)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
