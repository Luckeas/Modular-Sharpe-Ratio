"""
run_multiple_backtests.py - Script to run multiple backtests with ML regime detection

This script runs the ML-enhanced backtester multiple times with different seeds,
then analyzes and visualizes the aggregate results.
"""

import os
import sys
import json
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import subprocess
from tqdm import tqdm
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the current Python executable path
PYTHON_EXECUTABLE = sys.executable

def run_backtest(seed, output_dir=None, strategy='ml'):
    """
    Run a single backtest with the specified seed.
    
    Args:
        seed: Random seed for the backtest
        output_dir: Directory to save results
        strategy: Strategy to use (ml, standard, or compare)
        
    Returns:
        Dictionary with backtest results
    """
    # Create command
    cmd = [PYTHON_EXECUTABLE, 'strategy_runner.py',
           '--strategy', strategy,
           '--seed', str(seed)]
    
    if output_dir:
        cmd.extend(['--output', os.path.join(output_dir, f'backtest_seed_{seed}')])
    
    # Run backtest
    logger.debug(f"Running backtest with seed {seed}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if successful
        if result.returncode != 0:
            logger.error(f"Backtest failed with seed {seed}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None
        
        # Extract metrics from output
        trades_count = None
        profit_loss = None
        win_rate = None
        max_drawdown = None
        sharpe_ratio = None
        
        # Look for metrics section
        metrics_section = False
        metrics_end = False
        
        for line in result.stdout.split('\n'):
            if "===== METRICS FOR SEED TESTING =====" in line:
                metrics_section = True
                continue
            
            if metrics_section and "===== END METRICS =====" in line:
                metrics_end = True
                break
            
            if metrics_section and not metrics_end:
                # Extract metrics
                if "Trades:" in line:
                    trades_count = int(line.split(':')[1].strip())
                elif "Profit/Loss:" in line:
                    profit_loss = float(line.split('$')[1].strip())
                elif "Win Rate:" in line:
                    win_rate = float(line.split(':')[1].strip().replace('%', ''))
                elif "Maximum Drawdown:" in line:
                    max_drawdown = float(line.split(':')[1].strip().replace('%', ''))
                elif "Sharpe Ratio:" in line:
                    sharpe_ratio = float(line.split(':')[1].strip())
        
        # Find output directory from stdout if not provided
        if not output_dir:
            for line in result.stdout.split('\n'):
                if "Results saved to:" in line:
                    output_directory = line.split(':', 1)[1].strip()
                    break
        
        # Return results
        return {
            'seed': seed,
            'strategy': strategy,
            'duration': duration,
            'trades_count': trades_count,
            'profit_loss': profit_loss,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'returncode': result.returncode,
            'output_dir': output_dir
        }
    
    except Exception as e:
        logger.error(f"Error running backtest with seed {seed}: {e}")
        return None

def run_backtest_wrapper(args):
    """Wrapper for parallel processing"""
    return run_backtest(*args)

def run_multiple_backtests(num_runs=100, parallel=True, strategy='ml', base_output_dir=None):
    """
    Run multiple backtests with different seeds.
    
    Args:
        num_runs: Number of backtests to run
        parallel: Whether to use parallel processing
        strategy: Strategy to use
        base_output_dir: Base directory for outputs
        
    Returns:
        List of backtest results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base_output_dir is None:
        base_output_dir = f"multiple_backtests_{strategy}_{timestamp}"
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Generate seeds
    seeds = random.sample(range(1, 100000), num_runs)
    
    # Run backtests
    if parallel and mp.cpu_count() > 1:
        # Use multiprocessing for parallel execution
        pool = mp.Pool(processes=min(mp.cpu_count(), 8))  # Limit to 8 processes max
        args = [(seed, base_output_dir, strategy) for seed in seeds]
        
        logger.info(f"Running {num_runs} backtests in parallel with {min(mp.cpu_count(), 8)} processes...")
        results = list(tqdm(pool.imap(run_backtest_wrapper, args), total=num_runs))
        pool.close()
        pool.join()
    else:
        # Run sequentially
        results = []
        for seed in tqdm(seeds, desc=f"Running {num_runs} {strategy} backtests"):
            result = run_backtest(seed, base_output_dir, strategy)
            if result:
                results.append(result)
    
    # Filter out failed runs
    results = [r for r in results if r is not None]
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(base_output_dir, f"backtest_results_{strategy}_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Completed {len(results)} backtests successfully")
    logger.info(f"Results saved to {results_path}")
    
    return results, results_df, base_output_dir

def analyze_results(results_df, output_dir):
    """
    Analyze backtest results and generate visualizations.
    
    Args:
        results_df: DataFrame with backtest results
        output_dir: Directory to save visualizations
    """
    if results_df.empty:
        logger.error("No results to analyze")
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Calculate summary statistics
    summary = {
        'num_runs': len(results_df),
        'profit_loss_mean': results_df['profit_loss'].mean(),
        'profit_loss_median': results_df['profit_loss'].median(),
        'profit_loss_std': results_df['profit_loss'].std(),
        'profit_loss_min': results_df['profit_loss'].min(),
        'profit_loss_max': results_df['profit_loss'].max(),
        'win_rate_mean': results_df['win_rate'].mean(),
        'win_rate_median': results_df['win_rate'].median(),
        'win_rate_std': results_df['win_rate'].std(),
        'trades_mean': results_df['trades_count'].mean(),
        'trades_median': results_df['trades_count'].median(),
        'trades_std': results_df['trades_count'].std(),
        'profitable_runs': (results_df['profit_loss'] > 0).sum(),
        'profitable_pct': (results_df['profit_loss'] > 0).mean() * 100
    }
    
    # Add max drawdown and Sharpe ratio stats if available
    if 'max_drawdown' in results_df.columns and not results_df['max_drawdown'].isnull().all():
        summary['max_drawdown_mean'] = results_df['max_drawdown'].mean()
        summary['max_drawdown_median'] = results_df['max_drawdown'].median()
        summary['max_drawdown_std'] = results_df['max_drawdown'].std()
    
    if 'sharpe_ratio' in results_df.columns and not results_df['sharpe_ratio'].isnull().all():
        summary['sharpe_ratio_mean'] = results_df['sharpe_ratio'].mean()
        summary['sharpe_ratio_median'] = results_df['sharpe_ratio'].median()
        summary['sharpe_ratio_std'] = results_df['sharpe_ratio'].std()
    
    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        summary[f'profit_loss_p{p}'] = results_df['profit_loss'].quantile(p/100)
    
    # Save summary to file
    with open(os.path.join(analysis_dir, "summary_statistics.txt"), "w") as f:
        f.write("BACKTEST RESULTS SUMMARY\n")
        f.write("=======================\n\n")
        f.write(f"Number of runs: {summary['num_runs']}\n")
        f.write(f"Percentage of profitable runs: {summary['profitable_pct']:.2f}%\n\n")
        
        f.write("PROFIT/LOSS STATISTICS\n")
        f.write("---------------------\n")
        f.write(f"Mean: ${summary['profit_loss_mean']:.2f}\n")
        f.write(f"Median: ${summary['profit_loss_median']:.2f}\n")
        f.write(f"Standard Deviation: ${summary['profit_loss_std']:.2f}\n")
        f.write(f"Minimum: ${summary['profit_loss_min']:.2f}\n")
        f.write(f"Maximum: ${summary['profit_loss_max']:.2f}\n\n")
        
        f.write("WIN RATE STATISTICS\n")
        f.write("------------------\n")
        f.write(f"Mean: {summary['win_rate_mean']:.2f}%\n")
        f.write(f"Median: {summary['win_rate_median']:.2f}%\n")
        f.write(f"Standard Deviation: {summary['win_rate_std']:.2f}%\n\n")
        
        f.write("TRADES STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"Mean: {summary['trades_mean']:.2f}\n")
        f.write(f"Median: {summary['trades_median']:.2f}\n")
        f.write(f"Standard Deviation: {summary['trades_std']:.2f}\n\n")
        
        if 'max_drawdown_mean' in summary:
            f.write("DRAWDOWN STATISTICS\n")
            f.write("-------------------\n")
            f.write(f"Mean: {summary['max_drawdown_mean']:.2f}%\n")
            f.write(f"Median: {summary['max_drawdown_median']:.2f}%\n")
            f.write(f"Standard Deviation: {summary['max_drawdown_std']:.2f}%\n\n")
        
        if 'sharpe_ratio_mean' in summary:
            f.write("SHARPE RATIO STATISTICS\n")
            f.write("----------------------\n")
            f.write(f"Mean: {summary['sharpe_ratio_mean']:.2f}\n")
            f.write(f"Median: {summary['sharpe_ratio_median']:.2f}\n")
            f.write(f"Standard Deviation: {summary['sharpe_ratio_std']:.2f}\n\n")
        
        f.write("PERCENTILES (PROFIT/LOSS)\n")
        f.write("------------------------\n")
        for p in percentiles:
            f.write(f"{p}th percentile: ${summary[f'profit_loss_p{p}']:.2f}\n")
    
    # Create visualizations
    create_visualizations(results_df, analysis_dir)
    
    logger.info(f"Analysis complete. Results saved to {analysis_dir}")
    return summary

def create_visualizations(results_df, analysis_dir):
    """
    Create visualizations of backtest results.
    
    Args:
        results_df: DataFrame with backtest results
        analysis_dir: Directory to save visualizations
    """
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Profit/Loss Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['profit_loss'], kde=True, bins=30)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Distribution of Profit/Loss Across All Runs')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    # Add text with percentage of profitable runs
    profitable_pct = (results_df['profit_loss'] > 0).mean() * 100
    plt.text(0.05, 0.95, f"Profitable runs: {profitable_pct:.2f}%", 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(os.path.join(analysis_dir, "profit_loss_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Win Rate Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['win_rate'], kde=True, bins=20)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7)
    plt.title('Distribution of Win Rates Across All Runs')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, "win_rate_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Trades Count Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['trades_count'], kde=True, bins=20)
    plt.title('Distribution of Number of Trades Across All Runs')
    plt.xlabel('Number of Trades')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, "trades_distribution.png"), dpi=150)
    plt.close()
    
    # 4. Profit vs Win Rate Scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='win_rate', y='profit_loss', 
                    hue='trades_count', size='trades_count',
                    palette='viridis', data=results_df)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7)
    plt.title('Profit/Loss vs Win Rate')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Profit/Loss ($)')
    plt.savefig(os.path.join(analysis_dir, "profit_vs_winrate.png"), dpi=150)
    plt.close()
    
    # 5. Sharpe Ratio Distribution (if available)
    if 'sharpe_ratio' in results_df.columns and not results_df['sharpe_ratio'].isnull().all():
        plt.figure(figsize=(12, 6))
        sns.histplot(results_df['sharpe_ratio'], kde=True, bins=20)
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        plt.title('Distribution of Sharpe Ratios Across All Runs')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(analysis_dir, "sharpe_ratio_distribution.png"), dpi=150)
        plt.close()
    
    # 6. Max Drawdown Distribution (if available)
    if 'max_drawdown' in results_df.columns and not results_df['max_drawdown'].isnull().all():
        plt.figure(figsize=(12, 6))
        sns.histplot(results_df['max_drawdown'], kde=True, bins=20)
        plt.title('Distribution of Maximum Drawdowns Across All Runs')
        plt.xlabel('Maximum Drawdown (%)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(analysis_dir, "max_drawdown_distribution.png"), dpi=150)
        plt.close()
    
    # 7. Profit/Loss Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = [results_df['profit_loss'].quantile(p/100) for p in percentiles]
    
    plt.figure(figsize=(12, 6))
    plt.bar(percentiles, percentile_values, color='skyblue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Profit/Loss Percentiles')
    plt.xlabel('Percentile')
    plt.ylabel('Profit/Loss ($)')
    for i, v in enumerate(percentile_values):
        plt.text(i, v + (1 if v >= 0 else -1), f"${v:.2f}", ha='center')
    plt.savefig(os.path.join(analysis_dir, "profit_loss_percentiles.png"), dpi=150)
    plt.close()
    
    # 8. Profit vs Trade Count (to look for correlation)
    plt.figure(figsize=(12, 6))
    sns.regplot(x='trades_count', y='profit_loss', data=results_df, scatter_kws={'alpha':0.5})
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Profit/Loss vs Number of Trades')
    plt.xlabel('Number of Trades')
    plt.ylabel('Profit/Loss ($)')
    plt.savefig(os.path.join(analysis_dir, "profit_vs_trades.png"), dpi=150)
    plt.close()

def compare_strategies(num_runs=50, parallel=True, output_dir=None):
    """
    Run and compare both the standard and ML-enhanced strategies.
    
    Args:
        num_runs: Number of runs for each strategy
        parallel: Whether to use parallel processing
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary with comparison results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"strategy_comparison_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run standard strategy
    logger.info(f"Running {num_runs} backtests with standard strategy...")
    std_results, std_df, std_dir = run_multiple_backtests(
        num_runs=num_runs, 
        parallel=parallel, 
        strategy='standard',
        base_output_dir=os.path.join(output_dir, 'standard')
    )
    
    # Run ML-enhanced strategy
    logger.info(f"Running {num_runs} backtests with ML-enhanced strategy...")
    ml_results, ml_df, ml_dir = run_multiple_backtests(
        num_runs=num_runs, 
        parallel=parallel, 
        strategy='ml',
        base_output_dir=os.path.join(output_dir, 'ml')
    )
    
    # Analyze each strategy
    std_summary = analyze_results(std_df, std_dir)
    ml_summary = analyze_results(ml_df, ml_dir)
    
    # Create comparison visualizations
    create_comparison_visualizations(std_df, ml_df, output_dir)
    
    # Save comparison summary
    comparison = {
        'standard': std_summary,
        'ml': ml_summary,
        'improvement': {
            'profit_mean': ml_summary['profit_loss_mean'] - std_summary['profit_loss_mean'],
            'profit_median': ml_summary['profit_loss_median'] - std_summary['profit_loss_median'],
            'win_rate_mean': ml_summary['win_rate_mean'] - std_summary['win_rate_mean'],
            'profitable_pct': ml_summary['profitable_pct'] - std_summary['profitable_pct']
        }
    }
    
    # Calculate percent improvement for key metrics
    if std_summary['profit_loss_mean'] != 0:
        comparison['improvement']['profit_mean_pct'] = (ml_summary['profit_loss_mean'] - std_summary['profit_loss_mean']) / abs(std_summary['profit_loss_mean']) * 100
    else:
        comparison['improvement']['profit_mean_pct'] = float('inf')
        
    if std_summary['win_rate_mean'] != 0:
        comparison['improvement']['win_rate_mean_pct'] = (ml_summary['win_rate_mean'] - std_summary['win_rate_mean']) / std_summary['win_rate_mean'] * 100
    else:
        comparison['improvement']['win_rate_mean_pct'] = float('inf')
    
    # Save comparison to file
    with open(os.path.join(output_dir, "strategy_comparison.txt"), "w") as f:
        f.write("STRATEGY COMPARISON: STANDARD vs ML-ENHANCED\n")
        f.write("==========================================\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-----------------\n")
        f.write(f"{'Metric':<25} {'Standard':<15} {'ML-Enhanced':<15} {'Difference':<15} {'% Change':<10}\n")
        f.write("-" * 70 + "\n")
        
        metrics = [
            ('Profitable Runs (%)', 'profitable_pct', '%'),
            ('Mean Profit/Loss', 'profit_loss_mean', '$'),
            ('Median Profit/Loss', 'profit_loss_median', '$'),
            ('Mean Win Rate', 'win_rate_mean', '%'),
            ('Mean Trades Count', 'trades_mean', '')
        ]
        
        for label, key, unit in metrics:
            std_val = std_summary[key]
            ml_val = ml_summary[key]
            diff = ml_val - std_val
            
            if abs(std_val) > 0:
                pct_change = diff / abs(std_val) * 100
                pct_str = f"{pct_change:+.2f}%"
            else:
                pct_str = "N/A"
                
            if unit == '$':
                f.write(f"{label:<25} {unit}{std_val:<14.2f} {unit}{ml_val:<14.2f} {unit}{diff:<14.2f} {pct_str:<10}\n")
            else:
                f.write(f"{label:<25} {std_val:<14.2f}{unit} {ml_val:<14.2f}{unit} {diff:<14.2f}{unit} {pct_str:<10}\n")
        
        f.write("\n")
        f.write("PERCENTILE COMPARISON (PROFIT/LOSS)\n")
        f.write("---------------------------------\n")
        f.write(f"{'Percentile':<15} {'Standard':<15} {'ML-Enhanced':<15} {'Difference':<15}\n")
        f.write("-" * 60 + "\n")
        
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            std_key = f'profit_loss_p{p}'
            std_p = std_summary[std_key]
            ml_p = ml_summary[std_key]
            diff = ml_p - std_p
            f.write(f"{p}th percentile:          ${std_p:<14.2f} ${ml_p:<14.2f} ${diff:<14.2f}\n")
        f.write("\n")
        f.write("CONCLUSION\n")
        f.write("----------\n")
        if ml_summary['profit_loss_mean'] > std_summary['profit_loss_mean'] and ml_summary['profitable_pct'] > std_summary['profitable_pct']:
            f.write("The ML-enhanced strategy shows significant improvement over the standard strategy.\n")
        elif ml_summary['profit_loss_mean'] > std_summary['profit_loss_mean']:
            f.write("The ML-enhanced strategy shows improvement in average profit but not in consistency.\n")
        elif ml_summary['profitable_pct'] > std_summary['profitable_pct']:
            f.write("The ML-enhanced strategy is more consistent but doesn't improve average profit.\n")
        else:
            f.write("The ML-enhanced strategy doesn't show clear improvement over the standard strategy.\n")
    
    logger.info(f"Comparison complete. Results saved to {output_dir}")
    return comparison


def create_comparison_visualizations(std_df, ml_df, output_dir):
    """
    Create visualizations comparing standard and ML-enhanced strategies.

    Args:
        std_df: DataFrame with standard strategy results
        ml_df: DataFrame with ML-enhanced strategy results
        output_dir: Directory to save visualizations
    """
    # Set style
    sns.set_style("whitegrid")

    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "comparison_plots")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # 1. Profit/Loss Distribution Comparison
    plt.figure(figsize=(12, 6))
    sns.histplot(std_df['profit_loss'], kde=True, color='blue', alpha=0.5, label='Standard', bins=20)
    sns.histplot(ml_df['profit_loss'], kde=True, color='green', alpha=0.5, label='ML-Enhanced', bins=20)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title('Profit/Loss Distribution: Standard vs ML-Enhanced')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.legend()

    # Add text with percentage of profitable runs
    std_profitable = (std_df['profit_loss'] > 0).mean() * 100
    ml_profitable = (ml_df['profit_loss'] > 0).mean() * 100
    diff = ml_profitable - std_profitable
    plt.text(0.05, 0.95, f"Profitable runs (Standard): {std_profitable:.2f}%",
             transform=plt.gca().transAxes, fontsize=10, color='blue',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.90, f"Profitable runs (ML): {ml_profitable:.2f}%",
             transform=plt.gca().transAxes, fontsize=10, color='green',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.85, f"Difference: {diff:+.2f}%",
             transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig(os.path.join(analysis_dir, "profit_loss_comparison.png"), dpi=150)
    plt.close()

    # 2. Win Rate Distribution Comparison
    plt.figure(figsize=(12, 6))
    sns.histplot(std_df['win_rate'], kde=True, color='blue', alpha=0.5, label='Standard', bins=20)
    sns.histplot(ml_df['win_rate'], kde=True, color='green', alpha=0.5, label='ML-Enhanced', bins=20)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7)
    plt.title('Win Rate Distribution: Standard vs ML-Enhanced')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, "win_rate_comparison.png"), dpi=150)
    plt.close()

    # 3. Box Plots for Key Metrics
    metrics = ['profit_loss', 'win_rate', 'trades_count']
    labels = ['Profit/Loss ($)', 'Win Rate (%)', 'Number of Trades']

    for metric, label in zip(metrics, labels):
        plt.figure(figsize=(8, 6))

        # Prepare data for box plot
        data = [std_df[metric], ml_df[metric]]

        # Create box plot
        box = plt.boxplot(data, patch_artist=True, labels=['Standard', 'ML-Enhanced'])

        # Set colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add actual data points as jittered scatter
        for i, d in enumerate(data):
            # Add jitter
            x = np.random.normal(i + 1, 0.04, size=len(d))
            plt.scatter(x, d, alpha=0.3, s=10)

        # Add horizontal line at 0 or 50 for reference
        if metric == 'profit_loss':
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        elif metric == 'win_rate':
            plt.axhline(y=50, color='red', linestyle='--', alpha=0.7)

        plt.title(f'{label}: Standard vs ML-Enhanced')
        plt.ylabel(label)
        plt.grid(axis='y', alpha=0.3)

        # Add means as text
        mean_std = std_df[metric].mean()
        mean_ml = ml_df[metric].mean()
        diff = mean_ml - mean_std
        diff_pct = diff / abs(mean_std) * 100 if mean_std != 0 else float('inf')

        if metric == 'profit_loss':
            unit = '$'
            plt.text(0.02, 0.95, f"Standard mean: {unit}{mean_std:.2f}",
                     transform=plt.gca().transAxes, fontsize=9, color='blue')
            plt.text(0.02, 0.90, f"ML mean: {unit}{mean_ml:.2f}",
                     transform=plt.gca().transAxes, fontsize=9, color='green')
            plt.text(0.02, 0.85, f"Diff: {unit}{diff:.2f} ({diff_pct:+.1f}%)",
                     transform=plt.gca().transAxes, fontsize=9, fontweight='bold')
        else:
            unit = '%' if metric == 'win_rate' else ''
            plt.text(0.02, 0.95, f"Standard mean: {mean_std:.2f}{unit}",
                     transform=plt.gca().transAxes, fontsize=9, color='blue')
            plt.text(0.02, 0.90, f"ML mean: {mean_ml:.2f}{unit}",
                     transform=plt.gca().transAxes, fontsize=9, color='green')
            plt.text(0.02, 0.85, f"Diff: {diff:.2f}{unit} ({diff_pct:+.1f}%)",
                     transform=plt.gca().transAxes, fontsize=9, fontweight='bold')

        plt.savefig(os.path.join(analysis_dir, f"{metric}_boxplot_comparison.png"), dpi=150)
        plt.close()

    # 4. Cumulative Distribution Function Comparison
    plt.figure(figsize=(12, 6))

    # Sort values for CDF
    std_profit_sorted = np.sort(std_df['profit_loss'])
    ml_profit_sorted = np.sort(ml_df['profit_loss'])

    # Calculate CDF
    std_cdf = np.arange(1, len(std_profit_sorted) + 1) / len(std_profit_sorted)
    ml_cdf = np.arange(1, len(ml_profit_sorted) + 1) / len(ml_profit_sorted)

    # Plot CDF
    plt.plot(std_profit_sorted, std_cdf, label='Standard', color='blue', linewidth=2)
    plt.plot(ml_profit_sorted, ml_cdf, label='ML-Enhanced', color='green', linewidth=2)

    # Add vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # Mark the value at 0 profit
    if len(std_profit_sorted) > 0:
        std_zero_idx = (np.abs(std_profit_sorted)).argmin()
        std_zero_cdf = std_cdf[std_zero_idx]
        plt.plot([0], [std_zero_cdf], 'o', color='blue')
        plt.text(10, std_zero_cdf, f"{(1 - std_zero_cdf) * 100:.1f}% profitable", color='blue')

    if len(ml_profit_sorted) > 0:
        ml_zero_idx = (np.abs(ml_profit_sorted)).argmin()
        ml_zero_cdf = ml_cdf[ml_zero_idx]
        plt.plot([0], [ml_zero_cdf], 'o', color='green')
        plt.text(10, ml_zero_cdf, f"{(1 - ml_zero_cdf) * 100:.1f}% profitable", color='green')

    plt.title('Cumulative Distribution of Profit/Loss: Standard vs ML-Enhanced')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, "profit_cdf_comparison.png"), dpi=150)
    plt.close()


def main():
    """Main function to run multiple backtests and analyze results"""
    import argparse

    parser = argparse.ArgumentParser(description='Run multiple backtests with different seeds')
    parser.add_argument('--runs', type=int, default=100, help='Number of backtest runs')
    parser.add_argument('--strategy', type=str, choices=['ml', 'standard', 'compare'], default='ml',
                        help='Strategy to test (ml, standard, or compare)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--output', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    # HARDCODE PARALLEL TO TRUE REGARDLESS OF COMMAND LINE
    use_parallel = True  # Force parallel processing

    logger.info(f"Starting multiple backtests with {args.runs} runs of {args.strategy} strategy")
    logger.info(f"Parallel processing: {'enabled' if use_parallel else 'disabled'}")

    # Create progress bar for main operation
    if args.strategy == 'compare':
        # Run comparison (this will run both strategies)
        comparison = compare_strategies(
            num_runs=args.runs,
            parallel=use_parallel,  # Use our hardcoded value
            output_dir=args.output
        )

        # Print summary comparison
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON SUMMARY: STANDARD vs ML-ENHANCED")
        print("=" * 80)

        std_mean = comparison['standard']['profit_loss_mean']
        ml_mean = comparison['ml']['profit_loss_mean']
        diff = ml_mean - std_mean
        pct_diff = diff / abs(std_mean) * 100 if std_mean != 0 else float('inf')

        print(f"Mean Profit/Loss:")
        print(f"  Standard:    ${std_mean:.2f}")
        print(f"  ML-Enhanced: ${ml_mean:.2f}")
        print(f"  Difference:  ${diff:.2f} ({pct_diff:+.1f}%)")

        std_profitable = comparison['standard']['profitable_pct']
        ml_profitable = comparison['ml']['profitable_pct']
        diff = ml_profitable - std_profitable

        print(f"\nProfitable Runs:")
        print(f"  Standard:    {std_profitable:.2f}%")
        print(f"  ML-Enhanced: {ml_profitable:.2f}%")
        print(f"  Difference:  {diff:+.2f}%")

        if ml_mean > std_mean and ml_profitable > std_profitable:
            conclusion = "ML-ENHANCED STRATEGY SHOWS SIGNIFICANT IMPROVEMENT"
        elif ml_mean > std_mean:
            conclusion = "ML-ENHANCED STRATEGY SHOWS HIGHER AVERAGE PROFIT BUT SIMILAR CONSISTENCY"
        elif ml_profitable > std_profitable:
            conclusion = "ML-ENHANCED STRATEGY SHOWS BETTER CONSISTENCY BUT SIMILAR AVERAGE PROFIT"
        else:
            conclusion = "ML-ENHANCED STRATEGY DOESN'T SHOW CLEAR IMPROVEMENT"

        print("\n" + "=" * 80)
        print(conclusion)
        print("=" * 80 + "\n")

    else:
        # Run single strategy multiple times
        results, results_df, output_dir = run_multiple_backtests(
            num_runs=args.runs,
            parallel=use_parallel,  # Use our hardcoded value
            strategy=args.strategy,
            base_output_dir=args.output
        )

        # Analyze results
        summary = analyze_results(results_df, output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print(f"MULTIPLE BACKTEST SUMMARY: {args.strategy.upper()} STRATEGY")
        print("=" * 80)
        print(f"Number of successful runs: {len(results_df)}/{args.runs}")
        print(f"Profitable runs: {summary['profitable_pct']:.2f}%")
        print(f"Mean profit/loss: ${summary['profit_loss_mean']:.2f}")
        print(f"Mean win rate: {summary['win_rate_mean']:.2f}%")

        # Print percentile information
        print("\nProfit/Loss Percentiles:")
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            print(f"  {p}th percentile: ${summary[f'profit_loss_p{p}']:.2f}")

        print("\n" + "=" * 80)

        # Determine if strategy is robust
        if summary['profitable_pct'] > 80 and summary['profit_loss_p5'] > 0:
            print("STRATEGY IS HIGHLY ROBUST (>80% profitable runs, positive 5th percentile)")
        elif summary['profitable_pct'] > 70:
            print("STRATEGY IS MODERATELY ROBUST (>70% profitable runs)")
        elif summary['profitable_pct'] > 60:
            print("STRATEGY IS SOMEWHAT ROBUST (>60% profitable runs)")
        else:
            print("STRATEGY SHOWS LIMITED ROBUSTNESS (<60% profitable runs)")
        print("=" * 80 + "\n")

    logger.info("Multiple backtest analysis complete.")


if __name__ == "__main__":
    main()
