"""
test_sharpe_robustness.py - Run multiple backtests with optimized parameters

This script tests the robustness of the Sharpe ratio optimization by running
multiple backtests with different random seeds and analyzing the distribution
of performance metrics.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from multiprocessing import Pool, cpu_count

# Import from central config
from config import config

# Import utility functions
from utils import load_and_process_data, calculate_indicators, initialize_random_seeds

# Import backtester
from unified_backtester import run_backtest

# Import analysis functions
from trade_analysis import analyze_performance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_backtest(params):
    """
    Run a single backtest with the specified parameters.
    
    Args:
        params: Dictionary with backtest parameters
        
    Returns:
        Dictionary with backtest results
    """
    # Unpack parameters
    test_params = params['test_params']
    param_set = params['param_set']
    seed = params['seed']
    output_dir = params['output_dir']
    use_ml = params['use_ml']
    
    # Load original config
    from config import config
    
    # Set the seed
    config['global']['random_seed'] = seed
    config['global']['use_fixed_seed'] = True
    initialize_random_seeds(seed)
    
    # Update config with optimized parameters
    from parameter_optimizer import update_config_with_params
    for param, value in param_set.items():
        update_config_with_params(config, {param: value})
    
    # Load and process data
    df = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )
    
    if df is None or len(df) == 0:
        logger.error(f"No data available for seed {seed}. Skipping.")
        return None
    
    # Calculate indicators
    df = calculate_indicators(df, config)
    
    # Create output directory for this test
    test_dir = os.path.join(output_dir, f"seed_{seed}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Set up file paths
    file_paths = {
        'trade_log': os.path.join(test_dir, 'trade_log.csv'),
        'portfolio_value': os.path.join(test_dir, 'portfolio_value.csv'),
        'regime_log': os.path.join(test_dir, 'regime_log.csv'),
        'market_type_log': os.path.join(test_dir, 'market_type_log.csv'),
        'summary': os.path.join(test_dir, 'summary.txt')
    }
    
    # Run backtest
    try:
        # Reset HMM detector
        from backtester_common import reset_hmm_detector
        reset_hmm_detector()
        
        # Run the backtest
        trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
            df.copy(),
            visualize_trades=False,
            file_paths=file_paths,
            use_ml=use_ml
        )
        
        # Calculate performance metrics
        if trades and len(portfolio_values) > 0:
            portfolio_series = pd.Series(portfolio_values, index=df['date'][:len(portfolio_values)])
            metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
            
            # Add seed to metrics
            metrics['seed'] = seed
            metrics['number_of_trades'] = len(trades)
            
            # Calculate additional metrics
            if len(trades) > 0:
                # Calculate average trade duration
                durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in trades]
                metrics['avg_trade_duration_hours'] = np.mean(durations)
                
                # Calculate percentage of profitable days
                if len(portfolio_series) > 1:
                    daily_returns = portfolio_series.resample('D').last().pct_change().dropna()
                    metrics['profitable_days_pct'] = (daily_returns > 0).mean() * 100
            
            # Log quick summary
            logger.info(f"Seed {seed}: Sharpe={metrics['sharpe_ratio']:.3f}, P/L=${metrics['profit_loss']:.2f}, "
                       f"WR={metrics['win_rate']:.1f}%, DD={metrics['max_drawdown_pct']:.1f}%")
            
            return metrics
        else:
            logger.warning(f"No trades executed for seed {seed}")
            return {
                'seed': seed,
                'sharpe_ratio': 0,
                'profit_loss': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'max_drawdown_pct': 0,
                'number_of_trades': 0
            }
    except Exception as e:
        logger.error(f"Error in backtest for seed {seed}: {e}")
        return {
            'seed': seed,
            'error': str(e),
            'sharpe_ratio': 0,
            'profit_loss': 0,
            'total_return_pct': 0,
            'win_rate': 0,
            'max_drawdown_pct': 0,
            'number_of_trades': 0
        }

def analyze_robustness_results(results, output_dir):
    """
    Analyze the results of multiple backtest runs.
    
    Args:
        results: List of dictionaries with backtest results
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with summary statistics
    """
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Remove any runs with errors
    valid_results = results_df[~results_df['sharpe_ratio'].isnull()]
    
    if len(valid_results) == 0:
        logger.error("No valid backtest results to analyze")
        return None
    
    # Calculate summary statistics
    summary = {
        'count': len(valid_results),
        'sharpe_ratio_mean': valid_results['sharpe_ratio'].mean(),
        'sharpe_ratio_median': valid_results['sharpe_ratio'].median(),
        'sharpe_ratio_std': valid_results['sharpe_ratio'].std(),
        'sharpe_ratio_min': valid_results['sharpe_ratio'].min(),
        'sharpe_ratio_max': valid_results['sharpe_ratio'].max(),
        'profit_loss_mean': valid_results['profit_loss'].mean(),
        'profit_loss_median': valid_results['profit_loss'].median(),
        'profit_loss_std': valid_results['profit_loss'].std(),
        'win_rate_mean': valid_results['win_rate'].mean(),
        'win_rate_std': valid_results['win_rate'].std(),
        'max_drawdown_mean': valid_results['max_drawdown_pct'].mean(),
        'max_drawdown_std': valid_results['max_drawdown_pct'].std(),
        'trades_mean': valid_results['number_of_trades'].mean(),
        'trades_std': valid_results['number_of_trades'].std(),
        'profitable_runs': (valid_results['profit_loss'] > 0).sum(),
        'profitable_pct': (valid_results['profit_loss'] > 0).mean() * 100
    }
    
    # Calculate percentiles for key metrics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        summary[f'sharpe_ratio_p{p}'] = valid_results['sharpe_ratio'].quantile(p/100)
        summary[f'profit_loss_p{p}'] = valid_results['profit_loss'].quantile(p/100)
    
    # Save results to CSV
    valid_results.to_csv(os.path.join(output_dir, 'robustness_results.csv'), index=False)
    
    # Create summary report
    with open(os.path.join(output_dir, 'robustness_summary.txt'), 'w') as f:
        f.write("SHARPE RATIO OPTIMIZATION ROBUSTNESS ANALYSIS\n")
        f.write("==========================================\n\n")
        
        f.write(f"Number of valid runs: {summary['count']}\n")
        f.write(f"Percentage of profitable runs: {summary['profitable_pct']:.2f}%\n\n")
        
        f.write("SHARPE RATIO STATISTICS\n")
        f.write("----------------------\n")
        f.write(f"Mean: {summary['sharpe_ratio_mean']:.4f}\n")
        f.write(f"Median: {summary['sharpe_ratio_median']:.4f}\n")
        f.write(f"Standard Deviation: {summary['sharpe_ratio_std']:.4f}\n")
        f.write(f"Minimum: {summary['sharpe_ratio_min']:.4f}\n")
        f.write(f"Maximum: {summary['sharpe_ratio_max']:.4f}\n\n")
        
        f.write("PROFIT/LOSS STATISTICS\n")
        f.write("---------------------\n")
        f.write(f"Mean: ${summary['profit_loss_mean']:.2f}\n")
        f.write(f"Median: ${summary['profit_loss_median']:.2f}\n")
        f.write(f"Standard Deviation: ${summary['profit_loss_std']:.2f}\n\n")
        
        f.write("OTHER METRICS\n")
        f.write("------------\n")
        f.write(f"Mean Win Rate: {summary['win_rate_mean']:.2f}%\n")
        f.write(f"Mean Max Drawdown: {summary['max_drawdown_mean']:.2f}%\n")
        f.write(f"Mean Trade Count: {summary['trades_mean']:.1f}\n\n")
        
        f.write("PERCENTILE ANALYSIS (SHARPE RATIO)\n")
        f.write("--------------------------------\n")
        for p in percentiles:
            f.write(f"{p}th percentile: {summary[f'sharpe_ratio_p{p}']:.4f}\n")
            
        f.write("\nPERCENTILE ANALYSIS (PROFIT/LOSS)\n")
        f.write("-------------------------------\n")
        for p in percentiles:
            f.write(f"{p}th percentile: ${summary[f'profit_loss_p{p}']:.2f}\n")
    
    # Generate visualizations
    try:
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Sharpe Ratio Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_results['sharpe_ratio'], kde=True, bins=20)
        plt.axvline(x=summary['sharpe_ratio_mean'], color='red', linestyle='--', 
                   label=f"Mean: {summary['sharpe_ratio_mean']:.4f}")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Sharpe Ratio Distribution Across Random Seeds')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sharpe_ratio_distribution.png'), dpi=150)
        plt.close()
        
        # 2. Profit/Loss Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_results['profit_loss'], kde=True, bins=20)
        plt.axvline(x=summary['profit_loss_mean'], color='red', linestyle='--', 
                   label=f"Mean: ${summary['profit_loss_mean']:.2f}")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Profit/Loss Distribution Across Random Seeds')
        plt.xlabel('Profit/Loss ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'profit_loss_distribution.png'), dpi=150)
        plt.close()
        
        # 3. Win Rate Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_results['win_rate'], kde=True, bins=20)
        plt.axvline(x=summary['win_rate_mean'], color='red', linestyle='--', 
                   label=f"Mean: {summary['win_rate_mean']:.2f}%")
        plt.axvline(x=50, color='black', linestyle='-', alpha=0.3, label='50% Threshold')
        plt.title('Win Rate Distribution Across Random Seeds')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'win_rate_distribution.png'), dpi=150)
        plt.close()
        
        # 4. Drawdown Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_results['max_drawdown_pct'], kde=True, bins=20)
        plt.axvline(x=summary['max_drawdown_mean'], color='red', linestyle='--', 
                   label=f"Mean: {summary['max_drawdown_mean']:.2f}%")
        plt.title('Maximum Drawdown Distribution Across Random Seeds')
        plt.xlabel('Maximum Drawdown (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'max_drawdown_distribution.png'), dpi=150)
        plt.close()
        
        # 5. Scatter plot: Sharpe vs Profit
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_results['sharpe_ratio'], valid_results['profit_loss'], alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Sharpe Ratio vs. Profit/Loss')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Profit/Loss ($)')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sharpe_vs_profit.png'), dpi=150)
        plt.close()
        
        # 6. Scatter plot: Sharpe vs Drawdown
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_results['sharpe_ratio'], valid_results['max_drawdown_pct'], alpha=0.7)
        plt.title('Sharpe Ratio vs. Maximum Drawdown')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Maximum Drawdown (%)')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'sharpe_vs_drawdown.png'), dpi=150)
        plt.close()
        
        # 7. Trade Count vs. Sharpe Ratio
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_results['number_of_trades'], valid_results['sharpe_ratio'], alpha=0.7)
        plt.title('Number of Trades vs. Sharpe Ratio')
        plt.xlabel('Number of Trades')
        plt.ylabel('Sharpe Ratio')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'trades_vs_sharpe.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    return summary

def run_robustness_test(param_set, num_runs=50, parallel=True, use_ml=False):
    """
    Run multiple backtests with different seeds to test robustness.
    
    Args:
        param_set: Dictionary with optimized parameters
        num_runs: Number of backtest runs with different seeds
        parallel: Whether to use parallel processing
        use_ml: Whether to use ML enhancement
        
    Returns:
        Dictionary with robustness analysis results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sharpe_robustness_test_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save parameter set
    with open(os.path.join(output_dir, 'optimized_parameters.json'), 'w') as f:
        json.dump(param_set, f, indent=4)
    
    # Generate seeds
    seeds = list(range(1, num_runs + 1))
    
    # Create test parameters
    test_params = {
        'output_dir': output_dir,
        'param_set': param_set,
        'use_ml': use_ml
    }
    
    # Prepare tasks for parallel processing
    tasks = [{'test_params': test_params, 'param_set': param_set, 'seed': seed, 
             'output_dir': output_dir, 'use_ml': use_ml} for seed in seeds]
    
    logger.info(f"Starting robustness test with {num_runs} runs")
    
    # Run backtests
    results = []
    if parallel and cpu_count() > 1:
        # Use multiprocessing for parallel execution
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = pool.map(run_single_backtest, tasks)
    else:
        # Run sequentially
        for task in tasks:
            result = run_single_backtest(task)
            if result:
                results.append(result)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Analyze results
    summary = analyze_robustness_results(results, output_dir)
    
    if summary:
        logger.info("\n===== ROBUSTNESS TEST RESULTS =====")
        logger.info(f"Number of valid runs: {summary['count']}")
        logger.info(f"Sharpe Ratio: Mean={summary['sharpe_ratio_mean']:.4f}, Std={summary['sharpe_ratio_std']:.4f}")
        logger.info(f"Profit/Loss: Mean=${summary['profit_loss_mean']:.2f}, Std=${summary['profit_loss_std']:.2f}")
        logger.info(f"Win Rate: Mean={summary['win_rate_mean']:.2f}%, Std={summary['win_rate_std']:.2f}%")
        logger.info(f"Max Drawdown: Mean={summary['max_drawdown_mean']:.2f}%, Std={summary['max_drawdown_std']:.2f}%")
        logger.info(f"Profitable Runs: {summary['profitable_pct']:.2f}%")
    
    logger.info(f"Robustness test complete. Results saved to {output_dir}")
    return {'summary': summary, 'output_dir': output_dir}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Sharpe ratio robustness test')
    parser.add_argument('--params', type=str, required=True, help='JSON file with optimized parameters')
    parser.add_argument('--runs', type=int, default=50, help='Number of backtest runs')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancement')
    
    args = parser.parse_args()
    
    # Load optimized parameters
    try:
        with open(args.params, 'r') as f:
            param_set = json.load(f)
            
        # If param_set has a nested structure, extract the parameters
        if isinstance(param_set, dict) and 'params' in param_set:
            param_set = param_set['params']
        elif isinstance(param_set, dict) and 'best_params' in param_set:
            param_set = param_set['best_params']
    except Exception as e:
        logger.error(f"Error loading parameters file: {e}")
        exit(1)
    
    # Run robustness test
    run_robustness_test(param_set, num_runs=args.runs, parallel=args.parallel, use_ml=args.ml)
