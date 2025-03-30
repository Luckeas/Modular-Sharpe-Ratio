"""
simplified_compare_strategies.py - Compare ML-Enhanced vs. Original Strategy

This script runs both the original and ML-enhanced strategy backtests
on the same data, then compares their performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

# Import from centralized config
from config import config

# Import utility functions
from utils import (
    setup_directories, copy_project_files, load_and_process_data, 
    calculate_indicators
)

# Import trade analysis
from trade_analysis import (
    analyze_performance, analyze_quarterly_performance, analyze_by_regime, 
    compare_strategies, analyze_by_season
)

# Import backtesters
from unified_backtester import run_backtest

# Import visualization functions
from trade_visualization import generate_comparison_plots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_comparison(data_file=None, start_date=None, end_date=None, plots_dir=None):
    """
    Run both the original strategy and ML-enhanced strategy for comparison.

    Args:
        data_file: Path to input data file (overrides config)
        start_date: Start date string YYYY-MM-DD (overrides config)
        end_date: End date string YYYY-MM-DD (overrides config)
        plots_dir: Directory to save comparison plots

    Returns:
        Dictionary of comparison results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'strategy_comparison_{timestamp}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If plots_dir is not provided, create a default inside the output directory
    if plots_dir is None:
        plots_dir = os.path.join(output_dir, 'comparison_plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

    # Copy project files
    copy_project_files(output_dir)

    # Create file paths
    date_range_str = f"{config['data']['start_date'].split('-')[0]}{config['data']['start_date'].split('-')[1]}_{config['data']['end_date'].split('-')[0]}{config['data']['end_date'].split('-')[1]}"

    file_paths = {
        'trade_log': os.path.join(output_dir, f'trade_log_mes_5min_{date_range_str}.csv'),
        'portfolio_value': os.path.join(output_dir, f'portfolio_value_mes_5min_{date_range_str}.csv'),
        'regime_log': os.path.join(output_dir, f'market_regime_log_{date_range_str}.csv'),
        'market_type_log': os.path.join(output_dir, f'market_type_log_{date_range_str}.csv'),
        'summary': os.path.join(output_dir, f'summary_{date_range_str}_{timestamp}.txt')
    }

    # Override config with parameters if provided
    if data_file:
        config['data']['file_path'] = data_file

    if start_date:
        config['data']['start_date'] = start_date

    if end_date:
        config['data']['end_date'] = end_date

    logger.info(f"Running strategy comparison from {config['data']['start_date']} to {config['data']['end_date']}")

    # Load and prepare data
    df_5min = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    if df_5min is None or len(df_5min) == 0:
        logger.error("No data available after loading. Exiting.")
        return None

    # Calculate all indicators
    df_5min = calculate_indicators(df_5min, config)

    # Make a copy for each backtest to avoid cross-contamination
    df_original = df_5min.copy()
    df_ml = df_5min.copy()

    # Run original strategy backtest
    logger.info("Running original strategy backtest...")
    orig_trades, orig_portfolio_values, _, orig_regime_log, orig_market_type_log, _, orig_season_metrics, _ = run_backtest(
        df_original, visualize_trades=False, file_paths=file_paths, use_ml=False
    )

    # Create portfolio series for analysis
    orig_portfolio_df = pd.DataFrame({'date': df_original['date'], 'value': orig_portfolio_values})
    orig_portfolio_series = orig_portfolio_df.set_index('date')['value']

    # Analyze original strategy results
    orig_results = analyze_performance(orig_trades, orig_portfolio_series, config['account']['initial_capital'])

    # Create ML file paths
    ml_file_paths = {
        'trade_log': os.path.join(output_dir, f'ml_trade_log_mes_5min_{date_range_str}.csv'),
        'portfolio_value': os.path.join(output_dir, f'ml_portfolio_value_mes_5min_{date_range_str}.csv'),
        'regime_log': os.path.join(output_dir, f'ml_market_regime_log_{date_range_str}.csv'),
        'market_type_log': os.path.join(output_dir, f'ml_market_type_log_{date_range_str}.csv'),
        'summary': os.path.join(output_dir, f'ml_summary_{date_range_str}_{timestamp}.txt')
    }

    # Force ML setting for this run
    original_ml_enable = config['ml']['enable']
    config['ml']['enable'] = True

    # Run ML-enhanced strategy backtest
    logger.info("Running ML-enhanced strategy backtest...")
    ml_trades, ml_portfolio_values, _, _, _, _, ml_season_metrics, ml_metrics = run_backtest(
        df_ml, visualize_trades=False, file_paths=ml_file_paths, use_ml=True
    )

    # Restore original ML setting
    config['ml']['enable'] = original_ml_enable

    # Create portfolio series for analysis
    ml_portfolio_df = pd.DataFrame({'date': df_ml['date'], 'value': ml_portfolio_values})
    ml_portfolio_series = ml_portfolio_df.set_index('date')['value']
    
    # Analyze ML-enhanced strategy results
    ml_results = analyze_performance(ml_trades, ml_portfolio_series, config['account']['initial_capital'])
    
    # Analyze quarterly performance
    orig_quarterly_df = analyze_quarterly_performance(
        orig_trades, orig_portfolio_series, config['account']['initial_capital']
    )
    
    ml_quarterly_df = analyze_quarterly_performance(
        ml_trades, ml_portfolio_series, config['account']['initial_capital']
    )
    
    # Compare results
    comparison = compare_strategies(orig_results, ml_results, orig_trades, ml_trades)
    
    # Add ML metrics to comparison
    comparison['ml_metrics'] = ml_metrics
    
    # Add season metrics to comparison
    comparison['season_metrics'] = {
        'original': orig_season_metrics,
        'ml_enhanced': ml_season_metrics
    }
    
    # Generate comparison plots
    generate_comparison_plots(orig_portfolio_series, ml_portfolio_series, 
                             orig_trades, ml_trades, plots_dir)
    
    # Save detailed comparison report
    save_comparison_report(comparison, output_dir)
    
    # Print summary to console
    print_comparison_summary(comparison)
    
    return comparison

def generate_season_comparison_plots(orig_metrics, ml_metrics, plots_dir):
    """
    Generate plots comparing season performance between original and ML strategies.
    
    Args:
        orig_metrics: Season metrics for original strategy
        ml_metrics: Season metrics for ML strategy
        plots_dir: Directory to save plots
    """
    if not orig_metrics or not ml_metrics:
        return
    
    # Collect all unique seasons
    all_seasons = sorted(set(list(orig_metrics.keys()) + list(ml_metrics.keys())))
    
    # Prepare data for each metric
    metrics = {
        'win_rate': {'title': 'Win Rate Comparison by Season', 'ylabel': 'Win Rate (%)'},
        'avg_profit': {'title': 'Average Profit Comparison by Season', 'ylabel': 'Average Profit ($)'},
        'total_profit': {'title': 'Total Profit Comparison by Season', 'ylabel': 'Total Profit ($)'},
        'trade_count': {'title': 'Trade Count Comparison by Season', 'ylabel': 'Number of Trades'}
    }
    
    # Create a plot for each metric
    for metric_name, info in metrics.items():
        plt.figure(figsize=(12, 6))
        
        # Set width and positions for bars
        width = 0.35
        x = np.arange(len(all_seasons))
        
        # Collect data with proper handling of missing seasons
        orig_values = [orig_metrics.get(season, {}).get(metric_name, 0) for season in all_seasons]
        ml_values = [ml_metrics.get(season, {}).get(metric_name, 0) for season in all_seasons]
        
        # Create the grouped bar plot
        plt.bar(x - width/2, orig_values, width, label='Original Strategy', color='blue', alpha=0.7)
        plt.bar(x + width/2, ml_values, width, label='ML-Enhanced Strategy', color='green', alpha=0.7)
        
        # Add labels and legend
        plt.ylabel(info['ylabel'])
        plt.title(info['title'])
        plt.xticks(x, all_seasons)
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        
        # Save the plot
        plot_file = os.path.join(plots_dir, f'season_comparison_{metric_name}.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
    logger.info(f"Season comparison plots saved to {plots_dir}")

def save_comparison_report(comparison, output_dir):
    """
    Save a detailed comparison report.
    
    Args:
        comparison: Dictionary of comparison results
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'strategy_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("           STRATEGY COMPARISON REPORT: ORIGINAL VS. ML-ENHANCED\n")
        f.write("=" * 80 + "\n\n")
        
        # Main performance metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<20} {'Original':<15} {'ML-Enhanced':<15} {'Difference':<15} {'% Change':<10}\n")
        f.write("-" * 80 + "\n")
        
        metrics = [
            ('Total Trades', 'total_trades', ''),
            ('Win Rate', 'win_rate', '%'),
            ('Net Profit/Loss', 'profit_loss', '$'),
            ('Total Return', 'total_return_pct', '%'),
            ('Avg. Profit/Trade', 'avg_profit', '$'),
            ('Profit Factor', 'profit_factor', ''),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown_pct', '%')
        ]
        
        for label, key, unit in metrics:
            orig_val = comparison['original'][key]
            ml_val = comparison['ml_enhanced'][key]
            diff = comparison['difference'][key]
            
            # Format percent change for applicable metrics
            if key in comparison['percent_change']:
                pct_change = comparison['percent_change'][key]
                pct_str = f"{pct_change:.2f}%" if pct_change != float('inf') else "N/A"
            else:
                pct_str = "-"
            
            # Format the values based on type
            if isinstance(orig_val, float):
                f.write(f"{label:<20} {orig_val:.2f}{unit:<10} {ml_val:.2f}{unit:<10} {diff:.2f}{unit:<10} {pct_str:<10}\n")
            else:
                f.write(f"{label:<20} {orig_val}{unit:<10} {ml_val}{unit:<10} {diff}{unit:<10} {pct_str:<10}\n")
        
        f.write("\n")
        
        # Trade overlap analysis
        f.write("TRADE OVERLAP ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Common trades between strategies: {comparison['trade_overlap']['common_trades']}\n")
        f.write(f"Trades unique to original strategy: {comparison['trade_overlap']['orig_only']}\n")
        f.write(f"Trades unique to ML-enhanced strategy: {comparison['trade_overlap']['ml_only']}\n")
        f.write(f"Trade overlap percentage: {comparison['trade_overlap']['overlap_pct']:.2f}%\n\n")
        
        # ML metrics
        f.write("ML-SPECIFIC METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total potential trade setups: {comparison['ml_metrics']['potential_trades']}\n")
        f.write(f"Trades executed after ML filtering: {comparison['ml_metrics']['executed_trades']}\n")
        f.write(f"Trades rejected by ML: {comparison['ml_metrics']['skipped_by_ml']}\n")
        
        if 'prediction_accuracy' in comparison['ml_metrics']:
            f.write(f"ML prediction accuracy: {comparison['ml_metrics']['prediction_accuracy']*100:.2f}%\n")
        
        f.write(f"Model retraining count: {comparison['ml_metrics']['model_retrain_count']}\n\n")
        
        # HMM integration details if available
        if 'hmm_enabled' in comparison['ml_metrics'] and comparison['ml_metrics']['hmm_enabled']:
            f.write("HMM INTEGRATION METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"HMM influence weight: {comparison['ml_metrics']['hmm_influence']:.2f}\n")
            f.write(f"HMM-ML agreement rate: {comparison['ml_metrics'].get('hmm_ml_agreement', 0):.2f}%\n")
            
            # Add HMM state distribution if available
            if 'hmm_metrics' in comparison['ml_metrics'] and comparison['ml_metrics']['hmm_metrics']:
                hmm_metrics = comparison['ml_metrics']['hmm_metrics']
                
                if 'state_frequencies' in hmm_metrics:
                    f.write("\nHMM State Distribution:\n")
                    for state, freq in hmm_metrics['state_frequencies'].items():
                        f.write(f"  {state}: {freq*100:.2f}%\n")
                
                f.write("\n")
                
        # Conclusion
        f.write("CONCLUSION\n")
        f.write("-" * 80 + "\n")
        
        # Determine if ML improved the strategy
        profit_improved = comparison['difference']['profit_loss'] > 0
        win_rate_improved = comparison['difference']['win_rate'] > 0
        drawdown_improved = comparison['difference']['max_drawdown_pct'] < 0
        sharpe_improved = comparison['difference']['sharpe_ratio'] > 0
        
        improvements = [profit_improved, win_rate_improved, drawdown_improved, sharpe_improved]
        improvement_score = sum(improvements) / len(improvements)
        
        if improvement_score >= 0.75:
            conclusion = "The ML-enhanced strategy shows significant improvement over the original strategy."
        elif improvement_score >= 0.5:
            conclusion = "The ML-enhanced strategy shows moderate improvement over the original strategy."
        elif improvement_score >= 0.25:
            conclusion = "The ML-enhanced strategy shows slight improvement over the original strategy."
        else:
            conclusion = "The ML-enhanced strategy does not show clear improvement over the original strategy."
        
        f.write(f"{conclusion}\n\n")
        
        # Key observations
        f.write("Key observations:\n")
        if profit_improved:
            f.write(f"- Profit increased by ${comparison['difference']['profit_loss']:.2f} ({comparison['percent_change']['profit_loss']:.2f}%)\n")
        else:
            f.write(f"- Profit decreased by ${-comparison['difference']['profit_loss']:.2f} ({-comparison['percent_change']['profit_loss']:.2f}%)\n")
            
        if win_rate_improved:
            f.write(f"- Win rate improved by {comparison['difference']['win_rate']:.2f} percentage points\n")
        else:
            f.write(f"- Win rate decreased by {-comparison['difference']['win_rate']:.2f} percentage points\n")
            
        if drawdown_improved:
            f.write(f"- Maximum drawdown reduced by {-comparison['difference']['max_drawdown_pct']:.2f} percentage points\n")
        else:
            f.write(f"- Maximum drawdown increased by {comparison['difference']['max_drawdown_pct']:.2f} percentage points\n")
            
        # Trade frequency observation
        trade_reduction = comparison['original']['total_trades'] - comparison['ml_enhanced']['total_trades']
        if trade_reduction > 0:
            f.write(f"- ML filtering reduced the number of trades by {trade_reduction} ({trade_reduction/comparison['original']['total_trades']*100:.2f}%)\n")
        else:
            f.write(f"- ML filtering did not reduce the number of trades\n")
        
        f.write("\n")
        f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
    logger.info(f"Comparison report saved to {report_path}")

def print_comparison_summary(comparison):
    """Print a summary of the comparison results to the console."""
    print("\n" + "=" * 80)
    print("                  STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nMetric                Original        ML-Enhanced    Difference")
    print("-" * 80)
    print(f"Total Trades         {comparison['original']['total_trades']:<15d} {comparison['ml_enhanced']['total_trades']:<15d} {comparison['difference']['total_trades']}")
    print(f"Win Rate (%)         {comparison['original']['win_rate']:<15.2f} {comparison['ml_enhanced']['win_rate']:<15.2f} {comparison['difference']['win_rate']:.2f}")
    print(f"Net Profit/Loss ($)  {comparison['original']['profit_loss']:<15.2f} {comparison['ml_enhanced']['profit_loss']:<15.2f} {comparison['difference']['profit_loss']:.2f}")
    print(f"Total Return (%)     {comparison['original']['total_return_pct']:<15.2f} {comparison['ml_enhanced']['total_return_pct']:<15.2f} {comparison['difference']['total_return_pct']:.2f}")
    print(f"Avg Profit/Trade ($) {comparison['original']['avg_profit']:<15.2f} {comparison['ml_enhanced']['avg_profit']:<15.2f} {comparison['difference']['avg_profit']:.2f}")
    print(f"Profit Factor        {comparison['original']['profit_factor']:<15.2f} {comparison['ml_enhanced']['profit_factor']:<15.2f} {comparison['difference']['profit_factor']:.2f}")
    print(f"Sharpe Ratio         {comparison['original']['sharpe_ratio']:<15.2f} {comparison['ml_enhanced']['sharpe_ratio']:<15.2f} {comparison['difference']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown (%)     {comparison['original']['max_drawdown_pct']:<15.2f} {comparison['ml_enhanced']['max_drawdown_pct']:<15.2f} {comparison['difference']['max_drawdown_pct']:.2f}")
    
    print("\nML METRICS:")
    print(f"Potential trades identified: {comparison['ml_metrics']['potential_trades']}")
    print(f"Trades taken after ML filtering: {comparison['ml_metrics']['executed_trades']}")
    print(f"Trades rejected by ML: {comparison['ml_metrics']['skipped_by_ml']}")
    
    if 'prediction_accuracy' in comparison['ml_metrics']:
        print(f"ML prediction accuracy: {comparison['ml_metrics']['prediction_accuracy']*100:.2f}%")
    
    # Key observations
    print("\nKEY OBSERVATIONS:")
    profit_improved = comparison['difference']['profit_loss'] > 0
    win_rate_improved = comparison['difference']['win_rate'] > 0
    
    if profit_improved:
        print(f"✓ ML enhancement increased profit by ${comparison['difference']['profit_loss']:.2f}")
    else:
        print(f"✗ ML enhancement decreased profit by ${-comparison['difference']['profit_loss']:.2f}")
        
    if win_rate_improved:
        print(f"✓ ML enhancement improved win rate by {comparison['difference']['win_rate']:.2f}%")
    else:
        print(f"✗ ML enhancement decreased win rate by {-comparison['difference']['win_rate']:.2f}%")
        
    print("\nComplete comparison report saved to the output directory.")

# Main execution
if __name__ == "__main__":
    # Run with defaults from config
    comparison = run_comparison()
