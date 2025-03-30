"""
optimize_sharpe_ratio.py - Script to optimize strategy parameters specifically for Sharpe ratio

This script focuses on improving the Sharpe ratio of the trading strategy by using
targeted parameter optimization with constraints on drawdown and win rate.
"""

import os
import logging
import json
import pandas as pd
from datetime import datetime

# Import from centralized config
from config import config

# Import utility functions
from utils import load_and_process_data, calculate_indicators, initialize_random_seeds

# Import optimization functions
from walk_forward_optimizer import run_walk_forward_optimization

# Import backtest function
from unified_backtester import run_backtest

# Import analysis functions
from trade_analysis import analyze_performance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_final_backtest(df, params, output_dir, use_ml=False):
    """
    Run a final backtest with optimized parameters.

    Args:
        df: DataFrame with price data
        params: Dictionary of optimized parameters
        output_dir: Directory to save results
        use_ml: Whether to use ML enhancement

    Returns:
        Dictionary with backtest results
    """
    from parameter_optimizer import update_config_with_params

    # Save original config
    import copy
    original_config = copy.deepcopy(config)

    # Update config with parameters
    for param, value in params.items():
        update_config_with_params(config, {param: value})

    # Create directory for final backtest
    final_dir = os.path.join(output_dir, "final_backtest")
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)

    # Set up file paths
    file_paths = {
        'trade_log': os.path.join(final_dir, 'trade_log.csv'),
        'portfolio_value': os.path.join(final_dir, 'portfolio_value.csv'),
        'regime_log': os.path.join(final_dir, 'regime_log.csv'),
        'market_type_log': os.path.join(final_dir, 'market_type_log.csv'),
        'summary': os.path.join(final_dir, 'summary.txt')
    }

    # Run backtest
    try:
        from backtester_common import reset_hmm_detector
        reset_hmm_detector()  # Reset state

        trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
            df.copy(),
            visualize_trades=True,  # Enable visualizations for final result
            file_paths=file_paths,
            use_ml=use_ml
        )

        # Calculate performance metrics
        if trades and len(portfolio_values) > 0:
            portfolio_series = pd.Series(portfolio_values, index=df['date'][:len(portfolio_values)])
            metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])

            # Log results
            logger.info("\n===== FINAL BACKTEST RESULTS WITH OPTIMIZED PARAMETERS =====")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"Profit/Loss: ${metrics['profit_loss']:.2f}")
            logger.info(f"Total Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
            logger.info(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            logger.info(f"Number of Trades: {metrics['number_of_trades']}")

            return metrics
        else:
            logger.warning("No trades executed in final backtest")
            return None
    except Exception as e:
        logger.error(f"Error in final backtest: {e}")
        return None
    finally:
        # Restore original config
        for key in original_config:
            config[key] = original_config[key]


def run_sharpe_optimization():
    """Run the full Sharpe ratio optimization process"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sharpe_optimization_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize random seeds
    seed = config['global'].get('random_seed', 42)
    initialize_random_seeds(seed)
    logger.info(f"Initialized random seed: {seed}")

    # Load and process data
    logger.info(f"Loading data from {config['data']['file_path']}")
    df = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    if df is None or len(df) == 0:
        logger.error("No data available after loading. Exiting.")
        return None

    # Calculate indicators
    df = calculate_indicators(df, config)

    # Step 1: Run walk-forward optimization focused on Sharpe ratio
    logger.info("Step 1: Running walk-forward optimization for Sharpe ratio")
    wf_results = run_walk_forward_optimization(
        data_file=config['data']['file_path'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        num_periods=4,  # 4 periods for good cross-validation
        n_trials=75,  # 75 trials per period
        use_ml=config['ml']['enable'],
        output_dir=os.path.join(output_dir, "walk_forward")
    )

    # Extract the recommended parameters from walk-forward optimization
    if wf_results and 'median_params' in wf_results:
        wf_results['best_params'] = wf_results['median_params']  # Add this key for compatibility
    elif wf_results and 'best_sharpe_params' in wf_results:
        wf_results['best_params'] = wf_results['best_sharpe_params']

    # Ensure we have parameters, otherwise the optimization failed
    if not wf_results or 'best_params' not in wf_results:
        logger.error("Walk-forward optimization did not produce valid parameters")
        return None

    # Step 2: Run a final backtest with the optimized parameters
    logger.info("Step 2: Running final backtest with optimized parameters")
    best_params = wf_results['best_params']

    # Run final backtest
    final_results = run_final_backtest(
        df,
        best_params,
        output_dir,
        use_ml=config['ml']['enable']
    )

    if final_results:
        wf_results['final_backtest'] = final_results
        logger.info(f"Final backtest completed with Sharpe ratio: {final_results['sharpe_ratio']:.4f}")
    else:
        logger.warning("Final backtest did not produce results")

    # Set output directory in results
    wf_results['output_dir'] = output_dir

    return wf_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Sharpe ratio optimization')
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancement')

    args = parser.parse_args()

    # Override config settings if provided
    if args.data:
        config['data']['file_path'] = args.data
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end
    if args.seed:
        config['global']['random_seed'] = args.seed
        config['global']['use_fixed_seed'] = True
    if args.ml:
        config['ml']['enable'] = True

    # Run optimization
    results = run_sharpe_optimization()

    if results:
        logger.info(f"Optimization complete. Results saved to {results['output_dir']}")