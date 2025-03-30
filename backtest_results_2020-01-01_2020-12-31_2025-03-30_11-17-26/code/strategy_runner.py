"""
strategy_runner.py - Command Line Interface for Strategy Execution

This script provides a unified interface to run backtests using either
the standard or ML-enhanced strategy, or to compare both strategies.
"""

import argparse
import logging
from datetime import datetime
import sys
import os
import pandas as pd

# Import from centralized config
from config import config

# Import utility functions
from utils import load_and_process_data, calculate_indicators, setup_directories

# Import backtester functions
from unified_backtester import run_backtest
from simplified_compare_strategies import run_comparison

# Import analysis functions
from trade_analysis import analyze_performance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_seed_testing_metrics(trades, results):
    """Add standardized metrics output for seed testing."""
    print("\n===== METRICS FOR SEED TESTING =====")
    print(f"Trades: {len(trades)}")
    print(f"Profit/Loss: ${results['profit_loss']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    if 'max_drawdown_pct' in results:
        print(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
    if 'sharpe_ratio' in results:
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print("===== END METRICS =====")


def main():
    """Main entry point for the strategy runner"""
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')

    # Strategy selection
    parser.add_argument('--strategy', type=str, choices=['standard', 'ml', 'compare'], default='standard',
                        help='Which strategy to run (standard, ml, or compare)')

    # Data and date parameters
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    # HMM-specific parameters
    parser.add_argument('--hmm-lookback', type=int, help='HMM lookback days for training')
    parser.add_argument('--hmm-states', type=int, help='Number of HMM states')
    parser.add_argument('--hmm-retrain', type=int, help='Days between HMM retraining')
    parser.add_argument('--hmm-samples', type=int, help='Minimum HMM training samples')

    # ML-specific parameters (if using ML strategy)
    parser.add_argument('--ml-enable', action='store_true', help='Enable ML filtering')
    parser.add_argument('--ml-threshold', type=float, help='ML prediction threshold')
    parser.add_argument('--ml-model', type=str, choices=['xgboost', 'random_forest'],
                        help='ML model type to use')

    # Add random seed parameter
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # Visualization options
    parser.add_argument('--visualize', action='store_true', help='Generate charts and visualizations')

    # Output directory
    parser.add_argument('--output', type=str, help='Output directory for results')

    # Parse arguments
    args = parser.parse_args()

    # Override config settings based on command line arguments
    if args.data:
        config['data']['file_path'] = args.data
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end

    # Update visualization setting
    if args.visualize:
        config['visualization']['generate_png_charts'] = True

    # Update HMM parameters if provided
    if args.hmm_lookback:
        config['hmm_detector']['lookback_days'] = args.hmm_lookback
    if args.hmm_states:
        config['hmm_detector']['n_states'] = args.hmm_states
    if args.hmm_retrain:
        config['hmm_detector']['retrain_frequency'] = args.hmm_retrain
    if args.hmm_samples:
        config['hmm_detector']['min_samples'] = args.hmm_samples

    # Update ML parameters if provided
    if args.ml_enable:
        config['ml']['enable'] = True
    if args.ml_threshold:
        config['ml']['prediction_threshold'] = args.ml_threshold
    if args.ml_model:
        config['ml']['model_type'] = args.ml_model

    # Update random seed if provided
    if args.seed is not None:
        config['global']['random_seed'] = args.seed
        config['global']['use_fixed_seed'] = True
        logger.info(f"Using random seed: {args.seed}")

    # Initialize random seeds
    from utils import initialize_random_seeds
    initialize_random_seeds(config['global']['random_seed'])

    # Log the current configuration
    logger.info(f"Running with data file: {config['data']['file_path']}")
    logger.info(f"Date range: {config['data']['start_date']} to {config['data']['end_date']}")
    logger.info(f"Random seed: {config['global']['random_seed']}")
    logger.info(
        f"HMM detector: enabled={config['hmm_detector']['enable']}, states={config['hmm_detector']['n_states']}, "
        f"lookback={config['hmm_detector']['lookback_days']}, retraining={config['hmm_detector']['retrain_frequency']}")
    if args.strategy in ['ml', 'compare']:
        logger.info(f"ML settings: enabled={config['ml']['enable']}, model={config['ml']['model_type']}, "
                    f"threshold={config['ml']['prediction_threshold']}")

    # Run the selected strategy
    if args.strategy == 'standard':
        run_strategy(args.output, use_ml=False)
    elif args.strategy == 'ml':
        run_strategy(args.output, use_ml=True)
    elif args.strategy == 'compare':
        run_comparison(args.output)
    else:
        logger.error(f"Unknown strategy type: {args.strategy}")
        return 1

    return 0

def run_strategy(output_dir=None, use_ml=None):
    """Run the backtester strategy with optional ML enhancement"""
    logger.info("Running strategy backtest...")

    # Set ML flag based on function argument or config
    ml_enabled = use_ml if use_ml is not None else config['ml']['enable']

    # Create appropriate directory name
    strategy_type = "ml_enhanced_" if ml_enabled else "standard_"

    # Set up output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{strategy_type}backtest_{timestamp}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Create file paths
    file_paths = {
        'trade_log': os.path.join(output_dir, 'trade_log.csv'),
        'portfolio_value': os.path.join(output_dir, 'portfolio_value.csv'),
        'regime_log': os.path.join(output_dir, 'regime_log.csv'),
        'market_type_log': os.path.join(output_dir, 'market_type_log.csv'),
        'summary': os.path.join(output_dir, 'summary.txt')
    }

    # Load and process data
    df = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    if df is None or len(df) == 0:
        logger.error("No data available after loading. Exiting.")
        return False

    # Calculate indicators
    df = calculate_indicators(df, config)

    # Run backtest
    trades, portfolio_values, df, _, _, _, _, ml_metrics = run_backtest(
        df,
        visualize_trades=config['visualization']['generate_png_charts'],
        file_paths=file_paths,
        use_ml=ml_enabled
    )

    # Convert to series for analysis
    portfolio_df = pd.DataFrame({'date': df['date'][:len(portfolio_values)], 'value': portfolio_values})
    portfolio_series = portfolio_df.set_index('date')['value']

    # Analyze results
    results = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])

    # Add seed testing metrics
    add_seed_testing_metrics(trades, results)

    logger.info(f"Strategy backtest completed. Results saved to {output_dir}")
    return True


if __name__ == "__main__":
    sys.exit(main())