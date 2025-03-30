"""
parameter_optimizer.py - Grid search optimization for strategy parameters

This script provides functions to systematically test and optimize trading parameters
across multiple dimensions including HMM, regime detection, and exit strategies.
"""

import os
import copy
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import logging
import matplotlib.pyplot as plt

# Import from config
from config import config

# Import backtester functions
from unified_backtester import run_backtest

# Import utilities
from utils import load_and_process_data, calculate_indicators

# Import trade analysis
from trade_analysis import analyze_performance, analyze_exit_strategies

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#############################################################################
#                      USER CONFIGURATION SECTION                           #
#############################################################################
# Set to True to use this configuration instead of command line arguments
USE_CONFIG_SECTION = True

# Data settings
DATA_FILE = os.path.join('..', 'Candlestick_Data', 'MES_data', 'U19_H25.csv')  # Path to your data file
START_DATE = '2020-01-01'  # Start date for backtest
END_DATE = '2020-12-31'  # End date for backtest

# Optimization settings
OPTIMIZATION_TYPE = 'full'  # Options: 'full', 'regime_only', 'exit_only'
USE_ML = False  # Whether to use ML-enhanced backtester
WALK_FORWARD_PERIODS = 2  # Number of periods for walk-forward testing

# Output directory
OUTPUT_DIR = f"parameter_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Parameter grids for optimization
# Focus specifically on regime detection stability parameters
HMM_PARAM_GRID = {
    'hmm_detector.lookback_days': [15, 20, 25],
    'hmm_detector.retrain_frequency': [5, 7, 10],
    'market_type.update_frequency': [3, 5, 7]
}

# Exit strategy parameters
EXIT_PARAM_GRID = {
    'risk.enable_trailing_stop': [True, False],
    'risk.trailing_stop_atr_multiplier': [1.5, 2.0, 2.5],
    'risk.atr_stop_multiplier': [2.0, 2.5, 3.0],
    'risk.dynamic_target_enable': [True, False],
    'risk.dynamic_target_atr_multiplier': [1.0, 1.2, 1.5],
    'risk.max_bars_held': [8, 12, 16]
}


# To test specific parameter combinations, you can override the grids with smaller sets
# Example: To test just 2 combinations of HMM parameters:
# HMM_PARAM_GRID = {
#     'hmm_detector.lookback_days': [20],
#     'hmm_detector.retrain_frequency': [5, 10]
# }

#############################################################################
#                        END OF CONFIGURATION                               #
#############################################################################

def update_config_with_params(config, param_dict):
    """
    Update config with parameter values from param_dict.

    Args:
        config: Configuration dictionary to update
        param_dict: Dictionary with parameter paths and values

    Returns:
        Updated config dictionary
    """
    for param, value in param_dict.items():
        # Parse parameter path, e.g., "hmm_detector.min_samples" -> config["hmm_detector"]["min_samples"]
        path = param.split('.')
        curr = config
        for p in path[:-1]:
            if p not in curr:
                curr[p] = {}
            curr = curr[p]
        curr[path[-1]] = value
    return config


def calculate_composite_score(df):
    """
    Calculate a composite score balancing profit, risk, and robustness metrics.

    Args:
        df: DataFrame with results

    Returns:
        DataFrame with added composite score
    """
    # Create a copy to avoid modifying the original
    df_scored = df.copy()

    # Define metrics to normalize (profit metrics should be maximized)
    maximize_metrics = ['profit_loss', 'return_pct', 'win_rate', 'sharpe_ratio', 'profit_factor', 'num_trades']
    # Metrics to minimize (risk metrics)
    minimize_metrics = ['max_drawdown']

    # Normalize metrics to 0-1 scale
    for col in maximize_metrics:
        if col in df_scored.columns:
            min_val = df_scored[col].min()
            max_val = df_scored[col].max()
            if max_val > min_val:
                df_scored[f'{col}_norm'] = (df_scored[col] - min_val) / (max_val - min_val)
            else:
                df_scored[f'{col}_norm'] = 0.5

    # For metrics where lower is better
    for col in minimize_metrics:
        if col in df_scored.columns:
            min_val = df_scored[col].min()
            max_val = df_scored[col].max()
            if max_val > min_val:
                df_scored[f'{col}_norm'] = 1 - ((df_scored[col] - min_val) / (max_val - min_val))
            else:
                df_scored[f'{col}_norm'] = 0.5

    # Define weights for each metric
    weights = {
        'profit_loss_norm': 0.30,  # Emphasize profitability
        'return_pct_norm': 0.05,  # Total return
        'win_rate_norm': 0.20,  # Consistency of trades
        'max_drawdown_norm': 0.20,  # Risk control
        'sharpe_ratio_norm': 0.15,  # Risk-adjusted returns
        'profit_factor_norm': 0.05,  # Profit/loss ratio
        'num_trades_norm': 0.05  # Sample size significance
    }

    # Calculate weighted composite score using available metrics
    score_components = []
    total_weight = 0

    for metric, weight in weights.items():
        if metric in df_scored.columns:
            score_components.append(df_scored[metric] * weight)
            total_weight += weight

    # Normalize by actual total weight
    if total_weight > 0:
        df_scored['composite_score'] = sum(score_components) / total_weight
    else:
        # Fallback if no metrics are available
        df_scored['composite_score'] = 0

    # Add ranking based on composite score
    df_scored['rank'] = df_scored['composite_score'].rank(ascending=False, method='min').astype(int)

    return df_scored


def reset_state_variables():
    """Reset any stateful variables in the backtester to ensure clean runs"""
    # Reset hmm_detector in backtester_common module
    try:
        import backtester_common
        backtester_common.hmm_detector = None
        logger.info("Reset hmm_detector in backtester_common")
    except Exception as e:
        logger.warning(f"Could not reset hmm_detector: {e}")

    # Reset any function attributes that might store state
    try:
        from refactored_backtester import detect_market_type

        # Remove attributes that store state
        for attr in ['historical_scores', 'score_dates', 'market_type_counts']:
            if hasattr(detect_market_type, attr):
                delattr(detect_market_type, attr)

        logger.info("Reset market type detection state variables")
    except Exception as e:
        logger.warning(f"Could not reset market type detection state: {e}")


def run_parameter_grid_search(df, output_dir, parameter_grid, test_periods=None, use_ml=False):
    """
    Run a grid search across multiple parameter combinations.

    Args:
        df: DataFrame with price data and indicators
        output_dir: Directory to save results
        parameter_grid: Dictionary with parameter names and values to test
        test_periods: List of time periods for testing (optional)
        use_ml: Whether to use ML-enhanced backtester

    Returns:
        DataFrame of results sorted by performance metrics
    """
    original_config = copy.deepcopy(config)
    results = []

    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = [parameter_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))

    logger.info(f"Testing {len(param_combinations)} parameter combinations")

    # If no test periods specified, use the entire dataset
    if test_periods is None:
        test_periods = [{"name": "full_period", "start": None, "end": None}]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Choose backtester function
    backtest_func = run_ml_enhanced_backtest if use_ml else run_backtest

    # Run tests for each combination
    for i, combo in enumerate(param_combinations):
        param_dict = {name: value for name, value in zip(param_names, combo)}

        # Create parameter ID for logging
        param_id = "_".join([f"{k.split('.')[-1]}={v}" for k, v in param_dict.items()])
        logger.info(f"Testing combination {i + 1}/{len(param_combinations)}: {param_id}")

        # Test on each period
        for period in test_periods:
            period_name = period["name"]

            # Create period-specific output directory
            period_dir = os.path.join(output_dir, f"{period_name}_{param_id}")
            if not os.path.exists(period_dir):
                os.makedirs(period_dir)

            # Set up file paths
            file_paths = {
                'trade_log': os.path.join(period_dir, 'trade_log.csv'),
                'portfolio_value': os.path.join(period_dir, 'portfolio_value.csv'),
                'regime_log': os.path.join(period_dir, 'regime_log.csv'),
                'market_type_log': os.path.join(period_dir, 'market_type_log.csv'),
                'summary': os.path.join(period_dir, 'summary.txt')
            }

            # Filter data by period if needed
            period_df = df
            if period["start"] is not None and period["end"] is not None:
                period_df = df[(df['date'] >= period["start"]) & (df['date'] <= period["end"])].copy()

                if len(period_df) < 50:  # Ensure enough data
                    logger.warning(f"Insufficient data for period {period_name}. Skipping.")
                    continue

            # Update config with test parameters
            reset_state_variables()
            test_config = copy.deepcopy(original_config)
            update_config_with_params(test_config, param_dict)

            # Temporarily update global config (This is necessary because some functions directly access global config)
            for key in config:
                if key in test_config:
                    config[key] = test_config[key]

            # Run backtest
            try:
                trades, portfolio_values, _, _, _, _, _, ml_metrics = run_backtest(
                    period_df.copy(), visualize_trades=False, file_paths=file_paths, use_ml=use_ml
                )

                # Calculate performance metrics
                if trades and len(portfolio_values) > 0:
                    portfolio_series = pd.Series(portfolio_values, index=period_df['date'][:len(portfolio_values)])
                    metrics = analyze_performance(trades, portfolio_series, test_config['account']['initial_capital'])

                    # Calculate exit strategy metrics
                    exit_metrics = analyze_exit_strategies(trades)

                    # Extract key metrics
                    result = {
                        'period': period_name,
                        'num_trades': len(trades),
                        'profit_loss': metrics['profit_loss'],
                        'return_pct': metrics['total_return_pct'],
                        'win_rate': metrics['win_rate'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown': metrics['max_drawdown_pct'],
                        'profit_factor': metrics['profit_factor']
                    }

                    # Add exit strategy metrics
                    if 'price_capture_efficiency' in exit_metrics:
                        result['price_capture'] = exit_metrics['price_capture_efficiency'].get('avg_capture_pct', 0)
                        result['trailing_advantage'] = exit_metrics['price_capture_efficiency'].get(
                            'trailing_advantage', 0)

                    # Add ML metrics if applicable
                    if use_ml and ml_metrics:
                        result['ml_accuracy'] = ml_metrics.get('prediction_accuracy', 0) * 100
                        result['ml_skipped'] = ml_metrics.get('skipped_by_ml', 0)

                    # Add parameter values
                    for param, value in param_dict.items():
                        result[param] = value

                    # Add to results list
                    results.append(result)

                    # Log basic results
                    logger.info(f"Results for {param_id} on {period_name}: "
                                f"Profit: ${metrics['profit_loss']:.2f}, "
                                f"Win Rate: {metrics['win_rate']:.2f}%, "
                                f"Trades: {len(trades)}")

            except Exception as e:
                logger.error(f"Error testing {param_id} on {period_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Restore global config to avoid contamination
        for key in original_config:
            config[key] = original_config[key]

    # Restore original config completely
    for key in original_config:
        config[key] = original_config[key]

    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)

        # Add composite score
        results_df = calculate_composite_score(results_df)

        # Sort by composite score
        results_df = results_df.sort_values('composite_score', ascending=False)

        # Save results
        results_df.to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False)

        # Also save a more readable version with key columns first
        key_columns = ['period', 'composite_score', 'rank', 'profit_loss', 'return_pct',
                       'win_rate', 'sharpe_ratio', 'max_drawdown', 'num_trades']

        # Add parameter columns
        param_columns = [col for col in results_df.columns if any(col.startswith(p) for p in param_names)]

        # Reorder columns
        ordered_columns = key_columns + param_columns + [col for col in results_df.columns
                                                         if col not in key_columns and col not in param_columns]

        readable_df = results_df[ordered_columns]
        readable_df.to_csv(os.path.join(output_dir, "grid_search_readable.csv"), index=False)

        return results_df
    else:
        logger.warning("No valid results obtained from grid search")
        return pd.DataFrame()  # Return empty DataFrame


def perform_walk_forward_optimization(df, output_dir, param_grid, num_folds=3, use_ml=False):
    """
    Perform walk-forward optimization by testing parameters on sequential time periods.

    Args:
        df: DataFrame with price data
        output_dir: Directory to save results
        param_grid: Dictionary with parameter names and values to test
        num_folds: Number of time periods to test
        use_ml: Whether to use ML-enhanced backtester

    Returns:
        Dictionary with optimized parameters for each fold and overall
    """
    # Split data into time periods
    start_date = df['date'].min().to_pydatetime()
    end_date = df['date'].max().to_pydatetime()

    date_range = end_date - start_date
    period_length = date_range / num_folds

    test_periods = []
    for i in range(num_folds):
        period_start = start_date + (period_length * i)
        period_end = start_date + (period_length * (i + 1))

        # Ensure datetime objects
        if isinstance(period_start, timedelta):
            period_start = start_date + period_start
        if isinstance(period_end, timedelta):
            period_end = start_date + period_end

        test_periods.append({
            "name": f"period_{i + 1}",
            "start": period_start,
            "end": period_end
        })

    # Create walk-forward directory
    walk_forward_dir = os.path.join(output_dir, "walk_forward")
    if not os.path.exists(walk_forward_dir):
        os.makedirs(walk_forward_dir)

    # Run grid search
    logger.info(f"Running walk-forward optimization with {num_folds} folds")
    results = run_parameter_grid_search(df, walk_forward_dir, param_grid, test_periods, use_ml)

    if results.empty:
        logger.error("No valid results from walk-forward optimization")
        return {}

    # Find best params for each period
    best_by_period = {}
    for period in [p["name"] for p in test_periods]:
        period_results = results[results['period'] == period]
        if not period_results.empty:
            best_by_period[period] = period_results.iloc[0].to_dict()

    # Calculate consistency scores
    param_columns = [col for col in results.columns if any(col == p for p in param_grid.keys())]

    # Group by parameter combination
    grouped_scores = {}

    for _, row in results.iterrows():
        # Create parameter signature
        param_sig = tuple((col, row[col]) for col in param_columns)

        if param_sig not in grouped_scores:
            grouped_scores[param_sig] = {
                'params': {col: row[col] for col in param_columns},
                'total_score': 0,
                'avg_score': 0,
                'num_periods': 0,
                'total_profit': 0,
                'results': []
            }

        # Add score for this period
        grouped_scores[param_sig]['total_score'] += row['composite_score']
        grouped_scores[param_sig]['num_periods'] += 1
        grouped_scores[param_sig]['total_profit'] += row['profit_loss']
        grouped_scores[param_sig]['results'].append({
            'period': row['period'],
            'profit': row['profit_loss'],
            'score': row['composite_score']
        })

    # Calculate average scores
    for sig, data in grouped_scores.items():
        if data['num_periods'] > 0:
            data['avg_score'] = data['total_score'] / data['num_periods']

    # Find parameter sets that performed well across all periods
    consistent_params = []

    for sig, data in grouped_scores.items():
        # Only consider parameter sets that have results for all periods
        if data['num_periods'] == num_folds:
            consistent_params.append(data)

    # Sort by average score
    consistent_params.sort(key=lambda x: x['avg_score'], reverse=True)

    # Get best consistent parameters
    best_consistent = consistent_params[0] if consistent_params else None

    # Save walk-forward results
    results_output = {
        'best_by_period': best_by_period,
        'best_consistent': best_consistent,
        'periods': [
            {
                'name': p['name'],
                'start': p['start'].strftime('%Y-%m-%d'),
                'end': p['end'].strftime('%Y-%m-%d')
            } for p in test_periods
        ]
    }

    try:
        # Handle non-serializable objects
        for period, data in best_by_period.items():
            for key, value in list(data.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    data[key] = str(value)

        if best_consistent:
            for key, value in list(best_consistent.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    best_consistent[key] = str(value)

        with open(os.path.join(output_dir, "walk_forward_results.json"), 'w') as f:
            json.dump(results_output, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving walk-forward results: {e}")

    # Create a summary file with the best parameters
    with open(os.path.join(output_dir, "walk_forward_summary.txt"), 'w') as f:
        f.write("WALK-FORWARD OPTIMIZATION RESULTS\n")
        f.write("================================\n\n")

        # Write period information
        f.write("Test Periods:\n")
        for p in test_periods:
            f.write(f"  {p['name']}: {p['start'].strftime('%Y-%m-%d')} to {p['end'].strftime('%Y-%m-%d')}\n")
        f.write("\n")

        # Write best parameters for each period
        f.write("Best Parameters by Period:\n")
        for period, data in best_by_period.items():
            f.write(f"\n  {period}:\n")
            f.write(f"    Profit: ${data.get('profit_loss', 0):.2f}\n")
            f.write(f"    Win Rate: {data.get('win_rate', 0):.2f}%\n")
            f.write(f"    Composite Score: {data.get('composite_score', 0):.4f}\n")
            f.write("    Parameters:\n")

            # Write parameter values
            for param in param_grid.keys():
                if param in data:
                    f.write(f"      {param}: {data[param]}\n")

        # Write best consistent parameters
        f.write("\nMost Consistent Parameters Across All Periods:\n")
        if best_consistent:
            f.write(f"  Average Score: {best_consistent['avg_score']:.4f}\n")
            f.write(f"  Total Profit: ${best_consistent['total_profit']:.2f}\n")
            f.write("  Parameters:\n")

            # Write parameter values
            for param, value in best_consistent['params'].items():
                f.write(f"    {param}: {value}\n")

            # Write period-specific results
            f.write("\n  Performance by Period:\n")
            for result in best_consistent['results']:
                f.write(f"    {result['period']}: Profit ${result['profit']:.2f}, Score {result['score']:.4f}\n")
        else:
            f.write("  No consistent parameters found across all periods\n")

    # Generate visualizations
    try:
        generate_optimization_visualizations(results, output_dir, param_grid)
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

    # Return the best consistent parameters or best overall if no consistent set
    if best_consistent:
        return best_consistent['params']
    elif not results.empty:
        # Return overall best parameters
        best_overall = results.iloc[0]
        return {param: best_overall[param] for param in param_grid.keys() if param in best_overall}
    else:
        return {}


def generate_optimization_visualizations(results_df, output_dir, param_grid):
    """
    Generate visualizations of parameter optimization results.

    Args:
        results_df: DataFrame with optimization results
        output_dir: Directory to save visualizations
        param_grid: Dictionary with parameter names and values tested
    """
    if results_df.empty:
        return

    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # 1. Parameter impact charts
    for param in param_grid.keys():
        if param in results_df.columns:
            plt.figure(figsize=(12, 6))

            # Group by parameter value
            param_impact = results_df.groupby(param)['profit_loss'].mean().reset_index()

            # Sort by parameter value (convert to string for categorical axis)
            param_impact = param_impact.sort_values(param)

            # Create bar chart
            plt.bar(param_impact[param].astype(str), param_impact['profit_loss'], color='#3498db')
            plt.title(f'Impact of {param} on Average Profit')
            plt.xlabel(param)
            plt.ylabel('Average Profit ($)')
            plt.grid(axis='y', alpha=0.3)

            # Add value labels
            for i, row in param_impact.iterrows():
                plt.text(i, row['profit_loss'], f'${row["profit_loss"]:.0f}',
                         ha='center', va='bottom' if row['profit_loss'] > 0 else 'top')

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{param.replace(".", "_")}_impact.png'), dpi=150)
            plt.close()

    # 2. Period comparison chart
    if 'period' in results_df.columns:
        periods = results_df['period'].unique()

        if len(periods) > 1:
            plt.figure(figsize=(14, 8))

            # Best result for each period
            best_by_period = []
            for period in periods:
                period_best = results_df[results_df['period'] == period].iloc[0]
                best_by_period.append({
                    'period': period,
                    'profit': period_best['profit_loss'],
                    'win_rate': period_best['win_rate'],
                    'drawdown': period_best['max_drawdown']
                })

            period_df = pd.DataFrame(best_by_period)

            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Profit chart
            axes[0].bar(period_df['period'], period_df['profit'], color='#2ecc71')
            axes[0].set_title('Best Profit by Period')
            axes[0].set_ylabel('Profit ($)')
            axes[0].grid(axis='y', alpha=0.3)

            # Win rate chart
            axes[1].bar(period_df['period'], period_df['win_rate'], color='#3498db')
            axes[1].set_title('Win Rate by Period')
            axes[1].set_ylabel('Win Rate (%)')
            axes[1].grid(axis='y', alpha=0.3)

            # Drawdown chart
            axes[2].bar(period_df['period'], period_df['drawdown'], color='#e74c3c')
            axes[2].set_title('Max Drawdown by Period')
            axes[2].set_ylabel('Drawdown (%)')
            axes[2].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'period_comparison.png'), dpi=150)
            plt.close()

    # 3. Top 10 parameter combinations
    top10 = results_df.head(10)

    if not top10.empty:
        plt.figure(figsize=(12, 8))

        # Create index for ranking
        ranks = [f"{i + 1}" for i in range(len(top10))]

        # Plot profit for top combinations
        plt.bar(ranks, top10['profit_loss'], color='#2ecc71')
        plt.title('Top 10 Parameter Combinations by Composite Score')
        plt.xlabel('Rank')
        plt.ylabel('Profit ($)')
        plt.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, row in enumerate(top10.itertuples()):
            plt.text(i, row.profit_loss, f'${row.profit_loss:.0f}',
                     ha='center', va='bottom' if row.profit_loss > 0 else 'top')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'top10_combinations.png'), dpi=150)
        plt.close()

        # Also create a text-based visualization of parameter values
        param_columns = [col for col in top10.columns if any(col == p for p in param_grid.keys())]

        if param_columns:
            with open(os.path.join(viz_dir, 'top_combinations.txt'), 'w') as f:
                f.write("TOP PARAMETER COMBINATIONS\n")
                f.write("=========================\n\n")

                for i, row in enumerate(top10.itertuples()):
                    f.write(f"Rank {i + 1}:\n")
                    f.write(f"  Composite Score: {row.composite_score:.4f}\n")
                    f.write(f"  Profit: ${row.profit_loss:.2f}\n")
                    f.write(f"  Win Rate: {row.win_rate:.2f}%\n")
                    f.write(f"  Max Drawdown: {row.max_drawdown:.2f}%\n")
                    f.write("  Parameters:\n")

                    for param in param_columns:
                        f.write(f"    {param}: {getattr(row, param)}\n")

                    f.write("\n")

def optimize_with_optuna(df, output_dir, n_trials=100, focus_sharpe=True):
    """
    Optimize strategy parameters using Optuna to maximize Sharpe ratio or other objectives.

    Args:
        df: DataFrame with price data
        output_dir: Directory to save results
        n_trials: Number of optimization trials to run
        focus_sharpe: Whether to focus optimization on Sharpe ratio

    Returns:
        Dictionary with optimized parameters
    """
    import optuna
    from optuna.samplers import TPESampler
    import os
    import json
    import matplotlib.pyplot as plt

    # Create output directory for Optuna results
    optuna_dir = os.path.join(output_dir, "optuna_optimization")
    if not os.path.exists(optuna_dir):
        os.makedirs(optuna_dir)

    # Save original config
    original_config = copy.deepcopy(config)

    # Define objective function
    def objective(trial):
        # Reset HMM and other state before each trial
        reset_state_variables()

        # Define parameters to optimize
        params = {
            # Risk management parameters
            'risk.risk_per_trade': trial.suggest_float('risk_per_trade', 0.005, 0.015),
            'risk.atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 3.0),
            'risk.trailing_stop_atr_multiplier': trial.suggest_float('trailing_stop_multiplier', 1.0, 2.5),
            'risk.dynamic_target_atr_multiplier': trial.suggest_float('dynamic_target_multiplier', 0.8, 2.0),
            'risk.max_bars_held': trial.suggest_int('max_bars_held', 6, 20),

            # HMM detector parameters
            'hmm_detector.lookback_days': trial.suggest_int('lookback_days', 20, 40),
            'hmm_detector.retrain_frequency': trial.suggest_int('retrain_frequency', 5, 15),
            'hmm_detector.min_samples': trial.suggest_int('min_samples', 150, 300),

            # Market type parameters
            'market_type.trend_following.min_regime_score': trial.suggest_float('tf_min_score', 65, 80),
            'market_type.mean_reverting.min_regime_score': trial.suggest_float('mr_min_score', 35, 50),
            'market_type.neutral.min_regime_score': trial.suggest_float('neutral_min_score', 45, 60),

            # Position sizing parameters
            'position_sizing.max_size_adjustment': trial.suggest_float('max_size_adjustment', 1.0, 2.0),
            'position_sizing.min_size_adjustment': trial.suggest_float('min_size_adjustment', 0.5, 0.9)
        }

        # Update config with trial parameters
        for param, value in params.items():
            update_config_with_params(config, {param: value})

        # Set up file paths for this trial
        trial_dir = os.path.join(optuna_dir, f"trial_{trial.number}")
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        file_paths = {
            'trade_log': os.path.join(trial_dir, 'trade_log.csv'),
            'portfolio_value': os.path.join(trial_dir, 'portfolio_value.csv'),
            'regime_log': os.path.join(trial_dir, 'regime_log.csv'),
            'market_type_log': os.path.join(trial_dir, 'market_type_log.csv'),
            'summary': os.path.join(trial_dir, 'summary.txt')
        }

        # Run backtest
        try:
            trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
                df.copy(), visualize_trades=False, file_paths=file_paths
            )

            # Calculate performance metrics
            if trades and len(portfolio_values) > 0:
                portfolio_series = pd.Series(portfolio_values, index=df['date'][:len(portfolio_values)])
                metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])

                # Extract key metrics
                sharpe_ratio = metrics['sharpe_ratio']
                profit_loss = metrics['profit_loss']
                max_drawdown = metrics['max_drawdown_pct']
                win_rate = metrics['win_rate']

                # Decide what to optimize
                if focus_sharpe:
                    # Primarily optimize Sharpe ratio
                    objective_value = sharpe_ratio
                else:
                    # Create composite score (balancing profit and risk)
                    profit_component = profit_loss / config['account']['initial_capital']
                    risk_component = 1.0 - (max_drawdown / 100)  # Transform to 0-1 scale where higher is better

                    # Weight components: 50% profit, 30% Sharpe, 20% risk control
                    objective_value = (0.5 * profit_component) + (0.3 * sharpe_ratio) + (0.2 * risk_component)

                # Log the results
                logger.info(f"Trial {trial.number}: Sharpe={sharpe_ratio:.3f}, P/L=${profit_loss:.2f}, "
                            f"DD={max_drawdown:.2f}%, Win={win_rate:.2f}%")

                # Record additional metrics in trial user attributes
                trial.set_user_attr('profit_loss', float(profit_loss))
                trial.set_user_attr('max_drawdown', float(max_drawdown))
                trial.set_user_attr('win_rate', float(win_rate))
                trial.set_user_attr('trade_count', len(trades))

                return objective_value
            else:
                # No trades executed, return a poor score
                logger.warning(f"Trial {trial.number}: No trades executed")
                return -1.0 if focus_sharpe else 0.0

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return -1.0 if focus_sharpe else 0.0
        finally:
            # Restore original config
            for key in original_config:
                config[key] = original_config[key]

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config['global']['random_seed']),
        study_name="strategy_optimization"
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial

    # Print best results
    logger.info("\n===== OPTIMIZATION RESULTS =====")
    logger.info(f"Best trial: #{best_trial.number}")
    logger.info(f"Value: {best_trial.value:.4f}")
    logger.info(f"Params: {best_params}")
    logger.info(f"Profit/Loss: ${best_trial.user_attrs['profit_loss']:.2f}")
    logger.info(f"Maximum Drawdown: {best_trial.user_attrs['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {best_trial.user_attrs['win_rate']:.2f}%")

    # Export study results
    try:
        # Convert study to DataFrame
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(optuna_dir, "optimization_results.csv"), index=False)

        # Save best parameters
        best_dict = {
            'params': best_params,
            'metrics': {
                'objective_value': float(best_trial.value),
                'profit_loss': float(best_trial.user_attrs['profit_loss']),
                'max_drawdown': float(best_trial.user_attrs['max_drawdown']),
                'win_rate': float(best_trial.user_attrs['win_rate']),
                'trade_count': int(best_trial.user_attrs['trade_count'])
            }
        }

        with open(os.path.join(output_dir, "best_optuna_params.json"), 'w') as f:
            json.dump(best_dict, f, indent=4)

        # Generate optimization visualizations
        try:
            # Optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(os.path.join(optuna_dir, "optimization_history.png"))

            # Parameter importance
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(optuna_dir, "param_importances.png"))

            # Parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(optuna_dir, "parallel_coordinate.png"))

            # Slice plots for top parameters
            param_importances = optuna.importance.get_param_importances(study)
            top_params = list(param_importances.keys())[:3]
            for param in top_params:
                fig = optuna.visualization.plot_slice(study, params=[param])
                fig.write_image(os.path.join(optuna_dir, f"slice_{param}.png"))
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    except Exception as e:
        logger.error(f"Error exporting study results: {e}")

    # Return best parameters with correct formatting for config
    optimized_params = {}
    for param_name, value in best_params.items():
        # Map back to config parameter paths
        if param_name == 'risk_per_trade':
            optimized_params['risk.risk_per_trade'] = value
        elif param_name == 'atr_stop_multiplier':
            optimized_params['risk.atr_stop_multiplier'] = value
        elif param_name == 'trailing_stop_multiplier':
            optimized_params['risk.trailing_stop_atr_multiplier'] = value
        elif param_name == 'dynamic_target_multiplier':
            optimized_params['risk.dynamic_target_atr_multiplier'] = value
        elif param_name == 'max_bars_held':
            optimized_params['risk.max_bars_held'] = value
        elif param_name == 'lookback_days':
            optimized_params['hmm_detector.lookback_days'] = value
        elif param_name == 'retrain_frequency':
            optimized_params['hmm_detector.retrain_frequency'] = value
        elif param_name == 'min_samples':
            optimized_params['hmm_detector.min_samples'] = value
        elif param_name == 'tf_min_score':
            optimized_params['market_type.trend_following.min_regime_score'] = value
        elif param_name == 'mr_min_score':
            optimized_params['market_type.mean_reverting.min_regime_score'] = value
        elif param_name == 'neutral_min_score':
            optimized_params['market_type.neutral.min_regime_score'] = value
        elif param_name == 'max_size_adjustment':
            optimized_params['position_sizing.max_size_adjustment'] = value
        elif param_name == 'min_size_adjustment':
            optimized_params['position_sizing.min_size_adjustment'] = value

    return optimized_params

def optimize_strategy_parameters(df, output_dir, hmm_param_grid, exit_param_grid, use_ml=False):
    """
    Comprehensive optimization of both HMM and exit parameters.

    Uses a two-step process:
    1. First optimize HMM & regime detection parameters
    2. Then optimize exit parameters using the best HMM parameters

    Args:
        df: DataFrame with price data
        output_dir: Directory to save results
        hmm_param_grid: Dictionary with HMM parameter names and values to test
        exit_param_grid: Dictionary with exit parameter names and values to test
        use_ml: Whether to use ML-enhanced backtester

    Returns:
        Dictionary with optimized parameters
    """
    # Create output directories
    hmm_dir = os.path.join(output_dir, "hmm_optimization")
    exit_dir = os.path.join(output_dir, "exit_optimization")
    combined_dir = os.path.join(output_dir, "combined_optimization")

    for directory in [hmm_dir, exit_dir, combined_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save original config for restoring later
    original_config = copy.deepcopy(config)

    # Step 1: Optimize HMM parameters first
    logger.info("Step 1: Optimizing HMM and regime detection parameters...")
    best_hmm_params = perform_walk_forward_optimization(df, hmm_dir, hmm_param_grid, use_ml=use_ml)

    # Check if we got valid HMM parameters
    if not best_hmm_params:
        logger.error("HMM parameter optimization failed. Using default parameters.")
        # Use first value of each parameter as default
        best_hmm_params = {param: values[0] for param, values in hmm_param_grid.items()}

    # Update config with best HMM parameters
    update_config_with_params(config, best_hmm_params)

    # Step 2: Optimize exit parameters with best HMM parameters
    logger.info("Step 2: Optimizing exit parameters using best HMM parameters...")
    best_exit_params = perform_walk_forward_optimization(df, exit_dir, exit_param_grid, use_ml=use_ml)

    # Check if we got valid exit parameters
    if not best_exit_params:
        logger.error("Exit parameter optimization failed. Using default parameters.")
        # Use first value of each parameter as default
        best_exit_params = {param: values[0] for param, values in exit_param_grid.items()}

    # Combine best parameters
    optimized_params = {**best_hmm_params, **best_exit_params}

    # Step 3: Validate combined parameters
    logger.info("Step 3: Validating combined parameter set...")

    # Update config with all optimized parameters
    update_config_with_params(config, optimized_params)

    logger.info(f"Using unified backtester with ML {'enabled' if use_ml else 'disabled'}")

    try:
        # Set up file paths for final backtest
        file_paths = {
            'trade_log': os.path.join(combined_dir, 'trade_log.csv'),
            'portfolio_value': os.path.join(combined_dir, 'portfolio_value.csv'),
            'regime_log': os.path.join(combined_dir, 'regime_log.csv'),
            'market_type_log': os.path.join(combined_dir, 'market_type_log.csv'),
            'summary': os.path.join(combined_dir, 'summary.txt')
        }

        # Reset any state variables to ensure a clean run
        reset_state_variables()

        # Run final backtest
        if use_ml:
            trades, portfolio_values, _, ml_metrics, _ = backtest_func(
                df.copy(),
                visualize_trades=True,
                file_paths=file_paths
            )
        else:
            trades, portfolio_values, _, _, _, _, _ = backtest_func(
                df.copy(),
                visualize_trades=True,
                file_paths=file_paths
            )

        # Calculate final performance metrics
        if trades and len(portfolio_values) > 0:
            portfolio_series = pd.Series(portfolio_values, index=df['date'][:len(portfolio_values)])
            metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])

            # Add metrics to optimized parameters
            optimized_params['final_metrics'] = {
                'profit_loss': metrics['profit_loss'],
                'return_pct': metrics['total_return_pct'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'profit_factor': metrics['profit_factor'],
                'num_trades': len(trades)
            }

            # Log final results
            logger.info(f"Final optimization results:")
            logger.info(f"  Profit/Loss: ${metrics['profit_loss']:.2f}")
            logger.info(f"  Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
            logger.info(f"  Trades: {len(trades)}")
    except Exception as e:
        logger.error(f"Error running final backtest: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Save optimized parameters
    try:
        with open(os.path.join(output_dir, "optimized_parameters.json"), 'w') as f:
            json.dump(optimized_params, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving optimized parameters: {e}")

    # Write a summary file
    with open(os.path.join(output_dir, "optimization_summary.txt"), 'w') as f:
        f.write("PARAMETER OPTIMIZATION SUMMARY\n")
        f.write("=============================\n\n")

        f.write("OPTIMIZED PARAMETERS:\n")
        f.write("-------------------\n")

        # Write HMM parameters
        f.write("HMM & Regime Detection Parameters:\n")
        for param, value in best_hmm_params.items():
            f.write(f"  {param}: {value}\n")

        # Write exit parameters
        f.write("\nExit Strategy Parameters:\n")
        for param, value in best_exit_params.items():
            f.write(f"  {param}: {value}\n")

        # Write final performance metrics
        if 'final_metrics' in optimized_params:
            metrics = optimized_params['final_metrics']
            f.write("\nFINAL PERFORMANCE WITH OPTIMIZED PARAMETERS:\n")
            f.write("------------------------------------------\n")
            f.write(f"Profit/Loss: ${metrics['profit_loss']:.2f}\n")
            f.write(f"Return: {metrics['return_pct']:.2f}%\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Number of Trades: {metrics['num_trades']}\n")

    # Restore original config
    for key in original_config:
        config[key] = original_config[key]

    logger.info(f"Optimization complete. Results saved to {output_dir}")
    return optimized_params


def optimize_with_optuna(df, output_dir, n_trials=100, focus_sharpe=True):
    """
    Optimize strategy parameters using Optuna to maximize Sharpe ratio or other objectives.

    Args:
        df: DataFrame with price data
        output_dir: Directory to save results
        n_trials: Number of optimization trials to run
        focus_sharpe: Whether to focus optimization on Sharpe ratio

    Returns:
        Dictionary with optimized parameters
    """
    import optuna
    from optuna.samplers import TPESampler
    import os
    import json
    import matplotlib.pyplot as plt

    # Create output directory for Optuna results
    optuna_dir = os.path.join(output_dir, "optuna_optimization")
    if not os.path.exists(optuna_dir):
        os.makedirs(optuna_dir)

    # Save original config
    original_config = copy.deepcopy(config)

    # Define objective function
    def objective(trial):
        # Reset HMM and other state before each trial
        reset_state_variables()

        # Define parameters to optimize
        params = {
            # Risk management parameters
            'risk.risk_per_trade': trial.suggest_float('risk_per_trade', 0.005, 0.015),
            'risk.atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 3.0),
            'risk.trailing_stop_atr_multiplier': trial.suggest_float('trailing_stop_multiplier', 1.0, 2.5),
            'risk.dynamic_target_atr_multiplier': trial.suggest_float('dynamic_target_multiplier', 0.8, 2.0),
            'risk.max_bars_held': trial.suggest_int('max_bars_held', 6, 20),

            # HMM detector parameters
            'hmm_detector.lookback_days': trial.suggest_int('lookback_days', 20, 40),
            'hmm_detector.retrain_frequency': trial.suggest_int('retrain_frequency', 5, 15),
            'hmm_detector.min_samples': trial.suggest_int('min_samples', 150, 300),

            # Market type parameters
            'market_type.trend_following.min_regime_score': trial.suggest_float('tf_min_score', 65, 80),
            'market_type.mean_reverting.min_regime_score': trial.suggest_float('mr_min_score', 35, 50),
            'market_type.neutral.min_regime_score': trial.suggest_float('neutral_min_score', 45, 60),

            # Position sizing parameters
            'position_sizing.max_size_adjustment': trial.suggest_float('max_size_adjustment', 1.0, 2.0),
            'position_sizing.min_size_adjustment': trial.suggest_float('min_size_adjustment', 0.5, 0.9)
        }

        # Update config with trial parameters
        for param, value in params.items():
            update_config_with_params(config, {param: value})

        # Set up file paths for this trial
        trial_dir = os.path.join(optuna_dir, f"trial_{trial.number}")
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        file_paths = {
            'trade_log': os.path.join(trial_dir, 'trade_log.csv'),
            'portfolio_value': os.path.join(trial_dir, 'portfolio_value.csv'),
            'regime_log': os.path.join(trial_dir, 'regime_log.csv'),
            'market_type_log': os.path.join(trial_dir, 'market_type_log.csv'),
            'summary': os.path.join(trial_dir, 'summary.txt')
        }

        # Run backtest
        try:
            trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
                df.copy(), visualize_trades=False, file_paths=file_paths
            )

            # Calculate performance metrics
            if trades and len(portfolio_values) > 0:
                portfolio_series = pd.Series(portfolio_values, index=df['date'][:len(portfolio_values)])
                metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])

                # Extract key metrics
                sharpe_ratio = metrics['sharpe_ratio']
                profit_loss = metrics['profit_loss']
                max_drawdown = metrics['max_drawdown_pct']
                win_rate = metrics['win_rate']

                # Decide what to optimize
                if focus_sharpe:
                    # Primarily optimize Sharpe ratio
                    objective_value = sharpe_ratio
                else:
                    # Create composite score (balancing profit and risk)
                    profit_component = profit_loss / config['account']['initial_capital']
                    risk_component = 1.0 - (max_drawdown / 100)  # Transform to 0-1 scale where higher is better

                    # Weight components: 50% profit, 30% Sharpe, 20% risk control
                    objective_value = (0.5 * profit_component) + (0.3 * sharpe_ratio) + (0.2 * risk_component)

                # Log the results
                logger.info(f"Trial {trial.number}: Sharpe={sharpe_ratio:.3f}, P/L=${profit_loss:.2f}, "
                            f"DD={max_drawdown:.2f}%, Win={win_rate:.2f}%")

                # Record additional metrics in trial user attributes
                trial.set_user_attr('profit_loss', float(profit_loss))
                trial.set_user_attr('max_drawdown', float(max_drawdown))
                trial.set_user_attr('win_rate', float(win_rate))
                trial.set_user_attr('trade_count', len(trades))

                return objective_value
            else:
                # No trades executed, return a poor score
                logger.warning(f"Trial {trial.number}: No trades executed")
                return -1.0 if focus_sharpe else 0.0

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            return -1.0 if focus_sharpe else 0.0
        finally:
            # Restore original config
            for key in original_config:
                config[key] = original_config[key]

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config['global']['random_seed']),
        study_name="strategy_optimization"
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial

    # Print best results
    logger.info("\n===== OPTIMIZATION RESULTS =====")
    logger.info(f"Best trial: #{best_trial.number}")
    logger.info(f"Value: {best_trial.value:.4f}")
    logger.info(f"Params: {best_params}")
    logger.info(f"Profit/Loss: ${best_trial.user_attrs['profit_loss']:.2f}")
    logger.info(f"Maximum Drawdown: {best_trial.user_attrs['max_drawdown']:.2f}%")
    logger.info(f"Win Rate: {best_trial.user_attrs['win_rate']:.2f}%")

    # Export study results
    try:
        # Convert study to DataFrame
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(optuna_dir, "optimization_results.csv"), index=False)

        # Save best parameters
        best_dict = {
            'params': best_params,
            'metrics': {
                'objective_value': float(best_trial.value),
                'profit_loss': float(best_trial.user_attrs['profit_loss']),
                'max_drawdown': float(best_trial.user_attrs['max_drawdown']),
                'win_rate': float(best_trial.user_attrs['win_rate']),
                'trade_count': int(best_trial.user_attrs['trade_count'])
            }
        }

        with open(os.path.join(output_dir, "best_optuna_params.json"), 'w') as f:
            json.dump(best_dict, f, indent=4)

        # Generate optimization visualizations
        try:
            # Optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(os.path.join(optuna_dir, "optimization_history.png"))

            # Parameter importance
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(optuna_dir, "param_importances.png"))

            # Parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(optuna_dir, "parallel_coordinate.png"))

            # Slice plots for top parameters
            param_importances = optuna.importance.get_param_importances(study)
            top_params = list(param_importances.keys())[:3]
            for param in top_params:
                fig = optuna.visualization.plot_slice(study, params=[param])
                fig.write_image(os.path.join(optuna_dir, f"slice_{param}.png"))
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    except Exception as e:
        logger.error(f"Error exporting study results: {e}")

    # Return best parameters with correct formatting for config
    optimized_params = {}
    for param_name, value in best_params.items():
        # Map back to config parameter paths
        if param_name == 'risk_per_trade':
            optimized_params['risk.risk_per_trade'] = value
        elif param_name == 'atr_stop_multiplier':
            optimized_params['risk.atr_stop_multiplier'] = value
        elif param_name == 'trailing_stop_multiplier':
            optimized_params['risk.trailing_stop_atr_multiplier'] = value
        elif param_name == 'dynamic_target_multiplier':
            optimized_params['risk.dynamic_target_atr_multiplier'] = value
        elif param_name == 'max_bars_held':
            optimized_params['risk.max_bars_held'] = value
        elif param_name == 'lookback_days':
            optimized_params['hmm_detector.lookback_days'] = value
        elif param_name == 'retrain_frequency':
            optimized_params['hmm_detector.retrain_frequency'] = value
        elif param_name == 'min_samples':
            optimized_params['hmm_detector.min_samples'] = value
        elif param_name == 'tf_min_score':
            optimized_params['market_type.trend_following.min_regime_score'] = value
        elif param_name == 'mr_min_score':
            optimized_params['market_type.mean_reverting.min_regime_score'] = value
        elif param_name == 'neutral_min_score':
            optimized_params['market_type.neutral.min_regime_score'] = value
        elif param_name == 'max_size_adjustment':
            optimized_params['position_sizing.max_size_adjustment'] = value
        elif param_name == 'min_size_adjustment':
            optimized_params['position_sizing.min_size_adjustment'] = value

    return optimized_params

def update_config_from_json(json_file):
    """
    Update global config from a JSON file with optimized parameters.

    Args:
        json_file: Path to JSON file with parameters

    Returns:
        Dictionary with loaded parameters
    """
    try:
        with open(json_file, 'r') as f:
            params = json.load(f)

        # Update config
        if isinstance(params, dict):
            for param, value in params.items():
                # Skip non-parameter entries
                if param == 'final_metrics':
                    continue

                # Update config
                update_config_with_params(config, {param: value})

            logger.info(f"Updated config with parameters from {json_file}")
            return params
    except Exception as e:
        logger.error(f"Error loading parameters from {json_file}: {e}")

    return {}

def main():
    """Main execution function for parameter optimization"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Optimize trading strategy parameters")
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--ml', action='store_true', help='Use ML-enhanced backtester')
    parser.add_argument('--regime-only', action='store_true', help='Optimize only regime parameters')
    parser.add_argument('--exit-only', action='store_true', help='Optimize only exit parameters')
    parser.add_argument('--load-params', type=str, help='Load parameters from JSON file')
    parser.add_argument('--periods', type=int, default=3, help='Number of periods for walk-forward testing')

    args = parser.parse_args()

    # Check if we should use the configuration section instead of command line args
    if USE_CONFIG_SECTION and '--help' not in sys.argv and len(sys.argv) == 1:
        logger.info("Using parameters from configuration section")

        # Set data path and dates from configuration
        config['data']['file_path'] = DATA_FILE
        config['data']['start_date'] = START_DATE
        config['data']['end_date'] = END_DATE

        # Set output directory
        output_dir = OUTPUT_DIR

        # Set ML usage flag
        use_ml = USE_ML

        # Set walk-forward periods
        periods = WALK_FORWARD_PERIODS

        # Determine optimization type
        regime_only = OPTIMIZATION_TYPE == 'regime_only'
        exit_only = OPTIMIZATION_TYPE == 'exit_only'

        # Use parameter grids from config section
        hmm_param_grid = HMM_PARAM_GRID
        exit_param_grid = EXIT_PARAM_GRID
    else:
        # Override config settings if specified
        if args.data:
            config['data']['file_path'] = args.data
        if args.start:
            config['data']['start_date'] = args.start
        if args.end:
            config['data']['end_date'] = args.end

        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output if args.output else f"optimization_results_{timestamp}"

        # Load parameters from JSON if specified
        if args.load_params:
            update_config_from_json(args.load_params)

        # Set other parameters from command line
        use_ml = args.ml
        periods = args.periods
        regime_only = args.regime_only
        exit_only = args.exit_only

        # Use default parameter grids
        hmm_param_grid = {
            'hmm_detector.lookback_days': [15, 20, 25],
            'hmm_detector.retrain_frequency': [5, 7, 10],
            'market_type.update_frequency': [3, 5, 7]
        }

        exit_param_grid = {
            'risk.enable_trailing_stop': [True, False],
            'risk.trailing_stop_atr_multiplier': [1.5, 2.0, 2.5, 3.0],
            'risk.atr_stop_multiplier': [1.5, 2.0, 2.5],
            'risk.dynamic_target_enable': [True, False],
            'risk.dynamic_target_atr_multiplier': [0.8, 1.0, 1.2, 1.5],
            'risk.max_bars_held': [8, 12, 16]
        }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and prepare data
    logger.info("Loading and processing data...")
    df = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    if df is None or len(df) == 0:
        logger.error("No data available after loading. Exiting.")
        return

    # Calculate all indicators
    df = calculate_indicators(df, config)

    # Run optimization based on parameters
    if regime_only:
        # Only optimize regime parameters
        logger.info("Optimizing only regime detection parameters...")
        best_params = perform_walk_forward_optimization(
            df, output_dir, hmm_param_grid, num_folds=periods, use_ml=use_ml
        )
    elif exit_only:
        # Only optimize exit parameters
        logger.info("Optimizing only exit strategy parameters...")
        best_params = perform_walk_forward_optimization(
            df, output_dir, exit_param_grid, num_folds=periods, use_ml=use_ml
        )
    else:
        # Full optimization of both parameter sets
        logger.info("Running comprehensive parameter optimization...")
        best_params = optimize_strategy_parameters(
            df, output_dir, hmm_param_grid, exit_param_grid, use_ml=use_ml
        )

    # Print results
    if best_params:
        logger.info("\nOptimized Parameters:")
        for param, value in best_params.items():
            if param != 'final_metrics':
                logger.info(f"  {param}: {value}")

        if 'final_metrics' in best_params:
            metrics = best_params['final_metrics']
            logger.info("\nFinal Performance Metrics:")
            logger.info(f"  Profit/Loss: ${metrics['profit_loss']:.2f}")
            logger.info(f"  Return: {metrics['return_pct']:.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")

    logger.info(f"\nDetails saved to {output_dir}")

if __name__ == "__main__":
    import sys

    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback

        logger.error(traceback.format_exc())