"""
walk_forward_optimizer.py - Implements walk-forward optimization with Optuna

This module extends the parameter optimization framework with walk-forward validation
to ensure that the optimized parameters are robust across different market periods.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import copy

# Import from central config
from config import config

# Import backtester functions
from unified_backtester import run_backtest

# Import utility functions
from utils import load_and_process_data, calculate_indicators, setup_directories
# Import reset_hmm_detector from backtester_common
from backtester_common import reset_hmm_detector

# Import analysis functions
from trade_analysis import analyze_performance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_time_periods(start_date, end_date, num_periods=4, in_sample_pct=0.7):
    """
    Create time periods for walk-forward optimization.
    
    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        num_periods: Number of periods to create
        in_sample_pct: Percentage of each period to use for in-sample optimization
        
    Returns:
        List of dictionaries with period information
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    total_days = (end - start).days
    period_days = total_days // num_periods
    
    periods = []
    
    for i in range(num_periods):
        period_start = start + timedelta(days=i * period_days)
        period_end = start + timedelta(days=(i + 1) * period_days - 1)
        
        # Make sure the last period ends at the overall end date
        if i == num_periods - 1:
            period_end = end
            
        # Calculate in-sample and out-of-sample dates
        in_sample_days = int(period_days * in_sample_pct)
        in_sample_end = period_start + timedelta(days=in_sample_days - 1)
        out_sample_start = in_sample_end + timedelta(days=1)
        
        periods.append({
            'period': i + 1,
            'in_sample': {
                'start': period_start,
                'end': in_sample_end
            },
            'out_sample': {
                'start': out_sample_start,
                'end': period_end
            }
        })
    
    return periods

def reset_state_variables():
    """Reset all stateful variables before each optimization run"""
    # Reset hmm_detector in backtester_common module
    try:
        import backtester_common
        backtester_common.reset_hmm_detector()
        logger.info("Reset hmm_detector in backtester_common")
    except Exception as e:
        logger.warning(f"Could not reset hmm_detector: {e}")

def update_config_with_params(config, params):
    """
    Update config with parameter values from a dictionary.
    
    Args:
        config: Configuration dictionary to update
        params: Dictionary with parameter paths and values
        
    Returns:
        Updated config dictionary
    """
    for param, value in params.items():
        # Parse parameter path, e.g., "hmm_detector.min_samples" -> config["hmm_detector"]["min_samples"]
        path = param.split('.')
        curr = config
        for p in path[:-1]:
            if p not in curr:
                curr[p] = {}
            curr = curr[p]
        curr[path[-1]] = value
    return config

def create_study_for_period(period_num, optimization_params, df, output_dir, n_trials=50, seed=None): #50
    """
    Create and run an Optuna study for a specific in-sample period.
    
    Args:
        period_num: Period number for identification
        optimization_params: Dictionary with optimization parameters
        df: Full DataFrame with price data
        output_dir: Base output directory
        n_trials: Number of optimization trials
        seed: Random seed for reproducibility
        
    Returns:
        Optuna study object with optimization results
    """
    # Get period information
    period = optimization_params['periods'][period_num - 1]
    in_sample_start = period['in_sample']['start']
    in_sample_end = period['in_sample']['end']
    
    # Create period-specific output directory
    period_dir = os.path.join(output_dir, f"period_{period_num}")
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    # Filter data for this period
    period_df = df[(df['date'] >= in_sample_start) & (df['date'] <= in_sample_end)].copy().reset_index(drop=True)
    
    if len(period_df) < 100:  # Ensure enough data
        logger.warning(f"Insufficient data for period {period_num}. Only {len(period_df)} rows.")
        return None
    
    # Save original config
    original_config = copy.deepcopy(config)
    
    # Define objective function
    def objective(trial):
        # Reset state before each trial
        reset_state_variables()
        
        # Define parameters to optimize
        param_dict = {}
        for param_key, param_config in optimization_params['param_space'].items():
            if param_config['type'] == 'float':
                param_dict[param_key] = trial.suggest_float(
                    param_key, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', None)
                )
            elif param_config['type'] == 'int':
                param_dict[param_key] = trial.suggest_int(
                    param_key, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'categorical':
                param_dict[param_key] = trial.suggest_categorical(
                    param_key, 
                    param_config['choices']
                )
                
        # Update config with trial parameters
        for param, value in param_dict.items():
            update_config_with_params(config, {param: value})
        
        # Set up file paths for this trial
        trial_dir = os.path.join(period_dir, f"trial_{trial.number}")
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
                period_df.copy(), 
                visualize_trades=False, 
                file_paths=file_paths, 
                use_ml=config['ml']['enable']
            )
            
            # Calculate performance metrics
            if trades and len(portfolio_values) > 0:
                portfolio_series = pd.Series(portfolio_values, index=period_df['date'][:len(portfolio_values)])
                metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
                
                # Extract metrics
                sharpe_ratio = metrics['sharpe_ratio']
                profit_loss = metrics['profit_loss']
                win_rate = metrics['win_rate']
                max_drawdown = metrics['max_drawdown_pct']
                
                # Store metrics in trial
                trial.set_user_attr('profit_loss', float(profit_loss))
                trial.set_user_attr('win_rate', float(win_rate))
                trial.set_user_attr('max_drawdown', float(max_drawdown))
                trial.set_user_attr('trade_count', len(trades))
                
                # Return Sharpe ratio as objective (higher is better)
                # Add penalties for constraint violation
                objective_value = sharpe_ratio
                
                # Penalty for excessive drawdown
                if max_drawdown > 25:
                    objective_value *= 0.7
                elif max_drawdown > 20:
                    objective_value *= 0.85
                
                # Penalty for too few trades (less statistical significance)
                min_trades = optimization_params.get('min_trades', 20)
                if len(trades) < min_trades:
                    penalty_factor = max(0.5, len(trades) / min_trades)
                    objective_value *= penalty_factor
                
                # Penalty for low win rate
                min_win_rate = optimization_params.get('min_win_rate', 50)
                if win_rate < min_win_rate:
                    penalty_factor = max(0.5, win_rate / min_win_rate)
                    objective_value *= penalty_factor
                
                # Log results
                logger.info(f"Period {period_num}, Trial {trial.number}: "
                           f"Sharpe={sharpe_ratio:.3f}, P/L=${profit_loss:.2f}, "
                           f"Win={win_rate:.1f}%, Trades={len(trades)}, OBJ={objective_value:.3f}")
                
                return objective_value
            else:
                # No trades executed
                logger.warning(f"Period {period_num}, Trial {trial.number}: No trades executed")
                return -1.0  # Penalty for no trades
                
        except Exception as e:
            logger.error(f"Error in Period {period_num}, Trial {trial.number}: {e}")
            return -1.0
        finally:
            # Restore original config
            for key in original_config:
                config[key] = original_config[key]
    
    # Create and run study
    study_name = f"period_{period_num}_optimization"
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name
    )
    
    logger.info(f"Starting optimization for Period {period_num}: {in_sample_start.date()} to {in_sample_end.date()}")
    study.optimize(objective, n_trials=n_trials)
    
    # Save study results
    try:
        # Convert to DataFrame for analysis
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(period_dir, "trials.csv"), index=False)
        
        # Save best parameters
        best_params = study.best_params
        best_trial = study.best_trial
        
        best_dict = {
            'period': period_num,
            'in_sample_start': in_sample_start.strftime('%Y-%m-%d'),
            'in_sample_end': in_sample_end.strftime('%Y-%m-%d'),
            'params': best_params,
            'value': best_trial.value,
            'metrics': {
                'profit_loss': best_trial.user_attrs.get('profit_loss', 0),
                'win_rate': best_trial.user_attrs.get('win_rate', 0),
                'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0),
                'trade_count': best_trial.user_attrs.get('trade_count', 0)
            }
        }
        
        with open(os.path.join(period_dir, "best_params.json"), 'w') as f:
            json.dump(best_dict, f, indent=4)
        
        # Generate visualizations
        try:
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(period_dir, "optimization_history.png"))
            
            # Parameter importance
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(period_dir, "param_importances.png"))
        except Exception as e:
            logger.error(f"Error generating visualizations for period {period_num}: {e}")
    
    except Exception as e:
        logger.error(f"Error saving study results for period {period_num}: {e}")
    
    return study

def evaluate_parameters(params, df, period, output_dir, use_ml=False):
    """
    Evaluate a set of parameters on a specific period.
    
    Args:
        params: Dictionary of parameters to evaluate
        df: DataFrame with price data
        period: Dictionary with period information
        output_dir: Directory to save results
        use_ml: Whether to use ML enhancement
        
    Returns:
        Dictionary with evaluation results
    """
    # Save original config
    original_config = copy.deepcopy(config)
    
    # Extract period dates
    start_date = period['start']
    end_date = period['end']
    
    # Filter data for this period
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy().reset_index(drop=True)
    
    if len(period_df) < 50:  # Ensure enough data
        logger.warning(f"Insufficient data for evaluation. Only {len(period_df)} rows.")
        return None
    
    # Update config with parameters
    for param, value in params.items():
        update_config_with_params(config, {param: value})
    
    # Set up file paths
    test_dir = os.path.join(output_dir, f"test_{start_date.strftime('%Y%m%d')}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    file_paths = {
        'trade_log': os.path.join(test_dir, 'trade_log.csv'),
        'portfolio_value': os.path.join(test_dir, 'portfolio_value.csv'),
        'regime_log': os.path.join(test_dir, 'regime_log.csv'),
        'market_type_log': os.path.join(test_dir, 'market_type_log.csv'),
        'summary': os.path.join(test_dir, 'summary.txt')
    }
    
    # Reset state variables
    reset_state_variables()
    
    # Run backtest
    try:
        trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
            period_df.copy(), 
            visualize_trades=False, 
            file_paths=file_paths, 
            use_ml=use_ml
        )
        
        # Calculate performance metrics
        if trades and len(portfolio_values) > 0:
            portfolio_series = pd.Series(portfolio_values, index=period_df['date'][:len(portfolio_values)])
            metrics = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
            
            # Return evaluation results
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'sharpe_ratio': metrics['sharpe_ratio'],
                'profit_loss': metrics['profit_loss'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown_pct'],
                'trade_count': len(trades),
                'return_pct': metrics['total_return_pct']
            }
        else:
            # No trades executed
            logger.warning(f"No trades executed in evaluation period {start_date} to {end_date}")
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'sharpe_ratio': -1,
                'profit_loss': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'trade_count': 0,
                'return_pct': 0
            }
            
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return None
    finally:
        # Restore original config
        for key in original_config:
            config[key] = original_config[key]

def walk_forward_optimization(df, output_dir, optimization_params):
    """
    Perform walk-forward optimization.
    
    Args:
        df: DataFrame with price data
        output_dir: Directory to save results
        optimization_params: Dictionary with optimization parameters
        
    Returns:
        Dictionary with optimized parameters and performance metrics
    """
    # Initialize results storage
    period_results = []
    best_period_params = []
    out_of_sample_results = []
    
    # Get periods
    periods = optimization_params['periods']
    
    # Loop through periods
    for period_num in range(1, len(periods) + 1):
        logger.info(f"Processing Period {period_num}/{len(periods)}")
        
        # Optimize parameters on in-sample data
        study = create_study_for_period(
            period_num, 
            optimization_params, 
            df, 
            output_dir,
            n_trials=optimization_params.get('n_trials', 50), 
            seed=optimization_params.get('seed', 42)
        )
        
        if study is None:
            logger.warning(f"Skipping Period {period_num} due to optimization failure")
            continue
        
        # Get best parameters
        best_params = study.best_params
        best_trial = study.best_trial
        
        # Convert Optuna parameter names to config paths
        config_params = {}
        for param, value in best_params.items():
            config_params[param] = value
        
        # Store best parameters
        best_period_params.append({
            'period': period_num,
            'params': config_params,
            'in_sample_metrics': {
                'sharpe_ratio': best_trial.value,
                'profit_loss': best_trial.user_attrs.get('profit_loss', 0),
                'win_rate': best_trial.user_attrs.get('win_rate', 0),
                'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0),
                'trade_count': best_trial.user_attrs.get('trade_count', 0)
            }
        })
        
        # Evaluate on out-of-sample data
        out_sample_period = {
            'start': periods[period_num - 1]['out_sample']['start'],
            'end': periods[period_num - 1]['out_sample']['end']
        }
        
        out_sample_result = evaluate_parameters(
            config_params, 
            df, 
            out_sample_period, 
            os.path.join(output_dir, f"period_{period_num}", "out_of_sample"),
            use_ml=optimization_params.get('use_ml', False)
        )
        
        if out_sample_result:
            out_sample_result['period'] = period_num
            out_of_sample_results.append(out_sample_result)
            
            logger.info(f"Period {period_num} out-of-sample results: "
                      f"Sharpe={out_sample_result['sharpe_ratio']:.3f}, "
                      f"P/L=${out_sample_result['profit_loss']:.2f}")
    
    # Combine results
    results = {
        'periods': periods,
        'best_period_params': best_period_params,
        'out_of_sample_results': out_of_sample_results
    }
    
    # Calculate overall best parameters (most consistent across periods)
    if best_period_params:
        # Group parameters by parameter name
        param_groups = {}
        for period_result in best_period_params:
            for param, value in period_result['params'].items():
                if param not in param_groups:
                    param_groups[param] = []
                param_groups[param].append(value)
        
        # Calculate median value for each parameter
        median_params = {}
        for param, values in param_groups.items():
            if isinstance(values[0], (int, float)):
                median_params[param] = float(np.median(values))
                if param.endswith('.retrain_frequency') or param.endswith('.lookback_days') or param.endswith('.min_samples') or param.endswith('.max_bars_held'):
                    median_params[param] = int(median_params[param])
            else:
                # For categorical parameters, use mode
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                median_params[param] = most_common
        
        results['median_params'] = median_params
        
        # Find the best performing parameter set
        if out_of_sample_results:
            best_sharpe_result = max(out_of_sample_results, key=lambda x: x['sharpe_ratio'])
            best_sharpe_period = best_sharpe_result['period']
            
            # Get parameters from best performing period
            for period_result in best_period_params:
                if period_result['period'] == best_sharpe_period:
                    results['best_sharpe_params'] = period_result['params']
                    break
    
    # Save combined results
    try:
        with open(os.path.join(output_dir, "walk_forward_results.json"), 'w') as f:
            # Convert any non-serializable values to strings
            serializable_results = copy.deepcopy(results)
            for key, value in serializable_results.items():
                if isinstance(value, (pd.Timestamp, datetime)):
                    serializable_results[key] = value.strftime('%Y-%m-%d')
                    
            json.dump(serializable_results, f, indent=4)
            
        # Generate summary report
        create_walk_forward_summary(results, output_dir)
        
        # Generate visualizations
        create_walk_forward_visualizations(results, output_dir)
    
    except Exception as e:
        logger.error(f"Error saving walk-forward results: {e}")
    
    return results

def create_walk_forward_summary(results, output_dir):
    """
    Create a summary report for walk-forward optimization.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save the report
    """
    summary_path = os.path.join(output_dir, "walk_forward_summary.txt")
    
    try:
        with open(summary_path, 'w') as f:
            f.write("WALK-FORWARD OPTIMIZATION SUMMARY\n")
            f.write("================================\n\n")
            
            # Period information
            f.write("PERIOD INFORMATION\n")
            f.write("-----------------\n")
            for i, period in enumerate(results['periods']):
                f.write(f"Period {i+1}:\n")
                f.write(f"  In-Sample:  {period['in_sample']['start'].strftime('%Y-%m-%d')} to {period['in_sample']['end'].strftime('%Y-%m-%d')}\n")
                f.write(f"  Out-Sample: {period['out_sample']['start'].strftime('%Y-%m-%d')} to {period['out_sample']['end'].strftime('%Y-%m-%d')}\n")
            f.write("\n")
            
            # Out-of-sample performance
            if results.get('out_of_sample_results'):
                f.write("OUT-OF-SAMPLE PERFORMANCE\n")
                f.write("------------------------\n")
                f.write(f"{'Period':6} | {'Sharpe':10} | {'P/L':12} | {'Win Rate':10} | {'Drawdown':10} | {'Trades':6}\n")
                f.write("-" * 65 + "\n")
                
                for result in results['out_of_sample_results']:
                    f.write(f"{result['period']:6} | ")
                    f.write(f"{result['sharpe_ratio']:10.3f} | ")
                    f.write(f"${result['profit_loss']:10.2f} | ")
                    f.write(f"{result['win_rate']:10.2f}% | ")
                    f.write(f"{result['max_drawdown']:10.2f}% | ")
                    f.write(f"{result['trade_count']:6}\n")
                
                # Calculate averages
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in results['out_of_sample_results']])
                avg_pl = np.mean([r['profit_loss'] for r in results['out_of_sample_results']])
                avg_win = np.mean([r['win_rate'] for r in results['out_of_sample_results']])
                avg_dd = np.mean([r['max_drawdown'] for r in results['out_of_sample_results']])
                avg_trades = np.mean([r['trade_count'] for r in results['out_of_sample_results']])
                
                f.write("-" * 65 + "\n")
                f.write(f"{'Avg':6} | ")
                f.write(f"{avg_sharpe:10.3f} | ")
                f.write(f"${avg_pl:10.2f} | ")
                f.write(f"{avg_win:10.2f}% | ")
                f.write(f"{avg_dd:10.2f}% | ")
                f.write(f"{avg_trades:6.1f}\n\n")
            
            # Best parameters
            if results.get('median_params'):
                f.write("RECOMMENDED PARAMETERS (MEDIAN ACROSS PERIODS)\n")
                f.write("-------------------------------------------\n")
                for param, value in results['median_params'].items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")
            
            if results.get('best_sharpe_params'):
                f.write("BEST PERFORMING PARAMETERS (HIGHEST SHARPE RATIO)\n")
                f.write("----------------------------------------------\n")
                for param, value in results['best_sharpe_params'].items():
                    f.write(f"{param}: {value}\n")
                
            f.write("\n\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        logger.info(f"Walk-forward summary saved to {summary_path}")
    
    except Exception as e:
        logger.error(f"Error creating walk-forward summary: {e}")

def create_walk_forward_visualizations(results, output_dir):
    """
    Create visualizations for walk-forward optimization results.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save visualizations
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    try:
        # 1. Plot out-of-sample performance metrics
        if results.get('out_of_sample_results'):
            # Convert to DataFrame
            out_sample_df = pd.DataFrame(results['out_of_sample_results'])
            
            # Plot Sharpe ratio
            plt.figure(figsize=(10, 6))
            plt.bar(out_sample_df['period'], out_sample_df['sharpe_ratio'], color='blue')
            plt.axhline(y=out_sample_df['sharpe_ratio'].mean(), color='red', linestyle='--', 
                      label=f"Average: {out_sample_df['sharpe_ratio'].mean():.3f}")
            plt.title('Out-of-Sample Sharpe Ratio by Period')
            plt.xlabel('Period')
            plt.ylabel('Sharpe Ratio')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "out_sample_sharpe.png"), dpi=150)
            plt.close()
            
            # Plot profit/loss
            plt.figure(figsize=(10, 6))
            plt.bar(out_sample_df['period'], out_sample_df['profit_loss'], 
                   color=['green' if p > 0 else 'red' for p in out_sample_df['profit_loss']])
            plt.axhline(y=out_sample_df['profit_loss'].mean(), color='blue', linestyle='--', 
                      label=f"Average: ${out_sample_df['profit_loss'].mean():.2f}")
            plt.title('Out-of-Sample Profit/Loss by Period')
            plt.xlabel('Period')
            plt.ylabel('Profit/Loss ($)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "out_sample_profit.png"), dpi=150)
            plt.close()
            
            # Plot win rate
            plt.figure(figsize=(10, 6))
            plt.bar(out_sample_df['period'], out_sample_df['win_rate'], color='purple')
            plt.axhline(y=out_sample_df['win_rate'].mean(), color='red', linestyle='--', 
                      label=f"Average: {out_sample_df['win_rate'].mean():.2f}%")
            plt.axhline(y=50, color='black', linestyle=':', label="50% Break-even")
            plt.title('Out-of-Sample Win Rate by Period')
            plt.xlabel('Period')
            plt.ylabel('Win Rate (%)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "out_sample_win_rate.png"), dpi=150)
            plt.close()
            
            # Plot drawdown
            plt.figure(figsize=(10, 6))
            plt.bar(out_sample_df['period'], out_sample_df['max_drawdown'], color='orange')
            plt.axhline(y=out_sample_df['max_drawdown'].mean(), color='red', linestyle='--', 
                      label=f"Average: {out_sample_df['max_drawdown'].mean():.2f}%")
            plt.title('Out-of-Sample Maximum Drawdown by Period')
            plt.xlabel('Period')
            plt.ylabel('Maximum Drawdown (%)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(viz_dir, "out_sample_drawdown.png"), dpi=150)
            plt.close()
        
        # 2. Plot parameter consistency across periods
        if results.get('best_period_params'):
            # Extract parameters from each period
            param_df = pd.DataFrame([
                {**{'period': p['period']}, **p['params']} 
                for p in results['best_period_params']
            ])
            
            # Plot each parameter across periods
            for param in [col for col in param_df.columns if col != 'period']:
                plt.figure(figsize=(10, 6))
                
                # Check if parameter is numeric
                if pd.api.types.is_numeric_dtype(param_df[param]):
                    plt.plot(param_df['period'], param_df[param], 'o-', color='blue')
                    
                    # Add median line
                    median_value = np.median(param_df[param])
                    plt.axhline(y=median_value, color='red', linestyle='--', 
                               label=f"Median: {median_value:.4f}")
                    
                    # Add range
                    plt.fill_between(param_df['period'], 
                                    param_df[param].min(), 
                                    param_df[param].max(), 
                                    alpha=0.2, color='blue')
                else:
                    # For categorical parameters, create a bar chart with counts
                    value_counts = param_df[param].value_counts()
                    plt.bar(value_counts.index, value_counts.values, color='blue')
                    plt.xticks(rotation=45)
                
                plt.title(f'Parameter: {param} Across Periods')
                plt.xlabel('Period' if pd.api.types.is_numeric_dtype(param_df[param]) else 'Value')
                plt.ylabel('Value' if pd.api.types.is_numeric_dtype(param_df[param]) else 'Count')
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"param_{param.replace('.', '_')}.png"), dpi=150)
                plt.close()
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")


def save_walk_forward_results(results, output_dir):
    """
    Save walk-forward optimization results to a JSON file,
    handling Timestamp serialization.

    Args:
        results: Dictionary of optimization results
        output_dir: Directory to save results
    """
    import json
    import os
    from datetime import datetime

    # Import the serialization function from optimize_and_test
    from optimize_and_test import convert_to_serializable

    try:
        # Create serializable version of results
        serializable_results = convert_to_serializable(results)

        # Save to file
        output_file = os.path.join(output_dir, 'walk_forward_results.json')
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        logger.info(f"Walk-forward results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving walk-forward results: {e}")


def run_walk_forward_optimization(data_file, start_date, end_date, num_periods=4,
                                  n_trials=50, use_ml=False, output_dir=None):
    """
    Run the full walk-forward optimization process.

    Args:
        data_file: Path to the data file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        num_periods: Number of periods for walk-forward testing
        n_trials: Number of trials per period
        use_ml: Whether to use ML enhancement
        output_dir: Directory to save results (optional)

    Returns:
        Dictionary with optimization results
    """
    # Override config settings
    config['data']['file_path'] = data_file
    config['data']['start_date'] = start_date
    config['data']['end_date'] = end_date
    config['ml']['enable'] = use_ml

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"walk_forward_{timestamp}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and process data
    logger.info(f"Loading data from {data_file}")
    df = load_and_process_data(data_file, start_date, end_date)

    if df is None or len(df) == 0:
        logger.error("No data available after loading. Exiting.")
        return None

    # Calculate indicators
    df = calculate_indicators(df, config)

    # Create time periods
    periods = create_time_periods(start_date, end_date, num_periods)

    # Define optimization parameters
    optimization_params = {
        'periods': periods,
        'n_trials': n_trials,
        'min_trades': 20,
        'min_win_rate': 55,
        'use_ml': use_ml,
        'seed': config['global']['random_seed'],
        'param_space': {
            # Risk management parameters
            'risk.risk_per_trade': {'type': 'float', 'low': 0.006, 'high': 0.015, 'step': 0.001},
            'risk.atr_stop_multiplier': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.1},
            'risk.trailing_stop_atr_multiplier': {'type': 'float', 'low': 1.0, 'high': 2.5, 'step': 0.1},
            'risk.dynamic_target_atr_multiplier': {'type': 'float', 'low': 0.8, 'high': 2.0, 'step': 0.1},
            'risk.max_bars_held': {'type': 'int', 'low': 6, 'high': 20, 'step': 2},

            # HMM detector parameters
            'hmm_detector.lookback_days': {'type': 'int', 'low': 20, 'high': 40, 'step': 5},
            'hmm_detector.retrain_frequency': {'type': 'int', 'low': 5, 'high': 15, 'step': 2},
            'hmm_detector.min_samples': {'type': 'int', 'low': 150, 'high': 300, 'step': 25},

            # Market type parameters
            'market_type.trend_following.min_regime_score': {'type': 'float', 'low': 65, 'high': 80, 'step': 2.5},
            'market_type.mean_reverting.min_regime_score': {'type': 'float', 'low': 35, 'high': 50, 'step': 2.5},
            'market_type.neutral.min_regime_score': {'type': 'float', 'low': 45, 'high': 60, 'step': 2.5},

            # Position sizing parameters
            'position_sizing.max_size_adjustment': {'type': 'float', 'low': 1.0, 'high': 2.0, 'step': 0.1},
            'position_sizing.min_size_adjustment': {'type': 'float', 'low': 0.5, 'high': 0.9, 'step': 0.1}
        }
    }

    # Run walk-forward optimization
    logger.info("Starting walk-forward optimization")
    results = walk_forward_optimization(df, output_dir, optimization_params)

    # Import serialization function from optimize_and_test
    try:
        from optimize_and_test import convert_to_serializable

        # Save the results with proper serialization
        if results:
            # Create serializable version of results
            serializable_results = convert_to_serializable(results)

            # Save to file
            output_file = os.path.join(output_dir, 'walk_forward_results.json')
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)

            logger.info(f"Walk-forward results saved to {output_file}")

            if results.get('median_params'):
                logger.info("\n===== WALK-FORWARD OPTIMIZATION COMPLETE =====")
                logger.info("Recommended parameters (median values across periods):")
                for param, value in results['median_params'].items():
                    logger.info(f"  {param}: {value}")
    except Exception as e:
        logger.error(f"Error saving walk-forward results: {e}")

    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run walk-forward optimization')
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--periods', type=int, default=4, help='Number of periods for walk-forward testing')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials per period')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancement')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed:
        config['global']['random_seed'] = args.seed
        config['global']['use_fixed_seed'] = True
    
    # Run optimization
    run_walk_forward_optimization(
        data_file=args.data or config['data']['file_path'],
        start_date=args.start or config['data']['start_date'],
        end_date=args.end or config['data']['end_date'],
        num_periods=args.periods,
        n_trials=args.trials,
        use_ml=args.ml,
        output_dir=args.output
    )