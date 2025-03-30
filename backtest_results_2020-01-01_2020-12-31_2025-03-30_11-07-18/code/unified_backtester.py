"""
unified_backtester.py - Unified Backtester for Standard and ML-Enhanced Strategy

This module provides a single, unified backtester that supports both standard and
ML-enhanced strategy execution, eliminating the need for separate backtester files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import os
import logging
import sys
import json
import argparse

# Import from centralized config
from config import config

# Import common backtester functions
from backtester_common import (
    initialize_hmm_detector, detect_market_type, get_market_type_params,
    calculate_regime_score, calculate_position_size_adjustment, 
    is_in_trading_window, check_entry_signal, check_entry_signal_ml, check_exit_conditions,
    calculate_position_size, reset_hmm_detector, get_regime_parameters  # Add get_regime_parameters here
)

# Import utility functions
from utils import (
    setup_directories, copy_project_files, load_and_process_data, 
    calculate_indicators, find_closest_weekday, calculate_season_dates, is_in_season, define_explicit_seasons
)

# Import visualization functions
from trade_visualization import visualize_trade, generate_performance_charts, generate_quarterly_analysis_charts

# Import ML predictor (only used when ML is enabled)
from simplified_ml_predictor import MLPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize HMM detector (accessed from backtester_common)
hmm_detector = None

def run_backtest(df, visualize_trades=True, file_paths=None, use_ml=None):
    """
    Run the backtest on the provided DataFrame with optional ML enhancement.
    
    Args:
        df: DataFrame with price data and indicators
        visualize_trades: Whether to generate trade visualizations
        file_paths: Dictionary of file paths for logging (optional)
        use_ml: Override for ML setting (if None, uses config value)
        
    Returns:
        Tuple containing:
        - trades_list: List of executed trades
        - portfolio_values: List of portfolio values over time
        - updated_df: DataFrame with additional columns added during backtest
        - regime_log: List of regime information for each bar
        - market_type_log: List of market type changes
        - regime_score_bins: Dictionary with regime score distribution
        - season_metrics: Dictionary with performance metrics by season
        - ml_metrics: Dictionary with ML metrics (if ML enabled, otherwise None)
    """
    # Initialize random seeds for reproducibility
    from utils import initialize_random_seeds
    initialize_random_seeds()
    logger.info("Random seeds initialized for backtest")

    # Determine if we're using ML
    ml_enabled = use_ml if use_ml is not None else config['ml']['enable']
    logger.info(f"ML mode: {'enabled' if ml_enabled else 'disabled'}")

    # Always initialize HMM detector at the start
    initialize_hmm_detector(os.path.dirname(file_paths['trade_log']) if file_paths else None)
    logger.info("HMM detector initialized for backtest")

    # If file_paths is not provided, create a default
    if file_paths is None:
        # Use setup_directories to get file paths
        output_dir, file_paths = setup_directories(
            config['data']['start_date'],
            config['data']['end_date'],
            "backtest_results"
        )
    
    # Initialize ML predictor if needed
    ml_predictor = None
    if ml_enabled:
        ml_dir = os.path.join(os.path.dirname(file_paths['trade_log']), 'ml')
        if not os.path.exists(ml_dir):
            os.makedirs(ml_dir)
            
        ml_predictor = MLPredictor(
            output_dir=ml_dir,
            model_type=config['ml']['model_type'],
            prediction_threshold=config['ml']['prediction_threshold'],
            retrain_frequency_days=config['ml']['retrain_frequency'],
            min_training_samples=config['ml']['min_training_samples']
        )
        
        logger.info(f"ML Configuration: model={config['ml']['model_type']}, threshold={config['ml']['prediction_threshold']}")
        logger.info(
            f"ML Training Parameters: warmup_trades={config['ml']['warmup_trades']}, min_samples={config['ml']['min_training_samples']}")
    
    logger.info("Starting backtest...")
    
    # Initialize ML metrics tracking if ML is enabled
    ml_metrics = None
    if ml_enabled:
        ml_metrics = {
            'potential_trades': 0,
            'executed_trades': 0,
            'skipped_by_ml': 0,
            'model_retrain_count': 0,
            'prediction_accuracy': 0
        }
        
        # Add HMM-related fields to ML metrics
        ml_metrics.update({
            'hmm_enabled': config['hmm_detector']['enable'],
            'hmm_influence': config['ml']['hmm_confidence_weight'],
            'hmm_regime_changes': 0,
            'hmm_ml_agreement': 0,
            'hmm_metrics': {}
        })
    
    # Initialize variables for backtesting
    account_value = config['account']['initial_capital']
    position = 0
    entry_price = 0
    stop_loss = 0
    profit_target = 0
    entry_bar = 0
    trades = []
    executed_trades = []  # Used for ML training
    skipped_by_ml = []    # Used for ML analysis
    portfolio_value = []
    trade_count = 0
    warmup_complete = False
    current_portion = 1.0  # Track what portion of the position is still open
    partial_exits = []  # Store partial exit information

    # Variables for regime tracking
    regime_log = []
    trades_skipped_regime = 0
    trades_taken_favorable = 0
    trades_taken_unfavorable = 0
    regime_score_bins = {'0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0}
    
    # Current market type tracking
    current_market_type = 'neutral'
    market_type_params = get_market_type_params(current_market_type)
    last_market_type_update = None
    market_type_log = []
    in_warmup = True  # Start assuming we're in warmup phase
    
    # Season tracking
    season_active = not config.get('seasons', {}).get('enable', False)  # Default to active if filter disabled
    current_season = None
    season_dates = {}
    trades_by_season = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': [], 'Out-of-Season': []}

    # Calculate season dates for all years in the dataset if filter enabled
    if config.get('seasons', {}).get('enable', False):
        # Get unique years in the dataset
        years = sorted(set(d.year for d in df['date'].dt.date))

        # Use explicit season definition instead of calculate_season_dates
        season_dates = define_explicit_seasons(years, config['seasons']['definitions'])

        # Log season periods for reference
        for season, date_ranges in season_dates.items():
            for idx, (start, end) in enumerate(date_ranges):
                logger.info(f"Season {season} (period {idx + 1}/{len(date_ranges)}): {start} to {end}")
    
    # Determine trade log path based on ML setting
    trade_log_path = file_paths['trade_log']
    
    # Create a separate ML trade log if ML is enabled
    ml_trade_log_path = file_paths.get('ml_trade_log')
    if ml_enabled and ml_trade_log_path is None:
        ml_trade_log_path = os.path.join(os.path.dirname(file_paths['trade_log']), 'ml_trade_log.csv')
    
    # Create logs - Just create file headers, don't keep files open
    with open(file_paths['regime_log'], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'date', 'adx', 'ma_slope', 'volatility_regime', 'atr_ratio', 
            'adx_score', 'slope_score',
            'regime_score', 'favorable_regime', 'position_size_adj', 'in_season', 'season'
        ])
    
    # Create trade log header based on whether ML is enabled
    if ml_enabled and ml_trade_log_path:
        with open(ml_trade_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'type',
                'num_contracts', 'stop_loss', 'profit_target', 'bars_held', 'atr', 'rsi',
                'volume', 'avg_volume', 'entry_account_value', 'exit_reason', 'fees', 
                'exit_account_value', 'adx', 'ma_slope', 'volatility_regime', 'regime_score',
                'favorable_regime', 'position_size_adj', 'market_type', 'ml_probability', 'ml_approved',
                'in_season', 'season',
                'used_trailing_stop', 'used_dynamic_target', 'highest_price_reached', 'lowest_price_reached'
            ])
    else:
        with open(trade_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'type',
                'num_contracts', 'stop_loss', 'profit_target', 'bars_held', 'atr', 'rsi',
                'volume', 'avg_volume', 'entry_account_value', 'exit_reason', 'fees', 
                'exit_account_value', 'adx', 'ma_slope', 'volatility_regime', 'regime_score',
                'favorable_regime', 'position_size_adj', 'market_type', 'in_season', 'season',
                'used_trailing_stop', 'used_dynamic_target', 'highest_price_reached', 'lowest_price_reached'
            ])
    
    # Backtest loop
    for i, row in df.iterrows():
        current_portfolio_value = account_value
        current_date = row['date'].date()
        
        # Check if we're in a season
        if config.get('seasons', {}).get('enable', False):
            in_season, season_name = is_in_season(current_date, season_dates)
            season_active = in_season
            current_season = season_name if in_season else 'Out-of-Season'
        else:
            in_season = True
            current_season = None
        
        # Skip avoid dates
        if current_date in config['avoid_dates']:
            portfolio_value.append(current_portfolio_value)
            continue
        
        # Check market type detection - update if needed
        if (config['market_type']['enable'] and 
            (last_market_type_update is None or 
            (current_date - last_market_type_update).days >= config['market_type']['update_frequency'])):

            # Initialize duration outside the conditional blocks
            duration = None

            # Detect market type with warmup status
            detected_type, market_metrics, warmup_complete = detect_market_type(df,
                                                                               config['market_type']['window'],
                                                                               row['date'])

            # Track warmup status
            in_warmup = not warmup_complete

            # If in warmup, set special regime
            if in_warmup:
                current_market_type = 'warmup'
                logger.info(
                    f"Market regime detection in warmup phase: {market_metrics.get('classification_rationale', '')}")
            # Otherwise, proceed with normal market type update
            elif detected_type != current_market_type or last_market_type_update is None:
                # Format date for logging
                formatted_date = row['date'].strftime("%Y-%m-%d %H:%M:%S")

                # Extract classification rationale if it exists
                rationale = market_metrics.get('classification_rationale', '')

                if last_market_type_update is None or len(market_type_log) == 0:
                    # Initial market type
                    logger.info(f"Market type initialized as {detected_type.upper()} at {formatted_date} - {rationale}")
                else:
                    # Safely calculate regime duration
                    try:
                        duration = row['date'] - market_type_log[-1]['date']
                        duration_str = f"{duration.days}d {duration.seconds // 3600}h"
                        logger.info(
                            f"Market type changed from {current_market_type.upper()} to {detected_type.upper()} at {formatted_date} (lasted {duration_str}) - {rationale}")
                    except (IndexError, KeyError):
                        # Fallback if we can't calculate duration
                        logger.info(
                            f"Market type changed from {current_market_type.upper()} to {detected_type.upper()} at {formatted_date} - {rationale}")

                current_market_type = detected_type

                # Get confidence from HMM if available
                confidence = market_metrics.get('confidence')
                market_type_params = get_market_type_params(current_market_type, confidence)

                # Log the change with properly initialized duration
                market_type_log.append({
                    'date': row['date'],
                    'market_type': current_market_type,
                    'metrics': market_metrics,
                    'parameters': market_type_params.copy(),
                    'previous_duration': duration,  # This will be None or a timedelta object
                    'hmm_confidence': confidence  # Add HMM confidence to log
                })

            last_market_type_update = current_date
        
        # Calculate market regime for current bar
        adx_value = row['ADX']
        ma_slope_value = row['MA_slope']
        volatility_regime_value = row['volatility_regime']
        atr_ratio_value = row['atr_ratio']

        # Calculate regime score
        regime_score, regime_details = calculate_regime_score(
            adx_value, ma_slope_value,
            market_type_params
        )
        favorable = regime_details['favorable']

        # Store HMM confidence in regime details if available
        if hmm_detector is not None and config['hmm_detector']['enable']:
            # If HMM confidence is available in parameters, log it
            if 'confidence' in market_type_params:
                regime_details['hmm_confidence'] = market_type_params['confidence']
                
                # Store in DataFrame
                df.at[i, 'hmm_confidence'] = market_type_params['confidence']
        
        # Calculate position size adjustment
        position_size_adj = calculate_position_size_adjustment(regime_score, market_type_params)
        
        # Store regime data in dataframe
        df.at[i, 'regime_score'] = regime_score
        df.at[i, 'favorable_regime'] = int(favorable)
        df.at[i, 'position_size_adj'] = position_size_adj
        df.at[i, 'in_season'] = int(season_active)
        df.at[i, 'season'] = current_season
        
        # Track regime score distribution
        if regime_score <= 20:
            regime_score_bins['0-20'] += 1
        elif regime_score <= 40:
            regime_score_bins['21-40'] += 1
        elif regime_score <= 60:
            regime_score_bins['41-60'] += 1
        elif regime_score <= 80:
            regime_score_bins['61-80'] += 1
        else:
            regime_score_bins['81-100'] += 1
        
        # Log regime data
        row_log = {
            'date': row['date'],
            'adx': adx_value,
            'ma_slope': ma_slope_value,
            'volatility_regime': volatility_regime_value,
            'atr_ratio': atr_ratio_value,
            'adx_score': regime_details.get('adx_score', 0),
            'slope_score': regime_details.get('slope_score', 0),
            'regime_score': regime_score,
            'favorable_regime': favorable,
            'position_size_adj': position_size_adj,
            'in_season': int(season_active),
            'season': current_season
        }
        
        # Store HMM confidence in regime log if available
        if 'hmm_confidence' in regime_details:
            row_log['hmm_confidence'] = regime_details['hmm_confidence']
            
        regime_log.append(row_log)
        
        # HANDLE EXISTING POSITION - Check for exits
        if position != 0:
            # Call exit condition check with price tracking and current portion
            exit_result = check_exit_conditions(
                df, i, position, entry_price, stop_loss, profit_target, i - entry_bar,
                highest_price_since_entry, lowest_price_since_entry, trade_atr,
                market_type=current_market_type, current_portion=current_portion
            )

            # Update price tracking variables
            if 'highest_price' in exit_result:
                highest_price_since_entry = exit_result['highest_price']
            if 'lowest_price' in exit_result:
                lowest_price_since_entry = exit_result['lowest_price']

            # Update stop loss if trailing stop is active
            if 'updated_stop_loss' in exit_result:
                stop_loss = exit_result['updated_stop_loss']

            if exit_result['exit']:
                # Determine if this is a partial exit
                is_partial = 'exit_portion' in exit_result and exit_result['exit_portion'] < current_portion

                if is_partial:
                    # Calculate exit portion
                    exit_portion = exit_result['exit_portion']
                    exit_contracts = int(abs(position) * exit_portion / current_portion)

                    # Calculate profit for the exited portion
                    exit_price = exit_result['exit_price']
                    exit_reason = exit_result['exit_reason']
                    portion_profit = (exit_price - entry_price) * (
                        exit_contracts if position > 0 else -exit_contracts) * config['risk']['contract_multiplier']
                    fees = config['account']['transaction_cost'] * exit_contracts
                    portion_profit -= fees

                    # Update account
                    account_value += portion_profit

                    # Record partial exit
                    partial_exit_data = {
                        'entry_time': df.loc[entry_bar, 'date'],
                        'exit_time': row['date'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': portion_profit,
                        'type': 'long' if position > 0 else 'short',
                        'num_contracts': exit_contracts,
                        'stop_loss': stop_loss,
                        'profit_target': profit_target,
                        'bars_held': i - entry_bar,
                        'atr': trade_atr,
                        'rsi': trade_rsi,
                        'volume': trade_volume,
                        'avg_volume': trade_avg_volume,
                        'entry_account_value': trade_entry_account_value,
                        'exit_reason': exit_reason,
                        'fees': fees,
                        'exit_account_value': account_value,
                        'adx': trade_adx,
                        'ma_slope': trade_ma_slope,
                        'volatility_regime': trade_volatility_regime,
                        'regime_score': trade_regime_score,
                        'favorable_regime': trade_favorable_regime,
                        'position_size_adj': trade_position_size_adj,
                        'market_type': current_market_type,
                        'in_season': int(trade_in_season),
                        'season': trade_season,
                        'portion_exited': exit_portion,
                        'partial_exit': True,
                        'highest_price_reached': highest_price_since_entry if position > 0 else None,
                        'lowest_price_reached': lowest_price_since_entry if position < 0 else None
                    }

                    # Add ML-specific data if ML was used
                    if ml_enabled:
                        partial_exit_data.update({
                            'ml_probability': trade_ml_probability,
                            'ml_approved': trade_ml_approved
                        })

                    # Add to partial exits list
                    partial_exits.append(partial_exit_data)

                    # Update position size and portion
                    remaining_portion = exit_result['remaining_portion']
                    position = int(position * (remaining_portion / current_portion))
                    current_portion = remaining_portion

                else:
                    # Full exit (or exit of remaining portion)
                    exit_price = exit_result['exit_price']
                    exit_reason = exit_result['exit_reason']
                    num_contracts = abs(position)

                    # Calculate profit
                    profit = (exit_price - entry_price) * position * config['risk']['contract_multiplier']
                    fees = config['account']['transaction_cost'] * num_contracts
                    profit -= fees
                    account_value += profit
                
                # Record the trade
                trade_data = {
                    'entry_time': df.loc[entry_bar, 'date'],
                    'exit_time': row['date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'type': 'long' if position > 0 else 'short',
                    'num_contracts': num_contracts,
                    'stop_loss': stop_loss,
                    'profit_target': profit_target,
                    'bars_held': i - entry_bar,
                    'atr': trade_atr,
                    'rsi': trade_rsi,
                    'volume': trade_volume,
                    'avg_volume': trade_avg_volume,
                    'entry_account_value': trade_entry_account_value,
                    'exit_reason': exit_reason,
                    'fees': fees,
                    'exit_account_value': account_value,
                    'adx': trade_adx,
                    'ma_slope': trade_ma_slope,
                    'volatility_regime': trade_volatility_regime,
                    'regime_score': trade_regime_score,
                    'favorable_regime': trade_favorable_regime,
                    'position_size_adj': trade_position_size_adj,
                    'market_type': current_market_type,
                    'in_season': int(trade_in_season),
                    'season': trade_season,
                    'used_trailing_stop': exit_reason == 'trailing_stop',
                    'used_dynamic_target': config['risk'].get('dynamic_target_enable', False),
                    'highest_price_reached': highest_price_since_entry if position > 0 else None,
                    'lowest_price_reached': lowest_price_since_entry if position < 0 else None,
                    'partial_exits': partial_exits if partial_exits else None,
                }
                
                # Add ML-specific data if ML was used
                if ml_enabled:
                    trade_data.update({
                        'ml_probability': trade_ml_probability,
                        'ml_approved': trade_ml_approved
                    })
                
                # Add HMM information if available
                if 'hmm_confidence' in regime_details:
                    trade_data['hmm_confidence'] = regime_details['hmm_confidence']
                    trade_data['hmm_regime_score'] = regime_score  # This will be HMM-influenced score
                    trade_data['ml_hmm_combined'] = True
                
                # Write to trade log - REOPEN file in append mode
                if ml_enabled and ml_trade_log_path:
                    # Write to ML-specific trade log
                    with open(ml_trade_log_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        # Add all columns including ML-specific ones
                        writer.writerow([
                            trade_data['entry_time'], trade_data['exit_time'],
                            trade_data['entry_price'], trade_data['exit_price'], trade_data['profit'],
                            trade_data['type'], trade_data['num_contracts'], trade_data['stop_loss'],
                            trade_data['profit_target'], trade_data['bars_held'], trade_data['atr'],
                            trade_data['rsi'], trade_data['volume'], trade_data['avg_volume'],
                            trade_data['entry_account_value'], trade_data['exit_reason'],
                            trade_data['fees'], trade_data['exit_account_value'], trade_data['adx'],
                            trade_data['ma_slope'], trade_data['volatility_regime'], trade_data['regime_score'],
                            trade_data['favorable_regime'], trade_data['position_size_adj'], 
                            trade_data['market_type'], trade_data['ml_probability'], trade_data['ml_approved'],
                            trade_data['in_season'], trade_data['season'],
                            trade_data['used_trailing_stop'], trade_data['used_dynamic_target'],
                            trade_data['highest_price_reached'], trade_data['lowest_price_reached']
                        ])
                else:
                    # Write to standard trade log
                    with open(trade_log_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            trade_data['entry_time'], trade_data['exit_time'],
                            trade_data['entry_price'], trade_data['exit_price'], trade_data['profit'],
                            trade_data['type'], trade_data['num_contracts'], trade_data['stop_loss'],
                            trade_data['profit_target'], trade_data['bars_held'], trade_data['atr'],
                            trade_data['rsi'], trade_data['volume'], trade_data['avg_volume'],
                            trade_data['entry_account_value'], trade_data['exit_reason'],
                            trade_data['fees'], trade_data['exit_account_value'], trade_data['adx'],
                            trade_data['ma_slope'], trade_data['volatility_regime'], trade_data['regime_score'],
                            trade_data['favorable_regime'], trade_data['position_size_adj'], trade_data['market_type'],
                            trade_data['in_season'], trade_data['season'],
                            trade_data['used_trailing_stop'], trade_data['used_dynamic_target'],
                            trade_data['highest_price_reached'], trade_data['lowest_price_reached']
                        ])
                
                # Add to trades list
                trades.append(trade_data)
                
                # For ML: update executed trades list for ML training
                if ml_enabled:
                    executed_trades.append(trade_data.copy())
                    
                    # Update trade count
                    trade_count += 1
                    
                    # Record ML metrics
                    actual_outcome = profit > 0
                    ml_predictor.record_prediction_result(trade_ml_approved, actual_outcome)
                    
                    # ADDED: Periodically log ML warmup status regardless of training condition
                    if not warmup_complete and trade_count % 10 == 0:
                        logger.info(
                            f"[DEBUG] ML warmup status: {len(executed_trades)}/{config['ml']['warmup_trades']} trades collected, warmpup_complete={warmup_complete}")
                
                # Track trade by season
                season_key = trade_season if trade_season else 'Out-of-Season'
                if season_key in trades_by_season:
                    trades_by_season[season_key].append(trade_data)
                
                # Visualize trade if enabled
                if visualize_trades and config['visualization']['generate_png_charts']:
                    try:
                        visualize_trade(
                            df, trade_data, 
                            save_dir=os.path.join(os.path.dirname(file_paths['trade_log']), 'executed_trade_plots'),
                            rsi_oversold=config['strategy']['rsi_oversold'],
                            rsi_overbought=config['strategy']['rsi_overbought']
                        )
                    except Exception as e:
                        logger.error(f"Error visualizing trade: {e}")

                position = 0
                current_portfolio_value = account_value
                current_portion = 1.0  # Reset to full position for next trade
                partial_exits = []  # Clear partial exits list

                # Check if we have enough data for first ML training
                if ml_enabled and not warmup_complete and len(executed_trades) >= config['ml']['warmup_trades']:
                    logger.info(f"Warmup complete. Training initial ML model with {len(executed_trades)} trades.")

                    # Add debug information about the trades
                    logger.info(
                        f"Trade breakdown: {sum(1 for t in executed_trades if t['profit'] > 0)} profitable, {sum(1 for t in executed_trades if t['profit'] <= 0)} losing")

                    try:
                        # Generate training data from executed trades
                        X, y = ml_predictor.generate_training_data(df, executed_trades)

                        if X is not None:
                            logger.info(
                                f"Generated {len(X)} training samples with {sum(y)} positive examples ({sum(y) / len(y) * 100:.1f}%)")
                            logger.info(f"[DEBUG] X shape: {X.shape}, Feature names: {list(X.columns)[:5]}...")

                            if len(X) >= config['ml']['min_training_samples']:
                                # Train model
                                success = ml_predictor.train_model(X, y)
                                if success:
                                    warmup_complete = True
                                    if ml_metrics:
                                        ml_metrics['model_retrain_count'] += 1
                                    logger.info("Initial ML model training complete")
                                else:
                                    logger.error("[DEBUG] ML model training failed despite having enough samples")
                            else:
                                logger.info(
                                    f"Not enough training samples: {len(X)} < {config['ml']['min_training_samples']} required")

                                # ADDED: Temporary override for testing - force training with lower threshold
                                if len(X) >= 30:  # Much lower threshold for testing
                                    logger.info("[DEBUG] Attempting forced training with reduced sample threshold")
                                    success = ml_predictor.train_model(X, y)
                                    if success:
                                        warmup_complete = True
                                        if ml_metrics:
                                            ml_metrics['model_retrain_count'] += 1
                                        logger.info("[DEBUG] Forced ML training successful")
                                    else:
                                        logger.error("[DEBUG] Forced ML training failed")
                        else:
                            logger.info("[DEBUG] Failed to generate training samples - X is None")
                    except Exception as e:
                        logger.error(f"Error during ML training: {e}")
                        import traceback
                        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")

            else:
                # No exit yet, update unrealized P&L
                unrealized_pnl = (row['close'] - entry_price) * position * config['risk']['contract_multiplier']
                current_portfolio_value = account_value + unrealized_pnl
        
        # CHECK FOR NEW ENTRY SIGNALS (if not already in a position)
        elif position == 0 and i > 0:
            # Skip entry if in market regime warmup phase
            if in_warmup:
                portfolio_value.append(current_portfolio_value)
                continue
                
            # Check if we're in trading window and in an active season
            current_time = row['date'].time()
            in_trading_window = is_in_trading_window(current_time)
            
            # Skip entry check if not in season and season filter is enabled
            if not season_active and config.get('seasons', {}).get('enable', False):
                portfolio_value.append(current_portfolio_value)
                continue
            
            if not in_trading_window:
                portfolio_value.append(current_portfolio_value)
                continue
            
            # Get previous bar data
            prev_row = df.iloc[i-1]
            
            # Check for entry signal based on ML setting
            if ml_enabled:
                # Prepare ML features
                ml_features = ml_predictor.extract_features(df, i, 'unknown')

                # Add HMM features if enabled
                if ml_features is not None and config['ml']['use_hmm_features'] and hmm_detector is not None and config['hmm_detector']['enable']:
                    # Try to get HMM confidence from regime details
                    hmm_confidence = regime_details.get('hmm_confidence')
                    if hmm_confidence is not None:
                        ml_features['hmm_confidence'] = hmm_confidence

                        # Add market type as a one-hot encoded feature
                        ml_features['market_type_trend'] = 1 if current_market_type == 'trend_following' else 0
                        ml_features['market_type_mr'] = 1 if current_market_type == 'mean_reverting' else 0
                        ml_features['market_type_neutral'] = 1 if current_market_type == 'neutral' else 0

                # Check for ML entry signal
                entry_signal, ml_probability, ml_approved = check_entry_signal_ml(
                    prev_row, ml_features, ml_predictor, row['date'], current_market_type
                )
            else:
                # Regular entry check without ML
                entry_signal = check_entry_signal(prev_row, current_market_type, regime_score)
                ml_probability = 0.5  # Default value
                ml_approved = True    # Always approve if not using ML
            
            # Process potential entry
            if entry_signal:
                # Track potential trades for ML metrics
                if ml_enabled and ml_metrics:
                    ml_metrics['potential_trades'] += 1

                # Log entry signal details
                if ml_enabled and warmup_complete:
                    if ml_approved:
                        logger.info(
                            f"ML APPROVED {entry_signal} trade with probability {ml_probability:.2f} (threshold: {config['ml']['prediction_threshold']})")
                    else:
                        logger.info(
                            f"ML REJECTED {entry_signal} trade with probability {ml_probability:.2f} (threshold: {config['ml']['prediction_threshold']})")
                elif ml_enabled and not warmup_complete:
                    logger.info(f"ML in warmup phase, APPROVED {entry_signal} trade by default (no ML filtering yet)")

                # Determine if ML should be applied:
                use_ml_filtering = ml_enabled and warmup_complete

                # Check if we should take the trade (with clearer logic)
                if not favorable:
                    # Skip unfavorable regime regardless of ML
                    take_trade = False
                    trades_skipped_regime += 1
                    logger.info(f"Skipping {entry_signal} trade due to unfavorable regime (score: {regime_score:.1f})")
                elif use_ml_filtering:
                    # Apply ML filtering
                    if ml_approved:
                        take_trade = True
                        if ml_metrics:
                            ml_metrics['executed_trades'] += 1
                    else:
                        take_trade = False
                        if ml_metrics:
                            ml_metrics['skipped_by_ml'] += 1
                        # Track skipped trades for analysis
                        skipped_by_ml.append({
                            'entry_time': row['date'],
                            'type': entry_signal,
                            'rsi': prev_row['RSI'],
                            'regime_score': regime_score,
                            'ml_probability': ml_probability,
                            'in_season': int(season_active),
                            'season': current_season
                        })
                else:
                    # ML not active yet, take trade in favorable regime
                    take_trade = True
                    if ml_enabled:
                        logger.info(
                            f"ML in warmup phase, APPROVED {entry_signal} trade by default (warmup progress: {len(executed_trades)}/{config['ml']['warmup_trades']} trades)")

                if take_trade:
                    # Take the trade
                    entry_price = row['open']  # Enter at the open
                    entry_bar = i
                    atr = prev_row['ATR']

                    # Initialize price tracking for trailing stops
                    highest_price_since_entry = entry_price
                    lowest_price_since_entry = entry_price

                    # Set stop loss and profit target
                    if entry_signal == 'long':
                        # Get regime parameters for stop loss and target calculation
                        regime_params = get_regime_parameters(current_market_type, regime_score)

                        stop_loss = entry_price - regime_params['atr_stop_multiplier'] * atr

                        # Set profit target - either dynamic ATR-based or BB middle
                        if config['risk'].get('dynamic_target_enable', False):
                            profit_target = entry_price + (regime_params['dynamic_target_multiplier'] * atr)
                        else:
                            profit_target = prev_row['middle_band']
                    else:  # short
                        stop_loss = entry_price + config['risk']['atr_stop_multiplier'] * atr

                        # Set profit target - either dynamic ATR-based or BB middle
                        if config['risk'].get('dynamic_target_enable', False):
                            profit_target = entry_price - (config['risk']['dynamic_target_atr_multiplier'] * atr)
                        else:
                            profit_target = prev_row['middle_band']

                    # Calculate position size
                    num_contracts = calculate_position_size(
                        account_value,
                        atr,
                        position_size_adj
                    )
                    position = num_contracts if entry_signal == 'long' else -num_contracts

                    # Apply entry transaction cost
                    account_value -= config['account']['transaction_cost'] * num_contracts

                    # Store trade info for exit
                    trade_atr = atr
                    trade_rsi = prev_row['RSI']
                    trade_volume = prev_row['volume']
                    trade_avg_volume = prev_row['avg_volume']
                    trade_entry_account_value = account_value
                    trade_adx = adx_value
                    trade_ma_slope = ma_slope_value
                    trade_volatility_regime = volatility_regime_value
                    trade_regime_score = regime_score
                    trade_favorable_regime = favorable
                    trade_position_size_adj = position_size_adj
                    trade_in_season = season_active
                    trade_season = current_season
                    trade_regime_details = regime_details

                    # Store ML probability and approval for exit
                    if ml_enabled:
                        trade_ml_probability = ml_probability
                        trade_ml_approved = ml_approved
                    else:
                        trade_ml_probability = 0.5
                        trade_ml_approved = True

                    # Update metrics
                    if ml_enabled and ml_metrics:
                        ml_metrics['executed_trades'] += 1

                    # Track favorable vs unfavorable trades
                    trades_taken_favorable += 1

                    # Log trade entry
                    if ml_enabled:
                        logger.info(
                            f"Trade entered: {entry_signal} at {row['date']}, "
                            f"price={entry_price}, contracts={num_contracts}, "
                            f"ML prob={ml_probability:.2f}, regime_score={regime_score:.1f}, "
                            f"season={current_season if current_season else 'N/A'}"
                        )
                    else:
                        logger.info(
                            f"Trade entered: {entry_signal} at {row['date']}, "
                            f"price={entry_price}, contracts={num_contracts}, "
                            f"regime_score={regime_score:.1f}, season={current_season if current_season else 'N/A'}"
                        )

                # Check if we need to retrain the ML model
                if (ml_enabled and warmup_complete and
                    ml_predictor and ml_predictor.check_retrain_needed(row['date']) and
                    len(executed_trades) >= config['ml']['min_training_samples']):

                    logger.info(f"Retraining ML model with {len(executed_trades)} trades")

                    # Generate training data from executed trades
                    X, y = ml_predictor.generate_training_data(df, executed_trades)

                    if X is not None and len(X) >= config['ml']['min_training_samples']:
                        # Train model
                        success = ml_predictor.train_model(X, y)
                        if success and ml_metrics:
                            ml_metrics['model_retrain_count'] += 1
                            logger.info(f"ML model retrained successfully")

        # Record portfolio value
        portfolio_value.append(current_portfolio_value)

    # Save ML performance metrics
    if ml_enabled and ml_predictor:
        ml_predictor.save_performance_report()

        # Calculate additional metrics
        if ml_metrics:
            ml_metrics['prediction_accuracy'] = 0
            metrics = ml_predictor.get_performance_metrics()
            total_predictions = metrics['true_positives'] + metrics['true_negatives'] + \
                            metrics['false_positives'] + metrics['false_negatives']

            if total_predictions > 0:
                ml_metrics['prediction_accuracy'] = metrics['accuracy']

    # Save regime log
    regime_df = pd.DataFrame(regime_log)
    regime_df.to_csv(file_paths['regime_log'], index=False)

    # Save market type log
    try:
        with open(file_paths['market_type_log'], mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'date', 'market_type', 'trend_strength', 'volatility_ratio',
                'momentum_bias', 'market_type_score', 'rationale'
            ])

            for entry in market_type_log:
                metrics = entry['metrics']
                writer.writerow([
                    entry['date'],
                    entry['market_type'],
                    metrics.get('trend_strength', 0),
                    metrics.get('volatility_ratio', 0),
                    metrics.get('momentum_bias', 0),
                    metrics.get('market_type_score', 0),
                    metrics.get('classification_rationale', '')
                ])
    except Exception as e:
        logger.error(f"Error writing market type log: {e}")

    # Save portfolio value history
    portfolio_df = pd.DataFrame({'date': df['date'], 'portfolio_value': portfolio_value})
    portfolio_df.to_csv(file_paths['portfolio_value'], index=False)

    # Calculate season metrics
    season_metrics = {}
    for season, season_trades in trades_by_season.items():
        if season_trades:
            season_metrics[season] = {
                'trade_count': len(season_trades),
                'win_rate': sum(1 for t in season_trades if t['profit'] > 0) / len(season_trades) * 100,
                'total_profit': sum(t['profit'] for t in season_trades),
                'avg_profit': sum(t['profit'] for t in season_trades) / len(season_trades) if season_trades else 0,
                'profit_factor': (
                    sum(t['profit'] for t in season_trades if t['profit'] > 0) /
                    abs(sum(t['profit'] for t in season_trades if t['profit'] <= 0))
                ) if sum(t['profit'] for t in season_trades if t['profit'] <= 0) != 0 else float('inf')
            }

    # Save HMM logs if HMM was used
    if hmm_detector is not None and config['hmm_detector']['enable'] and ml_enabled:
        try:
            # Save HMM prediction history
            if hasattr(hmm_detector, 'prediction_history') and hmm_detector.prediction_history:
                # Convert to DataFrame
                hmm_pred_df = pd.DataFrame(hmm_detector.prediction_history)
                hmm_pred_path = os.path.join(os.path.dirname(file_paths['trade_log']), 'ml', 'hmm_predictions.csv')
                hmm_pred_df.to_csv(hmm_pred_path, index=False)

                # Save HMM model metrics
                if ml_metrics:
                    ml_metrics['hmm_metrics'] = hmm_detector.metrics

                logger.info(f"HMM logs saved to {os.path.dirname(file_paths['trade_log'])}/ml")
        except Exception as e:
            logger.error(f"Error saving HMM logs: {e}")

    # Return results
    return trades, portfolio_value, df, regime_log, market_type_log, regime_score_bins, season_metrics, ml_metrics


#############################################################################
#                          MAIN EXECUTION                                    #
#############################################################################

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run market regime-specific backtester')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancements')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Override config if command line arguments are provided
    if args.data:
        config['data']['file_path'] = args.data
    if args.start:
        config['data']['start_date'] = args.start
    if args.end:
        config['data']['end_date'] = args.end
    if args.ml:
        config['ml']['enable'] = True
    if args.seed is not None:
        config['global']['random_seed'] = args.seed
        config['global']['use_fixed_seed'] = True

    # Set up output directory and file paths
    output_dir, file_paths = setup_directories(
        config['data']['start_date'],
        config['data']['end_date']
    )

    # Copy Python files for reference
    copy_project_files(output_dir)

    # Load and process data
    df_5min = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )

    if df_5min is None or len(df_5min) == 0:
        logger.error("No data available after loading. Exiting.")
        sys.exit(1)

    # Calculate all indicators
    df_5min = calculate_indicators(df_5min, config)

    # Run backtest
    trades, portfolio_values, df_5min, regime_log, market_type_log, regime_score_bins, season_metrics, ml_metrics = run_backtest(
        df_5min,
        visualize_trades=config['visualization']['generate_png_charts'],
        file_paths=file_paths,
        use_ml=config['ml']['enable']
    )

    # Import analysis functions
    from trade_analysis import (
        analyze_performance, analyze_by_regime, analyze_trades_by_market_type,
        analyze_exit_reasons, create_summary_report, analyze_by_season,
        analyze_exit_strategies, analyze_quarterly_performance
    )

    # Convert to series for analysis
    portfolio_df = pd.DataFrame({'date': df_5min['date'], 'value': portfolio_values})
    portfolio_series = portfolio_df.set_index('date')['value']

    # Analyze results
    results = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
    trades_by_regime = analyze_by_regime(trades, regime_score_bins)
    trades_by_market = analyze_trades_by_market_type(trades)
    exit_reasons, profit_by_exit = analyze_exit_reasons(trades)
    trades_by_season = analyze_by_season(trades)  # Add season analysis

    # Run quarterly analysis
    quarterly_df = analyze_quarterly_performance(
        trades, portfolio_series, config['account']['initial_capital']
    )

    # Generate quarterly visualizations if enough data
    if not quarterly_df.empty:
        generate_quarterly_analysis_charts(
            quarterly_df,
            output_dir,
            f"{config['data']['start_date']}_{config['data']['end_date']}"
        )
        logger.info(f"Quarterly analysis saved to {os.path.join(output_dir, 'quarterly_analysis')}")

    # Generate charts
    if config['visualization']['generate_png_charts']:
        generate_performance_charts(
            portfolio_series,
            trades,
            output_dir,
            f"{config['data']['start_date']}_{config['data']['end_date']}"
        )

    # Calculate the exit strategy metrics
    exit_strategy_metrics = analyze_exit_strategies(trades)

    # Create summary report
    create_summary_report(
        results, trades_by_regime, trades_by_market,
        exit_reasons, profit_by_exit, regime_score_bins,
        season_metrics, exit_strategy_metrics,
        file_paths['summary']
    )

    # Print summary to console
    logger.info("\n===== PERFORMANCE SUMMARY =====")
    logger.info(f"Initial Capital: ${results['initial_capital']:.2f}")
    logger.info(f"Final Portfolio Value: ${results['final_value']:.2f}")
    logger.info(f"Profit/Loss: ${results['profit_loss']:.2f}")
    logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
    logger.info(f"Number of Trades: {results['number_of_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2f}%")
    logger.info(f"Average Trade P/L: ${results['avg_profit']:.2f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")

    # Print ML metrics if ML was enabled
    if config['ml']['enable'] and ml_metrics:
        logger.info("\n===== ML METRICS =====")
        logger.info(f"ML Filter Accuracy: {ml_metrics['prediction_accuracy']*100:.2f}%")
        logger.info(f"Potential Trades: {ml_metrics['potential_trades']}")
        logger.info(f"Trades Executed: {ml_metrics['executed_trades']}")
        logger.info(f"Trades Skipped by ML: {ml_metrics['skipped_by_ml']}")
        logger.info(f"Model Retrain Count: {ml_metrics['model_retrain_count']}")

    # Print season performance if enabled
    if config['seasons']['enable']:
        logger.info("\n===== SEASON PERFORMANCE =====")
        for season, metrics in season_metrics.items():
            if metrics['trade_count'] > 0:
                logger.info(f"{season}: {metrics['trade_count']} trades, "
                          f"Win Rate: {metrics['win_rate']:.2f}%, "
                          f"Total P/L: ${metrics['total_profit']:.2f}")

    logger.info(f"\n===== METRICS FOR SEED TESTING =====")
    logger.info(f"Trades: {len(trades)}")
    logger.info(f"Profit/Loss: ${results['profit_loss']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2f}%")
    logger.info(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"===== END METRICS =====")

    logger.info(f"Output directory: {output_dir}")