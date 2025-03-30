"""
backtester_common.py - Common functions for backtesters

This module provides shared functions used by both the standard and ML-enhanced
backtesters, eliminating code duplication.
"""

import numpy as np
import logging
import os
from datetime import datetime, timedelta

# Import the centralized configuration
from config import config

# Import HMM regime detector
from hmm_regime_detector import HMMRegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global HMM detector instance (shared by both backtesters)
hmm_detector = None

#############################################################################
#                    MARKET REGIME FUNCTIONS                                #
#############################################################################

def initialize_hmm_detector(output_dir=None):
    """Initialize the HMM regime detector"""
    global hmm_detector
    
    if output_dir:
        hmm_output_dir = os.path.join(output_dir, 'hmm_detector')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hmm_output_dir = f"hmm_detector_{timestamp}"
    
    if not os.path.exists(hmm_output_dir):
        os.makedirs(hmm_output_dir)
    
    # Create detector with config settings
    detector_settings = config['hmm_detector']
    hmm_detector = HMMRegimeDetector(
        n_states=detector_settings['n_states'],
        lookback_days=detector_settings['lookback_days'],
        retrain_frequency=detector_settings['retrain_frequency'],
        min_samples=detector_settings['min_samples'],
        output_dir=hmm_output_dir
    )
    
    logger.info(f"Initialized HMM regime detector with output dir: {hmm_output_dir}")
    return hmm_detector


def reset_hmm_detector():
    """
    Reset the HMM detector state completely between runs.
    This ensures that each backtest starts with a fresh HMM state.
    """
    global hmm_detector
    hmm_detector = None

    # Also clear any cached attributes in the detect_market_type function
    if hasattr(detect_market_type, 'historical_scores'):
        delattr(detect_market_type, 'historical_scores')
    if hasattr(detect_market_type, 'score_dates'):
        delattr(detect_market_type, 'score_dates')
    if hasattr(detect_market_type, 'market_type_counts'):
        delattr(detect_market_type, 'market_type_counts')

    logger.info("HMM detector state has been reset")

def calculate_regime_score(adx, ma_slope, market_params):
    """
    Calculate market regime score based on HMM confidence.
    
    Args:
        adx: ADX value (not used but kept for API compatibility)
        ma_slope: MA slope value (not used but kept for API compatibility)
        market_params: Parameters based on market type
        
    Returns:
        Tuple of (score, details_dict)
    """
    # If HMM confidence is available, use it directly
    if 'confidence' in market_params:
        confidence = market_params['confidence']
        # Scale confidence to 0-100 range to match existing score scale
        regime_score = confidence * 100
        
        # Create regime details with HMM information
        regime_details = {
            'adx': adx,  # Keep for compatibility
            'ma_slope': ma_slope,  # Keep for compatibility
            'total_score': regime_score,
            'favorable': regime_score >= market_params.get('min_regime_score', 40),
            'market_type': market_params.get('market_type', 'neutral'),
            'hmm_confidence': confidence
        }
        
        return regime_score, regime_details
    
    # If no HMM confidence (fallback during warmup), use neutral values
    default_score = 50  # Neutral score
    regime_details = {
        'adx': adx,
        'ma_slope': ma_slope,
        'total_score': default_score,
        'favorable': default_score >= market_params.get('min_regime_score', 40),
        'market_type': market_params.get('market_type', 'neutral')
    }
    
    return default_score, regime_details

def calculate_position_size_adjustment(regime_score, market_params):
    """Calculate position size adjustment based on regime score or HMM confidence"""
    # If using HMM, adjust position size based on confidence
    if 'confidence' in market_params:
        confidence = market_params['confidence']
        # Use confidence to determine position sizing
        if confidence > 0.85:  # Very high confidence
            return config['position_sizing']['max_size_adjustment']
        elif confidence > 0.70:  # Good confidence
            return 1.2  # Moderate increase
        elif confidence > 0.50:  # Reasonable confidence
            return 1.0  # Normal position size
        else:  # Low confidence
            return config['position_sizing']['min_size_adjustment']
    
    """
    Calculate position size adjustment based on regime score.
    
    Args:
        regime_score: The market regime score (0-100)
        market_params: Parameters for the current market type
        
    Returns:
        Adjustment factor for position sizing
    """
    if not config['position_sizing']['adjust_by_regime']:
        return 1.0  # No adjustment if feature disabled
    
    # Get position sizing parameters
    min_adjustment = config['position_sizing']['min_size_adjustment']
    max_adjustment = config['position_sizing']['max_size_adjustment']
    
    # Get market type specific parameters
    min_score = market_params.get('min_regime_score', 40)
    sweet_spot_min = market_params.get('sweet_spot_min', 45)
    sweet_spot_max = market_params.get('sweet_spot_max', 65)
    
    # Return minimum size for scores below minimum threshold
    if regime_score < min_score:
        return min_adjustment
    
    # Apply inverted bell curve approach
    if regime_score >= sweet_spot_min and regime_score <= sweet_spot_max:
        # In sweet spot range - maximum position size
        return max_adjustment
    
    elif regime_score < sweet_spot_min:
        # Ramp up from minimum score to sweet spot
        normalized_score = (regime_score - min_score) / (sweet_spot_min - min_score)
        return min_adjustment + normalized_score * (max_adjustment - min_adjustment)
    
    else:  # regime_score > sweet_spot_max
        # Ramp down from sweet spot to maximum score
        high_score_adjustment = 0.9  # Still good but not maximum
        normalized_score = min(1.0, (regime_score - sweet_spot_max) / (100 - sweet_spot_max))
        return max_adjustment - normalized_score * (max_adjustment - high_score_adjustment)

def detect_market_type(df, lookback_days=20, current_date=None):
    """
    Detect market type using HMM-only approach.
    
    Args:
        df: DataFrame with price and indicator data
        lookback_days: Days to look back (not used with HMM but kept for API compatibility)
        current_date: Current date for detection window
        
    Returns:
        Tuple of (market_type, metrics, warmup_complete)
    """
    global hmm_detector
    
    # Initialize HMM detector if needed
    if hmm_detector is None:
        hmm_detector = initialize_hmm_detector()
        logger.info("HMM detector initialized")
    
    # Default to last date in DataFrame if no current_date
    if current_date is None:
        current_date = df['date'].max()
    
    # Check if we need to train or retrain
    if hmm_detector.model is None or hmm_detector.check_retrain_needed(current_date):
        logger.info("Training/retraining HMM model...")
        hmm_detector.fit(df, current_date)
    
    # Get HMM prediction
    prediction = hmm_detector.predict_regime(df, current_date)
    
    # Check if we have a valid prediction
    if prediction.get('needs_training', False):
        # Still in warmup phase
        logger.info("HMM model in warmup phase, defaulting to neutral market type")
        return 'neutral', {'classification_rationale': 'HMM in warmup phase'}, False
    
    # Format return values for backtester
    market_type = prediction['regime']
    confidence = prediction['confidence']
    
    metrics = {
        'confidence': confidence,
        'hmm_state': prediction.get('state', 0),
        'classification_rationale': f"HMM detected {market_type} regime with {confidence:.2f} confidence"
    }
    
    # Add feature metrics if available
    if 'features' in prediction:
        for feature, value in prediction['features'].items():
            metrics[feature] = value
    
    # Get successful warmup status
    warmup_complete = hmm_detector.model is not None
    
    return market_type, metrics, warmup_complete

def get_regime_parameters(market_type, regime_score):
    """
    Get parameter set based on market regime and confidence score.

    Args:
        market_type: String with detected market type
        regime_score: Confidence score for the market regime (0-100)

    Returns:
        Dictionary with parameters for the current regime
    """
    # Import config once at the top of your file
    from config import config

    # Get base parameters for the market type
    if market_type in config['market_type_parameters']:
        params = config['market_type_parameters'][market_type].copy()
    else:
        params = config['market_type_parameters']['default'].copy()

    # Make adjustments based on regime score
    if regime_score > 90:
        # Very high confidence - enhance parameters
        params['base_sizing_factor'] *= 1.2  # 20% increase in position size

        # More aggressive parameters for high-confidence regimes
        if market_type == 'trend_following':
            params['trailing_activation_pct'] *= 0.8  # Earlier trailing stops
            params['dynamic_target_multiplier'] *= 1.1  # More aggressive targets

        elif market_type == 'mean_reverting':
            params['atr_stop_multiplier'] *= 1.1  # Wider stops
            params['bb_penetration_min'] *= 0.8  # Less band penetration required

    elif regime_score < 60:
        # Low confidence - more conservative
        params['base_sizing_factor'] *= 0.8  # 20% decrease in position size
        params['atr_stop_multiplier'] *= 0.9  # Tighter stops
        params['trailing_activation_pct'] *= 1.2  # Later trailing stops

        # More strict entry criteria for low-confidence regimes
        params['entry_volume_threshold'] *= 1.2  # Higher volume requirement

    # Add the regime score to the parameters for logging
    params['regime_score'] = regime_score
    params['market_type'] = market_type

    return params

def get_market_type_params(market_type, confidence=None):
    """Get parameters for the detected market type with confidence adjustment"""
    params = {}
    
    if market_type == 'trend_following':
        params = config['market_type']['trend_following'].copy()
    elif market_type == 'mean_reverting':
        params = config['market_type']['mean_reverting'].copy()
    else:  # neutral or unknown
        params = config['market_type']['neutral'].copy()
    
    # Add market type to parameters for use in regime scoring
    params['market_type'] = market_type
    
    # If HMM confidence is provided, add it to parameters
    if confidence is not None:
        params['confidence'] = confidence
    
    return params

#############################################################################
#                    TRADING STRATEGY FUNCTIONS                             #
#############################################################################

def is_in_trading_window(current_time):
    """Check if current time is within trading hours"""
    trading_hours = config['trading_hours']
    morning_session = (
        current_time >= trading_hours['morning_start'] and
        current_time <= trading_hours['morning_end']
    )
    afternoon_session = (
        current_time >= trading_hours['afternoon_start'] and
        current_time <= trading_hours['afternoon_end']
    )
    return morning_session or afternoon_session


def check_entry_signal(prev_row, current_market_type, regime_score=50):
    # Long entry conditions (mean reversion) - LOOSENED
    long_mean_reversion = (
            prev_row['low'] < prev_row['lower_band'] and
            prev_row['RSI'] < 35 and  # CHANGED: Back to 35 from 30
            prev_row['volume'] > 1.5 * prev_row['avg_volume'] and  # CHANGED: 1.5x from 1.8x
            prev_row['close'] > prev_row['low'] * 1.0005  # CHANGED: Minimal bounce requirement
    )

    # Short entry conditions (mean reversion) - LOOSENED
    short_mean_reversion = (
            prev_row['high'] > prev_row['upper_band'] and
            prev_row['RSI'] > 65 and  # CHANGED: Back to 65 from 70
            prev_row['volume'] > 1.5 * prev_row['avg_volume'] and  # CHANGED: 1.5x from 1.8x
            prev_row['close'] < prev_row['high'] * 0.9995  # CHANGED: Minimal pullback requirement
    )

    # Long entry conditions (trend following) - LOOSENED
    long_trend_following = (
            prev_row['close'] > prev_row['MA'] and
            prev_row['RSI'] > 50 and
            prev_row['RSI'] < 70 and
            prev_row['MA_slope'] > 0.1 and  # CHANGED: Back to 0.1 from 0.15
            prev_row['volume'] > prev_row['avg_volume'] * 1.2  # CHANGED: Reduced volume requirement
    )

    # Short entry conditions (trend following) - LOOSENED
    short_trend_following = (
            prev_row['close'] < prev_row['MA'] and
            prev_row['RSI'] < 50 and
            prev_row['RSI'] > 30 and
            prev_row['MA_slope'] < -0.1 and  # CHANGED: Back to -0.1 from -0.15
            prev_row['volume'] > prev_row['avg_volume'] * 1.2  # CHANGED: Reduced volume requirement
    )
    # Check conditions based on market type
    if current_market_type == 'mean_reverting':
        if long_mean_reversion:
            return 'long'
        elif short_mean_reversion:
            return 'short'

    elif current_market_type == 'trend_following':
        if long_trend_following:
            return 'long'
        elif short_trend_following:
            return 'short'

    elif current_market_type == 'neutral':
        # MODIFIED: More selective criteria for neutral markets
        # Require stronger signals with additional confirmation

        # For long entries in neutral markets
        if long_mean_reversion and prev_row['close'] > prev_row['MA']:
            # Mean reversion long with price above MA as confirmation
            return 'long'
        elif long_trend_following and prev_row['low'] < prev_row['middle_band']:
            # Trend following long with price near middle band as confirmation
            return 'long'

        # For short entries in neutral markets
        elif short_mean_reversion and prev_row['close'] < prev_row['MA']:
            # Mean reversion short with price below MA as confirmation
            return 'short'
        elif short_trend_following and prev_row['high'] > prev_row['middle_band']:
            # Trend following short with price near middle band as confirmation
            return 'short'

    return None

def check_entry_signal_ml(prev_row, ml_features, ml_predictor, current_date, current_market_type='neutral'):
    """
    Check for entry signals with ML filtering based on market type.
    
    Args:
        prev_row: Previous bar data
        ml_features: ML features for prediction
        ml_predictor: MLPredictor instance
        current_date: Current date for retraining check
        current_market_type: Current detected market type
        
    Returns:
        Tuple of (signal, probability, approved)
    """
    strategy = config['strategy']
    ml_settings = config['ml']
    
    # Long entry conditions (mean reversion)
    long_mean_reversion = (
        prev_row['low'] < prev_row['lower_band'] and 
        prev_row['RSI'] < strategy['rsi_oversold'] and 
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Short entry conditions (mean reversion)
    short_mean_reversion = (
        prev_row['high'] > prev_row['upper_band'] and 
        prev_row['RSI'] > strategy['rsi_overbought'] and 
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Long entry conditions (trend following)
    long_trend_following = (
        prev_row['close'] > prev_row['MA'] and
        prev_row['RSI'] > 50 and
        prev_row['RSI'] < 70 and  # Not overbought
        prev_row['MA_slope'] > 0.1 and  # Positive slope
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Short entry conditions (trend following)
    short_trend_following = (
        prev_row['close'] < prev_row['MA'] and
        prev_row['RSI'] < 50 and
        prev_row['RSI'] > 30 and  # Not oversold
        prev_row['MA_slope'] < -0.1 and  # Negative slope
        prev_row['volume'] > strategy['volume_multiplier'] * prev_row['avg_volume']
    )
    
    # Determine entry signal based on market type
    signal = None
    if current_market_type == 'mean_reverting':
        if long_mean_reversion:
            signal = 'long'
        elif short_mean_reversion:
            signal = 'short'
    
    elif current_market_type == 'trend_following':
        if long_trend_following:
            signal = 'long'
        elif short_trend_following:
            signal = 'short'
    
    elif current_market_type == 'neutral':
        # MODIFIED: More selective criteria for neutral markets
        # Require stronger signals with additional confirmation
        
        # For long entries in neutral markets
        if long_mean_reversion and prev_row['close'] > prev_row['MA']:
            # Mean reversion long with price above MA as confirmation
            signal = 'long'
        elif long_trend_following and prev_row['low'] < prev_row['middle_band']:
            # Trend following long with price near middle band as confirmation
            signal = 'long'
            
        # For short entries in neutral markets
        elif short_mean_reversion and prev_row['close'] < prev_row['MA']:
            # Mean reversion short with price below MA as confirmation
            signal = 'short'
        elif short_trend_following and prev_row['high'] > prev_row['middle_band']:
            # Trend following short with price near middle band as confirmation
            signal = 'short'
    
    if signal:
        # Check for ML approval if ML is enabled and model is trained
        if ml_settings['enable'] and ml_predictor.model is not None:
            # Check if retraining is needed
            if ml_predictor.check_retrain_needed(current_date):
                logger.info(f"ML model retraining needed. Last training: {ml_predictor.last_training_date}")
                # Retraining will be handled in the main loop
            
            # Update trade type in features
            ml_features['trade_type'] = 1 if signal == 'long' else 0
            
            # Get ML prediction
            probability, approved = ml_predictor.predict_trade_success(ml_features)
            
            # Adjust approval based on HMM confidence if available
            if 'hmm_confidence' in ml_features and ml_settings.get('use_hmm_features', True):
                hmm_confidence = ml_features['hmm_confidence']
                hmm_weight = ml_settings.get('hmm_confidence_weight', 0.3)
                
                # Weighted combination of ML probability and HMM confidence
                combined_probability = (probability * (1 - hmm_weight)) + (hmm_confidence * hmm_weight)
                
                # Only approve if combined probability exceeds threshold
                approved = combined_probability >= ml_settings['prediction_threshold']
                
                logger.debug(f"ML prob: {probability:.2f}, HMM conf: {hmm_confidence:.2f}, Combined: {combined_probability:.2f}, Approved: {approved}")
                
                # Return signal with combined probability
                return signal, combined_probability, approved
            
            # Return signal with ML information (original logic)
            return signal, probability, approved
        else:
            # ML not enabled or model not trained yet - approve all trades
            return signal, 0.5, True
    
    # No signal
    return None, 0.0, False

def check_exit_conditions(df, i, position, entry_price, stop_loss, profit_target, bars_held,
                          highest_price=None, lowest_price=None, atr=None, market_type='neutral',
                          current_portion=1.0):
    """
    Enhanced exit strategy combining multiple approaches.

    Args:
        df: DataFrame with price and indicator data
        i: Current bar index
        position: Position size (positive for long, negative for short)
        entry_price: Trade entry price
        stop_loss: Current stop loss level
        profit_target: Current profit target level
        bars_held: Number of bars position has been held
        highest_price: Highest price since entry (for trailing calculations)
        lowest_price: Lowest price since entry (for trailing calculations)
        atr: ATR value at entry
        market_type: Current market type classification
        current_portion: Current portion of the position still open

    Returns:
        Dictionary with exit information
    """
    # Initialize tracking variables if needed
    if highest_price is None:
        highest_price = entry_price
    if lowest_price is None:
        lowest_price = entry_price
    if atr is None:
        atr = df.iloc[i - bars_held]['ATR']

    current_row = df.iloc[i]
    entry_bar = i - bars_held

    # Update highest/lowest prices
    highest_price = max(highest_price, current_row['high'])
    lowest_price = min(lowest_price, current_row['low'])
    exit_data = {'exit': False, 'highest_price': highest_price, 'lowest_price': lowest_price}

    # Calculate current profit percentage
    current_price = current_row['close']
    profit_pips = (current_price - entry_price) if position > 0 else (entry_price - current_price)
    profit_pct = profit_pips / entry_price * 100

    # Check stop loss
    if (position > 0 and current_row['low'] <= stop_loss) or \
            (position < 0 and current_row['high'] >= stop_loss):
        exit_data['exit'] = True
        exit_data['exit_price'] = stop_loss
        exit_data['exit_reason'] = 'stop_loss'
        return exit_data

    # Check profit target
    if (position > 0 and current_row['high'] >= profit_target) or \
            (position < 0 and current_row['low'] <= profit_target):
        exit_data['exit'] = True
        exit_data['exit_price'] = profit_target
        exit_data['exit_reason'] = 'profit_target'
        return exit_data

    # Check time exit
    if bars_held >= config['risk']['max_bars_held']:
        exit_data['exit'] = True
        exit_data['exit_price'] = current_row['close']  # Exit at current bar's close
        exit_data['exit_reason'] = 'time_exit'
        return exit_data

    # No exit signal
    return exit_data

def calculate_position_size(account_value, atr, position_size_adj, market_type='neutral', regime_score=50):
    """
    Calculate appropriate position size based on risk, margin, and regime parameters.

    Args:
        account_value: Current account value
        atr: Current ATR value
        position_size_adj: Position size adjustment from standard calculation
        market_type: Current market type
        regime_score: Current regime score (0-100)

    Returns:
        Number of contracts to trade
    """
    from config import config
    risk = config['risk']
    account = config['account']

    # Get regime-specific parameters
    params = get_regime_parameters(market_type, regime_score)

    # Apply regime-specific sizing factor
    regime_position_adj = params['base_sizing_factor']

    # Combine with standard position size adjustment
    combined_adj = position_size_adj * regime_position_adj

    # Risk-based sizing with adjustment
    risk_per_contract = atr * risk['contract_multiplier']
    risk_based_contracts = max(1, int((account_value * risk['risk_per_trade'] * combined_adj) / risk_per_contract))

    # Margin-based sizing
    margin_based_contracts = int(account_value / account['initial_margin'])

    # Apply transaction costs
    temp_num_contracts = min(risk_based_contracts, margin_based_contracts)
    adjusted_account = account_value - account['transaction_cost'] * temp_num_contracts
    final_margin_contracts = int(adjusted_account / account['initial_margin'])

    # Use most conservative value with a reasonable cap
    max_contracts = 100

    # Add logging for position sizing decisions
    logger.debug(f"Position sizing: risk_based={risk_based_contracts}, margin_based={margin_based_contracts}, " +
                 f"regime_adj={regime_position_adj:.2f}, market_type={market_type}, regime_score={regime_score}")

    return min(risk_based_contracts, final_margin_contracts, max_contracts)

def calculate_dynamic_stop(row, entry_price, position, market_type):
    """
    Calculate a dynamic stop loss based on market conditions.

    Args:
        row: Current price bar data
        entry_price: Trade entry price
        position: Position size (positive for long, negative for short)
        market_type: Current market type classification

    Returns:
        Calculated stop loss price
    """
    atr = row['ATR']
    volatility_regime = row['volatility_regime']

    # Base multiplier adjusted by volatility regime
    base_multiplier = config['risk']['atr_stop_multiplier']
    vol_adjustment = 1.0 if volatility_regime == 1 else (0.8 if volatility_regime == 0 else 1.3)

    # Market type specific adjustments
    if market_type == 'trend_following':
        market_adjustment = 1.2  # Wider stops for trend following
    elif market_type == 'mean_reverting':
        market_adjustment = 0.9  # Tighter stops for mean reversion
    else:  # neutral
        market_adjustment = 1.0  # Standard stops for neutral

    # Calculate final multiplier and stop price
    final_multiplier = base_multiplier * vol_adjustment * market_adjustment

    if position > 0:  # Long position
        return entry_price - (final_multiplier * atr)
    else:  # Short position
        return entry_price + (final_multiplier * atr)

def calculate_intelligent_trailing_stop(df, current_index, position, entry_price, entry_index, atr, market_type):
    """
    Calculate an intelligent trailing stop that adapts to profit, momentum and time.

    Args:
        df: DataFrame with price and indicator data
        current_index: Current bar index
        position: Position size (positive for long, negative for short)
        entry_price: Trade entry price
        entry_index: Entry bar index
        atr: ATR value at entry
        market_type: Current market type classification

    Returns:
        Calculated trailing stop price
    """
    # Calculate current profit
    current_price = df.iloc[current_index]['close']
    pips_profit = (current_price - entry_price) if position > 0 else (entry_price - current_price)
    profit_multiple = pips_profit / atr  # Profit in terms of ATR multiples

    # Time factor - gradually tighten as trade progresses
    time_held = current_index - entry_index
    time_factor = min(1.5, 0.8 + (time_held / 20) * 0.7)  # Starts at 0.8, approaches 1.5

    # Momentum factor - check if momentum is fading
    if 'RSI' in df.columns:
        rsi = df.iloc[current_index]['RSI']
        prev_rsi = df.iloc[current_index - 1]['RSI']
        rsi_change = rsi - prev_rsi

        # For longs, we tighten when RSI starts falling from high levels
        if position > 0 and rsi > 70 and rsi_change < -2:
            momentum_factor = 0.7  # Tighter stop when momentum fades
        # For shorts, we tighten when RSI starts rising from low levels
        elif position < 0 and rsi < 30 and rsi_change > 2:
            momentum_factor = 0.7  # Tighter stop when momentum fades
        else:
            momentum_factor = 1.0  # Normal stop when momentum is steady
    else:
        momentum_factor = 1.0

    # Profit-based adjustment - tighter stops with more profit
    if profit_multiple <= 1.0:
        profit_factor = 1.5  # Wider stop when profit is small
    elif profit_multiple <= 2.0:
        profit_factor = 1.2  # Moderate stop when profit is medium
    else:
        profit_factor = 0.8  # Tight stop when profit is large

    # Market type specific base multiplier
    if market_type == 'trend_following':
        base_multiplier = 1.2
    elif market_type == 'mean_reverting':
        base_multiplier = 1.8
    else:  # neutral
        base_multiplier = 1.5

    # Calculate final trailing distance as ATR multiple
    trailing_multiple = base_multiplier * profit_factor * momentum_factor * time_factor

    # Calculate and return the trailing stop price
    highest_price = df.iloc[entry_index:current_index + 1]['high'].max()
    lowest_price = df.iloc[entry_index:current_index + 1]['low'].min()

    if position > 0:  # Long position
        return highest_price - (trailing_multiple * atr)
    else:  # Short position
        return lowest_price + (trailing_multiple * atr)


def check_multi_layered_exit(row, position, entry_price, atr, current_portion=1.0):
    """Check for exits with multi-layered approach."""
    exit_data = {'exit': False}

    # First target (30% of position)
    first_target = entry_price + (1.0 * atr) if position > 0 else entry_price - (1.0 * atr)
    # Second target (30% of position)
    second_target = entry_price + (2.0 * atr) if position > 0 else entry_price - (2.0 * atr)

    # Check if we've reached first target
    if (position > 0 and row['high'] >= first_target) or (position < 0 and row['low'] <= first_target):
        if current_portion == 1.0:  # Haven't taken partial exit yet
            exit_data['exit'] = True
            exit_data['exit_price'] = first_target  # Make sure exit_price is included
            exit_data['exit_reason'] = 'partial_profit_1'
            exit_data['exit_portion'] = 0.3
            exit_data['remaining_portion'] = 0.7

    # Check if we've reached second target
    elif (position > 0 and row['high'] >= second_target) or (position < 0 and row['low'] <= second_target):
        if current_portion <= 0.7 and current_portion > 0.4:  # Already took first exit
            exit_data['exit'] = True
            exit_data['exit_price'] = second_target  # Make sure exit_price is included
            exit_data['exit_reason'] = 'partial_profit_2'
            exit_data['exit_portion'] = 0.3
            exit_data['remaining_portion'] = 0.4

    return exit_data


def check_improved_time_exit(row, position, entry_bar, current_bar, profit_pct, market_type):
    """
    Check for improved time-based exit conditions.

    Args:
        row: Current price bar data
        position: Position size (positive for long, negative for short)
        entry_bar: Entry bar index
        current_bar: Current bar index
        profit_pct: Current profit percentage
        market_type: Current market type classification

    Returns:
        Dictionary with exit information
    """
    exit_data = {'exit': False}
    max_bars = config['risk']['max_bars_held']

    # Market-specific adjustments
    if market_type == 'trend_following':
        max_bars = int(max_bars * 1.5)  # Allow trend trades to run longer
    elif market_type == 'mean_reverting':
        max_bars = int(max_bars * 0.8)  # Exit mean reversion trades earlier

    # Don't exit profitable trades too early
    if profit_pct > 0.5:  # If we have a decent profit
        max_bars = int(max_bars * 1.3)  # Allow more time

    bars_held = current_bar - entry_bar

    # Standard time exit check
    if bars_held >= max_bars:
        # If in a trend and profitable, only exit part of position
        if market_type == 'trend_following' and profit_pct > 0.2:
            exit_data['exit'] = True
            exit_data['exit_price'] = row['close']  # Added exit_price
            exit_data['exit_reason'] = 'partial_time_exit'
            exit_data['exit_portion'] = 0.5  # Exit half the position
            exit_data['remaining_portion'] = 0.5
        else:
            exit_data['exit'] = True
            exit_data['exit_price'] = row['close']  # Added exit_price
            exit_data['exit_reason'] = 'time_exit'
            exit_data['exit_portion'] = 1.0  # Full exit
            exit_data['remaining_portion'] = 0.0

    return exit_data