"""
ml_enhanced_regime_detection.py - Integration of ML-based regime detection with existing backtester

This module demonstrates how to integrate the ML regime detector with your existing backtesting framework.
"""

import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import the ML regime detector
from ml_regime_detector import MLRegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_market_type_with_ml(df, lookback_days=20, current_date=None, ml_detector=None, 
                             use_ml_prediction=True, prediction_horizon=5):
    """
    Enhanced market type detection using HMM + ML for prediction.
    
    Args:
        df: DataFrame with price and indicator data
        lookback_days: Days to look back (used by HMM detector)
        current_date: Current date for detection window
        ml_detector: ML regime detector instance
        use_ml_prediction: Whether to use ML prediction for future regime
        prediction_horizon: Days ahead to predict (if not using ML detector's default)
        
    Returns:
        Tuple of (market_type, metrics, warmup_complete)
    """
    # Default to last date in DataFrame if no current_date
    if current_date is None:
        current_date = df['date'].max()
    
    # If ML detector is not provided or ML prediction not requested, 
    # fall back to standard HMM detection
    if ml_detector is None or not use_ml_prediction:
        # Import the standard detection function
        from backtester_common import detect_market_type as hmm_detect
        return hmm_detect(df, lookback_days, current_date)
    
    # Ensure ML detector is trained
    if ml_detector.model is None:
        logger.warning("ML model not trained. Falling back to HMM detection.")
        # Import the standard detection function
        from backtester_common import detect_market_type as hmm_detect
        return hmm_detect(df, lookback_days, current_date)
    
    # Get current regime from HMM (for comparison and as fallback)
    hmm_prediction = ml_detector.hmm_detector.predict_regime(df, current_date)
    current_regime = hmm_prediction['regime']
    
    # Get ML prediction for future regime
    ml_prediction = ml_detector.predict_future_regime(df, current_date)
    predicted_regime = ml_prediction['regime']
    confidence = ml_prediction['confidence']
    
    # Combine metrics
    metrics = {
        'current_regime': current_regime,
        'predicted_regime': predicted_regime,
        'hmm_confidence': hmm_prediction.get('confidence', 0.5),
        'ml_confidence': confidence,
        'prediction_date': ml_prediction['prediction_date'],
        'ml_probabilities': ml_prediction['probability'],
        'classification_rationale': f"ML predicts {predicted_regime} regime in {prediction_horizon} days with {confidence:.2f} confidence"
    }
    
    # Add feature metrics if available
    if 'features' in hmm_prediction:
        for feature, value in hmm_prediction['features'].items():
            metrics[feature] = value
    
    # Return the predicted regime, metrics, and warmup complete flag
    return predicted_regime, metrics, True

def initialize_ml_detector(output_dir=None, model_path=None, prediction_horizon=5):
    """
    Initialize the ML regime detector, either by loading an existing model or creating a new one.
    
    Args:
        output_dir: Directory to save model files and results
        model_path: Path to an existing model to load
        prediction_horizon: Days ahead to predict regime
        
    Returns:
        Initialized ML regime detector
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize detector
    ml_detector = MLRegimeDetector(
        output_dir=output_dir,
        model_type='xgboost',  # Can be 'xgboost' or 'random_forest'
        prediction_horizon=prediction_horizon
    )
    
    # Load model if path is provided
    if model_path and os.path.exists(model_path):
        success = ml_detector.load_model(model_path)
        if success:
            logger.info(f"Loaded existing ML model from {model_path}")
        else:
            logger.warning(f"Failed to load model from {model_path}. Will need to train a new one.")
    
    return ml_detector

def prepare_and_train_ml_detector(df, output_dir=None, test_size=0.2, prediction_horizon=5):
    """
    Prepare data and train the ML regime detector.
    
    Args:
        df: DataFrame with price and indicator data
        output_dir: Directory to save model files and results
        test_size: Portion of data to use for testing
        prediction_horizon: Days ahead to predict regime
        
    Returns:
        Trained ML regime detector
    """
    # Initialize detector
    ml_detector = initialize_ml_detector(output_dir, prediction_horizon=prediction_horizon)
    
    # Train the model
    metrics = ml_detector.train_model(df, test_size=test_size)
    
    # Log performance metrics
    logger.info(f"Model training complete with accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top predictive features: {', '.join(metrics['top_features'][:5])}")
    
    return ml_detector

def evaluate_ml_predictions(ml_detector, df, output_dir):
    """
    Evaluate ML predictions against actual regimes and visualize results.
    
    Args:
        ml_detector: Trained ML regime detector
        df: DataFrame with price and indicator data
        output_dir: Directory to save results
        
    Returns:
        DataFrame with evaluation results
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Evaluate predictions
    results_df = ml_detector.evaluate_predictions(df)
    
    if len(results_df) == 0:
        logger.warning("No prediction results available for evaluation.")
        return pd.DataFrame()
    
    # Calculate metrics by regime
    regime_accuracy = results_df.groupby('actual_regime')['correct'].mean()
    
    # Log regime-specific accuracy
    logger.info("Prediction accuracy by regime:")
    for regime, accuracy in regime_accuracy.items():
        logger.info(f"  {regime}: {accuracy:.4f}")
    
    # Create visualization of regime predictions vs actual
    try:
        import matplotlib.pyplot as plt
        
        # Plot regime comparison
        plt.figure(figsize=(15, 8))
        
        # Map regimes to numeric values for plotting
        regime_map = {
            'trend_following': 0,
            'mean_reverting': 1, 
            'neutral': 2
        }
        
        # Reverse map for labels
        reverse_map = {v: k for k, v in regime_map.items()}
        
        # Extract actual and predicted regimes
        results_df['actual_regime_num'] = results_df['actual_regime_idx']
        results_df['predicted_regime_num'] = results_df['predicted_regime_idx']
        
        # Plot actual regimes
        plt.plot(results_df['date'], results_df['actual_regime_num'], 'b-', label='Actual Regime')
        
        # Plot predicted regimes
        plt.plot(results_df['date'], results_df['predicted_regime_num'], 'r--', label='Predicted Regime')
        
        # Format plot
        plt.yticks([0, 1, 2], ['Trend Following', 'Mean Reverting', 'Neutral'])
        plt.title(f'Actual vs Predicted Regimes ({ml_detector.prediction_horizon}-Day Horizon)')
        plt.xlabel('Date')
        plt.ylabel('Market Regime')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight correct and incorrect predictions
        correct_mask = results_df['correct']
        incorrect_mask = ~correct_mask
        
        # Only add points for incorrect predictions to reduce clutter
        if sum(incorrect_mask) > 0:
            plt.scatter(results_df.loc[incorrect_mask, 'date'], 
                        results_df.loc[incorrect_mask, 'predicted_regime_num'],
                        color='red', alpha=0.5, s=30, marker='x', label='Incorrect Predictions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regime_prediction_comparison.png'), dpi=150)
        plt.close()
        
        # Plot confidence levels
        plt.figure(figsize=(15, 6))
        plt.plot(results_df['date'], results_df['confidence'], 'g-')
        plt.fill_between(results_df['date'], results_df['confidence'], alpha=0.3)
        plt.title(f'ML Model Prediction Confidence')
        plt.xlabel('Date')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_confidence.png'), dpi=150)
        plt.close()
        
        # Save results to CSV
        results_df.to_csv(os.path.join(output_dir, 'ml_prediction_results.csv'), index=False)
        
    except Exception as e:
        logger.error(f"Error creating evaluation visualizations: {e}")
    
    return results_df


def integrate_ml_with_backtester(config):
    """
    Main function to integrate ML-based regime detection with the backtester framework.

    This function demonstrates how to:
    1. Prepare data and train the ML model
    2. Set up the ML detector for use in backtesting
    3. Override the standard regime detection function

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (ml_detector, detection_function)
    """
    # Create output directory for ML model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ml_output_dir = os.path.join(config.get('output_dir', 'ml_regime_results'), f'ml_model_{timestamp}')

    if not os.path.exists(ml_output_dir):
        os.makedirs(ml_output_dir)

    # Load and prepare data
    from utils import load_and_process_data, calculate_indicators

    logger.info("Loading and processing data for ML training...")

    # Use a longer time period for ML training compared to backtest period
    train_start_date = config['data'].get('ml_train_start_date', config['data']['start_date'])

    df = load_and_process_data(
        config['data']['file_path'],
        train_start_date,
        config['data']['end_date']
    )

    if df is None or len(df) == 0:
        logger.error("No data available for ML training. Cannot proceed with ML integration.")
        return None, None

    # Calculate indicators
    df = calculate_indicators(df, config)

    # Train the ML model
    logger.info("Training ML regime detector...")
    prediction_horizon = config.get('ml', {}).get('prediction_horizon', 5)

    ml_detector = prepare_and_train_ml_detector(
        df,
        output_dir=ml_output_dir,
        prediction_horizon=prediction_horizon
    )

    # Evaluate the model
    logger.info("Evaluating ML regime predictions...")
    evaluation_results = evaluate_ml_predictions(ml_detector, df, ml_output_dir)

    # Create a wrapper function that will be used to override the standard detection function
    def ml_enhanced_detection(df, lookback_days=20, current_date=None):
        return detect_market_type_with_ml(
            df,
            lookback_days=lookback_days,
            current_date=current_date,
            ml_detector=ml_detector,
            use_ml_prediction=True,
            prediction_horizon=prediction_horizon
        )

    # Return the ML detector and detection function
    return ml_detector, ml_enhanced_detection


def apply_ml_regime_strategy(config):
    """
    Execute a backtest with ML-enhanced regime detection.

    Args:
        config: Configuration dictionary

    Returns:
        Results from the backtest
    """
    # Set up ML detector and detection function
    ml_detector, ml_detection_func = integrate_ml_with_backtester(config)

    if ml_detector is None or ml_detection_func is None:
        logger.error("Failed to set up ML regime detection. Exiting.")
        return None

    # Override the standard detection function in backtester_common
    import backtester_common
    original_detect_func = backtester_common.detect_market_type
    backtester_common.detect_market_type = ml_detection_func

    try:
        # Run backtest with ML-enhanced regime detection
        logger.info("Running backtest with ML-enhanced regime detection...")

        # Set up output directory and file paths
        from utils import setup_directories
        output_dir, file_paths = setup_directories(
            config['data']['start_date'],
            config['data']['end_date'],
            "ml_regime_backtest"
        )

        # Load and process data for the backtest period
        from utils import load_and_process_data, calculate_indicators

        df = load_and_process_data(
            config['data']['file_path'],
            config['data']['start_date'],
            config['data']['end_date']
        )

        if df is None or len(df) == 0:
            logger.error("No data available for backtest. Exiting.")
            return None

        # Calculate indicators
        df = calculate_indicators(df, config)

        # Run backtest with unified backtester and ML enabled
        trades, portfolio_values, df, regime_log, market_type_log, regime_score_bins, season_metrics, ml_metrics = run_backtest(
            df,
            visualize_trades=config['visualization']['generate_png_charts'],
            file_paths=file_paths,
            use_ml=True
        )

        # The remaining portion of the function can stay the same
        # ... code for analysis and reporting ...c
        
        # Analyze results
        portfolio_df = pd.DataFrame({'date': df['date'], 'value': portfolio_values})
        portfolio_series = portfolio_df.set_index('date')['value']
        
        # Import analysis functions
        from trade_analysis import analyze_performance, analyze_by_regime, analyze_trades_by_market_type
        
        # Calculate performance metrics
        results = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
        trades_by_regime = analyze_by_regime(trades, regime_score_bins)
        trades_by_market = analyze_trades_by_market_type(trades)
        
        # Log results
        logger.info("\n===== ML REGIME BACKTEST RESULTS =====")
        logger.info(f"Initial Capital: ${results['initial_capital']:.2f}")
        logger.info(f"Final Portfolio Value: ${results['final_value']:.2f}")
        logger.info(f"Profit/Loss: ${results['profit_loss']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Number of Trades: {results['number_of_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, 'ml_regime_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("ML-ENHANCED REGIME DETECTION BACKTEST SUMMARY\n")
            f.write("===========================================\n\n")
            f.write(f"Initial Capital: ${results['initial_capital']:.2f}\n")
            f.write(f"Final Portfolio Value: ${results['final_value']:.2f}\n")
            f.write(f"Profit/Loss: ${results['profit_loss']:.2f}\n")
            f.write(f"Total Return: {results['total_return_pct']:.2f}%\n")
            f.write(f"Number of Trades: {results['number_of_trades']}\n")
            f.write(f"Win Rate: {results['win_rate']:.2f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%\n\n")
            
            f.write("ML MODEL INFORMATION\n")
            f.write("===================\n")
            f.write(f"Model Type: {ml_detector.model_type}\n")
            f.write(f"Prediction Horizon: {ml_detector.prediction_horizon} days\n")
            f.write(f"Training Date: {ml_detector.last_training_date}\n")
            
            if ml_detector.feature_importance:
                f.write("\nTop 10 Features by Importance:\n")
                importance_items = sorted(ml_detector.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for feature, importance in importance_items:
                    f.write(f"  {feature}: {importance:.4f}\n")
        
        logger.info(f"Results saved to {output_dir}")
        
        # Return results
        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'regime_log': regime_log,
            'market_type_log': market_type_log,
            'ml_detector': ml_detector,
            'ml_metrics': ml_metrics
        }
    
    finally:
        # Restore original detection function
        backtester_common.detect_market_type = original_detect_func
