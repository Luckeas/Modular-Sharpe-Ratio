"""
run_ml_regime_backtest.py - Example script demonstrating how to use the ML-enhanced regime detection

This script shows how to train the ML regime detector and run a backtest with it.
"""

import os
import logging
from datetime import datetime
from config import config  # Import your existing config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the ML integration functions
from ml_enhanced_regime_detection import apply_ml_regime_strategy

def main():
    """Run a backtest with ML-enhanced regime detection"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ml_regime_backtest_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add output directory to config
    config['output_dir'] = output_dir
    
    # Add ML-specific configuration
    if 'ml' not in config:
        config['ml'] = {}
    
    # Configure ML settings
    config['ml'].update({
        'prediction_horizon': 5,  # Days ahead to predict regime
        'model_type': 'xgboost',  # 'xgboost' or 'random_forest'
        'enable': True,           # Enable ML regime prediction
        'use_hmm_features': True  # Use HMM confidence as a feature
    })
    
    # Set a longer training period for ML if available
    # Note: ML training benefits from longer history than the backtest period
    config['data']['ml_train_start_date'] = '2010-01-01'  # Optional: use longer history for ML training
    
    # Run the ML-enhanced backtest
    logger.info("Starting ML-enhanced regime backtest...")
    results = apply_ml_regime_strategy(config)
    
    if results is None:
        logger.error("Backtest failed. Exiting.")
        return
    
    logger.info("ML-enhanced regime backtest completed successfully.")
    logger.info(f"Results saved to: {output_dir}")
    
    # Summarize key results
    trades = results['trades']
    portfolio_values = results['portfolio_values']
    
    if trades:
        win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades) * 100
        total_profit = sum(t['profit'] for t in trades)
        max_profit = max(t['profit'] for t in trades)
        max_loss = min(t['profit'] for t in trades)
        
        logger.info(f"Number of trades: {len(trades)}")
        logger.info(f"Win rate: {win_rate:.2f}%")
        logger.info(f"Total profit: ${total_profit:.2f}")
        logger.info(f"Max profit: ${max_profit:.2f}")
        logger.info(f"Max loss: ${max_loss:.2f}")
    
    # Compare ML-enhanced performance to standard backtester (optional)
    # This requires running the standard backtest separately and comparing results
    try:
        # Implement a comparison between ML and standard approach if desired
        compare_with_standard = False
        
        if compare_with_standard:
            # Import standard backtest function
            from refactored_backtester import run_backtest
            from utils import load_and_process_data, calculate_indicators, setup_directories
            
            # Set up outputs for standard backtest
            std_output_dir, std_file_paths = setup_directories(
                config['data']['start_date'],
                config['data']['end_date'],
                "standard_backtest"
            )
            
            # Load and process data
            df = load_and_process_data(
                config['data']['file_path'],
                config['data']['start_date'],
                config['data']['end_date']
            )
            
            # Calculate indicators
            df = calculate_indicators(df, config)
            
            # Run standard backtest
            std_trades, std_portfolio_values, _, _, _, _, _ = run_backtest(
                df, 
                visualize_trades=False,
                file_paths=std_file_paths
            )
            
            # Compare results
            if std_trades and trades:
                ml_win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades) * 100
                std_win_rate = sum(1 for t in std_trades if t['profit'] > 0) / len(std_trades) * 100
                
                ml_profit = sum(t['profit'] for t in trades)
                std_profit = sum(t['profit'] for t in std_trades)
                
                logger.info("\n===== COMPARISON: ML vs STANDARD =====")
                logger.info(f"ML Win Rate: {ml_win_rate:.2f}% | Standard Win Rate: {std_win_rate:.2f}%")
                logger.info(f"ML Total Profit: ${ml_profit:.2f} | Standard Total Profit: ${std_profit:.2f}")
                logger.info(f"ML Trades: {len(trades)} | Standard Trades: {len(std_trades)}")
                
                # Calculate improvement percentages
                win_rate_improvement = (ml_win_rate - std_win_rate) / std_win_rate * 100
                profit_improvement = (ml_profit - std_profit) / abs(std_profit) * 100 if std_profit != 0 else float('inf')
                
                logger.info(f"Win Rate Improvement: {win_rate_improvement:.2f}%")
                logger.info(f"Profit Improvement: {profit_improvement:.2f}%")
    
    except Exception as e:
        logger.error(f"Error in comparison with standard backtest: {e}")

if __name__ == "__main__":
    main()
