"""
optimize_for_sharpe.py - Script to optimize strategy parameters for maximum Sharpe ratio
"""

import os
import logging
from datetime import datetime

# Import from centralized config
from config import config

# Import utility functions
from utils import load_and_process_data, calculate_indicators

# Import the optimization function
from parameter_optimizer import optimize_with_optuna

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sharpe_optimization_{timestamp}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and prepare data
    df = load_and_process_data(
        config['data']['file_path'],
        config['data']['start_date'],
        config['data']['end_date']
    )
    
    if df is None or len(df) == 0:
        logger.error("No data available after loading. Exiting.")
        exit(1)
    
    # Calculate all indicators
    df = calculate_indicators(df, config)
    
    # Run optimization focusing on Sharpe ratio
    logger.info("Starting parameter optimization for maximum Sharpe ratio...")
    best_params = optimize_with_optuna(
        df,
        output_dir=output_dir,
        n_trials=100,  # Adjust based on your computational resources
        focus_sharpe=True
    )
    
    logger.info(f"Optimization complete. Results saved to {output_dir}")
    logger.info(f"Best parameters: {best_params}")
    
    # Option to apply the best parameters and run a final backtest
    apply_best_params = True
    
    if apply_best_params:
        logger.info("Running final backtest with optimized parameters...")
        
        # Import unified backtester
        from unified_backtester import run_backtest
        
        # Update config with best parameters
        from parameter_optimizer import update_config_with_params
        update_config_with_params(config, best_params)
        
        # Set up file paths for final backtest
        final_dir = os.path.join(output_dir, "final_backtest")
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            
        file_paths = {
            'trade_log': os.path.join(final_dir, 'trade_log.csv'),
            'portfolio_value': os.path.join(final_dir, 'portfolio_value.csv'),
            'regime_log': os.path.join(final_dir, 'regime_log.csv'),
            'market_type_log': os.path.join(final_dir, 'market_type_log.csv'),
            'summary': os.path.join(final_dir, 'summary.txt')
        }
        
        # Run backtest with optimized parameters
        trades, portfolio_values, _, _, _, _, _, _ = run_backtest(
            df.copy(),
            visualize_trades=True,  # Enable visualizations for final result
            file_paths=file_paths
        )
        
        # Analyze and report results
        from trade_analysis import analyze_performance
        
        portfolio_df = pd.DataFrame({'date': df['date'], 'value': portfolio_values})
        portfolio_series = portfolio_df.set_index('date')['value']
        
        results = analyze_performance(trades, portfolio_series, config['account']['initial_capital'])
        
        logger.info("\n===== FINAL BACKTEST RESULTS WITH OPTIMIZED PARAMETERS =====")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        logger.info(f"Profit/Loss: ${results['profit_loss']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Maximum Drawdown: {results['max_drawdown_pct']:.2f}%")
