"""
optimize_and_test.py - Launcher for the full Sharpe ratio optimization workflow

This script runs the complete optimization and robustness testing workflow:
1. Optimizes strategy parameters for maximum Sharpe ratio
2. Tests the robustness of the optimized parameters
3. Summarizes the results
"""

import os
import argparse
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from central config
from config import config


def convert_to_serializable(obj):
    """
    Convert non-serializable objects to serializable format.

    Args:
        obj: Object to convert

    Returns:
        Serializable version of the object
    """
    # Handle datetime and Timestamp objects
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()

    # Primitive types
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]

    # Dictionaries
    if isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}

    # Fall back to string representation
    return str(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Sharpe ratio optimization and testing workflow')
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancement')
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Skip optimization and use existing parameters')
    parser.add_argument('--params', type=str, help='JSON file with existing parameters (if skipping optimization)')
    parser.add_argument('--robustness-runs', type=int, default=100, help='Number of runs for robustness testing')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = args.output if args.output else f"sharpe_optimization_workflow_{timestamp}"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

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

    # Save initial config with serialization handling
    with open(os.path.join(main_output_dir, 'initial_config.json'), 'w') as f:
        # Convert config to a serializable format
        serializable_config = convert_to_serializable(config)
        json.dump(serializable_config, f, indent=4)

    # Step 1: Parameter Optimization
    optimized_params = None

    if not args.skip_optimization:
        logger.info("Step 1: Running Sharpe ratio optimization")

        # Import and run optimization
        from optimize_sharpe_ratio import run_sharpe_optimization

        optimization_results = run_sharpe_optimization()

        if optimization_results and 'best_params' in optimization_results:
            optimized_params = optimization_results['best_params']

            # Save optimization results
            with open(os.path.join(main_output_dir, 'optimization_results.json'), 'w') as f:
                # Convert any non-serializable values to serializable format
                serializable_results = convert_to_serializable(optimization_results)
                json.dump(serializable_results, f, indent=4)

            logger.info(f"Optimization complete. Best parameters saved to {main_output_dir}/optimization_results.json")
        else:
            logger.error("Optimization failed. Exiting workflow.")
            exit(1)
    else:
        # Load existing parameters
        if not args.params:
            logger.error("Must provide --params when using --skip-optimization. Exiting workflow.")
            exit(1)

        try:
            with open(args.params, 'r') as f:
                param_data = json.load(f)

            # Extract parameters from different possible formats
            if isinstance(param_data, dict):
                if 'best_params' in param_data:
                    optimized_params = param_data['best_params']
                elif 'params' in param_data:
                    optimized_params = param_data['params']
                else:
                    # Assume the whole dict is parameters
                    optimized_params = param_data
            else:
                logger.error(f"Invalid parameter format in {args.params}")
                exit(1)

            logger.info(f"Loaded existing parameters from {args.params}")
        except Exception as e:
            logger.error(f"Error loading parameters file: {e}")
            exit(1)

    # Step 2: Robustness Testing
    if optimized_params:
        logger.info("Step 2: Running robustness testing")

        # Import and run robustness testing
        from test_sharpe_robustness import run_robustness_test

        robustness_results = run_robustness_test(
            optimized_params,
            num_runs=args.robustness_runs,
            parallel=not args.no_parallel,  # Default to parallel unless explicitly disabled
            use_ml=args.ml
        )

        if robustness_results and 'summary' in robustness_results:
            # Create link to robustness results
            with open(os.path.join(main_output_dir, 'robustness_results.json'), 'w') as f:
                # Convert summary to serializable format
                serializable_summary = convert_to_serializable(robustness_results['summary'])
                json.dump(serializable_summary, f, indent=4)

            logger.info(f"Robustness testing complete. Results saved to {robustness_results['output_dir']}")

            # Create main workflow summary
            with open(os.path.join(main_output_dir, 'workflow_summary.txt'), 'w') as f:
                f.write("SHARPE RATIO OPTIMIZATION WORKFLOW SUMMARY\n")
                f.write("=======================================\n\n")

                f.write("OPTIMIZED PARAMETERS\n")
                f.write("-------------------\n")
                for param, value in optimized_params.items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")

                f.write("ROBUSTNESS METRICS\n")
                f.write("------------------\n")
                summary = robustness_results['summary']
                f.write(f"Number of test runs: {summary['count']}\n")
                f.write(f"Profitable runs: {summary['profitable_pct']:.2f}%\n\n")

                f.write(
                    f"Sharpe Ratio: Mean={summary['sharpe_ratio_mean']:.4f}, Std={summary['sharpe_ratio_std']:.4f}\n")
                f.write(
                    f"Profit/Loss: Mean=${summary['profit_loss_mean']:.2f}, Std=${summary['profit_loss_std']:.2f}\n")
                f.write(f"Win Rate: Mean={summary['win_rate_mean']:.2f}%, Std={summary['win_rate_std']:.2f}%\n")
                f.write(
                    f"Max Drawdown: Mean={summary['max_drawdown_mean']:.2f}%, Std={summary['max_drawdown_std']:.2f}%\n\n")

                f.write("STABILITY ASSESSMENT\n")
                f.write("-------------------\n")

                # Calculate coefficient of variation for key metrics
                sharpe_cv = summary['sharpe_ratio_std'] / summary['sharpe_ratio_mean'] if summary[
                                                                                              'sharpe_ratio_mean'] != 0 else float(
                    'inf')
                profit_cv = summary['profit_loss_std'] / abs(summary['profit_loss_mean']) if summary[
                                                                                                 'profit_loss_mean'] != 0 else float(
                    'inf')

                # Assess stability
                if sharpe_cv < 0.3 and profit_cv < 0.5 and summary['profitable_pct'] > 80:
                    stability = "HIGHLY STABLE"
                elif sharpe_cv < 0.5 and profit_cv < 1.0 and summary['profitable_pct'] > 70:
                    stability = "MODERATELY STABLE"
                elif sharpe_cv < 0.8 and summary['profitable_pct'] > 60:
                    stability = "SOMEWHAT STABLE"
                else:
                    stability = "UNSTABLE"

                f.write(f"Strategy Stability Assessment: {stability}\n")
                f.write(f"Sharpe Ratio Coefficient of Variation: {sharpe_cv:.4f}\n")
                f.write(f"Profit/Loss Coefficient of Variation: {profit_cv:.4f}\n\n")

                f.write(f"5th percentile Sharpe Ratio: {summary['sharpe_ratio_p5']:.4f}\n")
                f.write(f"5th percentile Profit/Loss: ${summary['profit_loss_p5']:.2f}\n\n")

                f.write("RECOMMENDATION\n")
                f.write("-------------\n")
                if stability == "HIGHLY STABLE" or stability == "MODERATELY STABLE":
                    f.write("The optimized parameters show good stability across multiple runs.\n")
                    f.write("RECOMMENDATION: Proceed with implementing these parameters in live trading.\n")
                elif stability == "SOMEWHAT STABLE":
                    f.write("The optimized parameters show moderate stability, but with some variability.\n")
                    f.write("RECOMMENDATION: Consider using these parameters with reduced position sizing initially.\n")
                else:
                    f.write("The optimized parameters show high variability across different runs.\n")
                    f.write("RECOMMENDATION: Further refinement needed before live implementation.\n")

                f.write("\n\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            logger.info(f"Workflow complete. Summary saved to {main_output_dir}/workflow_summary.txt")
        else:
            logger.error("Robustness testing failed.")
    else:
        logger.error("No optimized parameters available for robustness testing.")