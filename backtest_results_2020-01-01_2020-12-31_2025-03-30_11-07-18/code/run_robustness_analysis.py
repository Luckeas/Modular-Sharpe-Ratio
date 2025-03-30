#!/usr/bin/env python
"""
run_robustness_analysis.py - Script to automate robustness testing and analysis

This script runs a strategy through robustness testing with multiple seeds, analyzes
the results, and generates comprehensive reports about the strategy's performance
characteristics.
"""

import os
import json
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robustness_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("robustness_analysis")

def run_robustness_test(param_file, num_runs=50, ml_enabled=False, output_dir=None):
    """
    Run robustness testing with multiple random seeds.
    
    Args:
        param_file: Path to parameter file (JSON)
        num_runs: Number of runs with different seeds
        ml_enabled: Whether to enable ML enhancement
        output_dir: Directory to save results
        
    Returns:
        Path to the output directory
    """
    logger.info(f"Starting robustness testing with {num_runs} runs")
    
    # Create command
    cmd = [
        "python", "test_sharpe_robustness.py",
        "--params", param_file,
        "--runs", str(num_runs)
    ]
    
    # Add ML flag if enabled
    if ml_enabled:
        cmd.append("--ml")
    
    # Add output directory if specified
    if output_dir:
        cmd.append("--output")
        cmd.append(output_dir)
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Error running robustness test: {stderr}")
            return None
        
        # Check for output directory in the output
        if output_dir:
            return output_dir
        else:
            # Find the output directory from the log
            return None  # Would need to parse the log to find this
            
    except Exception as e:
        logger.error(f"Error executing robustness test: {e}")
        return None

def run_loss_analysis(input_dir, output_dir=None):
    """
    Run loss analysis on robustness test results.
    
    Args:
        input_dir: Directory with robustness test results
        output_dir: Directory to save loss analysis
        
    Returns:
        Path to loss analysis directory
    """
    logger.info(f"Running loss analysis on {input_dir}")
    
    # Create command
    cmd = [
        "python", "analyze_losses.py",
        "--input", input_dir
    ]
    
    # Add output directory if specified
    if output_dir:
        cmd.append("--output")
        cmd.append(output_dir)
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Error running loss analysis: {stderr}")
            return None
        
        # Return output directory
        return output_dir
            
    except Exception as e:
        logger.error(f"Error executing loss analysis: {e}")
        return None

def run_profit_analysis(input_dir, percentile=25, output_dir=None):
    """
    Run profit analysis on robustness test results.
    
    Args:
        input_dir: Directory with robustness test results
        percentile: Top percentile to analyze
        output_dir: Directory to save profit analysis
        
    Returns:
        Path to profit analysis directory
    """
    logger.info(f"Running profit analysis on {input_dir} (top {percentile}%)")
    
    # Create command
    cmd = [
        "python", "analyze_profits.py",
        "--input", input_dir,
        "--percentile", str(percentile)
    ]
    
    # Add output directory if specified
    if output_dir:
        cmd.append("--output")
        cmd.append(output_dir)
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Error running profit analysis: {stderr}")
            return None
        
        # Return output directory
        return output_dir
            
    except Exception as e:
        logger.error(f"Error executing profit analysis: {e}")
        return None

def create_combined_summary(robustness_dir, loss_dir, profit_dir, output_dir):
    """
    Create a comprehensive summary report combining all analysis results.
    
    Args:
        robustness_dir: Directory with robustness test results
        loss_dir: Directory with loss analysis results
        profit_dir: Directory with profit analysis results
        output_dir: Directory to save the combined summary
        
    Returns:
        Path to the summary report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    summary_path = os.path.join(output_dir, "combined_analysis_summary.md")
    logger.info(f"Creating combined summary report: {summary_path}")
    
    # Load robustness results
    robustness_results = None
    robustness_summary = None
    try:
        robustness_results_path = os.path.join(robustness_dir, "robustness_results.csv")
        if os.path.exists(robustness_results_path):
            robustness_results = pd.read_csv(robustness_results_path)
        
        robustness_summary_path = os.path.join(robustness_dir, "robustness_summary.txt")
        if os.path.exists(robustness_summary_path):
            with open(robustness_summary_path, 'r') as f:
                robustness_summary = f.read()
    except Exception as e:
        logger.error(f"Error loading robustness results: {e}")
    
    # Load loss analysis
    loss_report = None
    try:
        loss_report_path = os.path.join(loss_dir, "loss_analysis_report.txt")
        if os.path.exists(loss_report_path):
            with open(loss_report_path, 'r') as f:
                loss_report = f.read()
    except Exception as e:
        logger.error(f"Error loading loss analysis report: {e}")
    
    # Load profit analysis
    profit_report = None
    try:
        profit_report_path = os.path.join(profit_dir, "profit_analysis_report.txt")
        if os.path.exists(profit_report_path):
            with open(profit_report_path, 'r') as f:
                profit_report = f.read()
    except Exception as e:
        logger.error(f"Error loading profit analysis report: {e}")
    
    # Create markdown report
    with open(summary_path, 'w') as f:
        f.write("# Comprehensive Strategy Analysis Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Add robustness statistics
        f.write("## 1. Robustness Analysis\n\n")
        
        if robustness_results is not None:
            # Calculate key statistics
            profitable_runs = (robustness_results['profit_loss'] > 0).sum()
            total_runs = len(robustness_results)
            profitable_pct = profitable_runs / total_runs * 100 if total_runs > 0 else 0
            
            f.write(f"* **Number of runs**: {total_runs}\n")
            f.write(f"* **Profitable runs**: {profitable_runs} ({profitable_pct:.2f}%)\n")
            f.write(f"* **Average profit**: ${robustness_results['profit_loss'].mean():.2f}\n")
            f.write(f"* **Median profit**: ${robustness_results['profit_loss'].median():.2f}\n")
            f.write(f"* **Standard deviation**: ${robustness_results['profit_loss'].std():.2f}\n")
            f.write(f"* **Average Sharpe ratio**: {robustness_results['sharpe_ratio'].mean():.4f}\n")
            f.write(f"* **Average win rate**: {robustness_results['win_rate'].mean():.2f}%\n")
            f.write(f"* **Average max drawdown**: {robustness_results['max_drawdown_pct'].mean():.2f}%\n\n")
            
            # Calculate coefficient of variation
            profit_cv = robustness_results['profit_loss'].std() / abs(robustness_results['profit_loss'].mean()) if robustness_results['profit_loss'].mean() != 0 else float('inf')
            sharpe_cv = robustness_results['sharpe_ratio'].std() / abs(robustness_results['sharpe_ratio'].mean()) if robustness_results['sharpe_ratio'].mean() != 0 else float('inf')
            
            f.write(f"* **Profit coefficient of variation**: {profit_cv:.4f}\n")
            f.write(f"* **Sharpe ratio coefficient of variation**: {sharpe_cv:.4f}\n\n")
            
            # Calculate 5th and 95th percentiles
            p05_profit = robustness_results['profit_loss'].quantile(0.05)
            p95_profit = robustness_results['profit_loss'].quantile(0.95)
            
            f.write(f"* **5th percentile profit**: ${p05_profit:.2f}\n")
            f.write(f"* **95th percentile profit**: ${p95_profit:.2f}\n\n")
            
            # Stability assessment
            if profit_cv < 0.3 and profitable_pct > 90:
                stability = "HIGHLY STABLE"
            elif profit_cv < 0.6 and profitable_pct > 80:
                stability = "MODERATELY STABLE"
            elif profit_cv < 1.0 and profitable_pct > 70:
                stability = "SOMEWHAT STABLE"
            else:
                stability = "UNSTABLE"
                
            f.write(f"* **Stability assessment**: {stability}\n\n")
            
            # Add plots
            plots_dir = os.path.join(output_dir, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            # Create profit distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(robustness_results['profit_loss'], bins=20, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Distribution of Profit/Loss Across All Runs')
            plt.xlabel('Profit/Loss ($)')
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(plots_dir, "profit_distribution.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            f.write(f"![Profit Distribution](plots/profit_distribution.png)\n\n")
            
            # Create scatterplot of Sharpe vs Profit
            plt.figure(figsize=(10, 6))
            plt.scatter(robustness_results['sharpe_ratio'], robustness_results['profit_loss'])
            plt.axhline(y=0, color='r', linestyle='--')
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Sharpe Ratio vs. Profit/Loss')
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Profit/Loss ($)')
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(plots_dir, "sharpe_vs_profit.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            f.write(f"![Sharpe vs Profit](plots/sharpe_vs_profit.png)\n\n")
        
        elif robustness_summary is not None:
            # If we couldn't load results but have summary, include that
            f.write("```\n")
            f.write(robustness_summary)
            f.write("\n```\n\n")
        
        else:
            f.write("*No robustness results available.*\n\n")
        
        # Add loss analysis
        f.write("## 2. Loss Analysis\n\n")
        
        if loss_report is not None:
            # Extract key sections
            sections = loss_report.split('\n\n')
            
            # Add overall stats
            overall_section = next((s for s in sections if s.startswith("OVERALL STATISTICS")), None)
            if overall_section:
                f.write("### Overall Loss Statistics\n\n")
                f.write("```\n")
                f.write(overall_section)
                f.write("\n```\n\n")
            
            # Add worst loss streaks
            loss_streaks = next((s for s in sections if s.startswith("WORST LOSS STREAKS")), None)
            if loss_streaks:
                f.write("### Worst Loss Streaks\n\n")
                f.write("```\n")
                f.write(loss_streaks)
                f.write("\n```\n\n")
            
            # Add recommendations
            recommendations = next((s for s in sections if s.startswith("RECOMMENDATIONS")), None)
            if recommendations:
                f.write("### Loss Analysis Recommendations\n\n")
                f.write("```\n")
                f.write(recommendations)
                f.write("\n```\n\n")
        
        else:
            f.write("*No loss analysis results available.*\n\n")
        
        # Add profit analysis
        f.write("## 3. Profit Analysis\n\n")
        
        if profit_report is not None:
            # Extract key sections
            sections = profit_report.split('\n\n')
            
            # Add overall stats
            overall_section = next((s for s in sections if s.startswith("OVERALL STATISTICS")), None)
            if overall_section:
                f.write("### Overall Profit Statistics\n\n")
                f.write("```\n")
                f.write(overall_section)
                f.write("\n```\n\n")
            
            # Add best profit streaks
            profit_streaks = next((s for s in sections if s.startswith("BEST PROFIT STREAKS")), None)
            if profit_streaks:
                f.write("### Best Profit Streaks\n\n")
                f.write("```\n")
                f.write(profit_streaks)
                f.write("\n```\n\n")
            
            # Add success factors
            success_factors = next((s for s in sections if s.startswith("SUCCESS FACTORS")), None)
            if success_factors:
                f.write("### Success Factors\n\n")
                f.write("```\n")
                f.write(success_factors)
                f.write("\n```\n\n")
        
        else:
            f.write("*No profit analysis results available.*\n\n")
        
        # Final recommendations
        f.write("## 4. Integrated Recommendations\n\n")
        
        # Combine recommendations from both analyses
        recommendations = []
        
        if loss_report is not None:
            loss_recs = extract_recommendations(loss_report)
            if loss_recs:
                recommendations.append("### Loss Prevention Recommendations\n")
                recommendations.extend(loss_recs)
                recommendations.append("")
        
        if profit_report is not None:
            profit_recs = extract_recommendations(profit_report)
            if profit_recs:
                recommendations.append("### Profit Enhancement Recommendations\n")
                recommendations.extend(profit_recs)
                recommendations.append("")
        
        if recommendations:
            f.write("\n".join(recommendations))
        else:
            f.write("*No integrated recommendations available.*\n\n")
        
        # Stability assessment and conclusion
        f.write("## 5. Conclusion\n\n")
        
        if robustness_results is not None:
            if stability == "HIGHLY STABLE":
                f.write("The strategy shows **high stability** across multiple runs. It is suitable for live trading with standard position sizing.\n\n")
            elif stability == "MODERATELY STABLE":
                f.write("The strategy shows **moderate stability**. It can be used for live trading with reduced position sizing and careful monitoring.\n\n")
            elif stability == "SOMEWHAT STABLE":
                f.write("The strategy shows **some stability** but has significant variability. If used for live trading, use substantially reduced position sizing and implement additional safeguards.\n\n")
            else:
                f.write("The strategy is **unstable** with high variability across runs. Further optimization and simplification is recommended before live implementation.\n\n")
        
        f.write("**Next Steps:**\n\n")
        f.write("1. Implement the recommended improvements from both loss and profit analysis\n")
        f.write("2. Simplify the strategy to reduce parameter sensitivity\n")
        f.write("3. Add runtime performance monitoring and drawdown protection\n")
        f.write("4. Run another round of robustness testing after improvements\n")
        f.write("5. Consider paper trading for 2-4 weeks before live implementation\n")
    
    logger.info(f"Combined summary report created at {summary_path}")
    return summary_path

def extract_recommendations(report_text):
    """Extract recommendation items from a report."""
    lines = report_text.split('\n')
    
    # Find the recommendations section
    rec_start = next((i for i, line in enumerate(lines) 
                     if line.startswith("RECOMMENDATIONS") or 
                     line.startswith("SUCCESS FACTORS")), -1)
    
    if rec_start == -1:
        return []
    
    # Extract bullet points (lines starting with "-" or "*")
    recs = []
    for i in range(rec_start, len(lines)):
        line = lines[i].strip()
        if line.startswith('-') or line.startswith('*'):
            recs.append(line)
        elif len(recs) > 0 and line == '':
            # Empty line after recommendations, stop here
            break
    
    return recs

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run automated robustness analysis")
    parser.add_argument('--params', type=str, required=True, help='Path to parameter file (JSON)')
    parser.add_argument('--runs', type=int, default=50, help='Number of runs')
    parser.add_argument('--ml', action='store_true', help='Enable ML enhancement')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"robustness_analysis_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the robustness test
    logger.info("Step 1: Running robustness testing")
    robustness_dir = os.path.join(output_dir, "robustness_results")
    robustness_dir = run_robustness_test(
        args.params, 
        num_runs=args.runs, 
        ml_enabled=args.ml, 
        output_dir=robustness_dir
    )
    
    if not robustness_dir or not os.path.exists(robustness_dir):
        logger.error("Robustness testing failed. Exiting.")
        return 1
    
    # Run loss analysis
    logger.info("Step 2: Running loss analysis")
    loss_dir = os.path.join(output_dir, "loss_analysis")
    loss_dir = run_loss_analysis(
        robustness_dir,
        output_dir=loss_dir
    )
    
    # Run profit analysis
    logger.info("Step 3: Running profit analysis")
    profit_dir = os.path.join(output_dir, "profit_analysis")
    profit_dir = run_profit_analysis(
        robustness_dir, 
        percentile=25,  # Analyze top 25%
        output_dir=profit_dir
    )
    
    # Create combined summary
    logger.info("Step 4: Creating combined analysis summary")
    summary_dir = os.path.join(output_dir, "summary")
    summary_path = create_combined_summary(
        robustness_dir, 
        loss_dir or os.path.join(output_dir, "loss_analysis"),
        profit_dir or os.path.join(output_dir, "profit_analysis"),
        summary_dir
    )
    
    logger.info(f"Analysis complete. Summary report: {summary_path}")
    logger.info(f"All results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
