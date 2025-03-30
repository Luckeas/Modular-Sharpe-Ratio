"""
simplified_ml_predictor.py - Streamlined ML module for trade prediction

This module provides a simplified machine learning functionality to enhance 
trade selection by predicting the probability of successful trades.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Machine learning predictor for enhancing mean reversion trade selection.
    """
    def __init__(self, output_dir, model_type='xgboost', prediction_threshold=0.6, 
                 retrain_frequency_days=30, min_training_samples=200):
        """
        Initialize the ML predictor.
        
        Args:
            output_dir: Directory to save model files and logs
            model_type: Type of model to use ('xgboost', 'random_forest')
            prediction_threshold: Threshold for prediction probability
            retrain_frequency_days: How often to retrain the model (days)
            min_training_samples: Minimum samples required for training
        """
        self.output_dir = output_dir
        self.model_type = model_type
        self.prediction_threshold = prediction_threshold
        self.retrain_frequency_days = retrain_frequency_days
        self.min_training_samples = min_training_samples
        
        # Create model directory
        self.model_dir = os.path.join(output_dir, 'ml_models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Initialize model properties
        self.model = None
        self.last_training_date = None
        self.feature_names = None
        self.feature_importance = None
        
        # Create performance tracking
        self.performance_metrics = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

    def _create_model(self):
        """Create ML model based on the selected type."""
        # Import global config for random seed
        from config import config

        # Get global random seed
        random_seed = config['global']['random_seed'] if config['global']['use_fixed_seed'] else 42

        if self.model_type == 'random_forest':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=random_seed,  # Use global seed
                    class_weight='balanced'
                ))
            ])
        else:  # Default to XGBoost
            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=100,
                min_child_weight=5,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=random_seed,  # Use global seed
                scale_pos_weight=1.5  # Adjust for class imbalance
            )

        logger.info(f"Created {self.model_type} model with random_seed={random_seed}")
        return model

    def train_model(self, X, y):
        """
        Train the machine learning model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            True if training successful, False otherwise
        """
        if X is None or y is None or len(X) < self.min_training_samples:
            logger.warning(f"Insufficient training data: {0 if X is None else len(X)} samples")
            return False

        logger.info(f"Training model with {len(X)} samples")

        # Create and train model
        self.model = self._create_model()

        # Import global config
        from config import config
        random_seed = config['global']['random_seed']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'steps') and hasattr(self.model.steps[-1][1], 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.steps[-1][1].feature_importances_))

        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(self.model_dir, f"trade_classifier_{timestamp}.pkl")

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save feature importance if available
            if self.feature_importance:
                importance_df = pd.DataFrame({
                    'Feature': list(self.feature_importance.keys()),
                    'Importance': list(self.feature_importance.values())
                }).sort_values('Importance', ascending=False)

                importance_path = os.path.join(self.model_dir, f"feature_importance_{timestamp}.csv")
                importance_df.to_csv(importance_path, index=False)
        except Exception as e:
            logger.error(f"Error saving model: {e}")

        # Update last training date
        self.last_training_date = datetime.now().date()

        return True
    
    def extract_features(self, df, index, trade_type):
        """
        Extract ML features for a potential trade.
        
        Args:
            df: DataFrame with price and indicator data
            index: Index of the potential entry bar
            trade_type: 'long' or 'short'
            
        Returns:
            Dictionary of features
        """
        if index < 20:  # Need enough history
            return None
            
        # Get current and previous bar data
        current = df.iloc[index]
        prev = df.iloc[index-1]  # Previous bar (signal bar)
        
        # Skip if missing key data
        required_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'middle_band', 'upper_band', 'lower_band', 'RSI', 'ATR']
        
        for col in required_cols:
            if col not in df.columns or pd.isna(current[col]) or pd.isna(prev[col]):
                return None
        
        # Calculate core features
        features = {
            # Price action features
            'bb_deviation': (prev['close'] - prev['middle_band']) / prev['std_dev'] if 'std_dev' in prev and prev['std_dev'] > 0 else 0,
            'rsi': prev['RSI'],
            'rsi_change': prev['RSI'] - df.iloc[index-2]['RSI'] if index > 1 else 0,
            'close_pct_change': (prev['close'] / df.iloc[index-2]['close'] - 1) * 100 if index > 1 else 0,
            
            # Volume features
            'volume_ratio': prev['volume'] / prev['avg_volume'] if 'avg_volume' in prev and prev['avg_volume'] > 0 else 1,
            
            # Volatility features
            'atr': prev['ATR'],
            'atr_pct': prev['ATR'] / prev['close'] * 100 if prev['close'] > 0 else 0,
            
            # Regime features
            'adx': prev['ADX'] if 'ADX' in prev else 0,
            'ma_slope': prev['MA_slope'] if 'MA_slope' in prev else 0,
            'volatility_regime': prev['volatility_regime'] if 'volatility_regime' in prev else 1,
            
            # Trade-specific features
            'trade_type': 1 if trade_type == 'long' else 0
        }
        
        # Add Bollinger Band specific features
        if trade_type == 'long':
            features['bb_penetration'] = (prev['lower_band'] - prev['low']) / prev['ATR'] if prev['ATR'] > 0 else 0
            features['bb_target_distance'] = (prev['middle_band'] - prev['close']) / prev['ATR'] if prev['ATR'] > 0 else 0
        else:  # Short
            features['bb_penetration'] = (prev['high'] - prev['upper_band']) / prev['ATR'] if prev['ATR'] > 0 else 0
            features['bb_target_distance'] = (prev['close'] - prev['middle_band']) / prev['ATR'] if prev['ATR'] > 0 else 0
            
        return features

    def generate_training_data(self, df, trade_logs):
        """
        Generate training data from historical trades.

        Args:
            df: DataFrame with price data and indicators
            trade_logs: List of executed trades with outcomes

        Returns:
            X: Feature matrix
            y: Target labels (1 for profitable trades, 0 for losing trades)
        """
        features_list = []
        outcomes = []
        failed_extractions = 0
        missing_entry_idx = 0

        # ADDED: Debug number of trades
        logger.info(f"[DEBUG] Generating training data from {len(trade_logs)} trades")

        for i, trade in enumerate(trade_logs):
            # ADDED: Debug sample trade format for first trade
            if i == 0:
                logger.info(f"[DEBUG] Sample trade format: {list(trade.keys())}")
                logger.info(f"[DEBUG] Entry time: {trade['entry_time']}, Type: {trade['type']}")

            # Find index of entry bar
            entry_time = trade['entry_time']
            entry_mask = df['date'] == entry_time

            if not entry_mask.any():
                missing_entry_idx += 1
                # ADDED: Debug entry time issues
                if missing_entry_idx <= 3:  # Just log first few
                    logger.warning(f"[DEBUG] Could not find entry time {entry_time} in dataframe")
                continue

            entry_idx = df[entry_mask].index[0]

            # Extract features
            features = self.extract_features(df, entry_idx, trade['type'])

            if features is None:
                failed_extractions += 1
                # ADDED: Debug feature extraction failures
                if failed_extractions <= 3:  # Just log first few
                    logger.warning(f"[DEBUG] Feature extraction failed for trade at {entry_time}")
                continue

            # Determine outcome (1 for profitable, 0 for losing)
            outcome = 1 if trade['profit'] > 0 else 0

            features_list.append(features)
            outcomes.append(outcome)

        # ADDED: Debug collection results
        logger.info(
            f"[DEBUG] Data generation results: {len(features_list)} valid samples, {failed_extractions} failed extractions, {missing_entry_idx} missing entry indices")

        # Convert to DataFrame and Series
        if len(features_list) == 0:
            logger.warning("[DEBUG] No valid training samples found")
            # ADDED: Check key columns in the dataframe
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'middle_band', 'upper_band', 'lower_band', 'RSI',
                             'ATR']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"[DEBUG] Missing required columns in dataframe: {missing_cols}")
            return None, None

        X = pd.DataFrame(features_list)
        y = pd.Series(outcomes)

        # ADDED: Debug feature information
        logger.info(f"[DEBUG] Generated features: {list(X.columns)}")
        logger.info(
            f"[DEBUG] Feature stats: min={X.min().min():.2f}, max={X.max().max():.2f}, mean={X.mean().mean():.2f}")
        logger.info(
            f"[DEBUG] Class distribution: positive={sum(y)}, negative={len(y) - sum(y)}, ratio={sum(y) / len(y):.2f}")

        # Save feature names
        self.feature_names = list(X.columns)

        return X, y
    
    def train_model(self, X, y):
        """
        Train the machine learning model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            True if training successful, False otherwise
        """
        if X is None or y is None or len(X) < self.min_training_samples:
            logger.warning(f"Insufficient training data: {0 if X is None else len(X)} samples")
            return False
        
        logger.info(f"Training model with {len(X)} samples")
        
        # Create and train model
        self.model = self._create_model()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'steps') and hasattr(self.model.steps[-1][1], 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.steps[-1][1].feature_importances_))
        
        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(self.model_dir, f"trade_classifier_{timestamp}.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save feature importance if available
            if self.feature_importance:
                importance_df = pd.DataFrame({
                    'Feature': list(self.feature_importance.keys()),
                    'Importance': list(self.feature_importance.values())
                }).sort_values('Importance', ascending=False)
                
                importance_path = os.path.join(self.model_dir, f"feature_importance_{timestamp}.csv")
                importance_df.to_csv(importance_path, index=False)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        # Update last training date
        self.last_training_date = datetime.now().date()
        
        return True
    
    def predict_trade_success(self, features):
        """
        Predict the probability of trade success.
        
        Args:
            features: Dictionary or DataFrame of features
            
        Returns:
            Tuple of (probability, take_trade)
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot make predictions.")
            return 0.5, False
            
        # Convert features to DataFrame if dictionary
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Ensure features match model's expected input
        if self.feature_names:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Select only the features used by the model
            features_df = features_df[self.feature_names]
        
        # Make prediction
        try:
            if hasattr(self.model, 'predict_proba'):
                # For models that output probability
                probability = self.model.predict_proba(features_df)[0][1]
            else:
                # For models without probability output
                probability = float(self.model.predict(features_df)[0])
                
            # Determine if trade should be taken
            take_trade = probability >= self.prediction_threshold
            
            return probability, take_trade
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5, False
    
    def check_retrain_needed(self, current_date):
        """
        Check if model retraining is needed based on time elapsed.
        
        Args:
            current_date: Current date to check against
            
        Returns:
            True if retraining needed, False otherwise
        """
        if self.last_training_date is None:
            return True
            
        days_elapsed = (current_date.date() - self.last_training_date).days
        return days_elapsed >= self.retrain_frequency_days
    
    def record_prediction_result(self, predicted_outcome, actual_outcome):
        """
        Record the accuracy of a prediction.
        
        Args:
            predicted_outcome: Whether the model predicted trade success (True/False)
            actual_outcome: Whether the trade was actually successful (True/False)
        """
        # Update confusion matrix counts
        if predicted_outcome and actual_outcome:
            self.performance_metrics['true_positives'] += 1
        elif predicted_outcome and not actual_outcome:
            self.performance_metrics['false_positives'] += 1
        elif not predicted_outcome and actual_outcome:
            self.performance_metrics['false_negatives'] += 1
        else:  # not predicted_outcome and not actual_outcome
            self.performance_metrics['true_negatives'] += 1
    
    def get_performance_metrics(self):
        """Get current performance metrics for the model."""
        metrics = self.performance_metrics.copy()
        
        # Calculate derived metrics
        total = sum(metrics.values())
        
        if total > 0:
            # Accuracy
            correct = metrics['true_positives'] + metrics['true_negatives']
            metrics['accuracy'] = correct / total
            
            # Precision
            if metrics['true_positives'] + metrics['false_positives'] > 0:
                metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            else:
                metrics['precision'] = 0
                
            # Recall
            if metrics['true_positives'] + metrics['false_negatives'] > 0:
                metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
            else:
                metrics['recall'] = 0
                
            # F1 Score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0
        else:
            metrics.update({
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            })
        
        return metrics
    
    def save_performance_report(self):
        """Generate and save performance report to file."""
        metrics = self.get_performance_metrics()
        
        report = []
        report.append("=" * 50)
        report.append("ML PREDICTOR PERFORMANCE REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 50)
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION")
        report.append("-" * 50)
        report.append(f"Model Type: {self.model_type}")
        report.append(f"Prediction Threshold: {self.prediction_threshold}")
        report.append(f"Last Training Date: {self.last_training_date}")
        report.append("")
        
        # Performance metrics
        report.append("PREDICTION PERFORMANCE")
        report.append("-" * 50)
        report.append(f"Total Predictions: {sum(metrics.values())}")
        report.append(f"True Positives: {metrics['true_positives']}")
        report.append(f"True Negatives: {metrics['true_negatives']}")
        report.append(f"False Positives: {metrics['false_positives']}")
        report.append(f"False Negatives: {metrics['false_negatives']}")
        report.append("")
        report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        report.append(f"Precision: {metrics.get('precision', 0):.4f}")
        report.append(f"Recall: {metrics.get('recall', 0):.4f}")
        report.append(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        report.append("")
        
        # Feature importance
        if self.feature_importance:
            report.append("TOP FEATURE IMPORTANCE")
            report.append("-" * 50)
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                report.append(f"{feature}: {importance:.4f}")
        
        # Write to file
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, f"ml_performance_report.txt")
        
        try:
            with open(report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            
        return report_text