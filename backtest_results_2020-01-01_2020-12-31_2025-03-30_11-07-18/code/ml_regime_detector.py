"""
ml_regime_detector.py - Machine Learning Enhanced Market Regime Detection

This module builds on the HMM-based regime detection by adding machine learning
capability to predict regime shifts before they occur.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import HMM detector for generating labels
from hmm_regime_detector import HMMRegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLRegimeDetector:
    """
    Machine learning extension to HMM-based regime detection.
    This class uses HMM to generate regime labels, then trains ML models
    to predict future regime changes.
    """
    
    def __init__(self, output_dir=None, model_type='xgboost', prediction_horizon=5, 
                 hmm_detector=None, feature_lookback=20):
        """
        Initialize the ML Regime Detector.
        
        Args:
            output_dir: Directory to save model files and results
            model_type: Type of ML model to use ('xgboost', 'random_forest')
            prediction_horizon: Days ahead to predict regime
            hmm_detector: Existing HMM detector to use (if None, creates a new one)
            feature_lookback: Number of days of historical data to use for features
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.feature_lookback = feature_lookback
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"ml_regime_detector_{timestamp}"
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize or use provided HMM detector
        if hmm_detector is None:
            self.hmm_detector = HMMRegimeDetector(
                n_states=3,
                lookback_days=30,
                retrain_frequency=7,
                min_samples=200,
                output_dir=self.output_dir
            )
        else:
            self.hmm_detector = hmm_detector
            
        # Initialize ML model and other properties
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.last_training_date = None
        self.performance_metrics = {}
        self.feature_columns = None
        
        logger.info(f"Initialized ML Regime Detector with {model_type} model")
        
    def _create_model(self):
        """Create and configure the ML model based on selected type."""
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model
    
    def _generate_regime_labels(self, df):
        """
        Use HMM detector to generate regime labels for the dataset.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            DataFrame with added regime labels
        """
        # Ensure HMM detector is trained
        if self.hmm_detector.model is None:
            logger.info("Training HMM detector to generate labels")
            self.hmm_detector.fit(df)
        
        # Create a copy of the dataframe for adding labels
        labeled_df = df.copy()
        
        # Add regime labels for each date
        regime_labels = []
        for i, row in df.iterrows():
            prediction = self.hmm_detector.predict_regime(df, row['date'])
            regime_labels.append(prediction['regime'])
            
        labeled_df['regime'] = regime_labels
        
        # Encode regimes as integers for ML
        regime_map = {
            'trend_following': 0,
            'mean_reverting': 1,
            'neutral': 2
        }
        labeled_df['regime_code'] = labeled_df['regime'].map(regime_map)
        
        # Also add HMM confidence if available
        if 'confidence' in prediction:
            labeled_df['hmm_confidence'] = prediction['confidence']
            
        return labeled_df
        
    def _extract_features(self, df):
        """
        Extract features for ML model from price data.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            DataFrame of features for ML training
        """
        # Standard technical indicators are already calculated in df
        
        # Calculate additional features that might help predict regime changes
        features_df = df.copy()
        
        # 1. Volatility features
        features_df['volatility_ratio'] = features_df['ATR'] / features_df['ATR'].rolling(self.feature_lookback).mean()
        features_df['volatility_change'] = features_df['ATR'].pct_change(5)
        
        # 2. Momentum features
        features_df['rsi_5d_change'] = features_df['RSI'] - features_df['RSI'].shift(5)
        features_df['price_distance_from_ma'] = (features_df['close'] - features_df['MA']) / features_df['MA'] * 100
        
        # 3. Trend strength features
        features_df['adx_change'] = features_df['ADX'] - features_df['ADX'].shift(5)
        
        # 4. Correlation features (if multiple assets are available)
        # This would require additional data
        
        # 5. Bollinger Band features
        features_df['bb_width'] = (features_df['upper_band'] - features_df['lower_band']) / features_df['middle_band']
        features_df['bb_position'] = (features_df['close'] - features_df['lower_band']) / (features_df['upper_band'] - features_df['lower_band'])
        
        # 6. Volume features
        features_df['volume_ratio'] = features_df['volume'] / features_df['avg_volume']
        features_df['volume_trend'] = features_df['volume'].pct_change(5)
        
        # Create lagged features for lookback
        for i in range(1, 6):
            features_df[f'close_lag_{i}'] = features_df['close'].pct_change(i)
            features_df[f'rsi_lag_{i}'] = features_df['RSI'].shift(i)
            features_df[f'atr_lag_{i}'] = features_df['ATR'].shift(i)
            features_df[f'adx_lag_{i}'] = features_df['ADX'].shift(i)
            
        # 7. Target variable with forward-looking regime
        # Shift the regime_code back by prediction_horizon days to predict future regimes
        features_df['future_regime'] = features_df['regime_code'].shift(-self.prediction_horizon)
        
        # Drop rows with NaN values (beginning and end of the dataset)
        features_df = features_df.dropna()
        
        return features_df
    
    def train_model(self, df, test_size=0.2, use_hmm_confidence=True):
        """
        Train the ML model to predict future regimes.
        
        Args:
            df: DataFrame with price and indicator data
            test_size: Portion of data to use for testing
            use_hmm_confidence: Whether to use HMM confidence as a feature
            
        Returns:
            Dictionary with training results and metrics
        """
        # Generate regime labels with HMM
        labeled_df = self._generate_regime_labels(df)
        
        # Extract features
        features_df = self._extract_features(labeled_df)
        
        # Select feature columns (excluding target and date)
        exclude_cols = ['date', 'future_regime', 'regime', 'regime_code']
        
        # If not using HMM confidence, exclude it
        if not use_hmm_confidence and 'hmm_confidence' in features_df.columns:
            exclude_cols.append('hmm_confidence')
            
        # Get feature columns
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        # Prepare X and y for training
        X = features_df[feature_cols]
        y = features_df['future_regime']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and train model
        self.model = self._create_model()
        
        # Train the model
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            
            # Create feature importance plot
            plt.figure(figsize=(10, 8))
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=importance_df[:15])
            plt.title(f'Top 15 Feature Importance - {self.model_type}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Trend', 'Mean Rev', 'Neutral'],
                    yticklabels=['Trend', 'Mean Rev', 'Neutral'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save the model
        self._save_model()
        self.last_training_date = datetime.now().date()
        
        # Store and return performance metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'training_date': self.last_training_date.strftime("%Y-%m-%d"),
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'feature_count': len(feature_cols),
            'top_features': list(importance_df['Feature'].head(10))
        }
        
        logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
        return self.performance_metrics
    
    def predict_future_regime(self, df, current_date=None):
        """
        Predict the future regime based on current market data.
        
        Args:
            df: DataFrame with price and indicator data
            current_date: Date to make prediction for (uses latest date if None)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot make predictions.")
            return {'regime': 'neutral', 'probability': {}, 'confidence': 0.0}
            
        # Use latest date if not specified
        if current_date is None:
            current_date = df['date'].max()
            
        # Generate features including regime labels
        labeled_df = self._generate_regime_labels(df)
        features_df = self._extract_features(labeled_df)
        
        # Find the row for the current date
        date_mask = features_df['date'] == current_date
        if not date_mask.any():
            logger.warning(f"Date {current_date} not found in dataset")
            return {'regime': 'neutral', 'probability': {}, 'confidence': 0.0}
            
        # Get current features
        current_features = features_df[date_mask][self.feature_columns]
        if len(current_features) == 0:
            logger.warning("No valid features for prediction")
            return {'regime': 'neutral', 'probability': {}, 'confidence': 0.0}
            
        # Scale features
        X_scaled = self.scaler.transform(current_features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map regime indices to names
        regime_map = {
            0: 'trend_following',
            1: 'mean_reverting',
            2: 'neutral'
        }
        
        # Find predicted regime (highest probability)
        predicted_regime_idx = np.argmax(probabilities)
        predicted_regime = regime_map[predicted_regime_idx]
        confidence = probabilities[predicted_regime_idx]
        
        # Create probability dictionary
        prob_dict = {regime_map[i]: probabilities[i] for i in range(len(probabilities))}
        
        # Return formatted prediction
        prediction = {
            'regime': predicted_regime,
            'probability': prob_dict,
            'confidence': float(confidence),
            'prediction_date': (current_date + timedelta(days=self.prediction_horizon)).strftime("%Y-%m-%d"),
            'current_date': current_date.strftime("%Y-%m-%d")
        }
        
        return prediction
    
    def _save_model(self):
        """Save the trained model and associated data."""
        if self.model is None:
            return
            
        try:
            # Create models directory
            model_dir = os.path.join(self.output_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"ml_regime_model_{timestamp}.pkl")
            
            # Save the model and associated data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'model_type': self.model_type,
                'prediction_horizon': self.prediction_horizon,
                'performance_metrics': self.performance_metrics,
                'last_training_date': self.last_training_date
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def load_model(self, model_path):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data['feature_importance']
            self.model_type = model_data['model_type']
            self.prediction_horizon = model_data['prediction_horizon']
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.last_training_date = model_data.get('last_training_date')
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def evaluate_predictions(self, df, start_date=None, end_date=None):
        """
        Evaluate prediction accuracy over a time period.
        
        Args:
            df: DataFrame with price and indicator data
            start_date: Start date for evaluation (uses earliest date if None)
            end_date: End date for evaluation (uses latest date if None)
            
        Returns:
            DataFrame with predictions and actual outcomes
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot evaluate predictions.")
            return pd.DataFrame()
            
        # Generate features with regime labels
        labeled_df = self._generate_regime_labels(df)
        features_df = self._extract_features(labeled_df)
        
        # Filter by date range if specified
        if start_date is not None:
            features_df = features_df[features_df['date'] >= start_date]
        if end_date is not None:
            features_df = features_df[features_df['date'] <= end_date]
            
        # Prepare results storage
        results = []
        
        # Loop through each date and make predictions
        for i, row in features_df.iterrows():
            # Skip rows with NaN future_regime (end of dataset)
            if pd.isna(row['future_regime']):
                continue
                
            # Get current features
            current_features = features_df.loc[[i], self.feature_columns]
            X_scaled = self.scaler.transform(current_features)
            
            # Get prediction
            probabilities = self.model.predict_proba(X_scaled)[0]
            predicted_regime_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_regime_idx]
            
            # Map regime indices to names
            regime_map = {
                0: 'trend_following',
                1: 'mean_reverting',
                2: 'neutral'
            }
            
            # Get actual future regime
            future_date = row['date'] + timedelta(days=self.prediction_horizon)
            actual_regime_idx = int(row['future_regime'])
            
            # Add to results
            results.append({
                'date': row['date'],
                'future_date': future_date,
                'predicted_regime_idx': predicted_regime_idx,
                'predicted_regime': regime_map[predicted_regime_idx],
                'actual_regime_idx': actual_regime_idx,
                'actual_regime': regime_map[actual_regime_idx],
                'confidence': confidence,
                'correct': predicted_regime_idx == actual_regime_idx
            })
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate rolling accuracy
        if len(results_df) > 0:
            results_df['rolling_accuracy'] = results_df['correct'].rolling(20).mean()
            
            # Create accuracy plot
            plt.figure(figsize=(12, 6))
            plt.plot(results_df['date'], results_df['rolling_accuracy'])
            plt.title(f'Rolling 20-Day Prediction Accuracy - {self.prediction_horizon} Day Horizon')
            plt.xlabel('Date')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'rolling_accuracy.png'))
            plt.close()
            
            # Log overall accuracy
            accuracy = results_df['correct'].mean()
            logger.info(f"Overall prediction accuracy: {accuracy:.4f}")
            
        return results_df
