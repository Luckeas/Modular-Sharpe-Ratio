"""
HMM Regime Detector - Market regime detection using Hidden Markov Models.

This module provides a concrete implementation of the RegimeDetector that uses
Hidden Markov Models to identify market regimes based on observable features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pickle
import os

from market_regimes.regime_detector import RegimeDetector

# Configure logging
logger = logging.getLogger(__name__)

class HMMRegimeDetector(RegimeDetector):
    """
    Market regime detector using Hidden Markov Models.
    
    This class uses HMM to identify market regimes (trend-following, 
    mean-reverting, neutral) based on observable market features.
    """
    
    def __init__(self, name: str = "HMMRegimeDetector", config: Optional[Dict] = None):
        """
        Initialize the HMM regime detector.
        
        Args:
            name: Name of the detector
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # HMM parameters
        self.n_states = config.get('n_states', 3)
        self.lookback_days = config.get('lookback_days', 30)
        self.retrain_frequency = config.get('retrain_frequency', 7)
        self.min_samples = config.get('min_samples', 200)
        
        # Default state mapping
        self.state_mapping = {
            0: 'trend_following',
            1: 'mean_reverting',
            2: 'neutral'
        }
        
        # HMM model and components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.last_training_date = None
        self.save_dir = config.get('save_dir', 'models/hmm')
        
        logger.info(f"HMMRegimeDetector initialized with {self.n_states} states")
    
    def _initialize_detector(self) -> None:
        """Initialize detector-specific components."""
        # Create save directory if it doesn't exist
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Try to load a pre-trained model if available
        if self.save_dir:
            self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """
        Try to load a pre-trained model.
        
        Returns:
            True if a model was loaded, False otherwise
        """
        model_files = self._get_model_files()
        
        if not model_files:
            return False
        
        # Get the most recent model
        latest_model = model_files[-1]
        
        try:
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler', StandardScaler())
            self.state_mapping = model_data.get('state_mapping', self.state_mapping)
            self.feature_names = model_data.get('feature_names')
            self.last_training_date = model_data.get('last_training_date')
            
            logger.info(f"Loaded pre-trained HMM model from {latest_model}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model from {latest_model}: {str(e)}")
            return False
    
    def _get_model_files(self) -> List[str]:
        """
        Get list of available model files sorted by timestamp.
        
        Returns:
            List of model file paths
        """
        if not self.save_dir or not os.path.exists(self.save_dir):
            return []
        
        # Get all model files
        model_files = [
            os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir)
            if f.startswith('hmm_model_') and f.endswith('.pkl')
        ]
        
        # Sort by timestamp
        model_files.sort()
        
        return model_files
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for the HMM model.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with extracted features
        """
        # Define features to extract
        features = [
            'ADX',       # Trend strength
            'MA_slope',  # Trend direction
            'atr_ratio', # Volatility ratio
            'RSI'        # Momentum
        ]
        
        # Calculate additional features if they don't exist
        if 'MA' in df.columns and 'close' in df.columns and 'price_to_ma' not in df.columns:
            # Distance from moving average (normalized)
            df['price_to_ma'] = (df['close'] - df['MA']) / df['close'] * 100
            features.append('price_to_ma')
        
        if all(col in df.columns for col in ['upper_band', 'lower_band', 'close']) and 'bb_width' not in df.columns:
            # Bollinger Band width as volatility indicator
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['close'] * 100
            features.append('bb_width')
        
        if 'RSI' in df.columns and 'rsi_change' not in df.columns:
            # RSI rate of change for momentum shift detection
            df['rsi_change'] = df['RSI'].diff(3)  # 3-bar RSI change
            features.append('rsi_change')
        
        # Make sure all features exist in the DataFrame
        features = [f for f in features if f in df.columns]
        
        # Store feature names
        self.feature_names = features
        
        # Extract features
        feature_df = df[features].copy()
        
        # Drop NaN values
        feature_df = feature_df.dropna()
        
        return feature_df
    
    def fit(self, df: pd.DataFrame, current_date: Optional[datetime] = None) -> bool:
        """
        Fit the HMM model.
        
        Args:
            df: DataFrame with market data
            current_date: Current date for training window
            
        Returns:
            True if fitting was successful, False otherwise
        """
        # Use latest date if not specified
        if current_date is None:
            current_date = df['date'].max()
        
        # Calculate training window
        days_offset = timedelta(days=self.lookback_days)
        start_date = current_date - days_offset
        
        # Extract training data
        if isinstance(current_date, datetime) and isinstance(df['date'].iloc[0], datetime):
            train_df = df[(df['date'] >= start_date) & (df['date'] <= current_date)].copy()
        else:
            # If dates are not datetime objects, use index-based selection
            current_idx = df.index[df['date'] == current_date][0] if current_date in df['date'].values else len(df) - 1
            lookback_idx = max(0, current_idx - self.lookback_days)
            train_df = df.iloc[lookback_idx:current_idx+1].copy()
        
        # Extract features
        features_df = self._extract_features(train_df)
        
        if len(features_df) < self.min_samples:
            logger.warning(f"Insufficient data for training: {len(features_df)} < {self.min_samples}")
            return False
        
        # Fit the model
        try:
            # Scale features
            X = features_df.values
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
                tol=1e-5,
                verbose=False
            )
            
            # Fit the model
            self.model.fit(X_scaled)
            
            # Update last training date
            self.last_training_date = current_date
            
            # Map states to regimes
            self._map_states_to_regimes(X_scaled, features_df)
            
            # Save the model
            self._save_model()
            
            logger.info(f"Successfully trained HMM with {self.n_states} states on {len(X_scaled)} samples")
            return True
        
        except Exception as e:
            logger.error(f"Error training HMM model: {str(e)}")
            return False
    
    def _map_states_to_regimes(self, X_scaled: np.ndarray, features_df: pd.DataFrame) -> None:
        """
        Map HMM states to market regimes based on feature characteristics.
        
        Args:
            X_scaled: Scaled feature matrix
            features_df: Original feature DataFrame
        """
        if self.model is None:
            return
        
        # Predict hidden states
        hidden_states = self.model.predict(X_scaled)
        
        # Add states to feature DataFrame
        features_with_states = features_df.copy()
        features_with_states['state'] = hidden_states
        
        # Calculate feature means for each state
        state_means = {}
        for state in range(self.n_states):
            state_filter = features_with_states['state'] == state
            if state_filter.any():
                state_data = features_with_states[state_filter]
                state_means[state] = {
                    feature: state_data[feature].mean() 
                    for feature in self.feature_names
                }
        
        # Define scoring system for regime classification
        state_scores = {}
        for state, means in state_means.items():
            # Trend following score
            trend_score = 0
            if 'ADX' in means:
                # Higher ADX indicates stronger trend
                trend_score += means['ADX'] * 0.5
            if 'MA_slope' in means:
                # Absolute slope indicates trend strength
                trend_score += abs(means['MA_slope']) * 20
            
            # Mean reversion score
            mean_rev_score = 0
            if 'RSI' in means:
                # Extreme RSI values favor mean reversion
                rsi_dev = abs(means['RSI'] - 50)
                mean_rev_score += rsi_dev * 0.5
            if 'bb_width' in means:
                # Wider Bollinger Bands favor mean reversion
                mean_rev_score += means['bb_width'] * 0.2
            
            # Neutral score (inverted from the other scores)
            neutral_score = 100 - (trend_score + mean_rev_score) / 2
            
            state_scores[state] = {
                'trend_score': trend_score,
                'mean_rev_score': mean_rev_score,
                'neutral_score': neutral_score
            }
        
        # Assign regimes based on highest score
        new_mapping = {}
        for state, scores in state_scores.items():
            max_score = max(scores.values())
            if max_score == scores['trend_score']:
                new_mapping[state] = 'trend_following'
            elif max_score == scores['mean_rev_score']:
                new_mapping[state] = 'mean_reverting'
            else:
                new_mapping[state] = 'neutral'
        
        # Update state mapping
        self.state_mapping = new_mapping
        
        # Log the mapping
        logger.info("HMM State to Market Regime mapping:")
        for state, regime in self.state_mapping.items():
            if state in state_means:
                features_str = ", ".join([f"{k}: {v:.2f}" for k, v in state_means[state].items()])
                logger.info(f"State {state} -> {regime.upper()} ({features_str})")
    
    def _save_model(self) -> None:
        """Save the trained model."""
        if self.model is None or not self.save_dir:
            return
        
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_dir, f"hmm_model_{timestamp}.pkl")
            
            # Save model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'state_mapping': self.state_mapping,
                'feature_names': self.feature_names,
                'last_training_date': self.last_training_date
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def check_retrain_needed(self, current_date):
        """
        Check if model retraining is needed.
        
        Args:
            current_date: Current date
            
        Returns:
            True if retraining is needed, False otherwise
        """
        if self.model is None:
            return True
        
        if self.last_training_date is None:
            return True
        
        # Calculate days elapsed
        if isinstance(current_date, datetime) and isinstance(self.last_training_date, datetime):
            days_elapsed = (current_date - self.last_training_date).days
        else:
            # If dates are not datetime objects, assume retraining is needed
            return True
        
        return days_elapsed >= self.retrain_frequency
    
    def detect_regime(self, bar: pd.Series, bar_index: int) -> Dict:
        """
        Detect the current market regime using HMM.
        
        Args:
            bar: Current bar data
            bar_index: Index of the current bar
            
        Returns:
            Dictionary with regime information
        """
        # Check if model needs to be trained
        if self.model is None:
            logger.warning("HMM model not trained yet")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': True}
        
        # Extract recent data for prediction
        lookback = 20  # Use recent data for prediction
        start_idx = max(0, bar_index - lookback)
        pred_df = self.data.iloc[start_idx:bar_index+1].copy()
        
        # Extract features
        features_df = self._extract_features(pred_df)
        
        if len(features_df) == 0:
            logger.warning("No valid features for prediction")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': False}
        
        try:
            # Scale features
            X = features_df.values
            X_scaled = self.scaler.transform(X)
            
            # Get state probabilities
            state_probs = None
            if hasattr(self.model, 'predict_proba'):
                state_probs = self.model.predict_proba(X_scaled)
            
            # Predict current state
            hidden_states = self.model.predict(X_scaled)
            current_state = hidden_states[-1]
            
            # Map state to regime
            current_regime = self.state_mapping.get(current_state, 'neutral')
            
            # Calculate prediction confidence
            if state_probs is not None:
                current_confidence = state_probs[-1, current_state]
            else:
                # Use model score as proxy for confidence
                log_likelihood = self.model.score(X_scaled[-10:])
                baseline = -10.0  # typical bad log-likelihood value
                max_ll = -1.0  # typical good log-likelihood value
                current_confidence = min(1.0, max(0.0, (log_likelihood - baseline) / (max_ll - baseline)))
            
            # Extract feature values for metrics
            feature_metrics = {name: features_df[name].iloc[-1] for name in self.feature_names}
            
            logger.info(f"Detected regime: {current_regime}, confidence: {current_confidence:.2f}")
            
            return {
                'regime': current_regime,
                'confidence': float(current_confidence),
                'state': int(current_state),
                'metrics': feature_metrics,
                'needs_training': self.check_retrain_needed(bar['date'] if 'date' in bar else bar_index)
            }
        
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': True}
