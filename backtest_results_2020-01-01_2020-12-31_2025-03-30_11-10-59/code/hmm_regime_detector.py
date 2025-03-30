"""
hmm_regime_detector.py - Market Regime Detection using Hidden Markov Models

This module implements an HMM-based approach to market regime detection,
which can identify different market states (trend-following, mean-reverting, 
neutral) based on observable market indicators.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HMMRegimeDetector:
    """
    A class to detect market regimes using Hidden Markov Models.
    This allows for probabilistic detection of market states based on 
    observable indicators.
    """

    def __init__(self, n_states=3, lookback_days=30, retrain_frequency=7,
                 min_samples=200, output_dir=None):
        """
        Initialize the HMM Regime Detector with improved defaults.

        Args:
            n_states: Number of hidden states (regimes) in the model
            lookback_days: Number of days to use for training (increased from 5 to 30)
            retrain_frequency: Days between model retraining
            min_samples: Minimum samples required for training
            output_dir: Directory to save model files and visualizations
        """
        # Increase lookback days for more robust training
        self.n_states = n_states
        self.lookback_days = lookback_days  # Increased from default
        self.retrain_frequency = retrain_frequency
        self.min_samples = min_samples
        
        # State mapping (default values, can be adjusted after fitting)
        self.state_mapping = {
            0: 'trend_following',
            1: 'mean_reverting',
            2: 'neutral'
        }
        
        # Set up output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Create a default directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"hmm_regime_detector_{timestamp}"
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Initialize model and history
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.last_training_date = None
        self.prediction_history = []
        self.confidence_history = []
        
        # Initialize metrics for regime detection quality
        self.metrics = {
            'log_likelihood': [],
            'state_frequencies': {},
            'transition_counts': {},
            'prediction_changes': 0
        }
        
        logger.info(f"Initialized HMM Regime Detector with {n_states} states")

    # In hmm_regime_detector.py, modify the _extract_features method to include more differentiating features:

    def _extract_features(self, df):
        """
        Extract features for the HMM model from market data with enhanced differentiation.

        Args:
            df: DataFrame with market data (price, indicators)

        Returns:
            Feature matrix X with shape (n_samples, n_features)
        """
        # Select core features that relate to market regimes
        features = [
            'ADX',  # Trend strength
            'MA_slope',  # Trend direction and steepness
            'atr_ratio',  # Volatility relative to history
            'RSI'  # Momentum/Overbought-Oversold
        ]

        # ADDED: Enhanced feature engineering for better state differentiation
        if 'MA' in df.columns and 'close' in df.columns:
            # Distance from moving average (normalized)
            df['price_to_ma'] = (df['close'] - df['MA']) / df['close'] * 100
            features.append('price_to_ma')

        if 'std_dev' in df.columns and 'close' in df.columns:
            # Volatility as percentage of price
            df['relative_volatility'] = df['std_dev'] / df['close'] * 100
            features.append('relative_volatility')

        # ADDED: Calculate Bollinger Band width as volatility indicator
        if all(col in df.columns for col in ['upper_band', 'lower_band', 'close']):
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['close'] * 100
            features.append('bb_width')

        # ADDED: RSI rate of change for momentum shift detection
        if 'RSI' in df.columns:
            df['rsi_change'] = df['RSI'].diff(3)  # 3-bar RSI change
            features.append('rsi_change')

        # ADDED: Extreme price movement detector
        if 'close' in df.columns:
            # Calculate 5-day returns
            df['returns_5d'] = df['close'].pct_change(5) * 100
            features.append('returns_5d')

        if 'volatility_regime' in df.columns:
            features.append('volatility_regime')

        # Store feature names for consistency
        self.feature_names = features

        # Drop rows with missing feature values
        feature_df = df[features].copy()
        feature_df = feature_df.dropna()

        # Log warning if too many rows are dropped
        if len(feature_df) < len(df) * 0.8:
            logger.warning(
                f"Lost {len(df) - len(feature_df)} rows ({(1 - len(feature_df) / len(df)) * 100:.1f}%) due to missing values")

        return feature_df

    def fit(self, df, current_date=None):
        """
        Fit the HMM model on historical data.

        Args:
            df: DataFrame with price and indicator data
            current_date: Current date for training window

        Returns:
            True if fitting was successful, False otherwise
        """
        # Use latest date if not specified
        if current_date is None:
            current_date = df['date'].max()

        # Calculate training window
        start_date = current_date - timedelta(days=self.lookback_days)
        train_df = df[(df['date'] >= start_date) & (df['date'] <= current_date)].copy()

        # Extract features
        features_df = self._extract_features(train_df)

        if len(features_df) < self.min_samples:
            logger.warning(
                f"Insufficient data for training: {len(features_df)} < {self.min_samples}. Need {self.min_samples - len(features_df)} more samples.")
            return False

        # Scale features for better HMM performance
        X = features_df.values

        # Initialize and fit HMM with early stopping on convergence issues
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # First fit the scaler
                X_scaled = self.scaler.fit_transform(X)

                # Adjust parameters based on attempt
                n_iter = 500 * (attempt + 1)  # Increase iterations on each attempt
                tol = 1e-6 / (10 ** attempt)  # Tighten tolerance on each attempt

                logger.info(f"HMM fitting attempt {attempt + 1}/{max_attempts} with {n_iter} iterations, tol={tol}")

                # Import global config
                from config import config

                # Get global seed if using fixed seed, otherwise use attempt-specific seed
                random_seed = config['global']['random_seed'] if config['global']['use_fixed_seed'] else \
                    config['global']['random_seed'] + attempt

                # Using Gaussian HMM for continuous observations
                self.model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=random_seed,  # Use global seed for reproducibility
                    tol=tol,
                    verbose=True
                )

                logger.info(f"HMM fitting with random_seed={random_seed}")

                # Fit model
                self.model.fit(X_scaled)
                self.last_training_date = current_date.date() if hasattr(current_date, 'date') else current_date

                # If we reached this point, fitting was successful
                logger.info(f"HMM fitting successful on attempt {attempt + 1}")

                # Save model
                self._save_model()

                # Analyze the fitted model to understand regime characteristics
                self._analyze_fitted_model(X_scaled, features_df)

                logger.info(f"Successfully fitted HMM with {self.n_states} states on {len(X_scaled)} samples")
                return True

            except Exception as e:
                logger.warning(f"HMM fitting attempt {attempt + 1} failed: {e}")

        # If we've tried all attempts, log error and return False
        logger.error(f"All HMM fitting attempts failed.")
        return False
    
    def _analyze_fitted_model(self, X_scaled, features_df):
        """
        Analyze the fitted model to understand regime characteristics.
        
        Args:
            X_scaled: Scaled feature matrix
            features_df: Original feature DataFrame
        """
        if self.model is None:
            return
            
        # Predict hidden states
        hidden_states = self.model.predict(X_scaled)
        
        # Calculate state frequencies
        state_count = np.bincount(hidden_states, minlength=self.n_states)
        state_freq = state_count / len(hidden_states)
        
        self.metrics['state_frequencies'] = {
            f"state_{i}": freq for i, freq in enumerate(state_freq)
        }
        
        # Calculate log likelihood as model fit quality
        log_likelihood = self.model.score(X_scaled)
        self.metrics['log_likelihood'].append(log_likelihood)
        
        # Add decoded states to features
        features_with_states = features_df.copy()
        features_with_states['predicted_state'] = hidden_states
        
        # Calculate feature means for each state to understand regime characteristics
        state_means = {}
        for state in range(self.n_states):
            state_filter = features_with_states['predicted_state'] == state
            if state_filter.any():
                state_data = features_with_states[state_filter]
                state_means[state] = {
                    col: state_data[col].mean() for col in self.feature_names
                }
        
        # Automatically map states to regimes based on feature characteristics
        self._map_states_to_regimes(state_means)
        
        # Count transitions between states
        transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(1, len(hidden_states)):
            prev_state = hidden_states[i-1]
            curr_state = hidden_states[i]
            transition_matrix[prev_state, curr_state] += 1
            
        # Normalize to get transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        empirical_transitions = np.divide(transition_matrix, row_sums, 
                                          out=np.zeros_like(transition_matrix), 
                                          where=row_sums!=0)
        
        # Save for comparison with model's transition matrix
        self.metrics['empirical_transitions'] = empirical_transitions
        self.metrics['model_transitions'] = self.model.transmat_
        
        # Generate visualization
        self._visualize_regimes(features_with_states)

    # In hmm_regime_detector.py, locate the _map_states_to_regimes method and replace it with this:

    def _map_states_to_regimes(self, state_means):
        """
        Map HMM states to market regimes based on feature characteristics.
        Ensures a more balanced distribution of regimes.

        Args:
            state_means: Dictionary with mean feature values for each state
        """
        if not state_means:
            return

        # Count samples in each state to analyze distribution
        state_counts = {}
        total_samples = 0
        for state in state_means:
            count = self.metrics.get('state_frequencies', {}).get(f'state_{state}', 0)
            # Convert from frequency to count if needed
            if count <= 1.0:  # If it's stored as frequency
                count = count * 1000  # Approximate count
            state_counts[state] = count
            total_samples += count

        logger.info(f"State distribution before balancing: {state_counts}")

        # Check for severely imbalanced states (any state > 50% of samples)
        imbalanced = any(count / total_samples > 0.5 for count in state_counts.values())

        # Enhanced feature importance weights for better state differentiation
        feature_weights = {
            'ADX': 2.0,  # Increase importance of trend strength
            'MA_slope': 3.0,  # Make slope more important for regime detection
            'RSI': 1.5,  # Increase momentum importance
            'atr_ratio': 2.0,  # Increase volatility importance
            'price_to_ma': 2.0  # Increase deviation importance
        }

        # Calculate weighted feature scores for more distinct regime characteristics
        state_scores = {}
        for state, means in state_means.items():
            state_scores[state] = {
                'trend_score': (
                        means.get('ADX', 0) * feature_weights.get('ADX', 1.0) +
                        abs(means.get('MA_slope', 0)) * feature_weights.get('MA_slope', 1.0)
                ),
                'mean_reversion_score': (
                        (1 / (means.get('ADX', 50) + 1)) * 50 * feature_weights.get('ADX', 1.0) +
                        means.get('atr_ratio', 0) * feature_weights.get('atr_ratio', 1.0) +
                        abs(means.get('price_to_ma', 0)) * feature_weights.get('price_to_ma', 1.0)
                ),
                'neutral_score': (
                        (50 - abs(means.get('RSI', 50) - 50)) * feature_weights.get('RSI', 1.0) +
                        (1 - abs(means.get('MA_slope', 0))) * 10 * feature_weights.get('MA_slope', 1.0)
                )
            }

        # Define trend following characteristics
        tf_characteristics = {
            'ADX': {'min': 25.0},
            'MA_slope_abs': {'min': 0.2},
            'RSI_extreme': True
        }

        # Define mean reversion characteristics
        mr_characteristics = {
            'ADX': {'max': 20.0},
            'MA_slope_abs': {'max': 0.15},
            'volatility_regime': {'min': 1.2}
        }

        # Define neutral characteristics
        neutral_characteristics = {
            'ADX': {'min': 15.0, 'max': 30.0},
            'MA_slope_abs': {'max': 0.2},
            'RSI_middle': True
        }

        # Determine the natural regime for each state
        natural_regime = {}
        for state, scores in state_scores.items():
            if scores['trend_score'] > scores['mean_reversion_score'] and scores['trend_score'] > scores[
                'neutral_score']:
                natural_regime[state] = 'trend_following'
            elif scores['mean_reversion_score'] > scores['trend_score'] and scores['mean_reversion_score'] > scores[
                'neutral_score']:
                natural_regime[state] = 'mean_reverting'
            else:
                natural_regime[state] = 'neutral'

        # Initial regime mapping - will be potentially overridden below
        state_to_regime = {}

        # Handle imbalanced states by forced redistribution when needed
        if imbalanced:
            logger.warning(f"Detected severely imbalanced state distribution. Enforcing balanced regime mapping.")

            # Sort states by sample count, descending
            sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)

            # Find dominant state (the one with most samples)
            dominant_state, _ = sorted_states[0]

            # Get all regimes we need to assign
            all_regimes = {'trend_following', 'mean_reverting', 'neutral'}

            # Start with non-dominant states, assigning them to their best natural regime
            assigned_regimes = set()
            for state, count in sorted_states[1:]:  # Skip dominant state
                best_regime = natural_regime[state]
                state_to_regime[state] = best_regime
                assigned_regimes.add(best_regime)

            # Determine which regimes are missing
            missing_regimes = all_regimes - assigned_regimes

            # If all regimes are already covered (unlikely with imbalanced data), use natural regime for dominant
            if not missing_regimes:
                state_to_regime[dominant_state] = natural_regime[dominant_state]
            else:
                # Assign the dominant state to the first missing regime
                # This ensures all regime types are represented
                state_to_regime[dominant_state] = next(iter(missing_regimes))

            logger.info(f"Balanced state mapping from severe imbalance detection: {state_to_regime}")
        else:
            # If not severely imbalanced, start with natural mapping
            state_to_regime = natural_regime.copy()
            logger.info(f"Using natural regime mapping: {state_to_regime}")

        # Additional balancing based on RSI characteristics for moderate imbalances
        # This runs regardless of whether the severe imbalance condition was triggered
        if any(count / total_samples > 0.4 for count in state_counts.values()):
            logger.info("Applying RSI-based regime balancing for moderate imbalance")

            # Sort states by their mean RSI value (proxy for market condition)
            rsi_values = {state: means.get('RSI', 50) for state, means in state_means.items()}

            # Assign trend following to the state with highest RSI (bullish momentum)
            trend_state = max(rsi_values.items(), key=lambda x: x[1])[0]

            # Assign mean reverting to the state with lowest RSI (oversold conditions)
            mr_state = min(rsi_values.items(), key=lambda x: x[1])[0]

            # Assign neutral to remaining state or state with RSI closest to 50
            remaining_states = set(state_means.keys()) - {trend_state, mr_state}
            if remaining_states:
                neutral_state = list(remaining_states)[0]
            else:
                # If we only have 2 states, choose the one closer to neutral RSI
                rsi_diff = {state: abs(rsi - 50) for state, rsi in rsi_values.items()}
                neutral_state = min(rsi_diff.items(), key=lambda x: x[1])[0]

            # Create forced mapping
            forced_mapping = {
                trend_state: 'trend_following',
                mr_state: 'mean_reverting',
                neutral_state: 'neutral'
            }

            # Handle case with < 3 states
            state_to_regime = {state: forced_mapping.get(state, 'neutral')
                               for state in state_means.keys()}

            logger.info(f"Forced balanced regime mapping based on RSI characteristics: {state_to_regime}")

        # Update the mapping
        self.state_mapping = state_to_regime.copy()

        # Log the mapping with detailed feature values
        logger.info("HMM State to Market Regime mapping:")
        for state, regime in self.state_mapping.items():
            if state in state_means:
                # Calculate frequency percentage for logging
                frequency = state_counts.get(state, 0) / total_samples * 100 if total_samples > 0 else 0
                features = ", ".join([f"{k}: {v:.2f}" for k, v in state_means[state].items()])
                logger.info(f"State {state} -> {regime.upper()} (freq: {frequency:.1f}%, {features})")

    def _visualize_regimes(self, features_with_states):
        """
        Generate visualizations of the detected regimes.

        Args:
            features_with_states: DataFrame with features and predicted states
        """
        if 'predicted_state' not in features_with_states.columns:
            return

        try:
            # Create directory for visualizations
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)

            # Plot state distribution
            plt.figure(figsize=(10, 6))
            state_counts = features_with_states['predicted_state'].value_counts().sort_index()
            state_labels = [f"{i}: {self.state_mapping.get(i, 'Unknown')}" for i in state_counts.index]
            plt.bar(state_labels, state_counts.values)
            plt.title('HMM State Distribution')
            plt.xlabel('State')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(viz_dir, 'state_distribution.png'), dpi=100)
            plt.close()

            # Plot feature distributions by state (with variance check)
            for feature in self.feature_names:
                # Skip plotting if feature has too little variance
                if features_with_states[feature].var() < 1e-6:
                    logger.warning(f"Skipping density plot for feature '{feature}' due to near-zero variance")
                    continue

                plt.figure(figsize=(12, 6))
                for state in range(self.n_states):
                    state_data = features_with_states[features_with_states['predicted_state'] == state][feature]
                    if len(state_data) > 0 and state_data.var() > 1e-6:  # Check variance again for state subset
                        sns.kdeplot(state_data, label=f"State {state}: {self.state_mapping.get(state, 'Unknown')}")
                    else:
                        # Draw a vertical line at the mean instead of a density plot
                        if len(state_data) > 0:
                            mean_value = state_data.mean()
                            plt.axvline(x=mean_value, color=f'C{state}',
                                        linestyle='--',
                                        label=f"State {state}: {self.state_mapping.get(state, 'Unknown')} (mean={mean_value:.2f})")

                plt.title(f'{feature} Distribution by State')
                plt.xlabel(feature)
                plt.ylabel('Density')
                plt.legend()
                plt.savefig(os.path.join(viz_dir, f'{feature}_by_state.png'), dpi=100)
                plt.close()

            # Plot transition diagram if we have enough data
            if hasattr(self, 'metrics') and 'model_transitions' in self.metrics:
                self._plot_transition_diagram(self.metrics['model_transitions'], viz_dir)

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_transition_diagram(self, transition_matrix, viz_dir):
        """
        Plot the transition diagram between states.
        
        Args:
            transition_matrix: Matrix of transition probabilities
            viz_dir: Directory to save visualization
        """
        try:
            plt.figure(figsize=(8, 8))
            
            # Create a directed graph visualization
            cmap = plt.cm.Blues
            plt.imshow(transition_matrix, cmap=cmap, vmin=0, vmax=1)
            
            # Add labels
            state_labels = [f"{i}: {self.state_mapping.get(i, 'Unknown')}" for i in range(self.n_states)]
            plt.xticks(range(self.n_states), state_labels, rotation=45)
            plt.yticks(range(self.n_states), state_labels)
            
            # Add colorbar
            plt.colorbar(label='Transition Probability')
            
            # Add text annotations
            for i in range(self.n_states):
                for j in range(self.n_states):
                    plt.text(j, i, f"{transition_matrix[i, j]:.2f}", 
                             ha="center", va="center", 
                             color="white" if transition_matrix[i, j] > 0.5 else "black")
            
            plt.title('State Transition Probabilities')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'state_transitions.png'), dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting transition diagram: {e}")

    def _save_model(self):
        """Save the trained model and metadata."""
        if self.model is None:
            return

        try:
            # Create model directory
            model_dir = os.path.join(self.output_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"hmm_regime_model_{timestamp}.pkl")

            # Save all necessary objects for later loading
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'state_mapping': self.state_mapping,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'n_states': self.n_states,
                'last_training_date': self.last_training_date
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Save summary statistics with error handling
            with open(os.path.join(model_dir, f"model_summary_{timestamp}.txt"), 'w') as f:
                f.write(f"HMM REGIME MODEL SUMMARY\n")
                f.write(f"=======================\n\n")
                f.write(f"Number of states: {self.n_states}\n")
                f.write(f"Training date: {self.last_training_date}\n")
                f.write(f"Feature names: {', '.join(self.feature_names)}\n\n")

                f.write(f"STATE MAPPING\n")
                f.write(f"------------\n")
                for state, regime in self.state_mapping.items():
                    f.write(f"State {state} -> {regime.upper()}\n")

                f.write(f"\nSTATE FREQUENCIES\n")
                f.write(f"----------------\n")
                for state_key, freq in self.metrics.get('state_frequencies', {}).items():
                    try:
                        mapped_state = int(state_key.split('_')[1])
                        regime = self.state_mapping.get(mapped_state, 'Unknown')
                        f.write(f"{state_key} ({regime}): {freq:.2%}\n")
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Could not parse state key: {state_key}. Error: {e}")
                        f.write(f"{state_key}: {freq:.2%}\n")

                f.write(f"\nMODEL PARAMETERS\n")
                f.write(f"---------------\n")
                if 'log_likelihood' in self.metrics and len(self.metrics['log_likelihood']) > 0:
                    f.write(f"Log likelihood: {self.metrics['log_likelihood'][-1]}\n")
                else:
                    f.write(f"Log likelihood: N/A\n")

                if hasattr(self.model, 'transmat_'):
                    f.write(f"\nTransition matrix:\n")
                    for i in range(self.n_states):
                        line = " ".join([f"{p:.3f}" for p in self.model.transmat_[i]])
                        f.write(f"  {line}\n")

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
            self.state_mapping = model_data['state_mapping']
            self.feature_names = model_data['feature_names']
            self.metrics = model_data.get('metrics', {})
            self.n_states = model_data.get('n_states', self.n_states)
            self.last_training_date = model_data.get('last_training_date')
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_regime(self, df, current_date=None):
        """Predict the current market regime with deterministic results."""
        from config import config

        # Reset random state before prediction
        np.random.seed(config['global']['random_seed'])
        """
        Predict the current market regime.

        Args:
            df: DataFrame with market data
            current_date: Date for prediction window (uses latest date if None)

        Returns:
            Dictionary with prediction results and confidence
        """
        if self.model is None:
            logger.warning("Model not trained. Cannot predict regime.")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': True}

        # Use latest date if not specified
        if current_date is None:
            current_date = df['date'].max()

        # Use recent data for prediction (last 20 days)
        window_size = 20  # days
        start_date = current_date - timedelta(days=window_size)
        pred_df = df[(df['date'] >= start_date) & (df['date'] <= current_date)].copy()

        # Extract features for prediction
        features_df = self._extract_features(pred_df)

        if len(features_df) == 0:
            logger.warning("No valid data for prediction.")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': False}

        try:
            # Scale features
            X = features_df.values
            X_scaled = self.scaler.transform(X)

            # Find most recent state
            hidden_states = self.model.predict(X_scaled)
            current_state = hidden_states[-1]

            # Map state to regime
            current_regime = self.state_mapping.get(current_state, 'neutral')

            # Calculate prediction confidence
            # Option 1: Use state posterior probabilities
            if hasattr(self.model, 'predict_proba'):
                state_probs = self.model.predict_proba(X_scaled)
                current_confidence = state_probs[-1, current_state]
                # ADDED: Use a more nuanced confidence formula that considers state balance
                if current_confidence > 0.95 and self.metrics.get('state_frequencies', {}).get(f'state_{current_state}',
                                                                                               0) > 0.7:
                    # High confidence but extremely common state - reduce confidence
                    current_confidence = current_confidence * 0.8
                elif current_confidence < 0.6:
                    # Low natural confidence - blend with baseline
                    current_confidence = 0.5 + (current_confidence - 0.5) * 0.8
            else:
                # Option 2: Use model score as proxy for confidence
                # Scale log-likelihood to 0-1 range (rough approximation)
                log_likelihood = self.model.score(X_scaled[-10:])  # Use last 10 points
                baseline = -10.0  # typical bad log-likelihood value
                max_ll = -1.0  # typical good log-likelihood value
                current_confidence = min(1.0, max(0.0, (log_likelihood - baseline) / (max_ll - baseline)))

            # Check if we should retrain
            needs_training = False
            if self.last_training_date is not None:
                current_date_conv = current_date.date() if hasattr(current_date, 'date') else current_date
                days_since_training = (current_date_conv - self.last_training_date).days
                if days_since_training >= self.retrain_frequency:
                    needs_training = True

            # Store prediction history
            self.prediction_history.append({
                'date': current_date,
                'state': current_state,
                'regime': current_regime,
                'confidence': current_confidence
            })

            # Return prediction result
            return {
                'regime': current_regime,
                'state': int(current_state),
                'confidence': float(current_confidence),
                'needs_training': needs_training,
                'features': {name: features_df.iloc[-1][name] for name in self.feature_names}
            }

        except (NotFittedError, ValueError, AttributeError) as e:
            # Handle the case when scaler or model is not properly fitted
            logger.warning(f"Prediction failed due to model fitting issue: {e}")
            return {'regime': 'neutral', 'confidence': 0.0, 'needs_training': True}
    
    def check_retrain_needed(self, current_date):
        """
        Check if model retraining is needed based on time or prediction quality.
        
        Args:
            current_date: Current date to check against
            
        Returns:
            True if retraining needed, False otherwise
        """
        if self.model is None:
            return True
            
        # Check time-based retraining
        if self.last_training_date is not None:
            current_date_conv = current_date.date() if hasattr(current_date, 'date') else current_date
            days_since_training = (current_date_conv - self.last_training_date).days
            if days_since_training >= self.retrain_frequency:
                logger.info(f"Time-based retraining needed. Days since last training: {days_since_training}")
                return True
        else:
            return True
            
        # TODO: Add prediction quality checks (e.g. regime flipping too frequently)
        
        return False
    
    def get_regime_params(self, detected_regime, confidence, config):
        """
        Get the appropriate parameters for the detected regime.
        
        Args:
            detected_regime: The detected market regime
            confidence: Confidence in the regime detection
            config: Configuration dictionary with market parameters
            
        Returns:
            Dictionary of parameters for the detected regime
        """
        # Validate regime
        if detected_regime not in ['trend_following', 'mean_reverting', 'neutral']:
            logger.warning(f"Unknown regime {detected_regime}. Falling back to neutral.")
            detected_regime = 'neutral'
            
        # Get base parameters from config
        if detected_regime in config['market_type']:
            params = config['market_type'][detected_regime].copy()
        else:
            # Default to neutral if specific regime not found
            params = config['market_type']['neutral'].copy()
            
        # Add regime type to parameters
        params['market_type'] = detected_regime
        
        # Adjust parameters based on confidence
        # If confidence is low, blend with neutral parameters
        if confidence < 0.7:
            # Get neutral parameters
            neutral_params = config['market_type']['neutral'].copy()
            
            # Blend parameters
            blend_weight = confidence
            for key in params:
                if key in neutral_params and isinstance(params[key], (int, float)):
                    params[key] = params[key] * blend_weight + neutral_params[key] * (1 - blend_weight)
            
            # Add confidence info
            params['confidence'] = confidence
            params['blended'] = True
            
        return params
        
    def get_best_market_params(self, df, current_date, config):
        """
        Get the best market parameters based on HMM regime detection.
        
        Args:
            df: DataFrame with market data
            current_date: Current date for detection window
            config: Configuration dictionary with market parameters
            
        Returns:
            Tuple of (market_type, parameters, detection_metrics)
        """
        # Check if we need to train or retrain
        if self.model is None or self.check_retrain_needed(current_date):
            logger.info("Training/retraining HMM model...")
            self.fit(df, current_date)
            
        # Predict regime
        prediction = self.predict_regime(df, current_date)
        
        # Get parameters for detected regime
        regime = prediction['regime']
        confidence = prediction['confidence']
        params = self.get_regime_params(regime, confidence, config)
        
        # Add metrics for logging
        metrics = {
            'hmm_state': prediction['state'],
            'confidence': confidence,
            'features': prediction.get('features', {})
        }
        
        return regime, params, metrics
