"""
Market regime detection using Hidden Markov Models and statistical methods.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. HMM regime detection unavailable.")


class RegimeDetector:
    """Detects market regimes using statistical and ML methods."""
    
    def __init__(self, returns: pd.Series, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            returns: Market return series
            n_regimes: Number of regimes to detect
        """
        self.returns = returns
        self.n_regimes = n_regimes
        self.regimes = None
        self.regime_stats = None
        self.model = None
        
    def detect_regimes_hmm(self, n_iter: int = 100, 
                          covariance_type: str = 'full') -> pd.Series:
        """
        Detect regimes using Hidden Markov Model.
        
        Args:
            n_iter: Number of EM iterations
            covariance_type: 'spherical', 'diag', 'full', 'tied'
            
        Returns:
            Series with regime labels
        """
        if not HMM_AVAILABLE:
            print("HMM not available, falling back to simple method")
            return self.detect_regimes_simple()
        
        print("Fitting Hidden Markov Model...")
        
        # Prepare features: returns and realized volatility
        features = self._prepare_hmm_features()
        
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )
        
        try:
            model.fit(features)
            regime_sequence = model.predict(features)
            
            # Sort regimes by mean return (0=Bear, 1=Neutral, 2=Bull)
            regime_sequence = self._sort_regimes_by_return(regime_sequence, features[:, 0])
            
            self.regimes = pd.Series(regime_sequence, index=self.returns.index)
            self.model = model
            
            self._calculate_regime_statistics()
            self._calculate_transition_matrix()
            
            print("✓ HMM regime detection completed")
            return self.regimes
            
        except Exception as e:
            print(f"HMM fitting failed: {e}")
            print("Falling back to simple method")
            return self.detect_regimes_simple()
    
    def _prepare_hmm_features(self, vol_window: int = 21) -> np.ndarray:
        """
        Prepare features for HMM.
        
        Args:
            vol_window: Window for volatility calculation
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Feature 1: Returns
        returns_array = self.returns.values.reshape(-1, 1)
        
        # Feature 2: Realized volatility
        realized_vol = self.returns.rolling(vol_window).std().values.reshape(-1, 1)
        
        # Combine features
        features = np.hstack([returns_array, realized_vol])
        
        # Handle NaNs (replace with mean)
        nan_mask = np.isnan(features)
        if nan_mask.any():
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                features[nan_mask[:, i], i] = col_means[i]
        
        return features
    
    def _sort_regimes_by_return(self, regime_sequence: np.ndarray, 
                                returns: np.ndarray) -> np.ndarray:
        """
        Sort regimes by mean return: 0=Bear, 1=Neutral, 2=Bull.
        
        Args:
            regime_sequence: Original regime sequence
            returns: Return array
            
        Returns:
            Sorted regime sequence
        """
        # Calculate mean return for each regime
        regime_means = {}
        for regime in range(self.n_regimes):
            mask = regime_sequence == regime
            regime_means[regime] = returns[mask].mean()
        
        # Sort regimes by mean return
        sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
        
        # Create mapping: old_regime -> new_regime
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        sorted_sequence = np.array([regime_mapping[r] for r in regime_sequence])
        
        return sorted_sequence
    
    def detect_regimes_simple(self, window: int = 63) -> pd.Series:
        """
        Simple regime detection based on rolling statistics.
        
        Args:
            window: Rolling window (default: 3 months)
            
        Returns:
            Series with regime labels
        """
        print("Using simple regime detection...")
        
        # Calculate rolling statistics
        rolling_mean = self.returns.rolling(window).mean()
        rolling_vol = self.returns.rolling(window).std()
        
        # Z-score for mean
        mean_zscore = (rolling_mean - rolling_mean.mean()) / rolling_mean.std()
        
        # Classify regimes
        regimes = pd.Series(index=self.returns.index, dtype=int)
        regimes[mean_zscore > 0.5] = 2  # Bull
        regimes[mean_zscore < -0.5] = 0  # Bear
        regimes[(mean_zscore >= -0.5) & (mean_zscore <= 0.5)] = 1  # Neutral
        
        # Forward fill NaNs
        regimes = regimes.fillna(method='bfill').fillna(1)
        
        self.regimes = regimes
        self._calculate_regime_statistics()
        
        print("✓ Simple regime detection completed")
        return regimes
    
    def detect_regimes_volatility(self, vol_window: int = 21) -> pd.Series:
        """
        Volatility-based regime detection.
        
        Args:
            vol_window: Window for volatility calculation
            
        Returns:
            Series with regime labels
        """
        # Calculate realized volatility
        realized_vol = self.returns.rolling(vol_window).std() * np.sqrt(252)
        
        # Percentile thresholds
        low_vol = realized_vol.quantile(0.33)
        high_vol = realized_vol.quantile(0.67)
        
        # Classify regimes
        regimes = pd.Series(index=self.returns.index, dtype=int)
        regimes[realized_vol < low_vol] = 2  # Low vol (Bull)
        regimes[realized_vol > high_vol] = 0  # High vol (Bear)
        regimes[(realized_vol >= low_vol) & (realized_vol <= high_vol)] = 1  # Neutral
        
        regimes = regimes.fillna(method='bfill').fillna(1)
        
        self.regimes = regimes
        self._calculate_regime_statistics()
        
        return regimes
    
    def _calculate_regime_statistics(self):
        """Calculate comprehensive statistics for each regime."""
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            regime_mask = self.regimes == regime
            regime_returns = self.returns[regime_mask]
            
            if len(regime_returns) > 0:
                # Basic statistics
                mean_return = regime_returns.mean() * 252
                volatility = regime_returns.std() * np.sqrt(252)
                sharpe = mean_return / volatility if volatility > 0 else 0
                
                # Additional statistics
                skewness = regime_returns.skew()
                kurtosis = regime_returns.kurtosis()
                max_return = regime_returns.max()
                min_return = regime_returns.min()
                
                regime_stats[regime] = {
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'max_return': max_return,
                    'min_return': min_return,
                    'count': regime_mask.sum(),
                    'percentage': regime_mask.sum() / len(self.regimes) * 100
                }
            else:
                regime_stats[regime] = {
                    'mean_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'skewness': 0,
                    'kurtosis': 0,
                    'max_return': 0,
                    'min_return': 0,
                    'count': 0,
                    'percentage': 0
                }
        
        self.regime_stats = pd.DataFrame(regime_stats).T
        self.regime_stats.index = ['Bear', 'Neutral', 'Bull']
        self.regime_stats.index.name = 'Regime'
    
    def _calculate_transition_matrix(self):
        """Calculate regime transition probability matrix."""
        if self.regimes is None:
            return None
        
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(self.regimes) - 1):
            current_regime = int(self.regimes.iloc[i])
            next_regime = int(self.regimes.iloc[i + 1])
            
            # FIX: Check if regimes are valid before indexing
            if 0 <= current_regime < self.n_regimes and 0 <= next_regime < self.n_regimes:
                transition_counts[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_counts / row_sums
        
        self.transition_matrix = pd.DataFrame(
            transition_probs,
            index=['Bear', 'Neutral', 'Bull'],
            columns=['Bear', 'Neutral', 'Bull']
        )
        
        return self.transition_matrix
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """Get regime statistics."""
        return self.regime_stats
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get regime transition matrix."""
        if not hasattr(self, 'transition_matrix'):
            self._calculate_transition_matrix()
        return self.transition_matrix
    
    def get_current_regime(self) -> int:
        """Get most recent regime."""
        return self.regimes.iloc[-1]
    
    def predict_next_regime(self) -> Dict[int, float]:
        """
        Predict next regime probabilities.
        
        Returns:
            Dictionary of regime probabilities
        """
        current_regime = self.get_current_regime()
        
        if hasattr(self, 'transition_matrix'):
            probs = self.transition_matrix.iloc[current_regime].to_dict()
            return probs
        else:
            # Uniform if no transition matrix
            return {i: 1/self.n_regimes for i in range(self.n_regimes)}
