import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List


class RegimeDetector:
    """Detects market regimes using statistical methods."""
    
    def __init__(self, returns: pd.Series, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            returns: Market return series (e.g., SPY returns)
            n_regimes: Number of regimes to detect (default: 3)
        """
        self.returns = returns
        self.n_regimes = n_regimes
        self.regimes = None
        self.regime_stats = None
        
    def detect_regimes_simple(self, window: int = 63) -> pd.Series:
        """
        Simple regime detection based on rolling statistics.
        
        Args:
            window: Rolling window for statistics (default: 63 = 3 months)
            
        Returns:
            Series with regime labels (0: Bear, 1: Neutral, 2: Bull)
        """
        # Calculate rolling mean and volatility
        rolling_mean = self.returns.rolling(window).mean()
        rolling_vol = self.returns.rolling(window).std()
        
        # Calculate z-score for mean
        mean_zscore = (rolling_mean - rolling_mean.mean()) / rolling_mean.std()
        
        # Classify regimes
        regimes = pd.Series(index=self.returns.index, dtype=int)
        regimes[mean_zscore > 0.5] = 2  # Bull
        regimes[mean_zscore < -0.5] = 0  # Bear
        regimes[(mean_zscore >= -0.5) & (mean_zscore <= 0.5)] = 1  # Neutral
        
        # Forward fill initial NaN values
        regimes = regimes.fillna(method='bfill')
        
        self.regimes = regimes
        self._calculate_regime_statistics()
        
        return regimes
    
    def detect_regimes_volatility(self, vol_window: int = 21) -> pd.Series:
        """
        Regime detection based on volatility clustering.
        
        Args:
            vol_window: Window for volatility calculation
            
        Returns:
            Series with regime labels
        """
        # Calculate realized volatility
        realized_vol = self.returns.rolling(vol_window).std() * np.sqrt(252)
        
        # Calculate percentiles
        low_vol_threshold = realized_vol.quantile(0.33)
        high_vol_threshold = realized_vol.quantile(0.67)
        
        # Classify regimes based on volatility
        regimes = pd.Series(index=self.returns.index, dtype=int)
        regimes[realized_vol < low_vol_threshold] = 2  # Low vol (Bull)
        regimes[realized_vol > high_vol_threshold] = 0  # High vol (Bear)
        regimes[(realized_vol >= low_vol_threshold) & 
                (realized_vol <= high_vol_threshold)] = 1  # Medium vol (Neutral)
        
        regimes = regimes.fillna(method='bfill')
        
        self.regimes = regimes
        self._calculate_regime_statistics()
        
        return regimes
    
    def _calculate_regime_statistics(self):
        """Calculate statistics for each regime."""
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            regime_mask = self.regimes == regime
            regime_returns = self.returns[regime_mask]
            
            regime_stats[regime] = {
                'mean_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) 
                                if regime_returns.std() > 0 else 0,
                'count': regime_mask.sum(),
                'percentage': regime_mask.sum() / len(self.regimes) * 100
            }
        
        self.regime_stats = pd.DataFrame(regime_stats).T
        self.regime_stats.index.name = 'Regime'
        
    def get_current_regime(self) -> int:
        """Get the most recent regime."""
        return self.regimes.iloc[-1]
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """Get statistics for each regime."""
        return self.regime_stats
