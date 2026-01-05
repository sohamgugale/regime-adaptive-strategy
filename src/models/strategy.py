"""
Multi-factor equity strategy with regime adaptation.
Combines factors with regime-dependent weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RegimeAdaptiveStrategy:
    """Multi-factor strategy with regime-adaptive weighting."""
    
    def __init__(self, factors: Dict[str, pd.DataFrame], regimes: pd.Series):
        """
        Initialize strategy.
        
        Args:
            factors: Dictionary of factor DataFrames
            regimes: Regime labels
        """
        self.factors = factors
        self.regimes = regimes
        self.regime_weights = None
        self.signals = None
        self.positions = None
        
    def optimize_regime_weights(self, returns: pd.DataFrame, 
                               train_end_date: pd.Timestamp) -> Dict[int, Dict[str, float]]:
        """
        Optimize factor weights for each regime using in-sample data.
        
        Args:
            returns: Stock returns
            train_end_date: End of training period
            
        Returns:
            Dictionary of optimized weights for each regime
        """
        print("\nOptimizing regime-specific factor weights...")
        
        # Split data
        train_factors = {name: df.loc[:train_end_date] 
                        for name, df in self.factors.items()}
        train_regimes = self.regimes.loc[:train_end_date]
        train_returns = returns.loc[:train_end_date]
        
        optimized_weights = {}
        
        for regime in range(3):
            print(f"  Optimizing weights for regime {regime}...")
            regime_dates = train_regimes[train_regimes == regime].index
            
            if len(regime_dates) < 20:
                # Not enough data, use default weights
                optimized_weights[regime] = self._get_default_weights()
                continue
            
            # Objective: maximize Sharpe ratio
            def objective(weights):
                # Calculate composite scores with these weights
                scores = self._calculate_scores_with_weights(
                    train_factors, regime_dates, weights
                )
                
                # Generate positions
                positions = self._scores_to_positions(scores, n_long=10, n_short=10)
                
                # Calculate returns
                strategy_returns = self._calculate_strategy_returns(
                    positions, train_returns.loc[regime_dates]
                )
                
                # Negative Sharpe (for minimization)
                if strategy_returns.std() > 0:
                    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    return -sharpe
                else:
                    return 0
            
            # Initial weights
            n_factors = len(train_factors)
            initial_weights = np.ones(n_factors) / n_factors
            
            # Constraints: weights sum to 1 (in absolute value)
            constraints = {'type': 'eq', 'fun': lambda w: np.abs(w).sum() - 1}
            
            # Bounds: each weight between -1 and 1
            bounds = [(-1, 1) for _ in range(n_factors)]
            
            # Optimize
            try:
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 50}
                )
                
                if result.success:
                    optimized = dict(zip(train_factors.keys(), result.x))
                else:
                    optimized = self._get_default_weights()
            except:
                optimized = self._get_default_weights()
            
            optimized_weights[regime] = optimized
            print(f"    Regime {regime} weights: {optimized}")
        
        self.regime_weights = optimized_weights
        return optimized_weights
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default factor weights."""
        factor_names = list(self.factors.keys())
        return {name: 1.0 / len(factor_names) for name in factor_names}
    
    def _calculate_scores_with_weights(self, factors: Dict[str, pd.DataFrame],
                                      dates: pd.DatetimeIndex,
                                      weights: np.ndarray) -> pd.DataFrame:
        """Calculate composite scores with given weights."""
        factor_names = list(factors.keys())
        weight_dict = dict(zip(factor_names, weights))
        
        # Initialize scores
        first_factor = list(factors.values())[0]
        scores = pd.DataFrame(0.0, index=dates, columns=first_factor.columns)
        
        # Combine factors
        for date in dates:
            if date not in first_factor.index:
                continue
            
            for factor_name, factor_df in factors.items():
                if date in factor_df.index and factor_name in weight_dict:
                    # Rank-based scoring
                    factor_values = factor_df.loc[date]
                    factor_ranks = factor_values.rank(pct=True, na_option='keep') - 0.5
                    
                    scores.loc[date] += weight_dict[factor_name] * factor_ranks
        
        return scores
    
    def _scores_to_positions(self, scores: pd.DataFrame, 
                           n_long: int, n_short: int) -> pd.DataFrame:
        """Convert scores to positions with PROPER NORMALIZATION."""
        positions = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        
        for date in scores.index:
            valid_scores = scores.loc[date].dropna()
            
            if len(valid_scores) >= n_long + n_short:
                # Long top N (FIXED: Equal weight normalized to 1.0 total)
                long_stocks = valid_scores.nlargest(n_long).index
                positions.loc[date, long_stocks] = 1.0 / n_long
                
                # Short bottom N (FIXED: Equal weight normalized to -1.0 total)
                short_stocks = valid_scores.nsmallest(n_short).index
                positions.loc[date, short_stocks] = -1.0 / n_short
        
        return positions
    
    def _calculate_strategy_returns(self, positions: pd.DataFrame, 
                                   returns: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from positions."""
        aligned_positions = positions.shift(1)
        strategy_returns = (aligned_positions * returns).sum(axis=1)
        return strategy_returns.dropna()
    
    def calculate_composite_score(self, use_optimized: bool = True) -> pd.DataFrame:
        """
        Calculate composite factor scores.
        
        Args:
            use_optimized: Use optimized weights if available
            
        Returns:
            DataFrame of composite scores
        """
        if use_optimized and self.regime_weights is None:
            print("Warning: No optimized weights, using default")
            use_optimized = False
        
        # Initialize scores
        first_factor = list(self.factors.values())[0]
        scores = pd.DataFrame(0.0, index=first_factor.index, columns=first_factor.columns)
        
        for date in scores.index:
            if date not in self.regimes.index:
                continue
            
            current_regime = self.regimes.loc[date]
            
            # Get weights for this regime
            if use_optimized and self.regime_weights:
                weights = self.regime_weights[current_regime]
            else:
                weights = self._get_default_weights()
            
            # Combine factors
            for factor_name, factor_df in self.factors.items():
                if date in factor_df.index and factor_name in weights:
                    # Rank-based scoring
                    factor_values = factor_df.loc[date]
                    factor_ranks = factor_values.rank(pct=True, na_option='keep') - 0.5
                    
                    scores.loc[date] += weights[factor_name] * factor_ranks
        
        self.signals = scores
        return scores
    
    def generate_positions(self, n_long: int = 10, n_short: int = 10,
                         rebalance_frequency: int = 5) -> pd.DataFrame:
        """
        Generate trading positions with PROPER NORMALIZATION.
        
        Args:
            n_long: Number of long positions
            n_short: Number of short positions
            rebalance_frequency: Rebalance every N days
            
        Returns:
            DataFrame of positions (sum to 0 for market neutral)
        """
        if self.signals is None:
            raise ValueError("Must calculate signals first")
        
        positions = pd.DataFrame(0.0, index=self.signals.index, columns=self.signals.columns)
        
        last_rebalance_date = None
        last_positions = None
        
        for i, date in enumerate(self.signals.index):
            # Check if we should rebalance
            if last_rebalance_date is None or i - self.signals.index.get_loc(last_rebalance_date) >= rebalance_frequency:
                # Rebalance
                valid_scores = self.signals.loc[date].dropna()
                
                if len(valid_scores) >= n_long + n_short:
                    # Long top N (FIXED: Equal weight, sum to 1.0)
                    long_stocks = valid_scores.nlargest(n_long).index
                    positions.loc[date, long_stocks] = 1.0 / n_long
                    
                    # Short bottom N (FIXED: Equal weight, sum to -1.0)
                    short_stocks = valid_scores.nsmallest(n_short).index
                    positions.loc[date, short_stocks] = -1.0 / n_short
                    
                    last_positions = positions.loc[date].copy()
                    last_rebalance_date = date
            else:
                # Hold previous positions
                if last_positions is not None:
                    positions.loc[date] = last_positions
        
        self.positions = positions
        return positions
    
    def get_factor_weights(self) -> pd.DataFrame:
        """Get current factor weights for each regime."""
        if self.regime_weights is None:
            return pd.DataFrame(self._get_default_weights(), index=[0, 1, 2])
        
        return pd.DataFrame(self.regime_weights).T
