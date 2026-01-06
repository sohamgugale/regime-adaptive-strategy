"""
Multi-factor equity strategy with regime adaptation.
FULLY DEBUGGED VERSION with position generation fix.
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
        """Initialize strategy."""
        self.factors = factors
        self.regimes = regimes
        self.regime_weights = None
        self.signals = None
        self.positions = None
        
    def optimize_regime_weights(self, returns: pd.DataFrame, 
                               train_end_date: pd.Timestamp) -> Dict[int, Dict[str, float]]:
        """Optimize factor weights for each regime."""
        print("\nOptimizing regime-specific factor weights...")
        
        train_factors = {name: df.loc[:train_end_date] for name, df in self.factors.items()}
        train_regimes = self.regimes.loc[:train_end_date]
        train_returns = returns.loc[:train_end_date]
        
        optimized_weights = {}
        
        for regime in range(3):
            print(f"  Optimizing weights for regime {regime}...")
            regime_dates = train_regimes[train_regimes == regime].index
            
            if len(regime_dates) < 20:
                optimized_weights[regime] = self._get_default_weights()
                continue
            
            def objective(weights):
                scores = self._calculate_scores_with_weights(train_factors, regime_dates, weights)
                positions = self._scores_to_positions(scores, n_long=10, n_short=10)
                strategy_returns = self._calculate_strategy_returns(positions, train_returns.loc[regime_dates])
                
                if strategy_returns.std() > 0:
                    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    return -sharpe
                else:
                    return 0
            
            n_factors = len(train_factors)
            initial_weights = np.ones(n_factors) / n_factors
            constraints = {'type': 'eq', 'fun': lambda w: np.abs(w).sum() - 1}
            bounds = [(-1, 1) for _ in range(n_factors)]
            
            try:
                result = minimize(objective, initial_weights, method='SLSQP', 
                                bounds=bounds, constraints=constraints, options={'maxiter': 50})
                
                if result.success:
                    optimized = dict(zip(train_factors.keys(), result.x))
                else:
                    optimized = self._get_default_weights()
            except:
                optimized = self._get_default_weights()
            
            optimized_weights[regime] = optimized
        
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
        
        first_factor = list(factors.values())[0]
        scores = pd.DataFrame(0.0, index=dates, columns=first_factor.columns)
        
        for date in dates:
            if date not in first_factor.index:
                continue
            
            for factor_name, factor_df in factors.items():
                if date in factor_df.index and factor_name in weight_dict:
                    factor_values = factor_df.loc[date]
                    # CRITICAL: Ensure we have valid values
                    if factor_values.notna().sum() > 0:
                        factor_ranks = factor_values.rank(pct=True, na_option='keep') - 0.5
                        scores.loc[date] += weight_dict[factor_name] * factor_ranks
        
        return scores
    
    def _scores_to_positions(self, scores: pd.DataFrame, 
                           n_long: int, n_short: int) -> pd.DataFrame:
        """Convert scores to positions - FIXED VERSION."""
        positions = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        
        for date in scores.index:
            valid_scores = scores.loc[date].dropna()
            
            if len(valid_scores) >= n_long + n_short:
                # CRITICAL FIX: Ensure scores have variation
                if valid_scores.std() > 0:
                    # Long top N
                    long_stocks = valid_scores.nlargest(n_long).index
                    positions.loc[date, long_stocks] = 1.0 / n_long
                    
                    # Short bottom N
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
        """Calculate composite factor scores - FIXED VERSION."""
        if use_optimized and self.regime_weights is None:
            print("Warning: No optimized weights, using default")
            use_optimized = False
        
        first_factor = list(self.factors.values())[0]
        scores = pd.DataFrame(0.0, index=first_factor.index, columns=first_factor.columns)
        
        print(f"\nCalculating composite scores for {len(scores)} dates...")
        
        valid_dates = 0
        for date in scores.index:
            if date not in self.regimes.index:
                continue
            
            current_regime = int(self.regimes.loc[date])
            
            # Get weights
            if use_optimized and self.regime_weights:
                weights = self.regime_weights[current_regime]
            else:
                weights = self._get_default_weights()
            
            # Combine factors
            date_score = pd.Series(0.0, index=scores.columns)
            
            for factor_name, factor_df in self.factors.items():
                if date in factor_df.index and factor_name in weights:
                    factor_values = factor_df.loc[date]
                    
                    # CRITICAL: Check for valid values
                    if factor_values.notna().sum() > 0:
                        factor_ranks = factor_values.rank(pct=True, na_option='keep') - 0.5
                        date_score += weights[factor_name] * factor_ranks
            
            scores.loc[date] = date_score
            
            if date_score.notna().sum() > 0:
                valid_dates += 1
        
        print(f"✓ Generated scores for {valid_dates} valid dates")
        self.signals = scores
        return scores
    
    def generate_positions(self, n_long: int = 10, n_short: int = 10,
                         rebalance_frequency: int = 5) -> pd.DataFrame:
        """Generate trading positions - FULLY DEBUGGED."""
        if self.signals is None:
            raise ValueError("Must calculate signals first")
        
        positions = pd.DataFrame(0.0, index=self.signals.index, columns=self.signals.columns)
        
        print(f"\nGenerating positions...")
        print(f"  Target: {n_long} longs, {n_short} shorts")
        print(f"  Rebalance frequency: {rebalance_frequency} days")
        
        last_rebalance_idx = None
        last_positions = None
        rebalance_count = 0
        
        for i, date in enumerate(self.signals.index):
            # Check if we should rebalance
            should_rebalance = (last_rebalance_idx is None or 
                               i - last_rebalance_idx >= rebalance_frequency)
            
            if should_rebalance:
                valid_scores = self.signals.loc[date].dropna()
                
                if len(valid_scores) >= n_long + n_short and valid_scores.std() > 0:
                    # Long top N
                    long_stocks = valid_scores.nlargest(n_long).index
                    positions.loc[date, long_stocks] = 1.0 / n_long
                    
                    # Short bottom N  
                    short_stocks = valid_scores.nsmallest(n_short).index
                    positions.loc[date, short_stocks] = -1.0 / n_short
                    
                    last_positions = positions.loc[date].copy()
                    last_rebalance_idx = i
                    rebalance_count += 1
                    
                    # DEBUG: Print first rebalance
                    if rebalance_count == 1:
                        print(f"\n  First rebalance on {date.date()}:")
                        print(f"    Longs: {list(long_stocks)}")
                        print(f"    Shorts: {list(short_stocks)}")
            else:
                # Hold previous positions
                if last_positions is not None:
                    positions.loc[date] = last_positions
        
        # Count final positions
        long_count = (positions > 0).any(axis=0).sum()
        short_count = (positions < 0).any(axis=0).sum()
        
        print(f"\n✓ Position generation completed:")
        print(f"  Total rebalances: {rebalance_count}")
        print(f"  Unique longs: {long_count}")
        print(f"  Unique shorts: {short_count}")
        print(f"  Total trading days: {(positions != 0).any(axis=1).sum()}")
        
        self.positions = positions
        return positions
    
    def get_factor_weights(self) -> pd.DataFrame:
        """Get current factor weights for each regime."""
        if self.regime_weights is None:
            return pd.DataFrame(self._get_default_weights(), index=[0, 1, 2])
        return pd.DataFrame(self.regime_weights).T
