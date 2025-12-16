"""
Multi-factor equity strategy with regime adaptation.
Combines factors with regime-dependent weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import cvxpy as cp


class RegimeAdaptiveStrategy:
    """Multi-factor strategy that adapts to market regimes."""
    
    def __init__(self, factors: Dict[str, pd.DataFrame], regimes: pd.Series):
        """
        Initialize strategy.
        
        Args:
            factors: Dictionary of factor DataFrames
            regimes: Series with regime labels
        """
        self.factors = factors
        self.regimes = regimes
        self.signals = None
        self.positions = None
        
        # Regime-dependent factor weights
        self.regime_weights = {
            0: {  # Bear market - defensive
                'momentum': -0.3,
                'volatility': -0.4,
                'mean_reversion': 0.5,
                'volume_momentum': 0.1
            },
            1: {  # Neutral market - balanced
                'momentum': 0.3,
                'volatility': -0.2,
                'mean_reversion': 0.3,
                'volume_momentum': 0.2
            },
            2: {  # Bull market - aggressive momentum
                'momentum': 0.5,
                'volatility': -0.1,
                'mean_reversion': 0.1,
                'volume_momentum': 0.3
            }
        }
    
    def calculate_composite_score(self) -> pd.DataFrame:
        """
        Calculate composite factor score for each stock.
        
        Returns:
            DataFrame of composite scores
        """
        # Initialize composite scores
        composite = pd.DataFrame(0, 
                                index=self.factors['momentum'].index,
                                columns=self.factors['momentum'].columns)
        
        # For each date, apply regime-specific weights
        for date in composite.index:
            if date not in self.regimes.index:
                continue
                
            current_regime = self.regimes.loc[date]
            weights = self.regime_weights[current_regime]
            
            # Combine factors with regime-specific weights
            for factor_name, factor_df in self.factors.items():
                if factor_name in weights and date in factor_df.index:
                    # Rank-based scoring (cross-sectional)
                    factor_values = factor_df.loc[date]
                    factor_ranks = factor_values.rank(pct=True) - 0.5  # Center around 0
                    
                    composite.loc[date] += weights[factor_name] * factor_ranks
        
        self.signals = composite
        return composite
    
    def generate_positions(self, n_long: int = 10, n_short: int = 10) -> pd.DataFrame:
        """
        Generate long/short positions based on composite scores.
        
        Args:
            n_long: Number of long positions
            n_short: Number of short positions
            
        Returns:
            DataFrame with positions (-1, 0, 1)
        """
        positions = pd.DataFrame(0, 
                                index=self.signals.index,
                                columns=self.signals.columns)
        
        for date in self.signals.index:
            scores = self.signals.loc[date].dropna()
            
            if len(scores) < n_long + n_short:
                continue
            
            # Long top N stocks
            long_stocks = scores.nlargest(n_long).index
            positions.loc[date, long_stocks] = 1
            
            # Short bottom N stocks
            short_stocks = scores.nsmallest(n_short).index
            positions.loc[date, short_stocks] = -1
        
        self.positions = positions
        return positions
    
    def optimize_positions(self, scores: pd.Series, 
                          max_position: float = 0.1,
                          target_leverage: float = 1.0) -> pd.Series:
        """
        Optimize positions using mean-variance optimization.
        
        Args:
            scores: Composite factor scores for date
            max_position: Maximum position size per stock
            target_leverage: Target gross leverage
            
        Returns:
            Series with optimal positions
        """
        valid_scores = scores.dropna()
        n_stocks = len(valid_scores)
        
        if n_stocks == 0:
            return pd.Series(0, index=scores.index)
        
        # Define optimization variables
        weights = cp.Variable(n_stocks)
        
        # Objective: maximize alignment with scores
        objective = cp.Maximize(valid_scores.values @ weights)
        
        # Constraints
        constraints = [
            cp.sum(cp.abs(weights)) <= target_leverage,  # Leverage constraint
            cp.abs(weights) <= max_position,  # Position size limits
            cp.sum(weights) == 0  # Market neutral
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        # Create result series
        optimized_positions = pd.Series(0.0, index=scores.index)
        optimized_positions[valid_scores.index] = weights.value
        
        return optimized_positions