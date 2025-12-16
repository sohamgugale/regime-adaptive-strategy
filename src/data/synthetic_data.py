"""
Synthetic market data generator for testing and demonstration.
Generates realistic price series with various market regimes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List


class SyntheticDataGenerator:
    """Generates synthetic stock price data with realistic properties."""
    
    def __init__(self, n_stocks: int, n_days: int, seed: int = 42):
        """
        Initialize synthetic data generator.
        
        Args:
            n_stocks: Number of stocks to generate
            n_days: Number of trading days
            seed: Random seed for reproducibility
        """
        self.n_stocks = n_stocks
        self.n_days = n_days
        np.random.seed(seed)
        
    def generate_prices(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic price and volume data.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Tuple of (prices DataFrame, volume DataFrame)
        """
        n_stocks = min(len(tickers), self.n_stocks)
        dates = pd.date_range(end=datetime.now(), periods=self.n_days, freq='D')
        
        # Generate returns with regime switching
        returns = self._generate_returns_with_regimes(n_stocks)
        
        # Convert to prices starting at 100
        initial_prices = np.random.uniform(50, 500, n_stocks)
        prices = np.zeros((self.n_days, n_stocks))
        prices[0] = initial_prices
        
        for t in range(1, self.n_days):
            prices[t] = prices[t-1] * np.exp(returns[t])
        
        # Create DataFrames
        price_df = pd.DataFrame(prices, index=dates, columns=tickers[:n_stocks])
        
        # Generate volume data (correlated with volatility)
        base_volume = np.random.uniform(1e6, 1e8, n_stocks)
        volume_multiplier = 1 + np.abs(returns) * 10  # Higher volume on volatile days
        volumes = base_volume * volume_multiplier
        
        volume_df = pd.DataFrame(volumes, index=dates, columns=tickers[:n_stocks])
        
        return price_df, volume_df
    
    def _generate_returns_with_regimes(self, n_stocks: int) -> np.ndarray:
        """Generate returns with bull/bear/neutral regimes."""
        returns = np.zeros((self.n_days, n_stocks))
        
        # Define regime parameters
        regimes = self._generate_regime_sequence()
        
        for i in range(n_stocks):
            # Stock-specific parameters
            base_vol = np.random.uniform(0.15, 0.35)  # Annual volatility
            beta = np.random.uniform(0.5, 1.5)  # Market beta
            
            for t in range(self.n_days):
                regime = regimes[t]
                
                # Regime-dependent drift and volatility
                if regime == 2:  # Bull
                    drift = 0.0008 * beta
                    vol = base_vol * 0.8
                elif regime == 0:  # Bear
                    drift = -0.0012 * beta
                    vol = base_vol * 1.3
                else:  # Neutral
                    drift = 0.0002 * beta
                    vol = base_vol
                
                # Daily return with some correlation to market
                market_return = np.random.normal(drift, vol / np.sqrt(252))
                idiosyncratic = np.random.normal(0, vol / np.sqrt(252) * 0.5)
                
                returns[t, i] = market_return + idiosyncratic
        
        return returns
    
    def _generate_regime_sequence(self) -> np.ndarray:
        """Generate regime sequence with persistence."""
        regimes = np.zeros(self.n_days, dtype=int)
        
        # Start in neutral regime
        current_regime = 1
        regime_duration = 0
        
        for t in range(self.n_days):
            # Regime persistence (stay in regime for ~60 days on average)
            if regime_duration > 0:
                regimes[t] = current_regime
                regime_duration -= 1
            else:
                # Transition to new regime
                transition_probs = {
                    0: [0.7, 0.2, 0.1],  # Bear -> [Bear, Neutral, Bull]
                    1: [0.2, 0.6, 0.2],  # Neutral -> [Bear, Neutral, Bull]
                    2: [0.1, 0.2, 0.7],  # Bull -> [Bear, Neutral, Bull]
                }
                current_regime = np.random.choice([0, 1, 2], p=transition_probs[current_regime])
                regimes[t] = current_regime
                regime_duration = np.random.poisson(60)  # Average regime duration
        
        return regimes
    
    def generate_benchmark(self) -> pd.DataFrame:
        """Generate benchmark (SPY-like) data."""
        dates = pd.date_range(end=datetime.now(), periods=self.n_days, freq='D')
        
        # Generate market returns
        regimes = self._generate_regime_sequence()
        returns = np.zeros(self.n_days)
        
        for t in range(self.n_days):
            regime = regimes[t]
            
            if regime == 2:  # Bull
                drift = 0.0008
                vol = 0.12
            elif regime == 0:  # Bear
                drift = -0.0012
                vol = 0.20
            else:  # Neutral
                drift = 0.0003
                vol = 0.15
            
            returns[t] = np.random.normal(drift, vol / np.sqrt(252))
        
        # Convert to prices
        prices = np.zeros(self.n_days)
        prices[0] = 400  # SPY-like starting price
        
        for t in range(1, self.n_days):
            prices[t] = prices[t-1] * np.exp(returns[t])
        
        return pd.DataFrame(prices, index=dates, columns=['SPY'])
