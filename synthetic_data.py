"""
Advanced synthetic data generator for demonstration and testing.
Generates realistic price data with regime changes, correlations, and market microstructure.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List


class SyntheticMarketGenerator:
    """Generates synthetic market data with realistic properties."""
    
    def __init__(self, n_assets: int, n_days: int, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            n_assets: Number of assets to generate
            n_days: Number of trading days
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_assets = n_assets
        self.n_days = n_days
        self.seed = seed
        
    def generate_regime_series(self) -> np.ndarray:
        """Generate regime sequence with persistence."""
        regimes = np.zeros(self.n_days, dtype=int)
        
        # Regime transition probabilities (high persistence)
        transition_matrix = np.array([
            [0.85, 0.10, 0.05],  # Bear
            [0.10, 0.80, 0.10],  # Neutral
            [0.05, 0.10, 0.85],  # Bull
        ])
        
        # Start in neutral regime
        current_regime = 1
        regimes[0] = current_regime
        
        for t in range(1, self.n_days):
            probs = transition_matrix[current_regime]
            current_regime = np.random.choice([0, 1, 2], p=probs)
            regimes[t] = current_regime
            
        return regimes
    
    def generate_correlated_returns(self, regimes: np.ndarray) -> np.ndarray:
        """
        Generate correlated asset returns with regime-dependent properties.
        
        Args:
            regimes: Regime sequence
            
        Returns:
            Array of returns (n_days x n_assets)
        """
        # Regime-dependent parameters
        regime_params = {
            0: {'mean': -0.0005, 'vol': 0.025, 'correlation': 0.7},  # Bear
            1: {'mean': 0.0002, 'vol': 0.015, 'correlation': 0.5},   # Neutral
            2: {'mean': 0.0008, 'vol': 0.012, 'correlation': 0.6},   # Bull
        }
        
        returns = np.zeros((self.n_days, self.n_assets))
        
        for t in range(self.n_days):
            regime = regimes[t]
            params = regime_params[regime]
            
            # Generate correlated returns using factor model
            market_return = np.random.normal(params['mean'], params['vol'])
            
            # Asset-specific betas (some high-beta, some low-beta)
            betas = np.random.uniform(0.5, 1.5, self.n_assets)
            
            # Idiosyncratic returns
            idio_vol = params['vol'] * 0.5
            idio_returns = np.random.normal(0, idio_vol, self.n_assets)
            
            # Total returns = beta * market + idiosyncratic
            returns[t] = betas * market_return + idio_returns
            
        return returns
    
    def generate_volume(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate trading volume with volatility correlation.
        
        Args:
            returns: Return series
            
        Returns:
            Volume data
        """
        # Base volume
        base_volume = np.random.lognormal(15, 0.5, (self.n_days, self.n_assets))
        
        # Volume spikes on high volatility days
        volatility = np.abs(returns)
        volume_multiplier = 1 + 2 * (volatility / volatility.mean(axis=0))
        
        return base_volume * volume_multiplier
    
    def generate_fundamental_data(self) -> pd.DataFrame:
        """Generate synthetic fundamental data for value/quality factors."""
        # Market cap (varying sizes)
        market_cap = np.random.lognormal(10, 2, self.n_assets)
        
        # Book-to-market (value factor)
        book_to_market = np.random.lognormal(0, 0.8, self.n_assets)
        
        # ROE (quality factor)
        roe = np.random.normal(0.12, 0.08, self.n_assets)
        
        # Debt-to-equity (quality factor)
        debt_to_equity = np.random.gamma(2, 0.5, self.n_assets)
        
        # Earnings yield (value factor)
        earnings_yield = np.random.normal(0.06, 0.03, self.n_assets)
        
        fundamentals = pd.DataFrame({
            'market_cap': market_cap,
            'book_to_market': book_to_market,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'earnings_yield': earnings_yield,
        })
        
        return fundamentals
    
    def generate_complete_dataset(self, tickers: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with prices, returns, volume, and fundamentals.
        
        Args:
            tickers: List of ticker symbols (optional)
            
        Returns:
            Tuple of (prices, returns, volume, fundamentals)
        """
        # Generate dates
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=self.n_days, freq='B')
        
        # Generate tickers if not provided
        if tickers is None:
            tickers = [f'STOCK_{i:03d}' for i in range(self.n_assets)]
        else:
            tickers = tickers[:self.n_assets]
        
        # Generate regimes
        regimes = self.generate_regime_series()
        
        # Generate returns
        returns_array = self.generate_correlated_returns(regimes)
        
        # Generate prices from returns
        initial_prices = np.random.uniform(50, 500, self.n_assets)
        prices_array = np.zeros((self.n_days, self.n_assets))
        prices_array[0] = initial_prices
        
        for t in range(1, self.n_days):
            prices_array[t] = prices_array[t-1] * (1 + returns_array[t])
        
        # Generate volume
        volume_array = self.generate_volume(returns_array)
        
        # Create DataFrames
        prices = pd.DataFrame(prices_array, index=dates, columns=tickers)
        returns = pd.DataFrame(returns_array, index=dates, columns=tickers)
        volume = pd.DataFrame(volume_array, index=dates, columns=tickers)
        
        # Generate fundamentals (constant for all dates in this simple version)
        fundamentals = self.generate_fundamental_data()
        fundamentals.index = tickers
        
        return prices, returns, volume, fundamentals
    
    def generate_benchmark(self, regimes: np.ndarray = None) -> pd.DataFrame:
        """
        Generate benchmark index (e.g., SPY).
        
        Args:
            regimes: Optional regime sequence (generates if not provided)
            
        Returns:
            Benchmark price series
        """
        if regimes is None:
            regimes = self.generate_regime_series()
        
        # Benchmark has similar regime properties but lower volatility
        regime_params = {
            0: {'mean': -0.0003, 'vol': 0.015},  # Bear
            1: {'mean': 0.0003, 'vol': 0.010},   # Neutral
            2: {'mean': 0.0007, 'vol': 0.008},   # Bull
        }
        
        returns = np.zeros(self.n_days)
        for t in range(self.n_days):
            regime = regimes[t]
            params = regime_params[regime]
            returns[t] = np.random.normal(params['mean'], params['vol'])
        
        # Generate prices
        initial_price = 400.0  # Similar to SPY
        prices = np.zeros(self.n_days)
        prices[0] = initial_price
        
        for t in range(1, self.n_days):
            prices[t] = prices[t-1] * (1 + returns[t])
        
        # Create DataFrame
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=self.n_days, freq='B')
        
        return pd.DataFrame(prices, index=dates, columns=['SPY'])


def test_generator():
    """Test the synthetic data generator."""
    generator = SyntheticMarketGenerator(n_assets=50, n_days=1260, seed=42)
    
    tickers = [f'STOCK_{i:02d}' for i in range(50)]
    prices, returns, volume, fundamentals = generator.generate_complete_dataset(tickers)
    
    print("Synthetic Data Generation Test")
    print("="*50)
    print(f"Prices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Volume shape: {volume.shape}")
    print(f"Fundamentals shape: {fundamentals.shape}")
    print(f"\nDate range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\nReturns statistics:")
    print(f"  Mean: {returns.mean().mean()*252:.2%}")
    print(f"  Volatility: {returns.std().mean()*np.sqrt(252):.2%}")
    print(f"  Sharpe: {returns.mean().mean() / returns.std().mean() * np.sqrt(252):.2f}")
    

if __name__ == "__main__":
    test_generator()
