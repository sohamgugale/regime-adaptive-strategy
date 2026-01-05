"""
Data loading and preprocessing module.
Handles both real market data and synthetic data with automatic fallback.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from .synthetic_data import SyntheticMarketGenerator


class DataLoader:
    """Loads and preprocesses market data."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, use_synthetic: bool = False):
        """
        Initialize data loader.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_synthetic: Force use of synthetic data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.use_synthetic = use_synthetic
        self.fundamentals = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load market data with automatic fallback to synthetic.
        
        Returns:
            Tuple of (prices, returns, volume)
        """
        if self.use_synthetic:
            print("Using synthetic data (configured)")
            return self._load_synthetic_data()
        
        try:
            print(f"Attempting to download data for {len(self.tickers)} tickers...")
            prices, volume = self._load_real_data()
            
            # Check if download was successful
            if prices.isnull().sum().sum() > 0.5 * prices.size:
                print("⚠️  Real data download failed or incomplete, using synthetic data")
                return self._load_synthetic_data()
            
            returns = np.log(prices / prices.shift(1))
            print(f"✓ Successfully loaded real market data")
            return prices, returns, volume
            
        except Exception as e:
            print(f"⚠️  Error downloading real data: {e}")
            print("Falling back to synthetic data")
            return self._load_synthetic_data()
    
    def _load_real_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load real market data from Yahoo Finance."""
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            threads=True
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
            volume = data['Volume']
        else:
            prices = data['Adj Close'].to_frame()
            volume = data['Volume'].to_frame()
        
        # Clean data
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        volume = volume.fillna(method='ffill').fillna(method='bfill')
        
        return prices, volume
    
    def _load_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic market data."""
        # Calculate number of days from date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n_days = len(date_range)
        
        generator = SyntheticMarketGenerator(
            n_assets=len(self.tickers),
            n_days=n_days,
            seed=42
        )
        
        prices, returns, volume, fundamentals = generator.generate_complete_dataset(self.tickers)
        self.fundamentals = fundamentals
        
        return prices, returns, volume
    
    def load_benchmark(self, benchmark_ticker: str = 'SPY') -> pd.DataFrame:
        """
        Load benchmark data.
        
        Args:
            benchmark_ticker: Benchmark ticker symbol
            
        Returns:
            Benchmark price series
        """
        if self.use_synthetic:
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            n_days = len(date_range)
            generator = SyntheticMarketGenerator(n_assets=1, n_days=n_days, seed=42)
            return generator.generate_benchmark()
        
        try:
            data = yf.download(
                benchmark_ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            return data['Adj Close'].to_frame(benchmark_ticker)
        except:
            # Fallback to synthetic
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            n_days = len(date_range)
            generator = SyntheticMarketGenerator(n_assets=1, n_days=n_days, seed=42)
            return generator.generate_benchmark()
    
    def get_fundamentals(self) -> pd.DataFrame:
        """Get fundamental data (only available for synthetic data)."""
        if self.fundamentals is not None:
            return self.fundamentals
        else:
            # For real data, return None or download from another source
            print("Warning: Fundamental data not available for real market data")
            return None


def test_data_loader():
    """Test the data loader."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("Testing Data Loader with Synthetic Data")
    print("="*50)
    
    loader = DataLoader(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        use_synthetic=True
    )
    
    prices, returns, volume = loader.load_data()
    benchmark = loader.load_benchmark()
    fundamentals = loader.get_fundamentals()
    
    print(f"\nPrices: {prices.shape}")
    print(f"Returns: {returns.shape}")
    print(f"Volume: {volume.shape}")
    print(f"Benchmark: {benchmark.shape}")
    if fundamentals is not None:
        print(f"Fundamentals: {fundamentals.shape}")
    
    print(f"\nDate range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Sample prices (first 5 rows):")
    print(prices.head())


if __name__ == "__main__":
    test_data_loader()
