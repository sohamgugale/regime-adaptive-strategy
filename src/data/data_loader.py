"""
Data acquisition and preprocessing for equity strategy research.
Handles data download, cleaning, and feature engineering.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta
import time
from .synthetic_data import SyntheticDataGenerator


class EquityDataLoader:
    """Downloads and processes equity data for quantitative research."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize data loader.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.using_synthetic = False
        
    def download_data(self, use_synthetic: bool = False, max_retries: int = 2) -> pd.DataFrame:
        """Download historical price data with synthetic fallback."""
        
        if use_synthetic:
            print("Using synthetic data generator...")
            return self._generate_synthetic_data()
        
        print(f"Attempting to download real data for {len(self.tickers)} tickers...")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 5
                    print(f"Retry attempt {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                
                # Download data
                data = yf.download(
                    self.tickers,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    group_by='column',
                    auto_adjust=True,
                    threads=False  # Reduce load
                )
                
                if data.empty:
                    print(f"No data returned on attempt {attempt + 1}")
                    continue
                
                # Extract close prices
                if len(self.tickers) == 1:
                    self.data = pd.DataFrame(data['Close'])
                    self.data.columns = self.tickers
                else:
                    self.data = data['Close']
                
                # Remove any tickers with insufficient data
                min_observations = 252
                if isinstance(self.data, pd.Series):
                    self.data = pd.DataFrame(self.data)
                    
                valid_tickers = self.data.columns[self.data.count() >= min_observations]
                self.data = self.data[valid_tickers]
                
                if len(self.data.columns) > 0:
                    print(f"✓ Successfully downloaded data for {len(self.data.columns)} tickers")
                    return self.data
                    
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {str(e)[:100]}")
        
        # Fallback to synthetic data
        print("\n⚠ Real data download failed. Using synthetic data for demonstration.")
        print("(This is common in quant research when APIs are rate-limited)\n")
        return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data as fallback."""
        self.using_synthetic = True
        
        # Calculate number of days
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        n_days = (end - start).days
        
        generator = SyntheticDataGenerator(
            n_stocks=len(self.tickers),
            n_days=n_days,
            seed=42
        )
        
        self.data, self.volume_data = generator.generate_prices(self.tickers)
        print(f"✓ Generated synthetic data for {len(self.data.columns)} tickers")
        
        return self.data
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate log returns."""
        if self.data is None or self.data.empty:
            raise ValueError("No price data available. Call download_data() first.")
        self.returns = np.log(self.data / self.data.shift(1))
        return self.returns
    
    def get_volume_data(self) -> pd.DataFrame:
        """Download volume data or use synthetic."""
        if self.using_synthetic:
            return self.volume_data
        
        try:
            volume_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by='column',
                threads=False
            )['Volume']
            
            if len(self.tickers) == 1:
                volume_data = pd.DataFrame(volume_data)
                volume_data.columns = self.tickers
                
            return volume_data
            
        except Exception as e:
            print(f"Volume download failed, using synthetic: {str(e)[:100]}")
            if hasattr(self, 'volume_data'):
                return self.volume_data
            return None


class FactorEngineer:
    """Engineers factors for multi-factor equity strategy."""
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        """
        Initialize factor engineer.
        
        Args:
            prices: DataFrame of adjusted close prices
            returns: DataFrame of log returns
        """
        self.prices = prices
        self.returns = returns
        self.factors = pd.DataFrame(index=prices.index)
        
    def calculate_momentum(self, window: int = 126) -> pd.DataFrame:
        """
        Calculate momentum factor (6-month return).
        
        Args:
            window: Lookback window in days (default: 126 = 6 months)
        """
        momentum = self.prices.pct_change(window)
        return momentum
    
    def calculate_volatility(self, window: int = 21) -> pd.DataFrame:
        """
        Calculate realized volatility.
        
        Args:
            window: Rolling window in days (default: 21 = 1 month)
        """
        volatility = self.returns.rolling(window).std() * np.sqrt(252)
        return volatility
    
    def calculate_mean_reversion(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate mean reversion score (distance from moving average).
        
        Args:
            window: Moving average window
        """
        ma = self.prices.rolling(window).mean()
        mean_reversion = (self.prices - ma) / ma
        return -mean_reversion
    
    def calculate_volume_momentum(self, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate volume momentum (unusual trading activity).
        
        Args:
            volume: Volume data
            window: Lookback window
        """
        avg_volume = volume.rolling(window).mean()
        volume_momentum = (volume - avg_volume) / avg_volume
        return volume_momentum
    
    def build_factor_matrix(self, volume: pd.DataFrame = None) -> dict:
        """
        Build complete factor matrix.
        
        Returns:
            Dictionary with factor DataFrames
        """
        factors = {}
        
        factors['momentum'] = self.calculate_momentum()
        factors['volatility'] = self.calculate_volatility()
        factors['mean_reversion'] = self.calculate_mean_reversion()
        
        if volume is not None:
            try:
                factors['volume_momentum'] = self.calculate_volume_momentum(volume)
            except:
                print("Skipping volume momentum factor")
        
        return factors
