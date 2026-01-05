"""
Advanced factor engineering with 10+ factors and orthogonalization.
Implements momentum, value, quality, volatility, and volume factors.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedFactorEngineer:
    """Engineers multiple factors and performs orthogonalization."""
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame, 
                 volume: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None):
        """
        Initialize factor engineer.
        
        Args:
            prices: Price data
            returns: Return data
            volume: Volume data
            fundamentals: Fundamental data (optional)
        """
        self.prices = prices
        self.returns = returns
        self.volume = volume
        self.fundamentals = fundamentals
        self.raw_factors = {}
        self.orthogonal_factors = None
        
    def calculate_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate all factors.
        
        Returns:
            Dictionary of factor DataFrames
        """
        print("\nCalculating factors...")
        
        # Momentum factors (3 different horizons)
        self.raw_factors['momentum_20d'] = self._momentum_factor(20)
        self.raw_factors['momentum_60d'] = self._momentum_factor(60)
        self.raw_factors['momentum_120d'] = self._momentum_factor(120)
        
        # Volatility factor
        self.raw_factors['volatility'] = self._volatility_factor(20)
        
        # Mean reversion
        self.raw_factors['mean_reversion'] = self._mean_reversion_factor(20)
        
        # Volume factors
        self.raw_factors['volume_momentum'] = self._volume_momentum_factor(20)
        self.raw_factors['volume_volatility'] = self._volume_volatility_factor(20)
        
        # Size factor (if prices available as proxy)
        self.raw_factors['size'] = self._size_factor()
        
        # Value factors (if fundamentals available)
        if self.fundamentals is not None:
            self.raw_factors['book_to_market'] = self._fundamental_factor('book_to_market')
            self.raw_factors['earnings_yield'] = self._fundamental_factor('earnings_yield')
            
            # Quality factors
            self.raw_factors['roe'] = self._fundamental_factor('roe')
            self.raw_factors['quality'] = self._quality_factor()
        
        # Residual momentum (momentum after controlling for volatility)
        self.raw_factors['residual_momentum'] = self._residual_momentum_factor()
        
        print(f"✓ Calculated {len(self.raw_factors)} factors")
        
        return self.raw_factors
    
    def _momentum_factor(self, window: int) -> pd.DataFrame:
        """Calculate momentum over specified window."""
        momentum = self.prices.pct_change(window)
        return momentum
    
    def _volatility_factor(self, window: int) -> pd.DataFrame:
        """Calculate realized volatility."""
        volatility = self.returns.rolling(window).std() * np.sqrt(252)
        return -volatility  # Negative because low vol is preferable
    
    def _mean_reversion_factor(self, window: int) -> pd.DataFrame:
        """Calculate mean reversion signal."""
        ma = self.prices.rolling(window).mean()
        deviation = (self.prices - ma) / ma
        return -deviation  # Negative because we buy oversold
    
    def _volume_momentum_factor(self, window: int) -> pd.DataFrame:
        """Calculate volume momentum."""
        avg_volume = self.volume.rolling(window).mean()
        volume_ratio = self.volume / avg_volume
        return volume_ratio - 1
    
    def _volume_volatility_factor(self, window: int) -> pd.DataFrame:
        """Calculate volume volatility."""
        volume_std = self.volume.rolling(window).std()
        avg_volume = self.volume.rolling(window).mean()
        volume_cv = volume_std / avg_volume
        return -volume_cv  # Negative because we prefer stable volume
    
    def _size_factor(self) -> pd.DataFrame:
        """
        Calculate size factor (using average price as proxy).
        In real implementation, use market cap.
        """
        # Use log of average price as proxy for size
        avg_price = self.prices.rolling(252).mean()
        log_size = np.log(avg_price)
        
        # Broadcast to all dates
        size_factor = pd.DataFrame(
            np.tile(log_size.iloc[-1].values, (len(self.prices), 1)),
            index=self.prices.index,
            columns=self.prices.columns
        )
        return -size_factor  # Negative for size premium (small cap)
    
    def _fundamental_factor(self, factor_name: str) -> pd.DataFrame:
        """Create factor from fundamental data."""
        if self.fundamentals is None:
            return pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)
        
        # Broadcast fundamental value to all dates
        factor_values = self.fundamentals[factor_name]
        factor_df = pd.DataFrame(
            np.tile(factor_values.values, (len(self.prices), 1)),
            index=self.prices.index,
            columns=self.prices.columns
        )
        return factor_df
    
    def _quality_factor(self) -> pd.DataFrame:
        """Composite quality factor from fundamentals."""
        if self.fundamentals is None:
            return pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)
        
        # Quality = high ROE + low debt
        quality = (
            self.fundamentals['roe'] - 
            0.5 * self.fundamentals['debt_to_equity']
        )
        
        # Broadcast to all dates
        quality_df = pd.DataFrame(
            np.tile(quality.values, (len(self.prices), 1)),
            index=self.prices.index,
            columns=self.prices.columns
        )
        return quality_df
    
    def _residual_momentum_factor(self) -> pd.DataFrame:
        """
        Momentum after controlling for volatility.
        This is momentum that can't be explained by risk-taking.
        """
        mom_60 = self._momentum_factor(60)
        vol_20 = -self._volatility_factor(20)  # Flip sign back
        
        # Residual momentum = momentum - beta * volatility
        # Simple version: momentum - correlation * volatility
        residual = mom_60 - 0.5 * vol_20
        return residual
    
    def orthogonalize_factors(self, n_components: int = 8) -> pd.DataFrame:
        """
        Orthogonalize factors using PCA.
        
        Args:
            n_components: Number of principal components to keep
            
        Returns:
            DataFrame of orthogonal factors
        """
        print(f"\nOrthogonalizing factors to {n_components} components...")
        
        # Stack all factors into a single matrix for each date
        orthogonal_factors = {}
        
        for date in self.prices.index:
            # Collect all factor values for this date
            factor_matrix = []
            valid_factors = []
            
            for name, factor_df in self.raw_factors.items():
                if date in factor_df.index:
                    values = factor_df.loc[date].values
                    if not np.all(np.isnan(values)):
                        factor_matrix.append(values)
                        valid_factors.append(name)
            
            if len(factor_matrix) < n_components:
                continue
            
            factor_matrix = np.array(factor_matrix).T  # (n_stocks, n_factors)
            
            # Handle NaNs
            if np.isnan(factor_matrix).any():
                factor_matrix = np.nan_to_num(factor_matrix, nan=0.0)
            
            # Standardize
            scaler = StandardScaler()
            try:
                factor_matrix_scaled = scaler.fit_transform(factor_matrix)
            except:
                continue
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, factor_matrix.shape[1]))
            try:
                principal_components = pca.fit_transform(factor_matrix_scaled)
            except:
                continue
            
            # Store principal components
            for i in range(principal_components.shape[1]):
                factor_name = f'PC{i+1}'
                if factor_name not in orthogonal_factors:
                    orthogonal_factors[factor_name] = {}
                orthogonal_factors[factor_name][date] = principal_components[:, i]
        
        # Convert to DataFrames
        orthogonal_dfs = {}
        for name, date_dict in orthogonal_factors.items():
            df = pd.DataFrame(date_dict, index=self.prices.columns).T
            df.index = pd.to_datetime(df.index)
            orthogonal_dfs[name] = df.sort_index()
        
        self.orthogonal_factors = orthogonal_dfs
        print(f"✓ Created {len(orthogonal_dfs)} orthogonal factors")
        
        return orthogonal_dfs
    
    def get_factor_matrix(self, orthogonalize: bool = True, 
                         n_components: int = 8) -> Dict[str, pd.DataFrame]:
        """
        Get complete factor matrix.
        
        Args:
            orthogonalize: Whether to orthogonalize factors
            n_components: Number of components for PCA
            
        Returns:
            Dictionary of factor DataFrames
        """
        if len(self.raw_factors) == 0:
            self.calculate_all_factors()
        
        if orthogonalize:
            if self.orthogonal_factors is None:
                self.orthogonalize_factors(n_components)
            return self.orthogonal_factors
        else:
            return self.raw_factors
    
    def get_factor_statistics(self) -> pd.DataFrame:
        """Calculate statistics for each factor."""
        stats = []
        
        for name, factor_df in self.raw_factors.items():
            factor_stats = {
                'factor': name,
                'mean': factor_df.mean().mean(),
                'std': factor_df.std().mean(),
                'min': factor_df.min().min(),
                'max': factor_df.max().max(),
                'coverage': factor_df.notna().sum().sum() / factor_df.size
            }
            stats.append(factor_stats)
        
        return pd.DataFrame(stats)


def test_factor_engineer():
    """Test the factor engineer."""
    from .synthetic_data import SyntheticMarketGenerator
    
    print("Testing Advanced Factor Engineer")
    print("="*50)
    
    # Generate test data
    generator = SyntheticMarketGenerator(n_assets=50, n_days=500, seed=42)
    tickers = [f'STOCK_{i:02d}' for i in range(50)]
    prices, returns, volume, fundamentals = generator.generate_complete_dataset(tickers)
    
    # Create engineer
    engineer = AdvancedFactorEngineer(prices, returns, volume, fundamentals)
    
    # Calculate factors
    raw_factors = engineer.calculate_all_factors()
    
    print("\nRaw Factors:")
    for name, df in raw_factors.items():
        print(f"  {name}: {df.shape}")
    
    # Orthogonalize
    orthogonal_factors = engineer.orthogonalize_factors(n_components=8)
    
    print("\nOrthogonal Factors:")
    for name, df in orthogonal_factors.items():
        print(f"  {name}: {df.shape}")
    
    # Statistics
    stats = engineer.get_factor_statistics()
    print("\nFactor Statistics:")
    print(stats.round(3))


if __name__ == "__main__":
    test_factor_engineer()
