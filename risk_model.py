"""
Risk model with covariance estimation, VaR, and CVaR calculation.
Implements Ledoit-Wolf shrinkage for robust covariance estimation.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class RiskModel:
    """Portfolio risk model with advanced covariance estimation."""
    
    def __init__(self, returns: pd.DataFrame, estimation_window: int = 252):
        """
        Initialize risk model.
        
        Args:
            returns: Return data (dates x assets)
            estimation_window: Window for covariance estimation
        """
        self.returns = returns
        self.estimation_window = estimation_window
        self.covariance_matrix = None
        self.correlation_matrix = None
        
    def estimate_covariance(self, method: str = 'ledoit_wolf', 
                          date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Estimate covariance matrix.
        
        Args:
            method: 'ledoit_wolf', 'sample', or 'constant_correlation'
            date: Date for estimation (uses all data if None)
            
        Returns:
            Covariance matrix DataFrame
        """
        if date is not None:
            # Use rolling window ending at date
            end_idx = self.returns.index.get_loc(date)
            start_idx = max(0, end_idx - self.estimation_window)
            returns_window = self.returns.iloc[start_idx:end_idx+1]
        else:
            # Use last estimation_window days
            returns_window = self.returns.iloc[-self.estimation_window:]
        
        # Remove NaN columns
        returns_clean = returns_window.dropna(axis=1, how='all')
        
        if method == 'ledoit_wolf':
            cov_matrix = self._ledoit_wolf_covariance(returns_clean)
        elif method == 'sample':
            cov_matrix = self._sample_covariance(returns_clean)
        elif method == 'constant_correlation':
            cov_matrix = self._constant_correlation_covariance(returns_clean)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.covariance_matrix = pd.DataFrame(
            cov_matrix,
            index=returns_clean.columns,
            columns=returns_clean.columns
        )
        
        # Also calculate correlation
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        self.correlation_matrix = pd.DataFrame(
            corr_matrix,
            index=returns_clean.columns,
            columns=returns_clean.columns
        )
        
        return self.covariance_matrix
    
    def _ledoit_wolf_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance using Ledoit-Wolf shrinkage."""
        returns_array = returns.fillna(0).values
        
        lw = LedoitWolf()
        try:
            cov_matrix = lw.fit(returns_array).covariance_
        except:
            # Fallback to sample covariance
            cov_matrix = np.cov(returns_array, rowvar=False)
        
        return cov_matrix
    
    def _sample_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate sample covariance matrix."""
        returns_array = returns.fillna(0).values
        cov_matrix = np.cov(returns_array, rowvar=False)
        return cov_matrix
    
    def _constant_correlation_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance assuming constant correlation.
        More stable for large universes.
        """
        returns_array = returns.fillna(0).values
        
        # Calculate standard deviations
        std_devs = np.std(returns_array, axis=0)
        
        # Calculate average correlation
        corr_matrix = np.corrcoef(returns_array, rowvar=False)
        avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix)**2 - len(corr_matrix))
        
        # Create constant correlation matrix
        n_assets = len(std_devs)
        const_corr_matrix = np.full((n_assets, n_assets), avg_corr)
        np.fill_diagonal(const_corr_matrix, 1.0)
        
        # Convert to covariance
        cov_matrix = np.outer(std_devs, std_devs) * const_corr_matrix
        
        return cov_matrix
    
    def calculate_portfolio_risk(self, weights: pd.Series, 
                                method: str = 'ledoit_wolf') -> Dict[str, float]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            weights: Portfolio weights
            method: Covariance estimation method
            
        Returns:
            Dictionary of risk metrics
        """
        # Ensure covariance matrix is estimated
        if self.covariance_matrix is None:
            self.estimate_covariance(method=method)
        
        # Align weights with covariance matrix
        aligned_weights = weights.reindex(self.covariance_matrix.index, fill_value=0)
        weight_array = aligned_weights.values
        
        # Portfolio variance
        portfolio_variance = weight_array @ self.covariance_matrix.values @ weight_array
        portfolio_vol = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
        
        # Calculate marginal risk contribution
        marginal_contrib = self.covariance_matrix.values @ weight_array
        risk_contrib = weight_array * marginal_contrib / np.sqrt(portfolio_variance)
        
        # Concentration (sum of squared weights)
        concentration = np.sum(weight_array**2)
        
        metrics = {
            'portfolio_volatility': portfolio_vol,
            'portfolio_variance': portfolio_variance * 252,
            'concentration': concentration,
            'effective_n_positions': 1 / concentration if concentration > 0 else 0
        }
        
        return metrics
    
    def calculate_var_cvar(self, weights: pd.Series, 
                          confidence: float = 0.95,
                          horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        
        Args:
            weights: Portfolio weights
            confidence: Confidence level
            horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR and CVaR
        """
        # Align weights
        aligned_weights = weights.reindex(self.returns.columns, fill_value=0)
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * aligned_weights).sum(axis=1)
        
        # Scale to horizon
        portfolio_returns_horizon = portfolio_returns * np.sqrt(horizon)
        
        # Calculate VaR (percentile)
        var = -np.percentile(portfolio_returns_horizon.dropna(), (1 - confidence) * 100)
        
        # Calculate CVaR (expected shortfall)
        returns_below_var = portfolio_returns_horizon[portfolio_returns_horizon <= -var]
        cvar = -returns_below_var.mean() if len(returns_below_var) > 0 else var
        
        return {
            'VaR': var * 100,  # As percentage
            'CVaR': cvar * 100,  # As percentage
            'confidence': confidence,
            'horizon_days': horizon
        }
    
    def decompose_risk(self, weights: pd.Series, 
                      method: str = 'ledoit_wolf') -> pd.DataFrame:
        """
        Decompose portfolio risk into asset contributions.
        
        Args:
            weights: Portfolio weights
            method: Covariance estimation method
            
        Returns:
            DataFrame with risk decomposition
        """
        if self.covariance_matrix is None:
            self.estimate_covariance(method=method)
        
        aligned_weights = weights.reindex(self.covariance_matrix.index, fill_value=0)
        weight_array = aligned_weights.values
        
        # Portfolio variance
        portfolio_variance = weight_array @ self.covariance_matrix.values @ weight_array
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Marginal contribution to risk
        marginal_contrib = (self.covariance_matrix.values @ weight_array) / portfolio_vol
        
        # Contribution to risk
        risk_contrib = weight_array * marginal_contrib
        
        # Percentage contribution
        risk_contrib_pct = risk_contrib / portfolio_vol
        
        decomposition = pd.DataFrame({
            'weight': weight_array,
            'volatility': np.sqrt(np.diag(self.covariance_matrix)) * np.sqrt(252),
            'marginal_contrib': marginal_contrib * np.sqrt(252),
            'risk_contrib': risk_contrib * np.sqrt(252),
            'risk_contrib_pct': risk_contrib_pct * 100
        }, index=self.covariance_matrix.index)
        
        return decomposition.sort_values('risk_contrib', ascending=False)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix."""
        if self.correlation_matrix is None:
            self.estimate_covariance()
        return self.correlation_matrix
    
    def get_factor_exposure(self, weights: pd.Series, 
                           factor_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio's factor exposures.
        
        Args:
            weights: Portfolio weights
            factor_returns: Factor return data
            
        Returns:
            Series of factor exposures (betas)
        """
        # Align weights and returns
        common_assets = weights.index.intersection(self.returns.columns)
        aligned_weights = weights.loc[common_assets]
        aligned_returns = self.returns[common_assets]
        
        # Portfolio returns
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # Regress portfolio returns on factors
        exposures = {}
        for factor_name in factor_returns.columns:
            factor_series = factor_returns[factor_name]
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(factor_series.index)
            if len(common_dates) > 0:
                y = portfolio_returns.loc[common_dates].values
                X = factor_series.loc[common_dates].values
                
                # Simple linear regression
                beta = np.cov(y, X)[0, 1] / np.var(X)
                exposures[factor_name] = beta
        
        return pd.Series(exposures)


def test_risk_model():
    """Test the risk model."""
    from ..data.synthetic_data import SyntheticMarketGenerator
    
    print("Testing Risk Model")
    print("="*50)
    
    # Generate test data
    generator = SyntheticMarketGenerator(n_assets=20, n_days=500, seed=42)
    tickers = [f'STOCK_{i:02d}' for i in range(20)]
    prices, returns, volume, _ = generator.generate_complete_dataset(tickers)
    
    # Create risk model
    risk_model = RiskModel(returns, estimation_window=252)
    
    # Estimate covariance
    cov_matrix = risk_model.estimate_covariance(method='ledoit_wolf')
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    
    # Create sample portfolio
    weights = pd.Series(np.random.uniform(-0.1, 0.1, len(tickers)), index=tickers)
    weights = weights / weights.abs().sum()  # Normalize
    
    # Calculate risk metrics
    risk_metrics = risk_model.calculate_portfolio_risk(weights)
    print("\nPortfolio Risk Metrics:")
    for key, value in risk_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate VaR and CVaR
    var_cvar = risk_model.calculate_var_cvar(weights, confidence=0.95)
    print("\nVaR/CVaR:")
    for key, value in var_cvar.items():
        print(f"  {key}: {value}")
    
    # Risk decomposition
    risk_decomp = risk_model.decompose_risk(weights)
    print("\nTop 5 Risk Contributors:")
    print(risk_decomp.head())


if __name__ == "__main__":
    test_risk_model()
