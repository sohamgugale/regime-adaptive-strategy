"""
Statistical tests for strategy performance validation.
Includes bootstrap confidence intervals and permutation tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """Statistical validation for trading strategies."""
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        """
        Initialize statistical tests.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        
    def bootstrap_sharpe(self, n_bootstrap: int = 1000, 
                        confidence: float = 0.95) -> Dict[str, float]:
        """
        Bootstrap confidence interval for Sharpe ratio.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with Sharpe statistics
        """
        print(f"Running bootstrap with {n_bootstrap} samples...")
        
        sharpe_samples = []
        n_obs = len(self.returns)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(self.returns.values, size=n_obs, replace=True)
            
            # Calculate Sharpe
            if sample.std() > 0:
                sharpe = sample.mean() / sample.std() * np.sqrt(252)
                sharpe_samples.append(sharpe)
        
        sharpe_samples = np.array(sharpe_samples)
        
        # Calculate statistics
        observed_sharpe = self.returns.mean() / self.returns.std() * np.sqrt(252)
        mean_sharpe = sharpe_samples.mean()
        std_sharpe = sharpe_samples.std()
        
        # Confidence interval
        alpha = 1 - confidence
        lower = np.percentile(sharpe_samples, alpha/2 * 100)
        upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)
        
        return {
            'observed_sharpe': observed_sharpe,
            'mean_sharpe': mean_sharpe,
            'std_sharpe': std_sharpe,
            'confidence_level': confidence,
            'lower_bound': lower,
            'upper_bound': upper,
            'is_significant': lower > 0  # Positive at confidence level
        }
    
    def permutation_test(self, n_permutations: int = 500) -> Dict[str, float]:
        """
        Permutation test for strategy performance.
        Tests if strategy performance is due to skill or luck.
        
        Args:
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with test results
        """
        print(f"Running permutation test with {n_permutations} permutations...")
        
        # Observed Sharpe ratio
        observed_sharpe = self.returns.mean() / self.returns.std() * np.sqrt(252)
        
        # Generate null distribution by permuting returns
        null_sharpes = []
        
        for _ in range(n_permutations):
            # Randomly permute returns
            permuted = np.random.permutation(self.returns.values)
            
            # Calculate Sharpe
            if permuted.std() > 0:
                sharpe = permuted.mean() / permuted.std() * np.sqrt(252)
                null_sharpes.append(sharpe)
        
        null_sharpes = np.array(null_sharpes)
        
        # Calculate p-value (one-sided test: strategy > random)
        p_value = (null_sharpes >= observed_sharpe).mean()
        
        return {
            'observed_sharpe': observed_sharpe,
            'null_mean': null_sharpes.mean(),
            'null_std': null_sharpes.std(),
            'p_value': p_value,
            'is_significant_05': p_value < 0.05,
            'is_significant_01': p_value < 0.01
        }
    
    def information_ratio_test(self) -> Dict[str, float]:
        """
        Calculate Information Ratio vs benchmark with confidence interval.
        
        Returns:
            Dictionary with IR statistics
        """
        if self.benchmark_returns is None:
            print("Warning: No benchmark provided")
            return {}
        
        # Align returns
        common_dates = self.returns.index.intersection(self.benchmark_returns.index)
        strategy_ret = self.returns.loc[common_dates]
        benchmark_ret = self.benchmark_returns.loc[common_dates]
        
        # Excess returns
        excess_returns = strategy_ret - benchmark_ret
        
        # Information ratio
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # T-statistic
        n = len(excess_returns)
        t_stat = excess_returns.mean() / (excess_returns.std() / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        # Confidence interval (95%)
        std_error = excess_returns.std() / np.sqrt(n)
        margin = 1.96 * std_error * np.sqrt(252)
        
        return {
            'information_ratio': ir,
            'excess_return_annual': excess_returns.mean() * 252 * 100,
            'tracking_error': excess_returns.std() * np.sqrt(252) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': (excess_returns.mean() - margin) * 100,
            'ci_upper': (excess_returns.mean() + margin) * 100,
            'is_significant': p_value < 0.05
        }
    
    def white_reality_check(self, n_bootstrap: int = 1000) -> Dict[str, float]:
        """
        White's Reality Check for data snooping bias.
        Conservative test for strategy performance.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Test results
        """
        print(f"Running White's Reality Check with {n_bootstrap} samples...")
        
        # Observed performance
        observed_mean = self.returns.mean() * 252
        
        # Bootstrap null distribution (centered at zero)
        centered_returns = self.returns - self.returns.mean()
        null_means = []
        
        n_obs = len(centered_returns)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(centered_returns.values, size=n_obs, replace=True)
            null_means.append(sample.mean() * 252)
        
        null_means = np.array(null_means)
        
        # P-value
        p_value = (null_means >= observed_mean).mean()
        
        return {
            'observed_return': observed_mean * 100,
            'p_value': p_value,
            'is_significant_05': p_value < 0.05,
            'is_significant_01': p_value < 0.01
        }
    
    def test_normality(self) -> Dict[str, float]:
        """
        Test if returns follow normal distribution.
        
        Returns:
            Dictionary with normality test results
        """
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        
        # Shapiro-Wilk test
        sw_stat, sw_pvalue = stats.shapiro(self.returns)
        
        # Descriptive statistics
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()
        
        return {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'shapiro_wilk_stat': sw_stat,
            'shapiro_wilk_pvalue': sw_pvalue,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'is_normal_jb': jb_pvalue > 0.05,
            'is_normal_sw': sw_pvalue > 0.05
        }
    
    def test_autocorrelation(self, max_lags: int = 20) -> Dict[str, any]:
        """
        Test for return autocorrelation.
        
        Args:
            max_lags: Maximum number of lags to test
            
        Returns:
            Dictionary with autocorrelation results
        """
        # Calculate autocorrelations
        autocorrs = [self.returns.autocorr(lag=i) for i in range(1, max_lags+1)]
        
        # Ljung-Box test
        lb_stat, lb_pvalue = stats.acorr_ljungbox(self.returns, lags=max_lags, return_df=False)
        
        return {
            'autocorrelations': autocorrs,
            'ljung_box_stat': lb_stat[-1],
            'ljung_box_pvalue': lb_pvalue[-1],
            'has_autocorrelation': lb_pvalue[-1] < 0.05
        }
    
    def run_all_tests(self, n_bootstrap: int = 1000, 
                     n_permutations: int = 500) -> pd.DataFrame:
        """
        Run all statistical tests.
        
        Args:
            n_bootstrap: Bootstrap samples
            n_permutations: Permutation samples
            
        Returns:
            DataFrame with all test results
        """
        print("\nRunning comprehensive statistical tests...")
        print("="*50)
        
        results = {}
        
        # Bootstrap Sharpe
        sharpe_results = self.bootstrap_sharpe(n_bootstrap)
        for key, value in sharpe_results.items():
            results[f'sharpe_{key}'] = value
        
        # Permutation test
        perm_results = self.permutation_test(n_permutations)
        for key, value in perm_results.items():
            results[f'perm_{key}'] = value
        
        # Information ratio (if benchmark available)
        if self.benchmark_returns is not None:
            ir_results = self.information_ratio_test()
            for key, value in ir_results.items():
                results[f'ir_{key}'] = value
        
        # White's reality check
        wrc_results = self.white_reality_check(n_bootstrap)
        for key, value in wrc_results.items():
            results[f'wrc_{key}'] = value
        
        # Normality test
        norm_results = self.test_normality()
        for key, value in norm_results.items():
            results[f'norm_{key}'] = value
        
        # Autocorrelation test
        autocorr_results = self.test_autocorrelation()
        results['autocorr_lb_stat'] = autocorr_results['ljung_box_stat']
        results['autocorr_lb_pvalue'] = autocorr_results['ljung_box_pvalue']
        results['autocorr_significant'] = autocorr_results['has_autocorrelation']
        
        print("✓ All tests completed")
        
        return pd.Series(results)


def test_statistical_tests():
    """Test the statistical testing module."""
    print("Testing Statistical Tests")
    print("="*50)
    
    # Generate synthetic returns
    np.random.seed(42)
    n_days = 500
    
    # Strategy returns with positive Sharpe
    strategy_returns = pd.Series(
        np.random.normal(0.0005, 0.01, n_days),
        index=pd.date_range(end='2024-01-01', periods=n_days, freq='B')
    )
    
    # Benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.008, n_days),
        index=strategy_returns.index
    )
    
    # Create tester
    tester = StatisticalTests(strategy_returns, benchmark_returns)
    
    # Run tests
    all_results = tester.run_all_tests(n_bootstrap=500, n_permutations=250)
    
    print("\nKey Results:")
    print("-"*50)
    print(f"Observed Sharpe: {all_results['sharpe_observed_sharpe']:.3f}")
    print(f"Sharpe 95% CI: [{all_results['sharpe_lower_bound']:.3f}, {all_results['sharpe_upper_bound']:.3f}]")
    print(f"Permutation p-value: {all_results['perm_p_value']:.3f}")
    print(f"Information Ratio: {all_results['ir_information_ratio']:.3f}")


if __name__ == "__main__":
    test_statistical_tests()
