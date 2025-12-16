"""
Visualization tools for strategy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List


class StrategyVisualizer:
    """Creates visualizations for strategy analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer."""
        plt.style.use('default')
        sns.set_palette("husl")
        self.fig_size = (12, 6)
        
    def plot_equity_curve(self, equity_curve: pd.Series, 
                         benchmark: pd.Series = None,
                         save_path: str = None):
        """
        Plot strategy equity curve.
        
        Args:
            equity_curve: Strategy equity curve
            benchmark: Optional benchmark equity curve
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(equity_curve.index, equity_curve.values, 
                label='Strategy', linewidth=2)
        
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values,
                   label='Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        ax.set_title('Strategy Performance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_drawdown(self, drawdown: pd.Series, save_path: str = None):
        """
        Plot drawdown series.
        
        Args:
            drawdown: Drawdown series
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, 
                color='red', linewidth=1.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_regime_distribution(self, regimes: pd.Series, 
                                 regime_names: List[str] = None,
                                 save_path: str = None):
        """
        Plot distribution of market regimes.
        
        Args:
            regimes: Series with regime labels
            regime_names: Names for regimes
            save_path: Path to save figure
        """
        if regime_names is None:
            regime_names = ['Bear', 'Neutral', 'Bull']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series plot
        ax1.plot(regimes.index, regimes.values, linewidth=1)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Regime', fontsize=12)
        ax1.set_title('Market Regime Over Time', fontsize=14, fontweight='bold')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(regime_names)
        ax1.grid(True, alpha=0.3)
        
        # Distribution plot
        regime_counts = regimes.value_counts().sort_index()
        ax2.bar([regime_names[i] for i in regime_counts.index], 
                regime_counts.values / len(regimes) * 100,
                color=['red', 'gray', 'green'])
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('Regime Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_returns_distribution(self, returns: pd.Series, 
                                  save_path: str = None):
        """
        Plot distribution of returns.
        
        Args:
            returns: Return series
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(returns.dropna() * 100, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean() * 100, color='red', 
                   linestyle='--', label=f'Mean: {returns.mean()*100:.3f}%')
        ax1.set_xlabel('Daily Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_factor_correlation(self, factors: Dict[str, pd.DataFrame],
                               date: pd.Timestamp = None,
                               save_path: str = None):
        """
        Plot correlation heatmap of factors.
        
        Args:
            factors: Dictionary of factor DataFrames
            date: Specific date to analyze (uses last date if None)
            save_path: Path to save figure
        """
        # Combine factors at specific date
        if date is None:
            date = list(factors.values())[0].index[-1]
        
        factor_data = pd.DataFrame({
            name: df.loc[date] for name, df in factors.items()
        })
        
        # Calculate correlation
        corr = factor_data.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title(f'Factor Correlation Matrix ({date.strftime("%Y-%m-%d")})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()