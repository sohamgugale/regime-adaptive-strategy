"""
Comprehensive visualization tools for strategy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StrategyVisualizer:
    """Creates professional visualizations for strategy analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 100):
        """Initialize visualizer."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_equity_curves(self, strategy_equity: pd.Series, 
                          benchmark_equity: pd.Series = None,
                          in_sample_end: pd.Timestamp = None,
                          save_path: str = None):
        """
        Plot strategy and benchmark equity curves with in/out-of-sample split.
        
        Args:
            strategy_equity: Strategy equity curve
            benchmark_equity: Benchmark equity curve
            in_sample_end: End of in-sample period
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot strategy
        ax.plot(strategy_equity.index, strategy_equity.values, 
                label='Strategy', linewidth=2, color='#2E86AB')
        
        # Plot benchmark
        if benchmark_equity is not None:
            aligned_benchmark = benchmark_equity.reindex(strategy_equity.index)
            ax.plot(aligned_benchmark.index, aligned_benchmark.values,
                   label='Benchmark', linewidth=2, alpha=0.7, color='#A23B72')
        
        # Mark in-sample/out-of-sample split
        if in_sample_end is not None:
            ax.axvline(in_sample_end, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Train/Test Split')
            
            # Add shaded regions
            ax.axvspan(strategy_equity.index[0], in_sample_end, 
                      alpha=0.1, color='green', label='In-Sample')
            ax.axvspan(in_sample_end, strategy_equity.index[-1], 
                      alpha=0.1, color='orange', label='Out-of-Sample')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
        ax.set_title('Strategy Performance: Equity Curve', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_drawdown(self, drawdown: pd.Series, save_path: str = None):
        """Plot underwater (drawdown) chart."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='#F18F01')
        ax.plot(drawdown.index, drawdown.values, 
                color='#C73E1D', linewidth=1.5)
        
        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.plot(max_dd_idx, max_dd_val, 'ro', markersize=10, 
               label=f'Max DD: {max_dd_val:.2f}%')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_regime_analysis(self, regimes: pd.Series, returns: pd.Series,
                            save_path: str = None):
        """Plot regime distribution and performance by regime."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        regime_names = ['Bear', 'Neutral', 'Bull']
        colors = ['#C73E1D', '#888888', '#06A77D']
        
        # 1. Regime time series
        ax1.plot(regimes.index, regimes.values, linewidth=1, color='#2E86AB')
        ax1.set_ylabel('Regime', fontsize=11, fontweight='bold')
        ax1.set_title('Market Regime Over Time', fontsize=12, fontweight='bold')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(regime_names)
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime distribution
        regime_counts = regimes.value_counts().sort_index()
        bars = ax2.bar([regime_names[i] for i in regime_counts.index], 
                      regime_counts.values / len(regimes) * 100,
                      color=colors)
        ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Regime Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 3. Returns by regime
        regime_returns = {}
        for regime in range(3):
            regime_mask = regimes == regime
            aligned_returns = returns.reindex(regimes.index)
            regime_returns[regime_names[regime]] = aligned_returns[regime_mask].values
        
        ax3.boxplot([regime_returns[name] for name in regime_names],
                   labels=regime_names,
                   patch_artist=True)
        
        # Color boxes
        for patch, color in zip(ax3.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax3.set_ylabel('Daily Returns', fontsize=11, fontweight='bold')
        ax3.set_title('Return Distribution by Regime', fontsize=12, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative returns by regime
        for regime in range(3):
            regime_mask = regimes == regime
            aligned_returns = returns.reindex(regimes.index)
            regime_ret = aligned_returns[regime_mask]
            cum_ret = (1 + regime_ret).cumprod()
            ax4.plot(cum_ret.index, cum_ret.values, 
                    label=regime_names[regime], linewidth=2, color=colors[regime])
        
        ax4.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
        ax4.set_title('Cumulative Performance by Regime', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_returns_distribution(self, returns: pd.Series, 
                                 benchmark_returns: pd.Series = None,
                                 save_path: str = None):
        """Plot distribution of returns with Q-Q plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # 1. Histogram
        ax1.hist(returns.dropna() * 100, bins=50, alpha=0.7, 
                color='#2E86AB', edgecolor='black')
        ax1.axvline(returns.mean() * 100, color='red', 
                   linestyle='--', linewidth=2,
                   label=f'Mean: {returns.mean()*100:.3f}%')
        ax1.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling volatility
        rolling_vol = returns.rolling(60).std() * np.sqrt(252) * 100
        ax3.plot(rolling_vol.index, rolling_vol.values, 
                linewidth=1.5, color='#F18F01')
        ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Volatility (%)', fontsize=11, fontweight='bold')
        ax3.set_title('60-Day Rolling Volatility', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Strategy vs Benchmark scatter
        if benchmark_returns is not None:
            common_dates = returns.index.intersection(benchmark_returns.index)
            ax4.scatter(benchmark_returns.loc[common_dates] * 100, 
                       returns.loc[common_dates] * 100,
                       alpha=0.5, s=20, color='#2E86AB')
            
            # Add regression line
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(
                benchmark_returns.loc[common_dates], 
                returns.loc[common_dates]
            )
            x_line = np.array([benchmark_returns.min(), benchmark_returns.max()]) * 100
            y_line = (slope * np.array([benchmark_returns.min(), benchmark_returns.max()]) + intercept) * 100
            ax4.plot(x_line, y_line, 'r--', linewidth=2, 
                    label=f'β={slope:.2f}, R²={r_value**2:.2f}')
            
            ax4.set_xlabel('Benchmark Return (%)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Strategy Return (%)', fontsize=11, fontweight='bold')
            ax4.set_title('Strategy vs Benchmark Returns', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_factor_analysis(self, factor_returns: Dict[str, pd.Series],
                           save_path: str = None):
        """Plot factor correlation and performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)
        
        # Create DataFrame from factors
        factor_df = pd.DataFrame(factor_returns)
        
        # 1. Correlation heatmap
        corr = factor_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax1, square=True, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Factor Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 2. Factor cumulative returns
        for col in factor_df.columns:
            cum_ret = (1 + factor_df[col]).cumprod()
            ax2.plot(cum_ret.index, cum_ret.values, label=col, linewidth=2)
        
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
        ax2.set_title('Factor Performance', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig
    
    def plot_performance_summary(self, metrics: Dict[str, float],
                                save_path: str = None):
        """Create visual summary of performance metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # Key metrics for display
        key_metrics = {
            'Annualized Return': metrics.get('Annualized Return (%)', 0),
            'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
            'Max Drawdown': abs(metrics.get('Max Drawdown (%)', 0)),
            'Win Rate': metrics.get('Win Rate (%)', 0)
        }
        
        # 1. Bar chart of key metrics
        bars = ax1.bar(range(len(key_metrics)), list(key_metrics.values()),
                      color=['#06A77D', '#2E86AB', '#C73E1D', '#F18F01'])
        ax1.set_xticks(range(len(key_metrics)))
        ax1.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
        ax1.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 2. Risk-Return scatter
        risk_metrics = ['Annualized Volatility (%)', 'Max Drawdown (%)']
        return_metrics = ['Annualized Return (%)', 'Sharpe Ratio']
        
        vol = metrics.get('Annualized Volatility (%)', 10)
        ret = metrics.get('Annualized Return (%)', 5)
        ax2.scatter([vol], [ret], s=200, c='#2E86AB', alpha=0.6)
        ax2.set_xlabel('Risk (Volatility %)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution metrics
        dist_metrics = {
            'Skewness': metrics.get('Skewness', 0),
            'Kurtosis': metrics.get('Kurtosis', 0)
        }
        ax3.bar(range(len(dist_metrics)), list(dist_metrics.values()),
               color=['#A23B72', '#F18F01'])
        ax3.set_xticks(range(len(dist_metrics)))
        ax3.set_xticklabels(list(dist_metrics.keys()))
        ax3.set_title('Return Distribution Characteristics', fontsize=12, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Text summary
        ax4.axis('off')
        summary_text = f"""
        PERFORMANCE SUMMARY
        {'='*40}
        
        Total Return: {metrics.get('Total Return (%)', 0):.2f}%
        Annualized Return: {metrics.get('Annualized Return (%)', 0):.2f}%
        Annualized Volatility: {metrics.get('Annualized Volatility (%)', 0):.2f}%
        
        Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}
        Sortino Ratio: {metrics.get('Sortino Ratio', 0):.3f}
        Calmar Ratio: {metrics.get('Calmar Ratio', 0):.3f}
        
        Max Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}%
        Win Rate: {metrics.get('Win Rate (%)', 0):.2f}%
        
        Best Day: {metrics.get('Best Day (%)', 0):.2f}%
        Worst Day: {metrics.get('Worst Day (%)', 0):.2f}%
        
        VaR (95%): {metrics.get('VaR 95% (%)', 0):.2f}%
        CVaR (95%): {metrics.get('CVaR 95% (%)', 0):.2f}%
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig


def test_visualizer():
    """Test the visualizer."""
    print("Testing Visualizer")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
    equity = (1 + returns).cumprod()
    
    visualizer = StrategyVisualizer()
    
    # Test equity curve
    fig = visualizer.plot_equity_curves(equity)
    print("✓ Equity curve plot created")
    
    plt.close('all')
    print("Tests completed")


if __name__ == "__main__":
    test_visualizer()
