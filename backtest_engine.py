"""
Backtesting engine with in-sample/out-of-sample testing and comprehensive performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    """Comprehensive backtesting engine with train/test split."""
    
    def __init__(self, prices: pd.DataFrame, positions: pd.DataFrame,
                 transaction_cost: float = 0.001, slippage: float = 0.0005,
                 initial_capital: float = 1_000_000):
        """
        Initialize backtest engine.
        
        Args:
            prices: Price data
            positions: Position data
            transaction_cost: Transaction cost (default: 10 bps)
            slippage: Slippage (default: 5 bps)
            initial_capital: Initial capital
        """
        self.prices = prices
        self.positions = positions
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_capital = initial_capital
        
        self.returns = None
        self.equity_curve = None
        self.metrics = {}
        
    def run_backtest(self, split_date: pd.Timestamp = None) -> Tuple[pd.Series, Dict, Dict]:
        """
        Run backtest with optional train/test split.
        
        Args:
            split_date: Date to split in-sample/out-of-sample
            
        Returns:
            Tuple of (returns, in_sample_metrics, out_of_sample_metrics)
        """
        # Calculate stock returns
        stock_returns = self.prices.pct_change()
        
        # Align positions (use previous day's positions)
        aligned_positions = self.positions.shift(1)
        
        # Calculate gross returns
        gross_returns = (aligned_positions * stock_returns).sum(axis=1)
        
        # Calculate turnover and transaction costs
        position_changes = self.positions.diff().abs()
        turnover = position_changes.sum(axis=1)
        costs = turnover * (self.transaction_cost + self.slippage)
        
        # Net returns
        net_returns = gross_returns - costs
        
        # Store returns
        self.returns = net_returns
        
        # Calculate equity curve
        self.equity_curve = (1 + net_returns).cumprod()
        
        # Calculate metrics
        if split_date is not None:
            # Split into in-sample and out-of-sample
            in_sample_returns = net_returns.loc[:split_date]
            out_sample_returns = net_returns.loc[split_date:]
            
            in_sample_metrics = self.calculate_metrics(in_sample_returns)
            out_sample_metrics = self.calculate_metrics(out_sample_returns)
            
            return net_returns, in_sample_metrics, out_sample_metrics
        else:
            # Full sample metrics
            metrics = self.calculate_metrics(net_returns)
            return net_returns, metrics, {}
    
    def calculate_metrics(self, returns: pd.Series = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Return series (uses self.returns if None)
            
        Returns:
            Dictionary of metrics
        """
        if returns is None:
            returns = self.returns
        
        returns = returns.dropna()
        
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Equity curve
        equity = (1 + returns).cumprod()
        
        # Basic metrics
        total_return = (equity.iloc[-1] - 1) * 100
        n_years = len(returns) / 252
        annualized_return = (np.power(equity.iloc[-1], 1/n_years) - 1) * 100 if n_years > 0 else 0
        annualized_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_vol / 100
        sortino_ratio = (returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        # Drawdown metrics
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() * 100 if len(wins) > 0 else 0
        avg_loss = losses.mean() * 100 if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Best/worst day
        best_day = returns.max() * 100
        worst_day = returns.min() * 100
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR and CVaR (95%)
        var_95 = -np.percentile(returns, 5) * 100
        cvar_95 = -returns[returns <= -var_95/100].mean() * 100
        
        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate (%)': win_rate,
            'Avg Win (%)': avg_win,
            'Avg Loss (%)': avg_loss,
            'Win/Loss Ratio': win_loss_ratio,
            'Best Day (%)': best_day,
            'Worst Day (%)': worst_day,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR 95% (%)': var_95,
            'CVaR 95% (%)': cvar_95,
            'Number of Trades': len(returns)
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {key: 0.0 for key in [
            'Total Return (%)', 'Annualized Return (%)', 'Annualized Volatility (%)',
            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Calmar Ratio',
            'Win Rate (%)', 'Avg Win (%)', 'Avg Loss (%)', 'Win/Loss Ratio',
            'Best Day (%)', 'Worst Day (%)', 'Skewness', 'Kurtosis',
            'VaR 95% (%)', 'CVaR 95% (%)', 'Number of Trades'
        ]}
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown series."""
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        return drawdown
    
    def get_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            window: Rolling window in days
            
        Returns:
            DataFrame with rolling metrics
        """
        if self.returns is None:
            raise ValueError("Must run backtest first")
        
        returns = self.returns.dropna()
        
        rolling_return = returns.rolling(window).mean() * 252 * 100
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        rolling_sharpe = rolling_return / rolling_vol
        
        rolling_metrics = pd.DataFrame({
            'Rolling Return (%)': rolling_return,
            'Rolling Volatility (%)': rolling_vol,
            'Rolling Sharpe': rolling_sharpe
        })
        
        return rolling_metrics
    
    def get_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        if self.returns is None:
            raise ValueError("Must run backtest first")
        
        monthly_returns = self.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        return monthly_returns
    
    def get_yearly_returns(self) -> pd.Series:
        """Calculate yearly returns."""
        if self.returns is None:
            raise ValueError("Must run backtest first")
        
        yearly_returns = self.returns.resample('Y').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        return yearly_returns
    
    def calculate_turnover_analysis(self) -> Dict[str, float]:
        """Analyze portfolio turnover."""
        position_changes = self.positions.diff().abs()
        daily_turnover = position_changes.sum(axis=1)
        
        avg_daily_turnover = daily_turnover.mean()
        avg_annual_turnover = avg_daily_turnover * 252
        
        # Cost impact
        total_costs = (daily_turnover * (self.transaction_cost + self.slippage)).sum()
        avg_daily_cost = total_costs / len(daily_turnover) * 100
        
        return {
            'Avg Daily Turnover': avg_daily_turnover,
            'Avg Annual Turnover': avg_annual_turnover,
            'Total Transaction Costs (%)': total_costs * 100,
            'Avg Daily Cost (bps)': avg_daily_cost * 100
        }


def test_backtest_engine():
    """Test the backtest engine."""
    from ..data.synthetic_data import SyntheticMarketGenerator
    
    print("Testing Backtest Engine")
    print("="*50)
    
    # Generate data
    generator = SyntheticMarketGenerator(n_assets=20, n_days=500, seed=42)
    tickers = [f'STOCK_{i:02d}' for i in range(20)]
    prices, _, _, _ = generator.generate_complete_dataset(tickers)
    
    # Generate random positions
    positions = pd.DataFrame(
        np.random.choice([-1, 0, 1], size=prices.shape, p=[0.1, 0.8, 0.1]),
        index=prices.index,
        columns=prices.columns
    )
    
    # Run backtest
    backtest = BacktestEngine(prices, positions)
    
    # Split date
    split_date = prices.index[int(len(prices) * 0.7)]
    
    returns, in_sample, out_sample = backtest.run_backtest(split_date)
    
    print("\nIn-Sample Metrics:")
    for key, value in in_sample.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nOut-of-Sample Metrics:")
    for key, value in out_sample.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    test_backtest_engine()
