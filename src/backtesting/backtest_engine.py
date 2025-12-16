"""
Backtesting engine with transaction costs and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class BacktestEngine:
    """Backtests trading strategies with realistic assumptions."""
    
    def __init__(self, prices: pd.DataFrame, positions: pd.DataFrame,
                 transaction_cost: float = 0.001, slippage: float = 0.0005):
        """
        Initialize backtest engine.
        
        Args:
            prices: DataFrame of stock prices
            positions: DataFrame of positions (-1, 0, 1 or continuous)
            transaction_cost: Transaction cost as fraction (default: 10 bps)
            slippage: Slippage as fraction (default: 5 bps)
        """
        self.prices = prices
        self.positions = positions
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.returns = None
        self.equity_curve = None
        self.metrics = None
        
    def run_backtest(self) -> pd.DataFrame:
        """
        Run backtest and calculate returns.
        
        Returns:
            DataFrame with portfolio returns
        """
        # Calculate daily returns for each stock
        stock_returns = self.prices.pct_change()
        
        # Align positions and returns
        aligned_positions = self.positions.shift(1)  # Use previous day's positions
        
        # Calculate gross returns
        gross_returns = (aligned_positions * stock_returns).sum(axis=1)
        
        # Calculate position changes (for transaction costs)
        position_changes = self.positions.diff().abs()
        
        # Calculate transaction costs
        turnover = position_changes.sum(axis=1)
        costs = turnover * (self.transaction_cost + self.slippage)
        
        # Net returns
        net_returns = gross_returns - costs
        
        # Create equity curve
        self.equity_curve = (1 + net_returns).cumprod()
        self.returns = net_returns
        
        return net_returns
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        returns = self.returns.dropna()
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] - 1) * 100
        annualized_return = (np.power(self.equity_curve.iloc[-1], 
                                     252 / len(returns)) - 1) * 100
        annualized_vol = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown metrics
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        self.metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate (%)': win_rate,
            'Number of Trades': len(returns)
        }
        
        return self.metrics
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown series."""
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        return drawdown