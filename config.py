"""
Configuration file for Regime-Adaptive Multi-Factor Strategy.
Centralizes all parameters for reproducibility and easy tuning.
"""

from datetime import datetime, timedelta

# Data Configuration - UPDATED TO USE REAL DATA
DATA_CONFIG = {
    'tickers': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
        'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE',
        'NFLX', 'CRM', 'CSCO', 'PFE', 'INTC', 'CMCSA', 'VZ', 'T', 'MRK',
        'ABT', 'NKE', 'TMO', 'COST', 'AVGO', 'LLY', 'ORCL', 'DHR', 'TXN',
        'NEE', 'PM', 'UNP', 'MDT', 'HON', 'LOW', 'QCOM', 'UPS', 'AMD',
        'AMGN', 'BMY', 'RTX', 'C', 'SBUX'
    ],
    'benchmark': 'SPY',
    'start_date': (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d'),
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'use_synthetic': False,  # CHANGED: Use real Yahoo Finance data
}

# Regime Detection Configuration
REGIME_CONFIG = {
    'n_regimes': 3,
    'method': 'simple',  # CHANGED: Use 'simple' instead of 'hmm' for stability
    'hmm_n_iter': 100,
    'hmm_covariance_type': 'full',
    'simple_window': 63,
    'vol_window': 21,
}

# Factor Configuration
FACTOR_CONFIG = {
    'momentum_windows': [20, 60, 120],
    'mean_reversion_window': 20,
    'volatility_window': 20,
    'volume_window': 20,
    'value_factors': False,  # CHANGED: Disable (no fundamental data)
    'quality_factors': False,  # CHANGED: Disable (no fundamental data)
    'orthogonalize': True,
    'n_components': 6,  # CHANGED: Reduced from 8 (fewer factors without fundamentals)
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'n_long': 10,
    'n_short': 10,
    'max_position_size': 0.1,
    'rebalance_frequency': 5,
    'optimize_weights': False,  # CHANGED: Disable optimization for stability
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 1_000_000,
    'transaction_cost': 0.0010,
    'slippage': 0.0005,
    'train_test_split': 0.7,
}

# Risk Model Configuration
RISK_CONFIG = {
    'estimation_window': 252,
    'shrinkage_method': 'ledoit_wolf',
    'var_confidence': 0.95,
    'cvar_confidence': 0.95,
}

# Statistical Testing Configuration
STATS_CONFIG = {
    'n_bootstrap': 1000,
    'confidence_level': 0.95,
    'n_permutations': 500,
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8-whitegrid',
}

# Output Paths
OUTPUT_PATHS = {
    'figures': 'results/figures/',
    'reports': 'results/reports/',
    'models': 'results/models/',
}
