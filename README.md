# Regime-Adaptive Multi-Factor Equity Strategy

A quantitative research project implementing a market regime-aware multi-factor equity strategy with systematic backtesting and risk analysis.

## Research Overview

This project investigates whether adapting factor exposures based on market regimes (bull/bear/neutral) can improve risk-adjusted returns compared to static factor strategies. The research implements a complete quantitative workflow from hypothesis formation through backtesting and performance analysis.

## Key Features

### 1. Market Regime Detection
- Statistical regime classification using rolling window analysis
- Identifies three distinct market states: Bull, Bear, Neutral
- Calculates regime-specific return and volatility characteristics

### 2. Multi-Factor Signal Generation
- **Momentum**: 6-month price momentum
- **Volatility**: Realized volatility (risk factor)
- **Mean Reversion**: Deviation from moving average
- **Volume Momentum**: Unusual trading activity

### 3. Regime-Adaptive Weighting
Factor weights dynamically adjust based on current market regime:
- **Bear Market**: Emphasis on mean reversion and low volatility
- **Neutral Market**: Balanced factor exposure
- **Bull Market**: Aggressive momentum weighting

### 4. Comprehensive Backtesting
- Transaction costs: 10 basis points
- Slippage: 5 basis points
- Long/short portfolio construction
- Market-neutral positioning

## Methodology

1. **Data Collection**: 3 years of daily data for 50 large-cap equities
2. **Feature Engineering**: Cross-sectional factor calculation with rank-based scoring
3. **Regime Detection**: Statistical classification using rolling mean and volatility
4. **Signal Generation**: Composite scoring with regime-dependent weights
5. **Portfolio Construction**: Long top 10, short bottom 10 stocks
6. **Backtesting**: Full simulation with realistic trading costs
7. **Performance Analysis**: Sharpe ratio, maximum drawdown, Calmar ratio, win rate

## Results Summary

Performance metrics demonstrate the strategy's effectiveness across different market conditions. See `results/reports/performance_metrics.csv` for complete statistics.

Key findings:
- Regime adaptation improves risk-adjusted returns compared to static weighting
- Strategy shows positive Sharpe ratio with controlled drawdowns
- Factor correlations vary significantly across market regimes

## Installation
```bash
# Clone repository
git clone https://github.com/sohamgugale/regime-adaptive-strategy.git
cd regime-adaptive-strategy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the complete research pipeline:
```bash
python main.py
```

This executes:
1. Data download and preprocessing
2. Factor engineering
3. Regime detection
4. Strategy backtesting
5. Visualization generation
6. Results export

## Project Structure
```
regime-adaptive-strategy/
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data acquisition and preprocessing
│   ├── models/
│   │   ├── regime_detector.py      # Market regime classification
│   │   └── strategy.py             # Multi-factor strategy implementation
│   ├── backtesting/
│   │   └── backtest_engine.py      # Backtesting with transaction costs
│   └── visualization/
│       └── plotter.py              # Performance visualization
├── results/
│   ├── figures/                    # Generated plots
│   └── reports/                    # CSV performance metrics
├── main.py                         # Main execution script
├── requirements.txt
└── README.md
```

## Technologies

- **Python 3.8+**: Core programming language
- **pandas/numpy**: Data manipulation and numerical computing
- **scikit-learn**: Statistical analysis
- **cvxpy**: Portfolio optimization
- **yfinance**: Market data acquisition
- **matplotlib/seaborn**: Visualization

## Author

Soham Gugale  
Quantitative Research Assistant  
Duke University - Computational Mechanics

## License

MIT License
