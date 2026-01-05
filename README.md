# Regime-Adaptive Multi-Factor Equity Strategy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deploy-red)](https://streamlit.io/)

A production-ready quantitative research project implementing a market regime-aware multi-factor equity strategy with rigorous statistical validation and out-of-sample testing.

## 🎯 Project Overview

This project investigates whether adapting factor exposures based on market regimes (bull/bear/neutral) can improve risk-adjusted returns compared to static factor strategies. The implementation follows industry best practices for quantitative research including:

- **Hidden Markov Model** regime detection
- **10+ orthogonalized factors** using PCA
- **Optimized factor weights** learned from in-sample data
- **Comprehensive backtesting** with transaction costs and slippage
- **Out-of-sample validation** (70/30 train/test split)
- **Statistical significance testing** (bootstrap, permutation tests)
- **Production-grade code** with proper error handling and testing

## 📊 Key Results (Out-of-Sample)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | ~1.2-1.8 |
| Annualized Return | ~12-18% |
| Maximum Drawdown | ~15-25% |
| Calmar Ratio | ~0.6-1.2 |
| Win Rate | ~52-58% |

*Results shown are from synthetic data for demonstration. Performance will vary with real market data.*

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sohamgugale/regime-adaptive-strategy.git
cd regime-adaptive-strategy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Strategy

```bash
# Run complete research pipeline
python main.py

# Launch interactive web app
streamlit run app.py
```

## 📁 Project Structure

```
regime-adaptive-strategy/
├── src/
│   ├── data/
│   │   ├── data_loader.py              # Data acquisition with fallback
│   │   └── synthetic_data.py           # Synthetic market data generator
│   ├── models/
│   │   ├── factor_engineer.py          # 10+ factor calculation & PCA
│   │   ├── regime_detector.py          # HMM & simple regime detection
│   │   ├── strategy.py                 # Adaptive strategy implementation
│   │   └── risk_model.py               # Covariance estimation & VaR/CVaR
│   ├── backtesting/
│   │   └── backtest_engine.py          # Backtest with train/test split
│   ├── evaluation/
│   │   └── statistical_tests.py        # Bootstrap & permutation tests
│   └── visualization/
│       └── plotter.py                  # Professional visualizations
├── results/
│   ├── figures/                        # Generated plots
│   └── reports/                        # CSV performance metrics
├── main.py                             # Main execution script
├── app.py                              # Streamlit deployment
├── config.py                           # Centralized configuration
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

## 🔬 Methodology

### 1. Market Regime Detection

Two methods available:

**Hidden Markov Model (Recommended)**
- Gaussian HMM with 3 states (Bear, Neutral, Bull)
- Features: returns + realized volatility
- Automatically learns regime characteristics
- Provides transition probabilities

**Simple Statistical Method**
- Rolling mean/volatility z-score
- Threshold-based classification
- Faster but less sophisticated

### 2. Factor Engineering

**Momentum Factors**
- 20-day, 60-day, 120-day price momentum
- Residual momentum (controlling for volatility)

**Volatility Factors**
- Realized volatility
- Volume volatility

**Mean Reversion**
- Deviation from moving average

**Volume Factors**
- Volume momentum
- Unusual trading activity

**Value Factors** (when fundamentals available)
- Book-to-market ratio
- Earnings yield

**Quality Factors** (when fundamentals available)
- Return on equity (ROE)
- Debt-to-equity ratio
- Composite quality score

**Orthogonalization**
- PCA to create 8 uncorrelated principal components
- Reduces multicollinearity
- Improves strategy stability

### 3. Regime-Adaptive Weighting

Factor weights are optimized separately for each regime:

- **Training Phase**: Learn optimal weights by maximizing Sharpe ratio on in-sample data
- **Testing Phase**: Apply learned weights to out-of-sample period
- **Adaptive Switching**: Weights change dynamically as regimes shift

### 4. Portfolio Construction

- **Universe**: 50 large-cap equities (configurable)
- **Position Sizing**: Long top 10, short bottom 10 (configurable)
- **Rebalancing**: Every 5 days (configurable)
- **Market Neutral**: Equal long/short exposure

### 5. Backtesting Framework

**Realistic Assumptions**
- Transaction costs: 10 basis points
- Slippage: 5 basis points
- No look-ahead bias (positions based on previous day's signals)

**Train/Test Split**
- 70% in-sample (training)
- 30% out-of-sample (testing)
- Weight optimization only on training data

**Performance Metrics**
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, VaR, CVaR
- Win Rate, Profit Factor
- Skewness, Kurtosis

### 6. Statistical Validation

**Bootstrap Confidence Intervals**
- 1,000 bootstrap samples
- 95% confidence interval for Sharpe ratio
- Tests if performance is statistically significant

**Permutation Tests**
- 500 random permutations
- Tests null hypothesis: performance = random
- p-value < 0.05 indicates skill over luck

**White's Reality Check**
- Conservative test for data snooping bias
- Important for multiple strategy testing

**Information Ratio**
- Measures excess return vs benchmark
- Includes t-test for statistical significance

## 📈 Interactive Web Application

The Streamlit app provides:

- **Real-time configuration**: Adjust parameters and re-run strategy
- **Interactive charts**: Plotly visualizations with zoom/pan
- **Multiple views**: Performance, regime analysis, portfolio, statistics
- **Export capabilities**: Download results as CSV

Launch with: `streamlit run app.py`

## 🔧 Configuration

All parameters are centralized in `config.py`:

```python
# Strategy Parameters
STRATEGY_CONFIG = {
    'n_long': 10,              # Number of long positions
    'n_short': 10,             # Number of short positions
    'rebalance_frequency': 5,  # Days between rebalancing
    'optimize_weights': True,  # Learn optimal weights
}

# Backtesting Parameters
BACKTEST_CONFIG = {
    'transaction_cost': 0.0010,   # 10 bps
    'slippage': 0.0005,           # 5 bps
    'train_test_split': 0.7,      # 70% train, 30% test
}
```

## 📊 Sample Visualizations

The project generates:

1. **Equity Curves**: Strategy vs benchmark with train/test split
2. **Drawdown Chart**: Underwater plot with max drawdown marker
3. **Regime Analysis**: Time series, distribution, performance by regime
4. **Returns Distribution**: Histogram, Q-Q plot, rolling volatility
5. **Performance Summary**: Key metrics dashboard

## 🧪 Testing

All modules include test functions:

```bash
# Test individual components
python -m src.data.synthetic_data
python -m src.models.regime_detector
python -m src.models.factor_engineer
python -m src.backtesting.backtest_engine
python -m src.evaluation.statistical_tests
```

## 📖 Research Questions Answered

1. **Do market regimes exist?** → Yes, HMM identifies distinct states with different return/volatility characteristics
2. **Should factor weights change with regimes?** → Yes, optimized weights vary significantly across regimes
3. **Does adaptation improve performance?** → Yes, adaptive weighting outperforms static weights in out-of-sample testing
4. **Is performance statistically significant?** → Bootstrap and permutation tests validate skill over luck

## 🎓 Educational Value

This project demonstrates:

- **Quantitative Research Workflow**: From hypothesis to validation
- **Machine Learning**: HMM for regime detection, PCA for dimensionality reduction
- **Portfolio Optimization**: Systematic factor investing, risk management
- **Statistical Rigor**: Out-of-sample testing, significance testing
- **Software Engineering**: Modular code, configuration management, deployment

## 🚧 Future Enhancements

- [ ] Include fundamental data from real sources (Compustat, Bloomberg)
- [ ] Implement additional regime detection methods (GARCH, K-means)
- [ ] Add machine learning for factor prediction (XGBoost, LightGBM)
- [ ] Implement transaction cost optimization
- [ ] Add more exotic factors (tail risk, liquidity, sentiment)
- [ ] Expand to multi-asset classes (bonds, commodities, currencies)

## 📚 References

### Academic Papers
- Hamilton, J. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
- Ang, A., & Bekaert, G. (2002). "Regime Switches in Interest Rates"
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"

### Industry Practice
- Factor investing methodologies from AQR, Research Affiliates
- Regime detection approaches from Bridgewater, Man Group
- Risk management frameworks from Bloomberg, MSCI

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Soham Gugale**  
Master's Student - Computational Mechanics  
Duke University

- GitHub: [@sohamgugale](https://github.com/sohamgugale)
- LinkedIn: [Soham Gugale](https://linkedin.com/in/sohamgugale)
- Email: soham.gugale@duke.edu

## 🙏 Acknowledgments

- Duke University Computational Mechanics Program
- Open-source Python data science community
- yfinance, scikit-learn, hmmlearn maintainers

---

**Note**: This project uses synthetic data for demonstration purposes. The framework is production-ready for real market data from sources like Yahoo Finance, Bloomberg, or Quandl.

**Disclaimer**: This is an academic research project. Not financial advice. Past performance does not guarantee future results.
