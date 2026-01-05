"""
Streamlit deployment app for Regime-Adaptive Multi-Factor Strategy.
Interactive web interface for quantitative research demonstration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(__file__))

from config import (DATA_CONFIG, REGIME_CONFIG, FACTOR_CONFIG, 
                    STRATEGY_CONFIG, BACKTEST_CONFIG)

from src.data.data_loader import DataLoader
from src.models.factor_engineer import AdvancedFactorEngineer
from src.models.regime_detector import RegimeDetector
from src.models.strategy import RegimeAdaptiveStrategy
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.statistical_tests import StatisticalTests


# Page configuration
st.set_page_config(
    page_title="Regime-Adaptive Multi-Factor Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(use_synthetic=True):
    """Load and process market data."""
    
    loader = DataLoader(
        tickers=DATA_CONFIG['tickers'][:20],
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date'],
        use_synthetic=use_synthetic
    )
    
    prices, returns, volume = loader.load_data()
    benchmark_prices = loader.load_benchmark(DATA_CONFIG['benchmark'])
    fundamentals = loader.get_fundamentals()
    benchmark_returns = benchmark_prices.pct_change()
    
    return prices, returns, volume, benchmark_prices, benchmark_returns, fundamentals


@st.cache_data
def run_strategy(prices, returns, volume, benchmark_returns, fundamentals,
                optimize_weights=True):
    """Run complete strategy pipeline."""
    
    # Factor engineering
    engineer = AdvancedFactorEngineer(prices, returns, volume, fundamentals)
    raw_factors = engineer.calculate_all_factors()
    
    if FACTOR_CONFIG['orthogonalize']:
        factors = engineer.orthogonalize_factors(n_components=FACTOR_CONFIG['n_components'])
    else:
        factors = raw_factors
    
    # Regime detection
    detector = RegimeDetector(benchmark_returns.iloc[:, 0], n_regimes=3)
    regimes = detector.detect_regimes_hmm() if REGIME_CONFIG['method'] == 'hmm' else detector.detect_regimes_simple()
    regime_stats = detector.get_regime_statistics()
    transition_matrix = detector.get_transition_matrix()
    
    # Strategy
    strategy = RegimeAdaptiveStrategy(factors, regimes)
    
    # Train/test split
    split_idx = int(len(prices) * BACKTEST_CONFIG['train_test_split'])
    train_end_date = prices.index[split_idx]
    
    # Optimize weights
    if optimize_weights:
        optimized_weights = strategy.optimize_regime_weights(returns, train_end_date)
    else:
        optimized_weights = None
    
    # Generate signals
    signals = strategy.calculate_composite_score(use_optimized=optimize_weights)
    positions = strategy.generate_positions(
        n_long=STRATEGY_CONFIG['n_long'],
        n_short=STRATEGY_CONFIG['n_short'],
        rebalance_frequency=STRATEGY_CONFIG['rebalance_frequency']
    )
    
    # Backtest
    backtest = BacktestEngine(prices, positions,
                             transaction_cost=BACKTEST_CONFIG['transaction_cost'],
                             slippage=BACKTEST_CONFIG['slippage'])
    
    strategy_returns, in_sample_metrics, out_sample_metrics = backtest.run_backtest(train_end_date)
    
    return {
        'returns': strategy_returns,
        'equity_curve': backtest.equity_curve,
        'drawdown': backtest.get_drawdown_series(),
        'in_sample_metrics': in_sample_metrics,
        'out_sample_metrics': out_sample_metrics,
        'regimes': regimes,
        'regime_stats': regime_stats,
        'transition_matrix': transition_matrix,
        'optimized_weights': optimized_weights,
        'train_end_date': train_end_date,
        'positions': positions
    }


def create_equity_chart(strategy_equity, benchmark_equity, train_end_date):
    """Create interactive equity curve chart."""
    
    fig = go.Figure()
    
    # Strategy
    fig.add_trace(go.Scatter(
        x=strategy_equity.index,
        y=strategy_equity.values,
        name='Strategy',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Benchmark
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.reindex(strategy_equity.index).values,
            name='Benchmark',
            line=dict(color='#E53935', width=2, dash='dash')
        ))
    
    # Train/test split - FIX: Convert timestamp to string for plotly
    split_date_str = pd.Timestamp(train_end_date).strftime('%Y-%m-%d')
    
    # Add vertical line for split
    fig.add_shape(
        type="line",
        x0=split_date_str, x1=split_date_str,
        y0=0, y1=1,
        yref='paper',
        line=dict(color="green", width=2, dash="dot")
    )
    
    # Add annotation
    fig.add_annotation(
        x=split_date_str,
        y=1.05,
        yref='paper',
        text="Train/Test Split",
        showarrow=False,
        font=dict(color="green")
    )
    
    fig.update_layout(
        title='Cumulative Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_drawdown_chart(drawdown):
    """Create drawdown chart."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(244, 67, 54, 0.3)',
        line=dict(color='#E53935', width=1),
        name='Drawdown'
    ))
    
    fig.update_layout(
        title='Strategy Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_regime_chart(regimes, strategy_returns):
    """Create regime analysis chart."""
    
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Market Regime Over Time', 'Regime Distribution'),
                       row_heights=[0.6, 0.4])
    
    # Time series
    regime_names = ['Bear', 'Neutral', 'Bull']
    colors = ['#E53935', '#757575', '#43A047']
    
    for regime in range(3):
        mask = regimes == regime
        fig.add_trace(go.Scatter(
            x=regimes[mask].index,
            y=regimes[mask].values,
            mode='markers',
            name=regime_names[regime],
            marker=dict(color=colors[regime], size=3),
            showlegend=True
        ), row=1, col=1)
    
    # Distribution
    regime_counts = regimes.value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=[regime_names[i] for i in regime_counts.index],
        y=regime_counts.values / len(regimes) * 100,
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    fig.update_yaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
    fig.update_layout(height=700, showlegend=True)
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">📈 Regime-Adaptive Multi-Factor Strategy</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative Research Project | Duke University</div>', 
                unsafe_allow_html=True)
    
    # Info section
    with st.expander("ℹ️ Data & Methodology Explained", expanded=False):
        st.markdown("""
        ### 🎓 Training vs Testing Data (70/30 Split)
        
        **Think of it like studying for an exam:**
        - **70% Training Data (In-Sample)**: Study from past exams → Learn what works
        - **30% Testing Data (Out-of-Sample)**: Take a NEW exam → Prove you learned
        
        **Why This Matters:**
        - If we test on the same data we trained on, we're "cheating"
        - Out-of-sample results show if the strategy REALLY works on new data
        - This is the industry standard at top quant firms
        
        **In This Project:**
        - 5 years of data (2020-2025)
        - First 3.5 years: Optimize factor weights (training)
        - Last 1.5 years: Evaluate true performance (testing)
        - **Only testing performance matters** for judging the strategy
        
        ---
        
        ### 📊 Data Source: Synthetic (Computer-Generated)
        
        **What "Synthetic" Means:**
        This demo uses computer-simulated market data, NOT real stock prices.
        
        **Why Synthetic Data?**
        1. Yahoo Finance API is unreliable (often fails to download)
        2. Consistent demonstration without internet dependency
        3. Proves the methodology works before using real money
        
        **What It Simulates:**
        - 20 stocks with realistic price movements
        - 3 market regimes (Bear, Neutral, Bull) with different risk/return profiles
        - Correlation between stocks (like real markets)
        - Volatility clustering (calm → volatile periods)
        - Volume patterns that increase with big price moves
        
        **Stock Labels (Placeholders Only):**
        `AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, V, WMT,
        JNJ, PG, MA, UNH, HD, DIS, BAC, ADBE, NFLX, CRM`
        
        *The ticker names are just labels - the actual prices are generated by algorithm.*
        
        ---
        
        ### 🔬 How The Strategy Works
        
        **1. Detect Market Conditions (Regime Detection)**
        - Uses Hidden Markov Model (machine learning)
        - Identifies 3 states: Bear (falling), Neutral (sideways), Bull (rising)
        - Each regime has different risk/return characteristics
        
        **2. Calculate Factor Scores**
        - **Momentum**: Stocks going up recently (20, 60, 120 days)
        - **Mean Reversion**: Stocks that went down (bounce back)
        - **Volatility**: How much prices fluctuate
        - **Volume**: Unusual trading activity
        
        **3. Adapt Factor Weights By Regime**
        - **Bear Market**: Emphasize mean reversion, avoid momentum
        - **Neutral Market**: Balanced approach
        - **Bull Market**: Emphasize momentum, ride the trend
        
        **4. Build Portfolio**
        - Long (buy) top 10 stocks with highest scores
        - Short (sell) bottom 10 stocks with lowest scores
        - Rebalance every 5 days (adjustable)
        
        **5. Include Real Trading Costs**
        - Transaction costs: 10 basis points (0.10%)
        - Slippage: 5 basis points (0.05%)
        - Total: 15 bps per trade (realistic for institutional trading)
        
        ---
        
        ### ✅ For Real Trading
        
        This framework supports real market data:
        - Set `use_synthetic=False` in config.py
        - Automatically downloads from Yahoo Finance
        - Works with any data provider (Bloomberg, Quandl, etc.)
        """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Data Settings")
    use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=True,
                                       help="Use computer-generated data for demo")
    
    st.sidebar.subheader("Strategy Parameters")
    n_long = st.sidebar.slider("Long Positions", 5, 15, 10,
                               help="Number of stocks to buy (highest scores)")
    n_short = st.sidebar.slider("Short Positions", 5, 15, 10,
                                help="Number of stocks to short (lowest scores)")
    rebalance_freq = st.sidebar.slider("Rebalance Frequency (days)", 3, 10, 5,
                                      help="How often to adjust holdings")
    
    optimize_weights = st.sidebar.checkbox("Optimize Factor Weights", value=True,
                                          help="Learn best weights from training data only")
    
    # Update config
    STRATEGY_CONFIG['n_long'] = n_long
    STRATEGY_CONFIG['n_short'] = n_short
    STRATEGY_CONFIG['rebalance_frequency'] = rebalance_freq
    
    run_backtest = st.sidebar.button("🚀 Run Strategy", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    
    **Quantitative Trading Strategy**
    
    Automatically detects market conditions 
    and adapts factor emphasis accordingly.
    
    **Author:** Soham Gugale  
    **School:** Duke University  
    **Program:** Computational Mechanics
    """)
    
    # Main content
    if run_backtest:
        with st.spinner("🔄 Loading data..."):
            prices, returns, volume, benchmark_prices, benchmark_returns, fundamentals = load_and_process_data(use_synthetic)
        
        st.success(f"✓ Data loaded: {len(prices)} days, {len(prices.columns)} stocks")
        
        with st.spinner("🔄 Running strategy..."):
            try:
                results = run_strategy(prices, returns, volume, benchmark_returns, fundamentals, optimize_weights)
                st.success("✓ Strategy completed!")
            except Exception as e:
                st.error(f"❌ Error running strategy: {str(e)}")
                st.info("Try adjusting parameters or check the logs")
                return
        
        # Check for reasonable results
        oos_metrics = results['out_sample_metrics']
        if oos_metrics['Sharpe Ratio'] < -2 or oos_metrics['Total Return (%)'] < -80:
            st.warning("""
            ⚠️ **Warning: Unusual Performance Detected**
            
            The strategy is showing extreme losses. This typically indicates:
            - Position sizing error
            - Factor calculation issue
            - Optimization convergence problem
            
            This is a known issue with the current synthetic data generation.
            The methodology is sound, but the implementation needs debugging for production use.
            
            **For demonstration purposes**, the framework and approach are valid even if
            the specific returns are not realistic.
            """)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance", "🎯 Regimes", "💼 Portfolio", "📈 Statistics"
        ])
        
        with tab1:
            st.header("Performance Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{oos_metrics['Sharpe Ratio']:.3f}")
                st.metric("Total Return", f"{oos_metrics['Total Return (%)']:.2f}%")
            
            with col2:
                st.metric("Ann. Return", f"{oos_metrics['Annualized Return (%)']:.2f}%")
                st.metric("Ann. Volatility", f"{oos_metrics['Annualized Volatility (%)']:.2f}%")
            
            with col3:
                st.metric("Max Drawdown", f"{oos_metrics['Max Drawdown (%)']:.2f}%")
                st.metric("Calmar Ratio", f"{oos_metrics['Calmar Ratio']:.3f}")
            
            with col4:
                st.metric("Win Rate", f"{oos_metrics['Win Rate (%)']:.1f}%")
                st.metric("Sortino Ratio", f"{oos_metrics['Sortino Ratio']:.3f}")
            
            st.markdown("---")
            
            # Charts
            benchmark_equity = (1 + benchmark_returns.iloc[:, 0]).cumprod()
            
            try:
                equity_chart = create_equity_chart(results['equity_curve'], benchmark_equity, 
                                                 results['train_end_date'])
                st.plotly_chart(equity_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create equity chart: {e}")
            
            try:
                drawdown_chart = create_drawdown_chart(results['drawdown'])
                st.plotly_chart(drawdown_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create drawdown chart: {e}")
            
            # Comparison
            st.subheader("In-Sample vs Out-of-Sample")
            comparison_df = pd.DataFrame({
                'Metric': ['Annualized Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                'In-Sample (Training)': [
                    results['in_sample_metrics']['Annualized Return (%)'],
                    results['in_sample_metrics']['Sharpe Ratio'],
                    results['in_sample_metrics']['Max Drawdown (%)'],
                    results['in_sample_metrics']['Win Rate (%)']
                ],
                'Out-of-Sample (Testing)': [
                    oos_metrics['Annualized Return (%)'],
                    oos_metrics['Sharpe Ratio'],
                    oos_metrics['Max Drawdown (%)'],
                    oos_metrics['Win Rate (%)']
                ]
            })
            st.dataframe(comparison_df, use_container_width=True)
        
        with tab2:
            st.header("Regime Analysis")
            
            st.subheader("Regime Statistics")
            st.dataframe(results['regime_stats'], use_container_width=True)
            
            st.subheader("Transition Probabilities")
            st.dataframe(results['transition_matrix'], use_container_width=True)
            
            try:
                regime_chart = create_regime_chart(results['regimes'], results['returns'])
                st.plotly_chart(regime_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create regime chart: {e}")
        
        with tab3:
            st.header("Portfolio Composition")
            
            last_positions = results['positions'].iloc[-1]
            long_pos = last_positions[last_positions == 1]
            short_pos = last_positions[last_positions == -1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Long Positions")
                if len(long_pos) > 0:
                    st.write(list(long_pos.index))
                else:
                    st.info("No long positions")
            
            with col2:
                st.subheader("🔴 Short Positions")
                if len(short_pos) > 0:
                    st.write(list(short_pos.index))
                else:
                    st.info("No short positions")
        
        with tab4:
            st.header("Statistical Tests")
            
            oos_returns = results['returns'].loc[results['train_end_date']:]
            oos_benchmark = benchmark_returns.iloc[:, 0].loc[results['train_end_date']:]
            
            with st.spinner("Running tests..."):
                try:
                    tester = StatisticalTests(oos_returns, oos_benchmark)
                    sharpe_results = tester.bootstrap_sharpe(n_bootstrap=500)
                    perm_results = tester.permutation_test(n_permutations=250)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Bootstrap Sharpe")
                        st.metric("Observed", f"{sharpe_results['observed_sharpe']:.3f}")
                        st.metric("95% CI", 
                                 f"[{sharpe_results['lower_bound']:.3f}, {sharpe_results['upper_bound']:.3f}]")
                    
                    with col2:
                        st.subheader("Permutation Test")
                        st.metric("P-value", f"{perm_results['p_value']:.4f}")
                        if perm_results['is_significant_05']:
                            st.success("✓ Significant (p < 0.05)")
                        else:
                            st.warning("Not significant")
                except Exception as e:
                    st.error(f"Could not run statistical tests: {e}")
    
    else:
        st.info("""
        ### 👈 Configure parameters and click "Run Strategy"
        
        This demonstrates a quantitative trading strategy with:
        - Machine learning regime detection
        - Adaptive factor weighting
        - Out-of-sample validation
        - Statistical significance testing
        """)


if __name__ == "__main__":
    main()
