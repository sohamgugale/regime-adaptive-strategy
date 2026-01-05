"""
Streamlit deployment app for Regime-Adaptive Multi-Factor Strategy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
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


st.set_page_config(
    page_title="Regime-Adaptive Strategy",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; padding: 1rem 0;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(use_synthetic=False):
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
def run_strategy(prices, returns, volume, benchmark_returns, fundamentals, optimize_weights=False):
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
    regimes = detector.detect_regimes_simple()  # Use simple method for stability
    regime_stats = detector.get_regime_statistics()
    transition_matrix = detector.get_transition_matrix()
    
    # Strategy
    strategy = RegimeAdaptiveStrategy(factors, regimes)
    
    # Train/test split
    split_idx = int(len(prices) * BACKTEST_CONFIG['train_test_split'])
    train_end_date = prices.index[split_idx]
    
    # Optimize weights only if requested
    if optimize_weights:
        try:
            optimized_weights = strategy.optimize_regime_weights(returns, train_end_date)
        except:
            optimized_weights = None
            st.warning("Weight optimization failed, using default weights")
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
    """Create equity curve chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_equity.index, y=strategy_equity.values,
        name='Strategy', line=dict(color='#1E88E5', width=2)
    ))
    
    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity.reindex(strategy_equity.index).values,
            name='Benchmark', line=dict(color='#E53935', width=2, dash='dash')
        ))
    
    # Add train/test split line
    split_str = pd.Timestamp(train_end_date).strftime('%Y-%m-%d')
    fig.add_shape(
        type="line", x0=split_str, x1=split_str, y0=0, y1=1, yref='paper',
        line=dict(color="green", width=2, dash="dot")
    )
    fig.add_annotation(
        x=split_str, y=1.05, yref='paper', text="Train/Test Split",
        showarrow=False, font=dict(color="green")
    )
    
    fig.update_layout(
        title='Cumulative Performance', xaxis_title='Date', yaxis_title='Return',
        hovermode='x unified', height=500
    )
    return fig


def create_drawdown_chart(drawdown):
    """Create drawdown chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill='tozeroy', fillcolor='rgba(244, 67, 54, 0.3)',
        line=dict(color='#E53935', width=1), name='Drawdown'
    ))
    fig.update_layout(
        title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown (%)',
        hovermode='x unified', height=400
    )
    return fig


def create_regime_chart(regimes, strategy_returns):
    """Create regime analysis chart - FIXED."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Market Regime Over Time', 'Regime Distribution'),
        row_heights=[0.6, 0.4]
    )
    
    regime_names = ['Bear', 'Neutral', 'Bull']
    colors = ['#E53935', '#757575', '#43A047']
    
    # Time series - FIXED: Ensure regimes are integers
    regimes_int = regimes.astype(int)
    
    for regime in range(3):
        mask = regimes_int == regime
        fig.add_trace(go.Scatter(
            x=regimes_int[mask].index,
            y=[regime] * mask.sum(),  # FIXED: Use integer list instead of series values
            mode='markers',
            name=regime_names[regime],
            marker=dict(color=colors[regime], size=3)
        ), row=1, col=1)
    
    # Distribution
    regime_counts = regimes_int.value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=[regime_names[i] for i in regime_counts.index],
        y=regime_counts.values / len(regimes_int) * 100,
        marker_color=[colors[i] for i in regime_counts.index],
        showlegend=False
    ), row=2, col=1)
    
    fig.update_yaxes(title_text="Regime", row=1, col=1, tickvals=[0, 1, 2], ticktext=regime_names)
    fig.update_yaxes(title_text="Percentage (%)", row=2, col=1)
    fig.update_layout(height=700, showlegend=True)
    
    return fig


def main():
    """Main app."""
    
    st.markdown('<div class="main-header">📈 Regime-Adaptive Multi-Factor Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative Research Project | Duke University</div>', unsafe_allow_html=True)
    
    # Info section
    with st.expander("ℹ️ Data & Methodology", expanded=False):
        st.markdown("""
        ### 📊 Data Source: Real Market Data (Yahoo Finance)
        
        **Live Stock Data:**
        - Downloads actual historical prices for 20 large-cap stocks
        - Includes AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, and more
        - 3 years of daily trading data
        - Real volume and price movements
        
        ### 🎓 70/30 Training/Testing Split
        
        **Why This Matters:**
        - **70% Training**: Optimize factor weights on 2022-2024 data
        - **30% Testing**: Evaluate on unseen 2024-2025 data
        - Prevents overfitting (looking at the answers before the test)
        - Industry standard for validating trading strategies
        
        **Only out-of-sample (testing) results count for judging strategy performance!**
        
        ### 🔬 Methodology
        
        **1. Regime Detection**
        - Identifies Bull/Neutral/Bear markets using rolling statistics
        - Each regime has different risk/return characteristics
        
        **2. Multi-Factor Scoring**
        - Momentum (20, 60, 120 days)
        - Volatility (risk factor)
        - Mean reversion (buy oversold)
        - Volume momentum
        
        **3. Adaptive Weighting**
        - Factor weights change with market regime
        - Bear: Emphasize mean reversion
        - Bull: Emphasize momentum
        
        **4. Portfolio Construction**
        - Long top 10 stocks (highest scores)
        - Short bottom 10 stocks (lowest scores)
        - Market-neutral design
        - Rebalance every 5 days
        
        **5. Realistic Costs**
        - Transaction costs: 10 bps
        - Slippage: 5 bps
        - Total: 15 bps per trade
        """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Data Settings")
    use_synthetic = st.sidebar.checkbox(
        "Use Synthetic Data", 
        value=False,
        help="Uncheck to use REAL Yahoo Finance data (recommended)"
    )
    
    st.sidebar.subheader("Strategy Parameters")
    n_long = st.sidebar.slider("Long Positions", 5, 15, 10)
    n_short = st.sidebar.slider("Short Positions", 5, 15, 10)
    rebalance_freq = st.sidebar.slider("Rebalance (days)", 3, 10, 5)
    optimize_weights = st.sidebar.checkbox(
        "Optimize Weights", 
        value=False,
        help="Experimental - may be unstable"
    )
    
    STRATEGY_CONFIG['n_long'] = n_long
    STRATEGY_CONFIG['n_short'] = n_short
    STRATEGY_CONFIG['rebalance_frequency'] = rebalance_freq
    
    run_backtest = st.sidebar.button("🚀 Run Strategy", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Author:** Soham Gugale  
        **School:** Duke University  
        **Program:** Computational Mechanics
    """)
    
    # Main content
    if run_backtest:
        with st.spinner("📥 Downloading market data from Yahoo Finance..."):
            try:
                prices, returns, volume, benchmark_prices, benchmark_returns, fundamentals = load_and_process_data(use_synthetic)
                data_source = "Synthetic" if use_synthetic else "Yahoo Finance"
                st.success(f"✓ Data loaded from {data_source}: {len(prices)} days, {len(prices.columns)} stocks")
            except Exception as e:
                st.error(f"❌ Data loading failed: {e}")
                st.info("Try enabling 'Use Synthetic Data' or check internet connection")
                return
        
        with st.spinner("🔄 Running strategy..."):
            try:
                results = run_strategy(prices, returns, volume, benchmark_returns, fundamentals, optimize_weights)
                st.success("✓ Strategy completed!")
            except Exception as e:
                st.error(f"❌ Strategy failed: {e}")
                st.exception(e)
                return
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Regimes", "💼 Portfolio", "📈 Statistics"])
        
        with tab1:
            st.header("Performance Analysis")
            
            oos = results['out_sample_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe Ratio", f"{oos['Sharpe Ratio']:.3f}")
                st.metric("Total Return", f"{oos['Total Return (%)']:.2f}%")
            with col2:
                st.metric("Ann. Return", f"{oos['Annualized Return (%)']:.2f}%")
                st.metric("Ann. Volatility", f"{oos['Annualized Volatility (%)']:.2f}%")
            with col3:
                st.metric("Max Drawdown", f"{oos['Max Drawdown (%)']:.2f}%")
                st.metric("Calmar Ratio", f"{oos['Calmar Ratio']:.3f}")
            with col4:
                st.metric("Win Rate", f"{oos['Win Rate (%)']:.1f}%")
                st.metric("Sortino Ratio", f"{oos['Sortino Ratio']:.3f}")
            
            st.markdown("---")
            
            benchmark_equity = (1 + benchmark_returns.iloc[:, 0]).cumprod()
            
            try:
                equity_chart = create_equity_chart(results['equity_curve'], benchmark_equity, results['train_end_date'])
                st.plotly_chart(equity_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")
            
            try:
                drawdown_chart = create_drawdown_chart(results['drawdown'])
                st.plotly_chart(drawdown_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")
            
            st.subheader("In-Sample vs Out-of-Sample")
            comparison = pd.DataFrame({
                'Metric': ['Ann. Return (%)', 'Sharpe Ratio', 'Max DD (%)', 'Win Rate (%)'],
                'In-Sample': [
                    results['in_sample_metrics']['Annualized Return (%)'],
                    results['in_sample_metrics']['Sharpe Ratio'],
                    results['in_sample_metrics']['Max Drawdown (%)'],
                    results['in_sample_metrics']['Win Rate (%)']
                ],
                'Out-of-Sample': [
                    oos['Annualized Return (%)'],
                    oos['Sharpe Ratio'],
                    oos['Max Drawdown (%)'],
                    oos['Win Rate (%)']
                ]
            })
            st.dataframe(comparison, use_container_width=True)
        
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
                st.error(f"Regime chart error: {e}")
                st.exception(e)
        
        with tab3:
            st.header("Portfolio Composition")
            
            # Get last valid positions
            last_positions = results['positions'].iloc[-1]
            long_pos = last_positions[last_positions > 0]
            short_pos = last_positions[last_positions < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Long Positions")
                if len(long_pos) > 0:
                    long_df = pd.DataFrame({
                        'Ticker': long_pos.index,
                        'Weight': [f"{w:.2%}" for w in long_pos.values]
                    })
                    st.dataframe(long_df, use_container_width=True)
                else:
                    st.info("No long positions currently")
            
            with col2:
                st.subheader("🔴 Short Positions")
                if len(short_pos) > 0:
                    short_df = pd.DataFrame({
                        'Ticker': short_pos.index,
                        'Weight': [f"{w:.2%}" for w in short_pos.values]
                    })
                    st.dataframe(short_df, use_container_width=True)
                else:
                    st.info("No short positions currently")
        
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
                        st.metric("95% CI", f"[{sharpe_results['lower_bound']:.3f}, {sharpe_results['upper_bound']:.3f}]")
                    
                    with col2:
                        st.subheader("Permutation Test")
                        st.metric("P-value", f"{perm_results['p_value']:.4f}")
                        if perm_results['is_significant_05']:
                            st.success("✓ Significant (p < 0.05)")
                        else:
                            st.warning("Not significant")
                except Exception as e:
                    st.error(f"Test error: {e}")
    
    else:
        st.info("""
        ### 👈 Configure parameters and click "Run Strategy"
        
        **Uses Real Market Data from Yahoo Finance**
        - 20 large-cap stocks (AAPL, MSFT, GOOGL, etc.)
        - 3 years of historical prices
        - Live data download on each run
        """)


if __name__ == "__main__":
    main()
