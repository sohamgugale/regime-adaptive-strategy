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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(use_synthetic=True):
    """Load and process market data."""
    
    loader = DataLoader(
        tickers=DATA_CONFIG['tickers'][:20],  # Limit for speed
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
    
    # Train/test split
    fig.add_vline(x=train_end_date, line_dash="dot", line_color="green",
                 annotation_text="Train/Test Split")
    
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
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Data Settings")
    use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=True,
                                       help="Use synthetic market data for demonstration")
    
    st.sidebar.subheader("Strategy Parameters")
    n_long = st.sidebar.slider("Long Positions", 5, 20, STRATEGY_CONFIG['n_long'])
    n_short = st.sidebar.slider("Short Positions", 5, 20, STRATEGY_CONFIG['n_short'])
    rebalance_freq = st.sidebar.slider("Rebalance Frequency (days)", 1, 10, 
                                      STRATEGY_CONFIG['rebalance_frequency'])
    
    optimize_weights = st.sidebar.checkbox("Optimize Factor Weights", value=True,
                                          help="Learn optimal weights from in-sample data")
    
    # Update strategy config
    STRATEGY_CONFIG['n_long'] = n_long
    STRATEGY_CONFIG['n_short'] = n_short
    STRATEGY_CONFIG['rebalance_frequency'] = rebalance_freq
    
    run_backtest = st.sidebar.button("🚀 Run Strategy", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This quantitative research project implements a regime-adaptive multi-factor equity strategy with:
    - HMM regime detection
    - 10+ orthogonalized factors
    - Optimized factor weights
    - Out-of-sample validation
    - Statistical significance testing
    
    **Author:** Soham Gugale  
    **Institution:** Duke University  
    **Program:** Computational Mechanics
    """)
    
    # Main content
    if run_backtest:
        with st.spinner("🔄 Loading data..."):
            prices, returns, volume, benchmark_prices, benchmark_returns, fundamentals = load_and_process_data(use_synthetic)
        
        st.success(f"✓ Data loaded: {len(prices)} days, {len(prices.columns)} stocks")
        
        with st.spinner("🔄 Running strategy pipeline..."):
            results = run_strategy(prices, returns, volume, benchmark_returns, fundamentals, optimize_weights)
        
        st.success("✓ Strategy execution completed!")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance", "🎯 Regime Analysis", "💼 Portfolio", "📈 Statistics", "⚙️ Technical"
        ])
        
        with tab1:
            st.header("Performance Analysis")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            oos_metrics = results['out_sample_metrics']
            
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
            
            # Equity curve
            benchmark_equity = (1 + benchmark_returns.iloc[:, 0]).cumprod()
            equity_chart = create_equity_chart(results['equity_curve'], benchmark_equity, 
                                             results['train_end_date'])
            st.plotly_chart(equity_chart, use_container_width=True)
            
            # Drawdown
            drawdown_chart = create_drawdown_chart(results['drawdown'])
            st.plotly_chart(drawdown_chart, use_container_width=True)
            
            # Comparison table
            st.subheader("In-Sample vs Out-of-Sample")
            
            comparison_df = pd.DataFrame({
                'Metric': ['Annualized Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                'In-Sample': [
                    results['in_sample_metrics']['Annualized Return (%)'],
                    results['in_sample_metrics']['Sharpe Ratio'],
                    results['in_sample_metrics']['Max Drawdown (%)'],
                    results['in_sample_metrics']['Win Rate (%)']
                ],
                'Out-of-Sample': [
                    oos_metrics['Annualized Return (%)'],
                    oos_metrics['Sharpe Ratio'],
                    oos_metrics['Max Drawdown (%)'],
                    oos_metrics['Win Rate (%)']
                ]
            })
            
            st.dataframe(comparison_df.style.format({
                'In-Sample': '{:.2f}',
                'Out-of-Sample': '{:.2f}'
            }), use_container_width=True)
        
        with tab2:
            st.header("Market Regime Analysis")
            
            # Regime stats
            st.subheader("Regime Statistics")
            st.dataframe(results['regime_stats'].style.format({
                'mean_return': '{:.3f}',
                'volatility': '{:.3f}',
                'sharpe_ratio': '{:.3f}',
                'percentage': '{:.1f}%'
            }), use_container_width=True)
            
            # Transition matrix
            st.subheader("Regime Transition Probabilities")
            st.dataframe(results['transition_matrix'].style.format('{:.2%}'), use_container_width=True)
            
            # Regime chart
            regime_chart = create_regime_chart(results['regimes'], results['returns'])
            st.plotly_chart(regime_chart, use_container_width=True)
        
        with tab3:
            st.header("Portfolio Composition")
            
            # Current positions
            last_positions = results['positions'].iloc[-1]
            long_positions = last_positions[last_positions == 1]
            short_positions = last_positions[last_positions == -1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Long Positions")
                if len(long_positions) > 0:
                    st.dataframe(pd.DataFrame({
                        'Ticker': long_positions.index,
                        'Position': ['Long'] * len(long_positions)
                    }), use_container_width=True)
                else:
                    st.info("No long positions")
            
            with col2:
                st.subheader("🔴 Short Positions")
                if len(short_positions) > 0:
                    st.dataframe(pd.DataFrame({
                        'Ticker': short_positions.index,
                        'Position': ['Short'] * len(short_positions)
                    }), use_container_width=True)
                else:
                    st.info("No short positions")
            
            # Turnover analysis
            position_changes = results['positions'].diff().abs()
            daily_turnover = position_changes.sum(axis=1)
            
            st.subheader("Portfolio Turnover")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_turnover.index,
                y=daily_turnover.values,
                mode='lines',
                name='Daily Turnover',
                line=dict(color='#1E88E5', width=1)
            ))
            
            fig.update_layout(
                title='Daily Portfolio Turnover',
                xaxis_title='Date',
                yaxis_title='Turnover',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Average Annual Turnover", f"{daily_turnover.mean() * 252:.1f}")
        
        with tab4:
            st.header("Statistical Validation")
            
            # Run statistical tests
            oos_returns = results['returns'].loc[results['train_end_date']:]
            oos_benchmark = benchmark_returns.iloc[:, 0].loc[results['train_end_date']:]
            
            with st.spinner("Running statistical tests..."):
                tester = StatisticalTests(oos_returns, oos_benchmark)
                
                # Bootstrap Sharpe
                sharpe_results = tester.bootstrap_sharpe(n_bootstrap=500)
                
                # Permutation test
                perm_results = tester.permutation_test(n_permutations=250)
            
            st.success("✓ Statistical tests completed")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bootstrap Analysis")
                st.metric("Observed Sharpe", f"{sharpe_results['observed_sharpe']:.3f}")
                st.metric("95% Confidence Interval", 
                         f"[{sharpe_results['lower_bound']:.3f}, {sharpe_results['upper_bound']:.3f}]")
                
                if sharpe_results['is_significant']:
                    st.success("✓ Sharpe ratio is statistically significant (95% CI > 0)")
                else:
                    st.warning("⚠️ Sharpe ratio not statistically significant")
            
            with col2:
                st.subheader("Permutation Test")
                st.metric("P-value", f"{perm_results['p_value']:.4f}")
                
                if perm_results['is_significant_05']:
                    st.success("✓ Performance is statistically significant (p < 0.05)")
                else:
                    st.warning("⚠️ Performance not statistically significant")
        
        with tab5:
            st.header("Technical Details")
            
            # Optimized weights
            if results['optimized_weights'] is not None:
                st.subheader("Optimized Factor Weights by Regime")
                weights_df = pd.DataFrame(results['optimized_weights'])
                st.dataframe(weights_df.style.format('{:.3f}'), use_container_width=True)
            
            # Configuration
            st.subheader("Strategy Configuration")
            
            config_data = {
                'Parameter': [
                    'Universe Size',
                    'Backtest Period',
                    'Long Positions',
                    'Short Positions',
                    'Rebalance Frequency',
                    'Transaction Cost',
                    'Slippage',
                    'Train/Test Split'
                ],
                'Value': [
                    len(prices.columns),
                    f"{len(prices)} days",
                    STRATEGY_CONFIG['n_long'],
                    STRATEGY_CONFIG['n_short'],
                    f"{STRATEGY_CONFIG['rebalance_frequency']} days",
                    f"{BACKTEST_CONFIG['transaction_cost']*10000:.0f} bps",
                    f"{BACKTEST_CONFIG['slippage']*10000:.0f} bps",
                    f"{BACKTEST_CONFIG['train_test_split']*100:.0f}% / {(1-BACKTEST_CONFIG['train_test_split'])*100:.0f}%"
                ]
            }
            
            st.dataframe(pd.DataFrame(config_data), use_container_width=True)
    
    else:
        # Welcome screen
        st.info("""
        ### 👈 Configure strategy parameters in the sidebar and click "Run Strategy"
        
        This application demonstrates a sophisticated quantitative trading strategy that:
        - Detects market regimes using Hidden Markov Models
        - Engineers 10+ orthogonalized factors
        - Optimizes factor weights for each regime
        - Validates performance with out-of-sample testing
        - Provides statistical significance testing
        
        **Note:** This demonstration uses synthetic data. The framework is ready for real market data.
        """)
        
        # Show some example visualizations or methodology
        st.markdown("### Methodology Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Data & Factors**
            - 50 large-cap equities
            - 10+ factors (momentum, value, quality, volatility)
            - PCA orthogonalization
            """)
        
        with col2:
            st.markdown("""
            **2. Regime Detection**
            - Hidden Markov Model
            - 3 regimes (Bear, Neutral, Bull)
            - Adaptive factor weights
            """)
        
        with col3:
            st.markdown("""
            **3. Validation**
            - 70/30 train/test split
            - Bootstrap confidence intervals
            - Permutation testing
            """)


if __name__ == "__main__":
    main()
