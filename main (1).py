"""
Main execution script for Regime-Adaptive Multi-Factor Strategy.
Complete quantitative research pipeline with out-of-sample testing.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import (DATA_CONFIG, REGIME_CONFIG, FACTOR_CONFIG, 
                    STRATEGY_CONFIG, BACKTEST_CONFIG, RISK_CONFIG, 
                    STATS_CONFIG, OUTPUT_PATHS)

from src.data.data_loader import DataLoader
from src.models.factor_engineer import AdvancedFactorEngineer
from src.models.regime_detector import RegimeDetector
from src.models.strategy import RegimeAdaptiveStrategy
from src.models.risk_model import RiskModel
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.statistical_tests import StatisticalTests
from src.visualization.plotter import StrategyVisualizer


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)


def main():
    """Execute complete quantitative research pipeline."""
    
    print("\n" + "█"*80)
    print("REGIME-ADAPTIVE MULTI-FACTOR EQUITY STRATEGY".center(80))
    print("Quantitative Research Project - Duke University".center(80))
    print("█"*80)
    
    # Create output directories
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ========================================
    # STEP 1: DATA LOADING
    # ========================================
    print_section("STEP 1: DATA ACQUISITION & PREPROCESSING")
    
    loader = DataLoader(
        tickers=DATA_CONFIG['tickers'],
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date'],
        use_synthetic=DATA_CONFIG['use_synthetic']
    )
    
    prices, returns, volume = loader.load_data()
    benchmark_prices = loader.load_benchmark(DATA_CONFIG['benchmark'])
    fundamentals = loader.get_fundamentals()
    
    benchmark_returns = benchmark_prices.pct_change()
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Universe: {len(DATA_CONFIG['tickers'])} stocks")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Total observations: {len(prices):,} days")
    print(f"  Using {'synthetic' if DATA_CONFIG['use_synthetic'] else 'real'} market data")
    
    # ========================================
    # STEP 2: FACTOR ENGINEERING
    # ========================================
    print_section("STEP 2: MULTI-FACTOR FEATURE ENGINEERING")
    
    engineer = AdvancedFactorEngineer(prices, returns, volume, fundamentals)
    raw_factors = engineer.calculate_all_factors()
    
    print(f"\n✓ Calculated {len(raw_factors)} raw factors")
    for name in raw_factors.keys():
        print(f"  • {name}")
    
    # Orthogonalize factors
    if FACTOR_CONFIG['orthogonalize']:
        factors = engineer.orthogonalize_factors(
            n_components=FACTOR_CONFIG['n_components']
        )
        print(f"\n✓ Orthogonalized to {len(factors)} principal components")
    else:
        factors = raw_factors
    
    # Factor statistics
    factor_stats = engineer.get_factor_statistics()
    factor_stats.to_csv(f"{OUTPUT_PATHS['reports']}factor_statistics.csv")
    
    # ========================================
    # STEP 3: REGIME DETECTION
    # ========================================
    print_section("STEP 3: MARKET REGIME DETECTION")
    
    detector = RegimeDetector(
        returns=benchmark_returns.iloc[:, 0],
        n_regimes=REGIME_CONFIG['n_regimes']
    )
    
    if REGIME_CONFIG['method'] == 'hmm':
        regimes = detector.detect_regimes_hmm(
            n_iter=REGIME_CONFIG['hmm_n_iter'],
            covariance_type=REGIME_CONFIG['hmm_covariance_type']
        )
    else:
        regimes = detector.detect_regimes_simple(
            window=REGIME_CONFIG['simple_window']
        )
    
    regime_stats = detector.get_regime_statistics()
    transition_matrix = detector.get_transition_matrix()
    
    print("\n✓ Regime detection completed")
    print("\nRegime Statistics:")
    print(regime_stats.round(3))
    print("\nTransition Probabilities:")
    print(transition_matrix.round(3))
    
    regime_stats.to_csv(f"{OUTPUT_PATHS['reports']}regime_statistics.csv")
    transition_matrix.to_csv(f"{OUTPUT_PATHS['reports']}transition_matrix.csv")
    
    # ========================================
    # STEP 4: STRATEGY CONSTRUCTION
    # ========================================
    print_section("STEP 4: REGIME-ADAPTIVE STRATEGY IMPLEMENTATION")
    
    strategy = RegimeAdaptiveStrategy(factors, regimes)
    
    # Calculate train/test split date
    split_idx = int(len(prices) * BACKTEST_CONFIG['train_test_split'])
    train_end_date = prices.index[split_idx]
    
    print(f"\n✓ Train/test split: {BACKTEST_CONFIG['train_test_split']*100:.0f}% / {(1-BACKTEST_CONFIG['train_test_split'])*100:.0f}%")
    print(f"  In-sample period: {prices.index[0].date()} to {train_end_date.date()}")
    print(f"  Out-of-sample period: {train_end_date.date()} to {prices.index[-1].date()}")
    
    # Optimize factor weights on in-sample data
    if STRATEGY_CONFIG['optimize_weights']:
        optimized_weights = strategy.optimize_regime_weights(returns, train_end_date)
        
        print("\n✓ Optimized regime-specific factor weights:")
        weights_df = pd.DataFrame(optimized_weights).T
        print(weights_df.round(3))
        weights_df.to_csv(f"{OUTPUT_PATHS['reports']}optimized_weights.csv")
    
    # Generate signals and positions
    signals = strategy.calculate_composite_score(use_optimized=True)
    positions = strategy.generate_positions(
        n_long=STRATEGY_CONFIG['n_long'],
        n_short=STRATEGY_CONFIG['n_short'],
        rebalance_frequency=STRATEGY_CONFIG['rebalance_frequency']
    )
    
    print(f"\n✓ Generated trading signals")
    print(f"  Long positions: {STRATEGY_CONFIG['n_long']}")
    print(f"  Short positions: {STRATEGY_CONFIG['n_short']}")
    print(f"  Rebalance frequency: {STRATEGY_CONFIG['rebalance_frequency']} days")
    
    # ========================================
    # STEP 5: BACKTESTING
    # ========================================
    print_section("STEP 5: BACKTEST EXECUTION")
    
    backtest = BacktestEngine(
        prices=prices,
        positions=positions,
        transaction_cost=BACKTEST_CONFIG['transaction_cost'],
        slippage=BACKTEST_CONFIG['slippage'],
        initial_capital=BACKTEST_CONFIG['initial_capital']
    )
    
    strategy_returns, in_sample_metrics, out_sample_metrics = backtest.run_backtest(train_end_date)
    
    print("\n📊 IN-SAMPLE PERFORMANCE (Training)")
    print("-" * 80)
    for metric, value in in_sample_metrics.items():
        print(f"{metric:.<50} {value:>10.3f}")
    
    print("\n📊 OUT-OF-SAMPLE PERFORMANCE (Testing)")
    print("-" * 80)
    for metric, value in out_sample_metrics.items():
        print(f"{metric:.<50} {value:>10.3f}")
    
    # Benchmark metrics
    benchmark_equity = (1 + benchmark_returns.iloc[:, 0]).cumprod()
    
    # Split benchmark metrics
    bench_in_sample = benchmark_returns.iloc[:, 0].loc[:train_end_date]
    bench_out_sample = benchmark_returns.iloc[:, 0].loc[train_end_date:]
    
    bench_in_sharpe = (bench_in_sample.mean() / bench_in_sample.std() * np.sqrt(252)) if len(bench_in_sample) > 0 else 0
    bench_out_sharpe = (bench_out_sample.mean() / bench_out_sample.std() * np.sqrt(252)) if len(bench_out_sample) > 0 else 0
    
    print("\n📈 BENCHMARK PERFORMANCE")
    print("-" * 80)
    print(f"{'In-Sample Sharpe':.<50} {bench_in_sharpe:>10.3f}")
    print(f"{'Out-of-Sample Sharpe':.<50} {bench_out_sharpe:>10.3f}")
    
    # Save metrics
    pd.DataFrame([in_sample_metrics]).to_csv(
        f"{OUTPUT_PATHS['reports']}in_sample_metrics.csv", index=False
    )
    pd.DataFrame([out_sample_metrics]).to_csv(
        f"{OUTPUT_PATHS['reports']}out_sample_metrics.csv", index=False
    )
    
    # ========================================
    # STEP 6: RISK ANALYSIS
    # ========================================
    print_section("STEP 6: RISK MODEL & PORTFOLIO ANALYTICS")
    
    risk_model = RiskModel(
        returns=returns,
        estimation_window=RISK_CONFIG['estimation_window']
    )
    
    # Estimate covariance
    cov_matrix = risk_model.estimate_covariance(method=RISK_CONFIG['shrinkage_method'])
    
    # Get last portfolio weights
    last_positions = positions.iloc[-1]
    portfolio_risk = risk_model.calculate_portfolio_risk(last_positions)
    var_cvar = risk_model.calculate_var_cvar(
        last_positions,
        confidence=RISK_CONFIG['var_confidence']
    )
    
    print("\n✓ Risk analysis completed")
    print("\nPortfolio Risk Metrics:")
    for key, value in portfolio_risk.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nValue at Risk:")
    for key, value in var_cvar.items():
        print(f"  {key}: {value}")
    
    # ========================================
    # STEP 7: STATISTICAL VALIDATION
    # ========================================
    print_section("STEP 7: STATISTICAL SIGNIFICANCE TESTING")
    
    # Out-of-sample returns for testing
    oos_returns = strategy_returns.loc[train_end_date:]
    oos_benchmark = benchmark_returns.iloc[:, 0].loc[train_end_date:]
    
    tester = StatisticalTests(oos_returns, oos_benchmark)
    
    # Run all tests
    test_results = tester.run_all_tests(
        n_bootstrap=STATS_CONFIG['n_bootstrap'],
        n_permutations=STATS_CONFIG['n_permutations']
    )
    
    print("\n✓ Statistical tests completed (out-of-sample)")
    print("\nKey Results:")
    print(f"  Sharpe Ratio: {test_results['sharpe_observed_sharpe']:.3f}")
    print(f"  95% CI: [{test_results['sharpe_lower_bound']:.3f}, {test_results['sharpe_upper_bound']:.3f}]")
    print(f"  Permutation p-value: {test_results['perm_p_value']:.4f}")
    print(f"  Statistically significant: {test_results['sharpe_is_significant']}")
    
    test_results.to_csv(f"{OUTPUT_PATHS['reports']}statistical_tests.csv")
    
    # ========================================
    # STEP 8: VISUALIZATION
    # ========================================
    print_section("STEP 8: GENERATING VISUALIZATIONS")
    
    visualizer = StrategyVisualizer()
    
    print("\nGenerating plots...")
    
    # 1. Equity curves
    visualizer.plot_equity_curves(
        backtest.equity_curve,
        benchmark_equity,
        in_sample_end=train_end_date,
        save_path=f"{OUTPUT_PATHS['figures']}equity_curve.png"
    )
    print("  ✓ Equity curve")
    
    # 2. Drawdown
    drawdown = backtest.get_drawdown_series()
    visualizer.plot_drawdown(
        drawdown,
        save_path=f"{OUTPUT_PATHS['figures']}drawdown.png"
    )
    print("  ✓ Drawdown chart")
    
    # 3. Regime analysis
    visualizer.plot_regime_analysis(
        regimes,
        strategy_returns,
        save_path=f"{OUTPUT_PATHS['figures']}regime_analysis.png"
    )
    print("  ✓ Regime analysis")
    
    # 4. Returns distribution
    visualizer.plot_returns_distribution(
        strategy_returns,
        benchmark_returns.iloc[:, 0],
        save_path=f"{OUTPUT_PATHS['figures']}returns_distribution.png"
    )
    print("  ✓ Returns distribution")
    
    # 5. Performance summary
    visualizer.plot_performance_summary(
        out_sample_metrics,
        save_path=f"{OUTPUT_PATHS['figures']}performance_summary.png"
    )
    print("  ✓ Performance summary")
    
    # ========================================
    # STEP 9: EXPORT RESULTS
    # ========================================
    print_section("STEP 9: EXPORTING RESULTS")
    
    # Complete backtest results
    results_df = pd.DataFrame({
        'Date': backtest.equity_curve.index,
        'Strategy_Equity': backtest.equity_curve.values,
        'Benchmark_Equity': benchmark_equity.reindex(backtest.equity_curve.index).values,
        'Strategy_Returns': strategy_returns.values,
        'Drawdown': drawdown.values,
        'Regime': regimes.reindex(backtest.equity_curve.index).values
    })
    results_df.to_csv(f"{OUTPUT_PATHS['reports']}complete_backtest_results.csv", index=False)
    
    print("\n✓ All results exported")
    print(f"  Reports: {OUTPUT_PATHS['reports']}")
    print(f"  Figures: {OUTPUT_PATHS['figures']}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "█"*80)
    print("✅ RESEARCH PIPELINE COMPLETED SUCCESSFULLY".center(80))
    print("█"*80)
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY".center(80))
    print("="*80)
    
    print(f"\nStrategy Performance (Out-of-Sample):")
    print(f"  • Annualized Return: {out_sample_metrics['Annualized Return (%)']:.2f}%")
    print(f"  • Annualized Volatility: {out_sample_metrics['Annualized Volatility (%)']:.2f}%")
    print(f"  • Sharpe Ratio: {out_sample_metrics['Sharpe Ratio']:.3f}")
    print(f"  • Maximum Drawdown: {out_sample_metrics['Max Drawdown (%)']:.2f}%")
    print(f"  • Calmar Ratio: {out_sample_metrics['Calmar Ratio']:.3f}")
    
    print(f"\nBenchmark Performance (Out-of-Sample):")
    print(f"  • Sharpe Ratio: {bench_out_sharpe:.3f}")
    
    print(f"\nStatistical Validation:")
    print(f"  • Sharpe 95% CI: [{test_results['sharpe_lower_bound']:.3f}, {test_results['sharpe_upper_bound']:.3f}]")
    print(f"  • Permutation Test p-value: {test_results['perm_p_value']:.4f}")
    print(f"  • Result: {'STATISTICALLY SIGNIFICANT' if test_results['sharpe_is_significant'] else 'NOT SIGNIFICANT'}")
    
    print(f"\nData Source: {'Synthetic (demonstration)' if DATA_CONFIG['use_synthetic'] else 'Real market data'}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {os.path.abspath('results/')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
