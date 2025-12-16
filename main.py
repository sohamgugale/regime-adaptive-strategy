"""
Main execution script for Regime-Adaptive Multi-Factor Strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import EquityDataLoader, FactorEngineer
from models.regime_detector import RegimeDetector
from models.strategy import RegimeAdaptiveStrategy
from backtesting.backtest_engine import BacktestEngine
from visualization.plotter import StrategyVisualizer


def main():
    """Execute complete research pipeline."""
    
    print("="*80)
    print("REGIME-ADAPTIVE MULTI-FACTOR EQUITY STRATEGY")
    print("="*80)
    
    # Configuration
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
        'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE',
        'NFLX', 'CRM', 'CSCO', 'PFE', 'INTC', 'CMCSA', 'VZ', 'T', 'MRK',
        'ABT', 'NKE', 'TMO', 'COST', 'AVGO', 'LLY', 'ORCL', 'DHR', 'TXN',
        'NEE', 'PM', 'UNP', 'MDT', 'HON', 'LOW', 'QCOM', 'UPS', 'AMD',
        'AMGN', 'BMY', 'RTX', 'C', 'SBUX'
    ]
    
    benchmark_ticker = 'SPY'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"\nBacktest Period: {start_date} to {end_date}")
    print(f"Universe Size: {len(tickers)} stocks")
    
    # Data Loading
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)
    
    loader = EquityDataLoader(tickers, start_date, end_date)
    prices = loader.download_data()  # Will fallback to synthetic if needed
    returns = loader.calculate_returns()
    volume = loader.get_volume_data()
    
    # Load benchmark
    from src.data.synthetic_data import SyntheticDataGenerator
    if loader.using_synthetic:
        generator = SyntheticDataGenerator(1, len(prices), seed=42)
        benchmark_prices = generator.generate_benchmark()
    else:
        benchmark_loader = EquityDataLoader([benchmark_ticker], start_date, end_date)
        benchmark_prices = benchmark_loader.download_data()
    
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))
    
    print(f"\nData shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Tickers: {', '.join(list(prices.columns[:5]))}...")
    
    # Factor Engineering
    print("\n" + "="*80)
    print("STEP 2: FACTOR ENGINEERING")
    print("="*80)
    
    engineer = FactorEngineer(prices, returns)
    factors = engineer.build_factor_matrix(volume)
    
    print("\nFactors calculated:")
    for factor_name, factor_df in factors.items():
        valid_pct = (factor_df.notna().sum().sum() / factor_df.size) * 100
        print(f"  - {factor_name}: {valid_pct:.1f}% valid observations")
    
    # Regime Detection
    print("\n" + "="*80)
    print("STEP 3: MARKET REGIME DETECTION")
    print("="*80)
    
    regime_detector = RegimeDetector(benchmark_returns.iloc[:, 0])
    regimes = regime_detector.detect_regimes_simple(window=63)
    regime_stats = regime_detector.get_regime_statistics()
    
    print("\nRegime Statistics:")
    print(regime_stats.round(3))
    
    # Strategy Implementation
    print("\n" + "="*80)
    print("STEP 4: STRATEGY SIGNAL GENERATION")
    print("="*80)
    
    strategy = RegimeAdaptiveStrategy(factors, regimes)
    composite_scores = strategy.calculate_composite_score()
    positions = strategy.generate_positions(n_long=10, n_short=10)
    
    print(f"\nComposite scores: {composite_scores.shape}")
    print(f"Positions: {positions.shape}")
    print(f"Avg positions/day: {(positions != 0).sum(axis=1).mean():.1f}")
    
    # Backtesting
    print("\n" + "="*80)
    print("STEP 5: BACKTEST EXECUTION")
    print("="*80)
    
    backtest = BacktestEngine(prices, positions, 
                              transaction_cost=0.0010,
                              slippage=0.0005)
    
    strategy_returns = backtest.run_backtest()
    metrics = backtest.calculate_metrics()
    
    print("\n📊 STRATEGY PERFORMANCE METRICS:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:.<40} {value:>8.2f}")
    
    # Benchmark metrics
    benchmark_equity = (1 + benchmark_returns.iloc[:, 0]).cumprod()
    benchmark_total_return = (benchmark_equity.iloc[-1] - 1) * 100
    benchmark_vol = benchmark_returns.iloc[:, 0].std() * np.sqrt(252) * 100
    benchmark_sharpe = (benchmark_returns.iloc[:, 0].mean() / 
                       benchmark_returns.iloc[:, 0].std() * np.sqrt(252))
    
    print("\n📈 BENCHMARK (SPY) METRICS:")
    print("-" * 50)
    print(f"{'Total Return (%)':.<40} {benchmark_total_return:>8.2f}")
    print(f"{'Annualized Volatility (%)':.<40} {benchmark_vol:>8.2f}")
    print(f"{'Sharpe Ratio':.<40} {benchmark_sharpe:>8.2f}")
    
    # Visualization
    print("\n" + "="*80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualizer = StrategyVisualizer()
    
    print("\nGenerating plots...")
    visualizer.plot_equity_curve(backtest.equity_curve, benchmark=benchmark_equity,
                                save_path='results/figures/equity_curve.png')
    
    drawdown = backtest.get_drawdown_series()
    visualizer.plot_drawdown(drawdown, save_path='results/figures/drawdown.png')
    
    visualizer.plot_regime_distribution(regimes, save_path='results/figures/regime_distribution.png')
    visualizer.plot_returns_distribution(strategy_returns, save_path='results/figures/returns_distribution.png')
    visualizer.plot_factor_correlation(factors, save_path='results/figures/factor_correlation.png')
    
    # Save Results
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)
    
    pd.DataFrame([metrics]).to_csv('results/reports/performance_metrics.csv', index=False)
    regime_stats.to_csv('results/reports/regime_statistics.csv')
    
    results_df = pd.DataFrame({
        'Date': backtest.equity_curve.index,
        'Strategy_Equity': backtest.equity_curve.values,
        'Benchmark_Equity': benchmark_equity.reindex(backtest.equity_curve.index).values,
        'Strategy_Returns': strategy_returns.values,
        'Drawdown': drawdown.values
    })
    results_df.to_csv('results/reports/backtest_results.csv', index=False)
    
    print("\n✓ Performance metrics saved")
    print("✓ Regime statistics saved")
    print("✓ Backtest results saved")
    
    print("\n" + "="*80)
    print("✅ RESEARCH PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults: results/")
    print(f"Figures: results/figures/")
    print(f"Reports: results/reports/")
    
    if loader.using_synthetic:
        print("\n📝 NOTE: Synthetic data used for demonstration")
        print("   (Real data download failed - common with Yahoo Finance API)")


if __name__ == "__main__":
    main()
