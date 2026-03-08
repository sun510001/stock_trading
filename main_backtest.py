import os
import traceback
from typing import List, Optional, Callable
from strategies.backtest_engine import BacktestEngine
from strategies.algorithms import RebalanceAlgorithms
from logger import logger

class BacktestRunner:
    """
    Runner class to execute portfolio backtest simulations.
    
    This class orchestrates the loading of aligned data, execution of the 
    backtest engine with a specific rebalancing algorithm, and generation 
    of performance statistics and visualizations.
    """

    def __init__(
        self, 
        data_file: str, 
        output_dir: str, 
        initial_capital: float = 100000.0,
        fees: float = 0.0005
    ) -> None:
        """Initialize the backtest runner with configuration settings."""
        self.data_file: str = data_file
        self.output_dir: str = output_dir
        self.initial_capital: float = initial_capital
        self.fees: float = fees

    def run(
        self, 
        start_date: Optional[str] = '2017-01-01', 
        end_date: Optional[str] = '2026-02-05',
        benchmarks: Optional[List[str]] = None,
        rebalance_fn: Optional[Callable] = None,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.35,
        model_type: str = "torch_mlp",
        model_path: Optional[str] = None,
        top_k: int = 3,
        target_volatility: float = 0.15,
        vol_lookback: int = 60,
        max_leverage: float = 1.0,
        safe_assets: Optional[List[str]] = None,
    ) -> None:
        """Execute the backtest and generate results.

        Args:
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD).
            benchmarks: List of benchmark column names for comparison plot.
            rebalance_fn: Callable ``(ctx: RebalanceContext) -> RebalanceResult``
                that implements the trading strategy.  Defaults to
                ``momentum_volatility_rebalance`` when None.
            use_trend_model: Whether to activate the ML trend overlay.
            model_lookback_days: Lookback window fed to the trend model.
            model_threshold: Trend score threshold for risk-off switching.
            model_type: Trend model flavour identifier string.
            model_path: Path to the persisted model file.
            top_k: Maximum assets to hold simultaneously.
            target_volatility: Target annualised portfolio volatility.
            vol_lookback: Lookback window for momentum / covariance estimation.
            max_leverage: Hard cap on gross portfolio exposure.
            safe_assets: Assets to hold during risk-off periods.
        """
        if benchmarks is None:
            benchmarks = ['Nasdaq100', 'GoldIndex', '20Y_Treasury_ETF', 'US3M', 'SP500']

        if not os.path.exists(self.data_file):
            logger.error(f"Data file not found: {self.data_file}")
            return

        try:
            # 1. Initialize Engine
            engine = BacktestEngine(self.data_file, self.initial_capital)
            
            # 2. Run simulation
            engine.run_backtest(
                start_date=start_date, 
                end_date=end_date,
                fees=self.fees,
                rebalance_fn=rebalance_fn,
                candidate_assets=None,
                use_trend_model=use_trend_model,
                model_lookback_days=model_lookback_days,
                model_threshold=model_threshold,
                model_type=model_type,
                model_path=model_path,
                top_k=top_k,
                target_volatility=target_volatility,
                vol_lookback=vol_lookback,
                max_leverage=max_leverage,
                safe_assets=safe_assets,
            )
            
            # 3. Output Performance Statistics
            stats = engine.get_performance_stats(benchmark_col="SP500")
            if not stats:
                logger.warning("No statistics generated. Please check the date range and data availability.")
                return

            print("\n" + "="*50)
            print(f"BACKTEST RESULTS ({start_date or 'Start'} to {end_date or 'End'})")
            print("="*50)
            print(f"Total Return:    {stats['Total Return']*100:.2f}%")
            print(f"CAGR:            {stats['CAGR']*100:.2f}%")
            print(f"Sharpe Ratio:    {stats['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown:    {stats['Max Drawdown']*100:.2f}%")
            print(f"Volatility:      {stats['Volatility']*100:.2f}%")
            print(f"Alpha:           {stats['Alpha']*100:.2f}%")
            print(f"Beta:            {stats['Beta']:.2f}")
            print(f"Info Ratio:      {stats['Information Ratio']:.2f}")
            print(f"Calmar Ratio:    {stats['Calmar Ratio']:.2f}")
            print(f"Max Recovery:    {stats['Max Recovery Days']} days")
            print(f"Win Rate:        {stats['Win Rate']*100:.2f}%")
            print(f"P/L Ratio:       {stats['P/L Ratio']:.2f}")
            print("="*50 + "\n")

            # 4. Generate Visualization
            saved_path = engine.plot_results(
                output_dir=self.output_dir, 
                benchmark_cols=benchmarks 
            )
            
            if saved_path:
                print(f"Visualization saved to: {saved_path}")
            
        except Exception as e:
            logger.error(f"Critical execution error: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data_processed', 'aligned_assets.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'data_processed')

    # Create and execute the runner
    runner = BacktestRunner(
        data_file=DATA_PATH,
        output_dir=OUTPUT_PATH,
        initial_capital=100000.0,
        fees=0.0005
    )
    
    # 示例1：双动量+目标波动率策略，不开趋势模型
    print("=== TEST 1: momentum_volatility (no trend model) ===")
    runner.run(
        start_date='2017-01-01',
        end_date='2026-02-05',
        rebalance_fn=RebalanceAlgorithms.momentum_volatility_rebalance,
        use_trend_model=False,
        top_k=3,
        target_volatility=0.15,
        vol_lookback=60,
    )

    # 示例2：双动量+目标波动率策略，开启 Torch MLP 趋势模型以避免熊市回撤
    print("=== TEST 2: momentum_volatility + Torch MLP model ===")
    runner.run(
        start_date='2017-01-01',
        end_date='2026-02-05',
        rebalance_fn=RebalanceAlgorithms.momentum_volatility_rebalance,
        use_trend_model=True,
        model_lookback_days=60,
        model_threshold=0.35,
        model_type="torch_mlp",
        model_path=None,  # Set to your model path, e.g. ".private_data/models/my_model.pt"
        top_k=3,
        target_volatility=0.15,
        vol_lookback=60,
        safe_assets=["20Y_Treasury_ETF", "GoldIndex", "US3M"],
    )

    # 示例3：永久组合策略（等权重）
    print("=== TEST 3: permanent_portfolio (equal weight) ===")
    runner.run(
        start_date='2017-01-01',
        end_date='2026-02-05',
        rebalance_fn=RebalanceAlgorithms.permanent_portfolio_rebalance,
        use_trend_model=False,
    )
