import os
import traceback
from typing import List, Optional
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
        fees: float = 0.0005,
        rebalance_freq: str = 'QE'
    ) -> None:
        """Initialize the backtest runner with configuration settings."""
        self.data_file: str = data_file
        self.output_dir: str = output_dir
        self.initial_capital: float = initial_capital
        self.fees: float = fees
        self.rebalance_freq: str = rebalance_freq

    def run(
        self, 
        start_date: Optional[str] = '2017-01-01', 
        end_date: Optional[str] = '2026-02-05',
        benchmarks: Optional[List[str]] = None,
        rebalance_fn = RebalanceAlgorithms.permanent_portfolio_rebalance,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.5,
        model_type: str = "kmeans",
    ) -> None:
        """Execute the backtest and generate results."""
        if benchmarks is None:
            benchmarks = ['Nasdaq100', 'GoldIndex', 'US30Y', 'US3M']

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
                rebalance_freq=self.rebalance_freq, 
                fees=self.fees,
                rebalance_fn=rebalance_fn,
                rebalance_interval_days=None,
                asset_cols=None,
                use_trend_model=use_trend_model,
                model_lookback_days=model_lookback_days,
                model_threshold=model_threshold,
                model_type=model_type,
            )
            
            # 3. Output Performance Statistics
            stats = engine.get_performance_stats()
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
        fees=0.0005,
        rebalance_freq='QE'
    )
    
    # 示例1：不开启趋势模型（保持原行为）
    # runner.run(
    #     start_date='2017-01-01',
    #     end_date='2026-02-05',
    # )

    # 示例2：开启趋势模型占位逻辑
    runner.run(
        start_date='2017-01-01',
        end_date='2026-02-05',
        use_trend_model=True,
        model_lookback_days=60,
        model_threshold=0.6,
        model_type="hmm",
    )
