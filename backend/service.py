import os
import sys
import inspect
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Ensure project root is in path so we can import strategies
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from strategies.backtest_engine import BacktestEngine
from strategies.algorithms import RebalanceAlgorithms


class BacktestConfig(BaseModel):
    """Configuration model for backtest execution."""

    data_file: str = Field(
        default="data_processed/aligned_assets.csv",
        description="Path to aligned data CSV relative to project root",
    )
    output_dir: str = Field(
        default="data_processed",
        description="Directory to save backtest HTML results",
    )
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100_000.0, description="Initial portfolio capital")
    fees: float = Field(0.0005, description="Transaction fee rate")
    rebalance_freq: str = Field("QE", description="Calendar frequency (e.g., QE, YE)")
    benchmark_cols: List[str] = Field(
        default_factory=lambda: ["Nasdaq100", "GoldIndex", "US30Y", "US3M"],
        description="Benchmark columns for comparison",
    )
    rebalance_interval_days: Optional[int] = Field(
        None, description="Fixed interval in days for rebalancing",
    )
    algorithm: str = Field(
        default="permanent_portfolio_rebalance",
        description="Algorithm function name ending with '_rebalance'",
    )
    asset_cols: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of columns to use as strategy assets; "
            "if None, all columns in the data file will be used"
        ),
    )
    # Trend model-related configuration
    use_trend_model: bool = Field(
        default=False,
        description="Whether to enable the unsupervised trend model to gate rebalancing",
    )
    model_type: str = Field(
        default="kmeans",
        description="Trend model type: 'kmeans', 'autoencoder', or 'hmm'",
    )
    model_lookback_days: int = Field(
        default=60,
        description="Number of past days used as the fixed window for the trend model",
    )
    model_threshold: float = Field(
        default=0.5,
        description="Trend score threshold in [0,1]; only rebalance when score >= threshold",
    )


class BacktestResult(BaseModel):
    """Response model for backtest results."""

    stats: Dict[str, Any]
    result_html_path: str
    result_url: str
    algorithm: str


class BacktestService:
    """Service class responsible for orchestrating backtest simulations."""

    def __init__(self) -> None:
        self.algorithm_map: Dict[str, Dict[str, Any]] = self._discover_algorithms()

    def _discover_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Discover rebalance algorithms from the RebalanceAlgorithms class."""
        algo_map: Dict[str, Dict[str, Any]] = {}

        for name, obj in inspect.getmembers(RebalanceAlgorithms, inspect.isfunction):
            if not name.endswith("_rebalance"):
                continue

            algo_map[name] = {
                "fn": obj,
                "label": name.replace("_", " ").title(),
                "description": f"Auto-discovered algorithm: {name}",
            }
        return algo_map

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path against the project root."""
        return os.path.join(project_root, path)

    def run_job(self, cfg: BacktestConfig) -> BacktestResult:
        """Run a complete backtest simulation job."""
        data_file_abs = self._resolve_path(cfg.data_file)
        output_dir_abs = self._resolve_path(cfg.output_dir)

        if not os.path.exists(data_file_abs):
            raise FileNotFoundError(f"Data file not found: {cfg.data_file}")

        os.makedirs(output_dir_abs, exist_ok=True)

        algo_info = self.algorithm_map.get(cfg.algorithm)
        if not algo_info:
            raise RuntimeError(f"Unsupported algorithm: {cfg.algorithm}")

        rebalance_fn = algo_info["fn"]

        # Initialize core engine and run simulation
        strategy = BacktestEngine(data_file_abs, cfg.initial_capital)
        strategy.run_backtest(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            rebalance_freq=cfg.rebalance_freq,
            fees=cfg.fees,
            rebalance_fn=rebalance_fn,
            rebalance_interval_days=cfg.rebalance_interval_days,
            asset_cols=cfg.asset_cols,
            use_trend_model=cfg.use_trend_model,
            model_lookback_days=cfg.model_lookback_days,
            model_threshold=cfg.model_threshold,
            model_type=cfg.model_type,
        )

        # Gather results
        stats = strategy.get_performance_stats()
        if not stats:
            raise RuntimeError("Backtest simulation produced no statistics.")

        html_path = strategy.plot_results(
            output_dir=output_dir_abs,
            benchmark_cols=cfg.benchmark_cols,
        )
        if not html_path:
            raise RuntimeError("Failed to generate performance chart.")

        # Prepare paths for response
        rel_html_path = os.path.relpath(html_path, project_root)
        result_url = f"/results/{os.path.basename(html_path)}"

        return BacktestResult(
            stats=stats,
            result_html_path=rel_html_path,
            result_url=result_url,
            algorithm=cfg.algorithm,
        )
