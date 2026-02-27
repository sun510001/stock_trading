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
from utils.naming import sanitize_filename


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
        None,
        description="Fixed interval in days for rebalancing",
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
    use_trend_model: bool = Field(
        default=False,
        description="Whether to enable the unsupervised trend model to gate rebalancing",
    )
    trend_model_type: str = Field(
        default="kmeans_simple",
        description="Trend model type: 'kmeans_simple', 'kmeans_window', 'random_forest', 'torch_mlp', 'autoencoder', or 'hmm'",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Relative path to a persisted trend model (.pkl or .pt) under project root",
    )
    model_lookback_days: int = Field(
        default=60,
        description="Number of past days used as the fixed window for the trend model",
    )
    model_threshold: float = Field(
        default=0.5,
        description="Trend score threshold in [0,1]; only rebalance when score >= threshold",
    )
    max_tilt: float = Field(
        default=0.1,
        description="Maximum weight tilt deviation from equal weight",
    )
    signal_weight_mode: str = Field(
        default="raw_signal_tilt",
        description=(
            "Signal-weighted mode: 'raw_signal_tilt' or "
            "'signal_tilt_with_risk_leverage'"
        ),
    )
    risk_asset_cols: Optional[List[str]] = Field(
        default=None,
        description="Optional risk-asset bucket columns (must be subset of strategy assets)",
    )
    safe_asset_cols: Optional[List[str]] = Field(
        default=None,
        description="Optional safe-asset bucket columns (must be subset of strategy assets)",
    )
    safe_allocation_mode: str = Field(
        default="equal_weight",
        description="Safe-bucket allocation mode: 'equal_weight', 'fixed_weight', or 'single_asset'",
    )
    safe_fixed_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-safe-asset weights when safe_allocation_mode='fixed_weight'",
    )
    safe_single_asset: Optional[str] = Field(
        default=None,
        description="Single safe asset name when safe_allocation_mode='single_asset'",
    )
    risk_leverage_min: float = Field(
        default=0.2,
        description="Minimum risk leverage when trend score is low",
    )
    risk_leverage_max: float = Field(
        default=1.0,
        description="Maximum risk leverage when trend score is high",
    )
    risk_leverage_score_lo: float = Field(
        default=0.2,
        description="Trend score lower bound mapped to risk_leverage_min",
    )
    risk_leverage_score_hi: float = Field(
        default=0.8,
        description="Trend score upper bound mapped to risk_leverage_max",
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

    def _build_aligned_filename_from_asset_cols(self, asset_cols: Optional[List[str]]) -> str:
        """Build aligned CSV filename based on strategy asset universe.

        If asset_cols is None or empty, fall back to the default global file.
        """
        if not asset_cols:
            return "data_processed/aligned_assets.csv"

        names_sorted = sorted(asset_cols)
        key = "_".join(names_sorted)
        safe_key = sanitize_filename(key)
        return os.path.join("data_processed", f"aligned_{safe_key}.csv")

    def run_job(self, cfg: BacktestConfig) -> BacktestResult:
        """Run a complete backtest simulation job."""
        # 1) 优先使用全局对齐文件 data_processed/aligned_assets.csv，
        #    其中可能包含比本次策略资产更多的列，供 Benchmarks 使用。
        global_rel = cfg.data_file or "data_processed/aligned_assets.csv"
        global_abs = self._resolve_path(global_rel)

        data_file_abs: str
        if os.path.exists(global_abs):
            data_file_abs = global_abs
        else:
            # 2) 如果找不到全局文件，则退回到按资产集合推导专属对齐文件的旧逻辑
            data_file_rel = self._build_aligned_filename_from_asset_cols(cfg.asset_cols)
            data_file_abs = self._resolve_path(data_file_rel)

        if not os.path.exists(data_file_abs):
            # 如果既没有全局文件也没有资产集合专属文件，提示用户先执行下载与处理
            raise FileNotFoundError("请先执行 Download & Process 生成 aligned_assets.csv 或该资产集合的对齐文件")

        output_dir_abs = self._resolve_path(cfg.output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)

        algo_info = self.algorithm_map.get(cfg.algorithm)
        if not algo_info:
            raise RuntimeError(f"Unsupported algorithm: {cfg.algorithm}")

        rebalance_fn = algo_info["fn"]

        strategy = BacktestEngine(data_file_abs, cfg.initial_capital)
        if cfg.signal_weight_mode not in {
            "raw_signal_tilt",
            "signal_tilt_with_risk_leverage",
        }:
            raise RuntimeError(
                "Unsupported signal_weight_mode. "
                "Use 'raw_signal_tilt' or 'signal_tilt_with_risk_leverage'."
            )

        risk_leverage_enabled = (
            cfg.algorithm == "signal_weighted_rebalance"
            and cfg.signal_weight_mode == "signal_tilt_with_risk_leverage"
        )
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
            model_type=cfg.trend_model_type,
            model_path=cfg.model_path,
            max_tilt=cfg.max_tilt,
            risk_asset_cols=cfg.risk_asset_cols,
            safe_asset_cols=cfg.safe_asset_cols,
            safe_allocation_mode=cfg.safe_allocation_mode,
            safe_fixed_weights=cfg.safe_fixed_weights,
            safe_single_asset=cfg.safe_single_asset,
            risk_leverage_enabled=risk_leverage_enabled,
            risk_leverage_min=cfg.risk_leverage_min,
            risk_leverage_max=cfg.risk_leverage_max,
            risk_leverage_score_lo=cfg.risk_leverage_score_lo,
            risk_leverage_score_hi=cfg.risk_leverage_score_hi,
        )

        stats = strategy.get_performance_stats()
        if not stats:
            raise RuntimeError("Backtest simulation produced no statistics.")

        html_path = strategy.plot_results(
            output_dir=output_dir_abs,
            benchmark_cols=cfg.benchmark_cols,
        )
        if not html_path:
            raise RuntimeError("Failed to generate performance chart.")

        # Save configuration to YAML
        import yaml
        from datetime import datetime
        configs_dir = os.path.join(output_dir_abs, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_short = cfg.algorithm.replace("_rebalance", "")
        model_info = f"_{cfg.trend_model_type}" if cfg.use_trend_model else ""
        yaml_filename = f"config_{timestamp}_{algo_short}{model_info}.yaml"
        yaml_path = os.path.join(configs_dir, yaml_filename)
        
        try:
            cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save config YAML: {e}")

        rel_html_path = os.path.relpath(html_path, project_root)
        result_url = f"/results/{os.path.basename(html_path)}"

        return BacktestResult(
            stats=stats,
            result_html_path=rel_html_path,
            result_url=result_url,
            algorithm=cfg.algorithm,
        )
