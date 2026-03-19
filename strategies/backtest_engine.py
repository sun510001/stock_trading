import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Union, Optional, List, Callable
from logger import logger
from utils.csv_utils import load_date_indexed_csv

from strategies.algorithms import RebalanceAlgorithms, RebalanceContext, RebalanceResult


class BacktestEngine:
    """Core backtest engine handling simulation and visualization."""

    _LOOKBACK_FREE_ALGORITHMS = {
        "permanent_portfolio_rebalance",
        "signal_weighted_rebalance",
    }

    def __init__(self, data_path: str, initial_capital: float = 10000.0) -> None:
        self.data_path: str = data_path
        self.initial_capital: float = initial_capital
        self.data: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.Series] = None
        self.drawdown: Optional[pd.Series] = None
        self.asset_weights: Optional[pd.DataFrame] = None
        self.regime_state_weights: Optional[pd.DataFrame] = None
        self.regime_confidence_margin: Optional[pd.Series] = None
        self.future_regime_forecast: Optional[Dict[str, object]] = None
        self.strategy_diagnostics: Optional[Dict[str, object]] = None

        self._load_data()

    def _normalize_forecast_payload(self, value: object) -> object:
        """Convert nested numpy/pandas scalars into JSON-safe Python values."""
        if isinstance(value, dict):
            return {
                str(key): self._normalize_forecast_payload(child)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [self._normalize_forecast_payload(child) for child in value]
        if isinstance(value, tuple):
            return [self._normalize_forecast_payload(child) for child in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _load_data(self) -> None:
        """Load price data from the specified CSV file."""
        try:
            logger.info(f"Loading data from {self.data_path}...")
            df = load_date_indexed_csv(self.data_path)
            self.data = df.astype(float)
            logger.info(f"Data loaded. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    @classmethod
    def _algorithm_requires_lookback_warmup(cls, rebalance_fn: Callable) -> bool:
        """Return whether the selected algorithm depends on lookback windows."""
        algo_name = getattr(rebalance_fn, "__name__", str(rebalance_fn))
        return algo_name not in cls._LOOKBACK_FREE_ALGORITHMS

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fees: float = 0.0005,
        rebalance_fn: Optional[Callable] = None,
        rebalance_interval_days: int = 30,
        candidate_assets: Optional[List[str]] = None,
        candidate_asset_weights: Optional[Dict[str, float]] = None,
        regime_candidate_assets: Optional[Dict[str, List[str]]] = None,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.5,
        model_type: str = "kmeans_simple",
        model_path: Optional[str] = None,
        top_k: int = 3,
        target_volatility: float = 0.10,
        vol_lookback: int = 60,
        max_leverage: float = 1.0,
        safe_assets: Optional[List[str]] = None,
        regime_safe_assets: Optional[Dict[str, List[str]]] = None,
        max_asset_weight: float = 1.0,
        vol_scale_lookback: int = 0,
        momentum_threshold: float = 0.0,
        use_sharpe_weighting: bool = False,
        min_blend: float = 0.0,
        fill_residual_with_safe: bool = True,
        use_regime_position_sizing: bool = False,
        strategy_b_base_nasdaq100_weight: float = 0.80,
        strategy_b_base_us3m_weight: float = 0.20,
    ) -> None:
        """Execute the backtest simulation by delegating trade logic to ``rebalance_fn``.

        BacktestEngine is responsible **only** for:
        - Slicing and pre-processing data for the requested date range.
        - Loading and validating the optional ML trend model.
        - Iterating through trading days (the Time Loop).
        - On rebalance days: gathering context (prices, returns windows, trend
          score) and delegating weight/unit decisions to ``rebalance_fn``.
        - Applying the returned unit positions and recording portfolio history.

        All trading-strategy logic lives inside ``rebalance_fn``.  If
        ``rebalance_fn`` is None the engine falls back to
        :meth:`~strategies.algorithms.RebalanceAlgorithms.permanent_portfolio_rebalance`
        (equal-weight), which matches the default algorithm exposed by
        ``BacktestConfig`` in the backend service.

        Args:
            start_date: ISO date string for backtest start (inclusive).  If
                None, uses the first available date in the data.
            end_date: ISO date string for backtest end (inclusive).  If None,
                uses the last available date in the data.
            fees: Per-trade transaction fee as a fraction of traded value
                (e.g. 0.0005 for 5 bps).
            rebalance_fn: Callable with signature
                ``(ctx: RebalanceContext) -> RebalanceResult``.
                Receives all market context and returns target positions.
            rebalance_interval_days: Minimum number of calendar days between
                consecutive executed rebalance events (e.g. 30 for monthly).
            candidate_assets: Optional list of column names to restrict the
                asset universe; if None all columns are used.
            candidate_asset_weights: Optional manual weight map keyed by
                candidate asset name. Used by compatible strategies such as
                permanent_portfolio_rebalance and normalized at runtime.
            regime_candidate_assets: Optional mapping from macro regime name
                to candidate asset lists. Compatible strategies can use this
                to switch the investable risk bucket by inferred macro state.
            use_trend_model: Whether to activate the ML trend overlay model.
            model_lookback_days: Number of past days fed to the trend model as
                its feature window.
            model_threshold: Trend score threshold passed to ``rebalance_fn``
                via :class:`~strategies.algorithms.RebalanceContext`.
            model_type: Identifier string for the trend model flavour.
            model_path: Optional path to a persisted model artifact.
            top_k: Maximum number of assets to select; forwarded to
                ``RebalanceContext``.
            target_volatility: Target annualised portfolio volatility;
                forwarded to ``RebalanceContext``.
            vol_lookback: Lookback window (days) for momentum and covariance
                estimation; forwarded to ``RebalanceContext``.
            max_leverage: Hard cap on gross exposure; forwarded to
                ``RebalanceContext``.
            safe_assets: Column names of safe-haven assets to hold during
                risk-off periods; forwarded to ``RebalanceContext``.
            regime_safe_assets: Optional mapping from macro regime name to
                state-specific safe-asset lists.
            max_asset_weight: Hard cap on the weight of any single asset in
                the final portfolio (e.g. 0.30 = 30%).  Excess weight is
                redistributed iteratively among other selected assets.
                Defaults to 1.0 (no cap); forwarded to ``RebalanceContext``.
            vol_scale_lookback: Short trailing window (days) used exclusively
                for covariance estimation in the vol-scaling layer.  0 means
                use the full ``vol_lookback`` window.  Forwarded to
                ``RebalanceContext``.
            momentum_threshold: Minimum cumulative return for an asset to pass
                the absolute-momentum filter.  0.0 = original hard-zero.
                Forwarded to ``RebalanceContext``.
            use_sharpe_weighting: If True, allocation weights are proportional
                to Sharpe-proxy (return / vol) instead of raw return.
                Forwarded to ``RebalanceContext``.
            fill_residual_with_safe: If True, route unused portfolio weight
                into the configured safe assets instead of leaving it as cash.
            use_regime_position_sizing: If True, allow compatible trend models
                to pass a four-state macro regime into the strategy so
                position sizing and volatility targeting can be adjusted.
        """
        logger.info(
            f"Preparing simulation for range: {start_date or 'Start'} to {end_date or 'End'}..."
        )
        logger.info(
            f"BacktestEngine.run_backtest params | rebalance_fn={getattr(rebalance_fn, '__name__', rebalance_fn)}, "
            f"top_k={top_k}, target_volatility={target_volatility:.2f}, "
            f"vol_lookback={vol_lookback}, max_leverage={max_leverage:.2f}, "
            f"fill_residual_with_safe={fill_residual_with_safe}"
        )

        # Default strategy: use equal-weight, matching BacktestConfig.algorithm default.
        if rebalance_fn is None:
            rebalance_fn = RebalanceAlgorithms.permanent_portfolio_rebalance
            logger.info(
                "rebalance_fn not provided; defaulting to permanent_portfolio_rebalance (equal-weight)."
            )

        # ── Load optional ML trend model ───────────────────────────────────────
        trend_model = None
        if use_trend_model:
            from strategies.trend_models import get_trend_model

            trend_model = get_trend_model(
                model_type=model_type,
                model_path=model_path,
            )

        if self.data is None:
            logger.error("No data available for backtest.")
            return

        # ── 1. Slice & pre-process data ────────────────────────────────────────
        df_slice = self.data.copy()
        if start_date:
            df_slice = df_slice.loc[start_date:]
        if end_date:
            df_slice = df_slice.loc[:end_date]

        if df_slice.empty:
            logger.error("No data found for the specified date range!")
            return

        # Build the full asset universe from all assets that can appear in any
        # risk or safe sleeve.  This keeps regime-specific assets tradeable even
        # when they are not present in the top-level candidate/safe lists.
        safe_assets_list: List[str] = list(safe_assets) if safe_assets else []
        regime_candidate_assets_map = regime_candidate_assets or {}
        regime_safe_assets_map = regime_safe_assets or {}

        ordered_candidate_assets: List[str] = list(candidate_assets) if candidate_assets else []
        for asset_names in regime_candidate_assets_map.values():
            for asset_name in asset_names:
                if asset_name not in ordered_candidate_assets:
                    ordered_candidate_assets.append(asset_name)

        ordered_safe_assets: List[str] = list(safe_assets_list)
        for asset_names in regime_safe_assets_map.values():
            for asset_name in asset_names:
                if asset_name not in ordered_safe_assets:
                    ordered_safe_assets.append(asset_name)

        if ordered_candidate_assets:
            extra_safe = [s for s in ordered_safe_assets if s not in ordered_candidate_assets]
            universe_cols = ordered_candidate_assets + extra_safe
            missing = [c for c in universe_cols if c not in df_slice.columns]
            if missing:
                raise RuntimeError(f"Missing asset columns in data: {missing}")
            df_slice = df_slice[universe_cols]
            # Indices of all risk-candidate assets within the universe.
            candidate_col_indices: List[int] = list(range(len(ordered_candidate_assets)))
            df_slice = df_slice.dropna(how="any")
        else:
            df_slice = df_slice.dropna(axis=1, how="all")
            candidate_col_indices = list(range(len(df_slice.columns)))

        df_slice = df_slice.dropna(how="all").ffill()
        data_filled = self.data.ffill()

        actual_start = df_slice.index[0].strftime("%Y-%m-%d")
        actual_end = df_slice.index[-1].strftime("%Y-%m-%d")
        logger.info(
            f"Actual Backtest Range: {actual_start} to {actual_end} ({len(df_slice)} days)"
        )

        # Validate trend model feature columns against loaded data
        trend_feature_cols: Optional[List[str]] = None
        trend_model_type_name = str(model_type).strip().lower() if model_type else ""
        if use_trend_model and trend_model is not None and hasattr(trend_model, "feature_cols"):
            trend_feature_cols = getattr(trend_model, "feature_cols") or None
            if trend_feature_cols:
                missing_feat = [c for c in trend_feature_cols if c not in self.data.columns]
                if missing_feat:
                    raise RuntimeError(
                        "aligned_assets.csv is missing columns required by the trend model: "
                        f"{missing_feat}. Re-train the model with the current data columns, "
                        "or select a model file that matches the current dataset."
                    )
                logger.info(
                    f"Trend model feature_cols validated: {trend_feature_cols}"
                )

        def _build_trend_window(end_idx: int) -> Optional[pd.DataFrame]:
            if (
                not use_trend_model
                or trend_model is None
                or model_lookback_days <= 0
                or end_idx < model_lookback_days
            ):
                return None
            window = df_slice.iloc[end_idx - model_lookback_days : end_idx]
            if window.empty:
                return None
            if trend_model_type_name == "regime_horizon_router":
                # Router inference is date-driven and uses the window end date as
                # the anchor into its precomputed regime table; it does not
                # require a dense feature matrix across every aligned column.
                return window
            if trend_feature_cols:
                full_window = data_filled.loc[window.index, trend_feature_cols].dropna(how="any")
            else:
                full_window = data_filled.loc[window.index].dropna(how="any")
            if full_window.empty or len(full_window) < model_lookback_days:
                return None
            return full_window

        # ── 2. Setup arrays ────────────────────────────────────────────────────
        prices = df_slice.values
        dates = df_slice.index
        n_days, n_assets = prices.shape
        col_names: List[str] = list(df_slice.columns)

        regime_candidate_indices_by_name: Dict[str, List[int]] = {}
        if regime_candidate_assets:
            for regime_name, asset_names in regime_candidate_assets.items():
                indices = [
                    col_names.index(asset_name)
                    for asset_name in asset_names
                    if asset_name in col_names
                ]
                if indices:
                    regime_candidate_indices_by_name[str(regime_name).strip().lower()] = indices

        regime_safe_indices_by_name: Dict[str, List[int]] = {}
        if regime_safe_assets:
            for regime_name, asset_names in regime_safe_assets.items():
                indices = [
                    col_names.index(asset_name)
                    for asset_name in asset_names
                    if asset_name in col_names
                ]
                if indices:
                    regime_safe_indices_by_name[str(regime_name).strip().lower()] = indices

        candidate_asset_weights_by_name: Dict[str, float] = {}
        if candidate_asset_weights:
            for asset_name, raw_weight in candidate_asset_weights.items():
                asset_key = str(asset_name).strip()
                if asset_key not in col_names:
                    continue
                weight_value = float(raw_weight)
                if np.isfinite(weight_value) and weight_value > 0.0:
                    candidate_asset_weights_by_name[asset_key] = weight_value

        returns_arr = df_slice.pct_change(fill_method=None).fillna(0).values

        interval = rebalance_interval_days if rebalance_interval_days and rebalance_interval_days > 0 else 30

        # Pre-compute safe asset index positions (within the full universe col_names)
        safe_asset_indices: List[int] = []
        if safe_assets_list:
            for col in safe_assets_list:
                if col in col_names:
                    safe_asset_indices.append(col_names.index(col))

        # Only lookback-driven strategies need a history warm-up.
        history_warmup = vol_lookback if self._algorithm_requires_lookback_warmup(rebalance_fn) else 0
        trend_warmup = model_lookback_days if use_trend_model else 0
        warmup = max(trend_warmup, history_warmup)

        # ── 3. Initialise portfolio ────────────────────────────────────────────
        portfolio_history = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))
        regime_history: List[str] = []
        regime_confidence_history: List[float] = []
        current_units = np.zeros(n_assets)
        cash_balance = self.initial_capital
        current_regime = "sideways"
        current_regime_mode = "default"
        current_regime_anchor_requested = None
        current_regime_anchor_used = None
        current_regime_confidence_margin = float("nan")
        current_recession_probability = float("nan")
        current_prosperity_share_20d = float("nan")
        current_recession_share_20d = float("nan")
        current_nasdaq100_drawdown_from_peak = float("nan")
        last_decision_info: Dict[str, str] = {}
        strategy_state: Dict[str, object] = {}
        last_rebalance_date: Optional[pd.Timestamp] = None

        # ── 4. Time Loop ───────────────────────────────────────────────────────
        for i in range(n_days):
            today_prices = prices[i]
            current_asset_vals = current_units * today_prices
            current_val = float(np.sum(current_asset_vals) + cash_balance)
            trend_window_df = _build_trend_window(i)
            if trend_window_df is not None and trend_model is not None:
                latest_regime_payload = None
                if hasattr(
                    trend_model,
                    "predict_latest_regime_payload_from_dataframe",
                ):
                    latest_regime_payload = trend_model.predict_latest_regime_payload_from_dataframe(
                        trend_window_df
                    )
                if isinstance(latest_regime_payload, dict):
                    current_regime = str(
                        latest_regime_payload.get("regime", current_regime)
                    )
                    inferred_margin = latest_regime_payload.get("confidence_margin")
                    if inferred_margin is not None:
                        current_regime_confidence_margin = float(inferred_margin)
                    inferred_probabilities = latest_regime_payload.get("probabilities")
                    if isinstance(inferred_probabilities, dict):
                        current_recession_probability = float(
                            inferred_probabilities.get("recession", float("nan"))
                        )
                    current_regime_mode = str(
                        latest_regime_payload.get("mode", current_regime_mode)
                    )
                    current_regime_anchor_requested = latest_regime_payload.get(
                        "anchor_date_requested"
                    )
                    current_regime_anchor_used = latest_regime_payload.get(
                        "anchor_date_used"
                    )
                if hasattr(
                    trend_model,
                    "predict_latest_regime_from_dataframe",
                ):
                    if not isinstance(latest_regime_payload, dict):
                        current_regime = str(
                            trend_model.predict_latest_regime_from_dataframe(
                                trend_window_df
                            )
                        )
                if hasattr(
                    trend_model,
                    "predict_latest_regime_confidence_margin_from_dataframe",
                ):
                    if not isinstance(latest_regime_payload, dict):
                        inferred_margin = trend_model.predict_latest_regime_confidence_margin_from_dataframe(
                            trend_window_df
                        )
                        if inferred_margin is not None:
                            current_regime_confidence_margin = float(inferred_margin)
                if hasattr(
                    trend_model,
                    "predict_latest_regime_probabilities_from_dataframe",
                ):
                    if not isinstance(latest_regime_payload, dict):
                        inferred_probabilities = trend_model.predict_latest_regime_probabilities_from_dataframe(
                            trend_window_df
                        )
                        if inferred_probabilities is not None:
                            current_recession_probability = float(
                                inferred_probabilities.get("recession", float("nan"))
                            )

            rolling_regime_window = regime_history[-19:] + [current_regime]
            current_recession_share_20d = float(
                np.mean(
                    [
                        1.0 if regime_name == "recession" else 0.0
                        for regime_name in rolling_regime_window
                    ]
                )
            ) if rolling_regime_window else float("nan")
            current_prosperity_share_20d = float(
                np.mean(
                    [
                        1.0 if regime_name == "prosperity" else 0.0
                        for regime_name in rolling_regime_window
                    ]
                )
            ) if rolling_regime_window else float("nan")
            current_deflation_share_20d = float(
                np.mean(
                    [
                        1.0 if regime_name == "deflation" else 0.0
                        for regime_name in rolling_regime_window
                    ]
                )
            ) if rolling_regime_window else float("nan")

            should_rebalance = last_rebalance_date is None or (dates[i] - last_rebalance_date).days >= interval
            if should_rebalance:
                # Skip rebalance until warm-up period has elapsed
                if i < warmup:
                    portfolio_history[i] = current_val
                    weights_history[i] = (
                        current_asset_vals / current_val
                        if current_val > 0
                        else np.zeros(n_assets)
                    )
                    regime_history.append(current_regime)
                    regime_confidence_history.append(current_regime_confidence_margin)
                    continue

                # ── Compute ML trend score ─────────────────────────────────────
                trend_score = 1.0
                trend_regime = current_regime
                trend_confidence_margin = current_regime_confidence_margin
                trend_recession_probability = current_recession_probability
                if use_trend_model and trend_model is not None and model_lookback_days > 0:
                    window = df_slice.iloc[i - model_lookback_days : i]
                    if not window.empty:
                        if hasattr(trend_model, "predict_latest_score_from_dataframe"):
                            full_window = trend_window_df
                            if full_window is not None:
                                trend_score = float(
                                    trend_model.predict_latest_score_from_dataframe(
                                        full_window
                                    )
                                )
                                if hasattr(
                                    trend_model,
                                    "predict_latest_regime_from_dataframe",
                                ):
                                    trend_regime = str(
                                        trend_model.predict_latest_regime_from_dataframe(
                                            full_window
                                        )
                                    )
                                if hasattr(
                                    trend_model,
                                    "predict_latest_regime_confidence_margin_from_dataframe",
                                ):
                                    inferred_margin = trend_model.predict_latest_regime_confidence_margin_from_dataframe(
                                        full_window
                                    )
                                    if inferred_margin is not None:
                                        trend_confidence_margin = float(inferred_margin)
                                if hasattr(
                                    trend_model,
                                    "predict_latest_regime_probabilities_from_dataframe",
                                ):
                                    inferred_probabilities = trend_model.predict_latest_regime_probabilities_from_dataframe(
                                        full_window
                                    )
                                    if inferred_probabilities is not None:
                                        trend_recession_probability = float(
                                            inferred_probabilities.get(
                                                "recession",
                                                float("nan"),
                                            )
                                        )
                        elif trend_feature_cols:
                            full_window = data_filled.loc[
                                window.index, trend_feature_cols
                            ].dropna(how="any")
                            if (
                                not full_window.empty
                                and len(full_window) >= model_lookback_days
                            ):
                                mat = full_window.values.astype(float)
                                if hasattr(
                                    trend_model, "predict_latest_score_from_matrix"
                                ):
                                    trend_score = float(
                                        trend_model.predict_latest_score_from_matrix(mat)
                                    )
                                else:
                                    close_series = full_window.iloc[:, 0].values.astype(
                                        float
                                    )
                                    trend_score = float(
                                        trend_model.predict_latest_score_from_series(
                                            close=close_series
                                        )
                                    )
                        elif hasattr(trend_model, "predict_latest_score_from_series"):
                            close_series = window.iloc[:, 0].values.astype(float)
                            trend_score = float(
                                trend_model.predict_latest_score_from_series(
                                    close=close_series
                                )
                            )
                        else:
                            trend_score = float(
                                trend_model.predict_score(window.values.astype(float))
                            )
                    logger.info(
                        f"Rebalance | idx={i}, Market Trend Score {trend_score:.3f}, regime={trend_regime}"
                    )
                current_regime = trend_regime
                current_regime_confidence_margin = trend_confidence_margin
                current_recession_probability = trend_recession_probability

                nasdaq100_drawdown_from_peak = float("nan")
                if "Nasdaq100" in col_names:
                    nasdaq100_idx = col_names.index("Nasdaq100")
                    nasdaq_history = prices[: i + 1, nasdaq100_idx]
                    positive_history = nasdaq_history[np.isfinite(nasdaq_history) & (nasdaq_history > 0.0)]
                    if positive_history.size > 0 and np.isfinite(today_prices[nasdaq100_idx]) and today_prices[nasdaq100_idx] > 0.0:
                        running_peak = float(np.max(positive_history))
                        if running_peak > 0.0:
                            nasdaq100_drawdown_from_peak = max(
                                0.0,
                                1.0 - (float(today_prices[nasdaq100_idx]) / running_peak),
                            )
                current_nasdaq100_drawdown_from_peak = nasdaq100_drawdown_from_peak

                # ── Build price / return windows for the rebalance function ────
                # price_window / returns_window are restricted to candidate assets
                # only, so safe-haven instruments never appear as momentum winners.
                lookback_start = max(0, i - vol_lookback)
                price_window = prices[lookback_start:i][:, candidate_col_indices]
                returns_window = returns_arr[lookback_start:i][:, candidate_col_indices]

                # ── Assemble context and delegate to rebalance_fn ──────────────
                ctx = RebalanceContext(
                    current_units=current_units.copy(),
                    today_prices=today_prices,
                    fees=fees,
                    cash_balance=cash_balance,
                    price_window=price_window,
                    returns_window=returns_window,
                    trend_score=trend_score,
                    trend_regime=trend_regime,
                    trend_regime_confidence_margin=trend_confidence_margin,
                    trend_regime_recession_probability=trend_recession_probability,
                    trend_regime_prosperity_share_20d=current_prosperity_share_20d,
                    trend_regime_recession_share_20d=current_recession_share_20d,
                    nasdaq100_drawdown_from_peak=nasdaq100_drawdown_from_peak,
                    model_threshold=model_threshold,
                    top_k=top_k,
                    target_volatility=target_volatility,
                    max_leverage=max_leverage,
                    safe_asset_indices=safe_asset_indices,
                    candidate_indices=candidate_col_indices,
                    candidate_asset_weights=candidate_asset_weights_by_name,
                    col_names=col_names,
                    max_asset_weight=max_asset_weight,
                    vol_scale_lookback=vol_scale_lookback,
                    momentum_threshold=momentum_threshold,
                    use_sharpe_weighting=use_sharpe_weighting,
                    min_blend=min_blend,
                    fill_residual_with_safe=fill_residual_with_safe,
                    use_regime_position_sizing=use_regime_position_sizing,
                    strategy_b_base_nasdaq100_weight=strategy_b_base_nasdaq100_weight,
                    strategy_b_base_us3m_weight=strategy_b_base_us3m_weight,
                    regime_candidate_indices_by_name=regime_candidate_indices_by_name,
                    regime_safe_indices_by_name=regime_safe_indices_by_name,
                    strategy_state=dict(strategy_state),
                )

                result: RebalanceResult = rebalance_fn(ctx)
                last_decision_info = dict(result.decision_info or {})
                strategy_state = dict(result.updated_state or {})

                # Apply result
                current_units = result.new_units
                cash_balance = result.new_cash
                current_val = float(np.sum(current_units * today_prices) + cash_balance)
                last_rebalance_date = dates[i]

                total_exposure = float(np.sum(current_units * today_prices)) / current_val if current_val > 0 else 0.0
                logger.info(
                    f"Rebalance | idx={i}, exposure={total_exposure:.2f}, "
                    f"cash_budget={max(0.0, 1.0 - total_exposure):.2f}"
                )
                if result.decision_info:
                    decision_str = ", ".join(
                        f"{key}={value}" for key, value in result.decision_info.items()
                    )
                    logger.info(f"Rebalance | idx={i}, decision: {decision_str}")

                # ── Log algorithm name, selected assets and weights ────────────
                algo_name = getattr(rebalance_fn, "__name__", str(rebalance_fn))
                asset_vals = current_units * today_prices
                nonzero_mask = asset_vals > 1e-8
                if current_val > 0 and nonzero_mask.any():
                    held_names = [col_names[j] for j in range(n_assets) if nonzero_mask[j]]
                    held_weights = asset_vals[nonzero_mask] / current_val
                    weight_str = ", ".join(
                        f"{name}={w:.1%}" for name, w in zip(held_names, held_weights)
                    )
                    logger.info(
                        f"Rebalance | algo={algo_name} | "
                        f"date={dates[i].strftime('%Y-%m-%d')} | "
                        f"holdings: {weight_str}"
                    )
                else:
                    logger.info(
                        f"Rebalance | algo={algo_name} | "
                        f"date={dates[i].strftime('%Y-%m-%d')} | "
                        f"holdings: CASH 100.0%"
                    )

            portfolio_history[i] = current_val
            weights_history[i] = (
                (current_units * today_prices) / current_val
                if current_val > 0
                else np.zeros(n_assets)
            )
            regime_history.append(current_regime)
            regime_confidence_history.append(current_regime_confidence_margin)

        self.future_regime_forecast = None
        self.strategy_diagnostics = None
        if (
            use_trend_model
            and trend_model is not None
            and hasattr(trend_model, "predict_future_regime_payload_from_dataframe")
        ):
            if trend_feature_cols:
                future_window = data_filled.loc[df_slice.index, trend_feature_cols].dropna(
                    how="any"
                )
            else:
                future_window = data_filled.loc[df_slice.index].dropna(how="any")

            if not future_window.empty:
                forecast_window = future_window.tail(max(int(model_lookback_days), 1))
                try:
                    forecast_payload = trend_model.predict_future_regime_payload_from_dataframe(
                        forecast_window
                    )
                except Exception as exc:
                    logger.error("Future regime forecast failed: %s", exc)
                    forecast_payload = None
                if forecast_payload is not None:
                    normalized_payload = self._normalize_forecast_payload(
                        forecast_payload
                    )
                    if isinstance(normalized_payload, dict):
                        normalized_payload["mode"] = "online_future_inference"
                        self.future_regime_forecast = normalized_payload

        final_asset_vals = current_units * prices[-1]
        final_portfolio_value = float(np.sum(final_asset_vals) + cash_balance)
        final_weight_lookup = {
            col_name: (
                float(final_asset_vals[idx] / final_portfolio_value)
                if final_portfolio_value > 0.0
                else 0.0
            )
            for idx, col_name in enumerate(col_names)
        }
        self.strategy_diagnostics = self._normalize_forecast_payload(
            {
                "regime": str(current_regime),
                "regime_source_mode": current_regime_mode,
                "regime_anchor_requested": current_regime_anchor_requested,
                "regime_anchor_used": current_regime_anchor_used,
                "prosperity_share_20d": current_prosperity_share_20d,
                "recession_share_20d": current_recession_share_20d,
                "nasdaq100_drawdown_from_peak": current_nasdaq100_drawdown_from_peak,
                "drawdown_layer_count": last_decision_info.get("drawdown_layers"),
                "prosperity_seen_since_scale_in": last_decision_info.get(
                    "prosperity_seen_since_scale_in"
                ),
                "mode": last_decision_info.get("mode"),
                "decision_info": last_decision_info,
                "state": strategy_state,
                "weights": {
                    "Nasdaq100": float(final_weight_lookup.get("Nasdaq100", 0.0)),
                    "US3M": float(final_weight_lookup.get("US3M", 0.0)),
                },
            }
        )

        # ── 5. Finalise ────────────────────────────────────────────────────────
        self.portfolio_value = pd.Series(
            portfolio_history, index=dates, name="Portfolio Value"
        )
        self.asset_weights = pd.DataFrame(
            weights_history, index=dates, columns=df_slice.columns
        )
        regime_df = pd.DataFrame(index=dates)
        regime_series = pd.Series(regime_history, index=dates, dtype="object")
        self.regime_confidence_margin = pd.Series(
            regime_confidence_history,
            index=dates,
            name="Regime Confidence Margin",
            dtype="float64",
        )
        core_regimes = ["prosperity", "inflation", "deflation", "recession"]
        for regime_name in core_regimes:
            regime_df[regime_name] = (regime_series == regime_name).astype(float)
        regime_df["sideways"] = (~regime_series.isin(core_regimes)).astype(float)
        self.regime_state_weights = regime_df
        running_max = self.portfolio_value.cummax()
        self.drawdown = (self.portfolio_value / running_max) - 1

        logger.info("Simulation complete.")

    def get_performance_stats(self, benchmark_col: str = "SP500") -> Dict[str, float]:
        """Calculate performance statistics for the backtest."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            return {}

        returns = self.portfolio_value.pct_change(fill_method=None).dropna()
        start_val = self.portfolio_value.iloc[0]
        end_val = self.portfolio_value.iloc[-1]

        n_years = (
            self.portfolio_value.index[-1] - self.portfolio_value.index[0]
        ).days / 365.25
        cagr = (end_val / start_val) ** (1 / n_years) - 1
        sharpe = (
            (returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() != 0
            else 0
        )
        max_dd = self.drawdown.min() if self.drawdown is not None else 0
        vol = returns.std() * np.sqrt(252)

        # New metrics calculation
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Max Recovery Days: longest calendar-day span from a peak to the next
        # point where the portfolio fully recovers to (or above) that peak.
        # Algorithm:
        #   1. Find every date that sets a new all-time high (peak).
        #   2. For each consecutive pair of peaks (peak_i, peak_{i+1}), the
        #      recovery period length = (peak_{i+1} - peak_i).days.
        #      If the portfolio never fully recovers after the last peak, that
        #      ongoing drawdown period stretches to the final date.
        #   3. max_recovery_days = max over all such periods.
        running_max = self.portfolio_value.cummax()
        max_recovery_days = 0
        if not running_max.empty:
            # Dates where the portfolio value equals the running maximum (new high)
            peak_dates = self.portfolio_value.index[
                self.portfolio_value >= running_max - running_max * 1e-9
            ]
            if len(peak_dates) >= 2:
                # Gaps between consecutive new-high dates
                for i in range(1, len(peak_dates)):
                    gap = (peak_dates[i] - peak_dates[i - 1]).days
                    if gap > max_recovery_days:
                        max_recovery_days = gap
            # Also account for an ongoing drawdown that has not yet recovered
            last_peak_date = peak_dates[-1] if len(peak_dates) > 0 else self.portfolio_value.index[0]
            ongoing = (self.portfolio_value.index[-1] - last_peak_date).days
            if ongoing > max_recovery_days:
                max_recovery_days = ongoing

        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0

        avg_win = returns[returns > 0].mean() if not returns[returns > 0].empty else 0.0
        avg_loss = returns[returns < 0].mean() if not returns[returns < 0].empty else 0.0
        pl_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0

        # Alpha, Beta, Information Ratio w.r.t Benchmark
        alpha = 0.0
        beta = 0.0
        info_ratio = 0.0

        if self.data is not None and benchmark_col in self.data.columns:
            # Align dates
            bench_prices = self.data[benchmark_col].loc[self.portfolio_value.index].dropna()
            bench_rets = bench_prices.pct_change(fill_method=None).dropna()

            # Match lengths
            aligned_rets = pd.concat([returns, bench_rets], axis=1).dropna()
            aligned_rets.columns = ['Strategy', 'Benchmark']

            strat_rets_align = aligned_rets['Strategy']
            bench_rets_align = aligned_rets['Benchmark']

            if not bench_rets_align.empty and bench_rets_align.std() != 0:
                covar = strat_rets_align.cov(bench_rets_align)
                ben_var = bench_rets_align.var()
                beta = covar / ben_var if ben_var != 0 else 0

                annual_strat_ret = strat_rets_align.mean() * 252
                annual_bench_ret = bench_rets_align.mean() * 252
                alpha = annual_strat_ret - beta * annual_bench_ret

                active_rets = strat_rets_align - bench_rets_align
                if active_rets.std() != 0:
                    info_ratio = (active_rets.mean() / active_rets.std()) * np.sqrt(252)

        return {
            "Start Value": float(start_val),
            "End Value": float(end_val),
            "Total Return": float((end_val / start_val) - 1),
            "CAGR": float(cagr),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown": float(max_dd),
            "Volatility": float(vol),
            "Years": float(n_years),
            "Alpha": float(alpha),
            "Beta": float(beta),
            "Information Ratio": float(info_ratio),
            "Calmar Ratio": float(calmar),
            "Max Recovery Days": int(max_recovery_days),
            "Win Rate": float(win_rate),
            "P/L Ratio": float(pl_ratio),
        }

    def plot_results(
        self,
        output_dir: str,
        benchmark_cols: Union[str, List[str]] = "Stocks",
        output_filename: str = "backtest_results.html",
    ) -> str:
        """Generate a comparative plot of strategy vs benchmarks."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            logger.warning("Portfolio value is empty. Cannot plot.")
            return ""

        filename = output_filename or "backtest_results.html"
        full_path = os.path.join(output_dir, filename)

        logger.info(f"Generating comparative plot to {full_path}...")

        start_date = self.portfolio_value.index[0]
        end_date = self.portfolio_value.index[-1]
        strategy_norm = (self.portfolio_value / self.portfolio_value.iloc[0]) - 1

        if isinstance(benchmark_cols, str):
            benchmark_list = [benchmark_cols]
        else:
            benchmark_list = list(benchmark_cols)

        has_confidence_chart = (
            self.regime_confidence_margin is not None
            and not self.regime_confidence_margin.dropna().empty
        )
        subplot_titles = [
            f"Performance: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "Portfolio Weights",
            "20-Day Rolling Macro Regime Share",
        ]
        row_heights = [0.55, 0.25, 0.20]
        total_rows = 3
        if has_confidence_chart:
            subplot_titles.append("Regime Confidence Margin vs 20-Day Recession Share")
            row_heights = [0.50, 0.22, 0.16, 0.12]
            total_rows = 4

        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=row_heights,
            subplot_titles=tuple(subplot_titles),
        )

        # Strategy stats
        strat_rets = self.portfolio_value.pct_change(fill_method=None).dropna()
        strat_years = (end_date - start_date).days / 365.25
        strat_cagr = (
            (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0])
            ** (1 / strat_years)
            - 1
        )
        strat_sharpe = (
            (strat_rets.mean() / strat_rets.std()) * np.sqrt(252)
            if strat_rets.std() != 0
            else 0
        )
        strat_running_max = self.portfolio_value.cummax()
        strat_dd = (self.portfolio_value / strat_running_max) - 1
        strat_max_dd = strat_dd.min()
        strat_total_ret = strategy_norm.iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=strategy_norm.index,
                y=strategy_norm,
                mode="lines",
                name=(
                    "Strategy<br>"
                    f"Total: {strat_total_ret:.1%}, CAGR: {strat_cagr:.1%}<br>"
                    f"Sharpe: {strat_sharpe:.2f}, MaxDD: {strat_max_dd:.1%}"
                ),
                line=dict(color="#00CC96", width=2),
            ),
            row=1,
            col=1,
        )

        default_colors = [
            "#EF553B",
            "#AB63FA",
            "#19D3F3",
            "#636EFA",
            "#FFA15A",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
            "#00CC96",
            "#A777F1",
            "#2E91E5",
        ]
        asset_color_map: dict[str, str] = {}
        for idx, col in enumerate(benchmark_list):
            if self.data is not None and col in self.data.columns:
                bench_series = self.data[col].loc[start_date:end_date]
                if not bench_series.empty:
                    bench_norm = (bench_series / bench_series.iloc[0]) - 1
                    bench_total_ret = bench_norm.iloc[-1]

                    bench_rets = bench_series.pct_change(fill_method=None).dropna()
                    years = (bench_series.index[-1] - bench_series.index[0]).days / 365.25
                    cagr = (
                        (bench_series.iloc[-1] / bench_series.iloc[0])
                        ** (1 / years)
                        - 1
                    )
                    sharpe = (
                        (bench_rets.mean() / bench_rets.std()) * np.sqrt(252)
                        if bench_rets.std() != 0
                        else 0
                    )
                    running_max = bench_series.cummax()
                    dd = (bench_series / running_max) - 1
                    max_dd = dd.min()

                    color = asset_color_map.setdefault(
                        col, default_colors[len(asset_color_map) % len(default_colors)]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=bench_norm.index,
                            y=bench_norm,
                            mode="lines",
                            name=(
                                f"Benchmark ({col})<br>"
                                f"Total: {bench_total_ret:.1%}, CAGR: {cagr:.1%}<br>"
                                f"Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.1%}"
                            ),
                            line=dict(color=color, width=1, dash="dot"),
                        ),
                        row=1,
                        col=1,
                    )

        # Weights subplot (stacked area)
        if self.asset_weights is not None and not self.asset_weights.empty:
            w_df = self.asset_weights.loc[start_date:end_date]
            for idx, col in enumerate(w_df.columns):
                color = asset_color_map.setdefault(
                    col, default_colors[len(asset_color_map) % len(default_colors)]
                )
                fig.add_trace(
                    go.Scatter(
                        x=w_df.index,
                        y=w_df[col],
                        mode="lines",
                        name=f"Weight {col}",
                        stackgroup="weights",
                        line=dict(width=0.5, color=color),
                        opacity=0.8,
                    ),
                    row=2,
                    col=1,
                )

        regime_colors = {
            "prosperity": "#00CC96",
            "inflation": "#FFA15A",
            "deflation": "#636EFA",
            "recession": "#EF553B",
            "sideways": "#9CA3AF",
        }
        rolling_regime_df: Optional[pd.DataFrame] = None
        if self.regime_state_weights is not None and not self.regime_state_weights.empty:
            regime_df = self.regime_state_weights.loc[start_date:end_date]
            rolling_regime_df = regime_df.rolling(window=20, min_periods=1).mean()
            for regime_name in [
                "prosperity",
                "inflation",
                "deflation",
                "recession",
                "sideways",
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_regime_df.index,
                        y=rolling_regime_df[regime_name],
                        mode="lines",
                        name=f"Regime {regime_name.title()}",
                        stackgroup="regime_share",
                        line=dict(width=0.5, color=regime_colors[regime_name]),
                        opacity=0.85,
                    ),
                    row=3,
                    col=1,
                )

        if has_confidence_chart and self.regime_confidence_margin is not None:
            confidence_series = self.regime_confidence_margin.loc[start_date:end_date]
            fig.add_trace(
                go.Scatter(
                    x=confidence_series.index,
                    y=confidence_series,
                    mode="lines",
                    name="Regime Confidence Margin",
                    line=dict(color="#FECB52", width=1.8),
                ),
                row=4,
                col=1,
            )
            threshold_x = confidence_series.index
            if rolling_regime_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_regime_df.index,
                        y=rolling_regime_df["recession"],
                        mode="lines",
                        name="20-Day Recession Share",
                        line=dict(color="#EF553B", width=1.8, dash="dash"),
                    ),
                    row=4,
                    col=1,
                )
                threshold_x = rolling_regime_df.index
            if len(threshold_x) > 0:
                for line_name, line_value, line_color, line_dash in [
                    ("Margin Threshold 0.3", 0.3, "#FECB52", "dot"),
                    ("Recession Share Threshold 0.5", 0.5, "#EF553B", "dot"),
                    ("Recession Share Threshold 0.8", 0.8, "#EF553B", "dashdot"),
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=threshold_x,
                            y=[line_value] * len(threshold_x),
                            mode="lines",
                            name=line_name,
                            line=dict(color=line_color, width=1.0, dash=line_dash),
                            opacity=0.7,
                            hovertemplate=f"{line_name}: {line_value:.1f}<extra></extra>",
                        ),
                        row=4,
                        col=1,
                    )

        fig.update_xaxes(title_text="Date", row=total_rows, col=1)
        fig.update_yaxes(
            title_text="Cumulative Return (%)", tickformat=".0%", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Portfolio Weights", tickformat=".0%", row=2, col=1
        )
        fig.update_yaxes(
            title_text="20-Day Share", tickformat=".0%", range=[0.0, 1.0], row=3, col=1
        )
        if has_confidence_chart:
            fig.update_yaxes(title_text="Margin / Share", range=[0.0, 1.0], row=4, col=1)

        fig.update_layout(
            title=(
                f"Strategy & Benchmarks with Weights | "
                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ),
            height=1450 if has_confidence_chart else 1200,
            template="plotly_dark",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0.5)",
            ),
        )

        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(full_path)
        return full_path
