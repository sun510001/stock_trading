import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Union, Optional, List, Callable
from logger import logger

from strategies.algorithms import RebalanceAlgorithms
from strategies.trend_models import get_trend_model


class BacktestEngine:
    """Core backtest engine handling simulation and visualization."""

    def __init__(self, data_path: str, initial_capital: float = 10000.0) -> None:
        self.data_path: str = data_path
        self.initial_capital: float = initial_capital
        self.data: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.Series] = None
        self.drawdown: Optional[pd.Series] = None
        self.asset_weights: Optional[pd.DataFrame] = None

        self._load_data()

    def _load_data(self) -> None:
        """Load price data from the specified CSV file."""
        try:
            logger.info(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path, index_col="Date", parse_dates=True)
            self.data = df.astype(float)
            logger.info(f"Data loaded. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: str = "QE",
        fees: float = 0.0005,
        rebalance_fn: Callable = RebalanceAlgorithms.permanent_portfolio_rebalance,
        rebalance_interval_days: Optional[int] = None,
        asset_cols: Optional[List[str]] = None,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.5,
        model_type: str = "kmeans",
    ) -> None:
        """Execute the backtest simulation."""
        logger.info(
            f"Preparing simulation for range: {start_date or 'Start'} to {end_date or 'End'}..."
        )

        if self.data is None:
            logger.error("No data available for backtest.")
            return

        # 1. Slice Data (Time Filtering)
        df_slice = self.data.copy()
        if start_date:
            df_slice = df_slice.loc[start_date:]
        if end_date:
            df_slice = df_slice.loc[:end_date]

        if df_slice.empty:
            logger.error("No data found for the specified date range!")
            return

        # Optionally restrict to a subset of asset columns (strategy asset universe)
        if asset_cols:
            missing = [c for c in asset_cols if c not in df_slice.columns]
            if missing:
                raise RuntimeError(f"Missing asset columns in data: {missing}")
            df_slice = df_slice[asset_cols]

        actual_start = df_slice.index[0].strftime("%Y-%m-%d")
        actual_end = df_slice.index[-1].strftime("%Y-%m-%d")
        logger.info(
            f"Actual Backtest Range: {actual_start} to {actual_end} ({len(df_slice)} days)"
        )

        # 2. Setup Variables
        prices = df_slice.values
        dates = df_slice.index
        n_days, n_assets = prices.shape

        # Identify rebalance indices
        if rebalance_interval_days is not None and rebalance_interval_days > 0:
            rb_indices = set(range(0, n_days, rebalance_interval_days))
        else:
            rb_dates = df_slice.index.to_series().resample(rebalance_freq).last().index
            rb_indices = set(df_slice.index.get_indexer(rb_dates, method="ffill"))

        # 3. Initialization
        portfolio_history = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))

        start_prices = prices[0]
        target_allocation = self.initial_capital / n_assets
        current_units = (target_allocation / start_prices) * (1 - fees)
        portfolio_history[0] = float(np.sum(current_units * start_prices))
        weights_history[0] = (current_units * start_prices) / portfolio_history[0]

        # 4. Time Loop
        for i in range(1, n_days):
            today_prices = prices[i]
            current_val = float(np.sum(current_units * today_prices))

            if i in rb_indices:
                signals = None

                if use_trend_model and model_lookback_days > 0:
                    if i < model_lookback_days:
                        portfolio_history[i] = current_val
                        weights_history[i] = (current_units * today_prices) / current_val
                        continue

                    window = df_slice.iloc[i - model_lookback_days : i]
                    if window.empty:
                        portfolio_history[i] = current_val
                        weights_history[i] = (current_units * today_prices) / current_val
                        continue

                    window_arr = window.values.astype(float)
                    model = get_trend_model(model_type=model_type)

                    # Global trend score used to decide whether to allow rebalancing at this index
                    trend_score = float(model.predict_score(window_arr))
                    trend_score = max(0.0, min(1.0, trend_score))
                    logger.info(
                        f"Trend model | day_idx={i}, lookback={model_lookback_days}, "
                        f"score={trend_score:.3f}, model_type={model_type}"
                    )
                    if trend_score < model_threshold:
                        logger.info(
                            f"Rebalance skipped at index {i} due to low trend_score={trend_score:.3f} "
                            f"(threshold={model_threshold:.3f})"
                        )
                        portfolio_history[i] = current_val
                        weights_history[i] = (current_units * today_prices) / current_val
                        continue

                    # 每个资产的信号, 供信号加权算法使用
                    signals = model.predict_asset_scores(window_arr)

                # 根据是否为信号加权算法选择不同调用方式
                if (
                    rebalance_fn is RebalanceAlgorithms.signal_weighted_rebalance
                    and signals is not None
                ):
                    current_units, current_val = rebalance_fn(
                        current_units=current_units,
                        prices=today_prices,
                        fees=fees,
                        signals=signals,
                    )
                else:
                    current_units, current_val = rebalance_fn(
                        current_units=current_units,
                        prices=today_prices,
                        fees=fees,
                    )

            portfolio_history[i] = current_val
            weights_history[i] = (current_units * today_prices) / current_val

        # 5. Finalize
        self.portfolio_value = pd.Series(
            portfolio_history, index=dates, name="Portfolio Value"
        )
        self.asset_weights = pd.DataFrame(
            weights_history, index=dates, columns=df_slice.columns
        )
        running_max = self.portfolio_value.cummax()
        self.drawdown = (self.portfolio_value / running_max) - 1

        logger.info("Simulation complete.")

    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics for the backtest."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            return {}

        returns = self.portfolio_value.pct_change().dropna()
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

        return {
            "Start Value": float(start_val),
            "End Value": float(end_val),
            "Total Return": float((end_val / start_val) - 1),
            "CAGR": float(cagr),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown": float(max_dd),
            "Volatility": float(vol),
            "Years": float(n_years),
        }

    def plot_results(
        self, output_dir: str, benchmark_cols: Union[str, List[str]] = "Stocks"
    ) -> str:
        """Generate a comparative plot of strategy vs benchmarks."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            logger.warning("Portfolio value is empty. Cannot plot.")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.html"
        full_path = os.path.join(output_dir, filename)

        logger.info(f"Generating comparative plot to {full_path}...")

        start_date = self.portfolio_value.index[0]
        end_date = self.portfolio_value.index[-1]
        strategy_norm = (self.portfolio_value / self.portfolio_value.iloc[0]) - 1

        if isinstance(benchmark_cols, str):
            benchmark_list = [benchmark_cols]
        else:
            benchmark_list = list(benchmark_cols)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=(
                f"Performance: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "Portfolio Weights",
            ),
        )

        # Strategy stats
        strat_rets = self.portfolio_value.pct_change().dropna()
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
                    f"Strategy | Total: {strat_total_ret:.1%}, "
                    f"CAGR: {strat_cagr:.1%}, "
                    f"Sharpe: {strat_sharpe:.2f}, "
                    f"MaxDD: {strat_max_dd:.1%}"
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
        ]
        for idx, col in enumerate(benchmark_list):
            if self.data is not None and col in self.data.columns:
                bench_series = self.data[col].loc[start_date:end_date]
                if not bench_series.empty:
                    bench_norm = (bench_series / bench_series.iloc[0]) - 1
                    bench_total_ret = bench_norm.iloc[-1]

                    bench_rets = bench_series.pct_change().dropna()
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

                    color = default_colors[idx % len(default_colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=bench_norm.index,
                            y=bench_norm,
                            mode="lines",
                            name=(
                                f"Benchmark ({col}) | Total: {bench_total_ret:.1%}, "
                                f"CAGR: {cagr:.1%}, "
                                f"Sharpe: {sharpe:.2f}, "
                                f"MaxDD: {max_dd:.1%}"
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
                color = default_colors[idx % len(default_colors)]
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

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(
            title_text="Cumulative Return (%)", tickformat=".0%", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Portfolio Weights", tickformat=".0%", row=2, col=1
        )

        fig.update_layout(
            title= (
                f"Strategy & Benchmarks with Weights | "
                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ),
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
            ),
        )

        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(full_path)
        return full_path
