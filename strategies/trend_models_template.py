# =============================================================================
# trend_models_template.py  —  Custom Trend Model Template
#
# This file shows the public interface contract for trend models in TrendAlloc.
# It documents the base class and provides a minimal example implementation.
#
# HOW IT WORKS
# ------------
# BacktestEngine optionally accepts a `trend_model` that produces a score
# ∈ [0, 1] each rebalance period.  The score is passed to the rebalance
# function via `ctx.trend_score`.
#
# Engine discovery order (checks `hasattr` in sequence):
#   1. predict_latest_score_from_matrix(matrix)  — preferred, full price matrix
#   2. predict_latest_score_from_series(close, us3m, us30y)  — single-asset
#   3. predict_score(window_features)             — raw feature fallback
#
# QUICK START
# -----------
# 1. Subclass TrendModelBase in .private_data/trend_models.py
# 2. Implement at least `predict_score`
# 3. Optionally implement `predict_latest_score_from_matrix` for matrix input
# 4. Pass an instance to BacktestEngine via the `trend_model` parameter
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TrendModelBase(ABC):
    """Public interface contract for all trend models.

    BacktestEngine calls one of the following methods (in priority order) to
    obtain a per-period trend score that is forwarded to the rebalance
    function as ``ctx.trend_score``:

    1. :meth:`predict_latest_score_from_matrix` — accepts a 2-D price matrix
       ``(lookback, n_cols)``; preferred when multiple feature columns are used.
    2. ``predict_latest_score_from_series`` — accepts 1-D ``close`` / rate
       series; used for single-asset signal models.
    3. :meth:`predict_score` — raw feature vector fallback; must be implemented
       by all subclasses (abstract).

    Conventions:
    - All scores are in ``[0, 1]``: ``1.0`` = strong uptrend, ``0.0`` = bear.
    - ``0.5`` is the neutral / sideways value.
    - Return ``default_score`` (e.g. ``0.5``) on insufficient data.
    """

    @abstractmethod
    def predict_score(self, window_features: np.ndarray) -> float:
        """Return a scalar trend score from a pre-extracted feature vector.

        Args:
            window_features: 1-D feature array whose length matches the model's
                expected input dimension.

        Returns:
            float score in [0, 1].
        """
        raise NotImplementedError

    def predict_latest_score_from_matrix(self, matrix: np.ndarray) -> float:
        """Return a trend score from a raw multi-column price/rate matrix.

        Override this method when your model needs access to the full
        ``(lookback, n_cols)`` price matrix to extract features itself.
        The default implementation raises ``NotImplementedError`` so that
        BacktestEngine falls back to the next method in the discovery chain.

        Args:
            matrix: 2-D array of shape ``(n_timesteps, n_cols)``.

        Returns:
            float score in [0, 1].
        """
        raise NotImplementedError

    def predict_asset_scores(self, window_features: np.ndarray) -> np.ndarray:
        """Return per-asset trend scores (optional helper, not used by engine).

        Default implementation: maps mean simple return over the window into
        [0, 1] using a ±0.5 % linear scale.

        Args:
            window_features: Price matrix of shape ``(window_len, n_assets)``.

        Returns:
            np.ndarray of shape ``(n_assets,)`` with scores in [0, 1].
        """
        if window_features.size == 0:
            return np.zeros((window_features.shape[1],), dtype=float)

        returns = np.diff(window_features, axis=0) / (window_features[:-1] + 1e-12)
        if returns.size == 0:
            return np.zeros((window_features.shape[1],), dtype=float)

        mean_rets = np.nanmean(returns, axis=0)
        scaled = (mean_rets + 0.005) / 0.01  # map [-0.5%, +0.5%] → [0, 1]
        return np.clip(scaled, 0.0, 1.0)


class SimpleMomentumTrendModel(TrendModelBase):
    """Minimal example: momentum-based trend model using mean simple return.

    This is a fully self-contained reference implementation that requires no
    trained model file.  It is intentionally simple — for production use,
    see the concrete model classes in ``trend_models.py``.
    """

    def __init__(self, default_score: float = 0.5) -> None:
        """Initialise with an optional fallback score.

        Args:
            default_score: Score returned when insufficient data is available.
        """
        self.default_score = default_score

    def predict_score(self, window_features: np.ndarray) -> float:
        """Score based on mean simple return of the feature window.

        Args:
            window_features: 1-D or 2-D array.  If 2-D, the mean across all
                columns is used.

        Returns:
            float score in [0, 1].
        """
        if window_features.size == 0:
            return self.default_score

        arr = np.asarray(window_features, dtype=float)
        # Accept both 1-D feature vectors and 2-D price windows
        if arr.ndim == 2:
            rets = np.diff(arr, axis=0) / (arr[:-1] + 1e-12)
            mean_ret = float(np.nanmean(rets))
        else:
            mean_ret = float(np.nanmean(arr))

        scaled = (mean_ret + 0.005) / 0.01
        return float(np.clip(scaled, 0.0, 1.0))
