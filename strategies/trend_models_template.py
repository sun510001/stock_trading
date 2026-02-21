from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

ModelType = Literal["kmeans"]


class TrendModelBase(ABC):
    """Unified interface for unsupervised trend models.

    Conventions:
    - Input: feature matrix with shape (window_len, n_features) or (window_len, n_assets)
    - Output: trend_score ∈ [0, 1], where larger values indicate stronger uptrend likelihood.
    """

    @abstractmethod
    def predict_score(self, window_features: np.ndarray) -> float:
        """Return a scalar trend score based on the given window of features."""
        raise NotImplementedError

    def predict_asset_scores(self, window_features: np.ndarray) -> np.ndarray:
        """Return per-asset trend scores.

        Default placeholder implementation:
        - Compute the average simple return for each asset over the window
        - Linearly map the result into [0, 1]

        Args:
            window_features: Price matrix with shape (window_len, n_assets).

        Returns:
            np.ndarray: shape=(n_assets,), per-asset scores in [0, 1].
        """
        if window_features.size == 0:
            return np.zeros((window_features.shape[1],), dtype=float)

        returns = np.diff(window_features, axis=0) / window_features[:-1]
        if returns.size == 0:
            return np.zeros((window_features.shape[1],), dtype=float)

        mean_rets = np.nanmean(returns, axis=0)  # (n_assets,)
        scaled = (mean_rets + 0.005) / 0.01  # map approx [-0.5%, +0.5%] to [0, 1]
        return np.clip(scaled, 0.0, 1.0)


class KMeansTrendModel(TrendModelBase):
    """K-means pattern-clustering based trend model (placeholder implementation)."""

    def __init__(self) -> None:
        # In a real implementation, this could load a trained KMeans model
        # and a mapping from cluster labels to trend scores.
        pass

    def predict_score(self, window_features: np.ndarray) -> float:
        if window_features.size == 0:
            return 0.0

        returns = np.diff(window_features, axis=0) / window_features[:-1]
        if returns.size == 0:
            return 0.0

        mean_ret = float(np.nanmean(returns))
        scaled = (mean_ret + 0.005) / 0.01
        return float(max(0.0, min(1.0, scaled)))


def get_trend_model(model_type: ModelType) -> TrendModelBase:
    """Return a trend model instance based on the given model_type string."""
    if model_type == "kmeans":
        return KMeansTrendModel()
