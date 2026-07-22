"""Coreset subsampling strategies for memory-bank compression.

Provides three strategies:
- ``IdentityCoreset``        – keep everything (no compression)
- ``AdaptiveCoreset``        – our improvement: auto-selects retention rate by
                               dataset size using approximate greedy farthest-point
- ``ApproximateGreedyCoreset`` – fixed-rate greedy coreset (paper baseline)
"""
import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentityCoreset:
    """No-op – keeps all features."""

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class _BaseCoreset(abc.ABC):
    """Abstract base for percentage-based coreset samplers."""

    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("percentage must be in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class ApproximateGreedyCoreset(_BaseCoreset):
    """Memory-efficient approximate greedy coreset (paper baseline).

    Avoids the full N×N distance matrix by initialising anchor distances
    against a small random subset of starting points, then greedily picks
    the farthest point at each step.

    Args:
        percentage: Fraction of features to retain (0, 1).
        device: Torch device for distance computation.
        number_of_starting_points: Size of the initial random anchor set.
        dimension_to_project_features_to: Random-projection dim before greedy
            selection – reduces memory without hurting coverage quality.
    """

    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        super().__init__(percentage)
        self.device = device
        self.number_of_starting_points = number_of_starting_points
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        ).to(self.device)
        return mapper(features.to(self.device))

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        a_sq = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_sq = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        ab = matrix_a.mm(matrix_b.T)
        return (-2 * ab + a_sq + b_sq).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        n_start = int(np.clip(self.number_of_starting_points, None, len(features)))
        start_pts = np.random.choice(len(features), n_start, replace=False).tolist()

        approx_dists = self._compute_batchwise_differences(
            features, features[start_pts]
        )
        approx_dists = torch.mean(approx_dists, axis=-1).reshape(-1, 1)
        indices = []
        n_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_samples), desc="Coreset subsampling..."):
                idx = torch.argmax(approx_dists).item()
                indices.append(idx)
                new_dist = self._compute_batchwise_differences(
                    features, features[idx : idx + 1]  # noqa: E203
                )
                approx_dists = torch.cat([approx_dists, new_dist], dim=-1)
                approx_dists = torch.min(approx_dists, dim=1).values.reshape(-1, 1)

        return np.array(indices)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.percentage >= 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced = self._reduce_features(features)
        indices = self._compute_greedy_coreset_indices(reduced)
        return self._restore_type(features[indices])


class AdaptiveCoreset(ApproximateGreedyCoreset):
    """Our improvement: automatically selects the coreset retention rate.

    Instead of requiring the user to hand-tune ``coreset_pct``, this sampler
    adapts based on the total number of patch features in the memory bank:

    - Small memory bank  (< 30 000 patches)  → retain 15 %
    - Medium             (30 000 – 80 000)    → retain 10 %
    - Large              (> 80 000 patches)   → retain  7 %

    The farthest-point greedy strategy is the same as ``ApproximateGreedyCoreset``.

    Args:
        device: Torch device for distance computation.
        number_of_starting_points: Size of the initial random anchor set.
        dimension_to_project_features_to: Random-projection dim.
    """

    _THRESHOLDS = [
        (30_000, 0.15),
        (80_000, 0.10),
        (float("inf"), 0.07),
    ]

    def __init__(
        self,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        # Dummy percentage; will be overridden at runtime in `run()`
        super().__init__(
            percentage=0.10,
            device=device,
            number_of_starting_points=number_of_starting_points,
            dimension_to_project_features_to=dimension_to_project_features_to,
        )

    def _select_rate(self, n: int) -> float:
        for threshold, rate in self._THRESHOLDS:
            if n < threshold:
                return rate
        return 0.07

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        n = len(features)
        self.percentage = self._select_rate(n)
        return super().run(features)
