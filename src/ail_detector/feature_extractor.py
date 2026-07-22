"""Network-level feature extraction, scoring, and segmentation utilities.

Key components
--------------
FaissNN                  – exact L2 nearest-neighbour search via FAISS
NetworkFeatureAggregator – hook-based multi-layer feature extraction
Preprocessing / Aggregator – dimensionality mapping modules
NearestNeighbourScorer   – baseline k-NN scorer (mean distance)
SoftmaxNNScorer          – our improvement: softmax-weighted distance
RescaleSegmentor         – upsample patch scores to image resolution
AdaptiveRescaleSegmentor – our improvement: adaptive Gaussian sigma
"""
import copy
import math
import os
import pickle
from typing import List, Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FAISS nearest-neighbour search wrappers
# ---------------------------------------------------------------------------

class FaissNN:
    """Exact L2 nearest-neighbour search via FAISS.

    Args:
        on_gpu: Run index on GPU (requires faiss-gpu).
        num_workers: CPU threads used by FAISS.
    """

    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension: int):
        if self.on_gpu and hasattr(faiss, "GpuIndexFlatL2"):
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(),
                dimension,
                faiss.GpuIndexFlatConfig(),
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """Add feature vectors to the search index."""
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass  # FlatL2 does not require training

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


# ---------------------------------------------------------------------------
# Feature mergers
# ---------------------------------------------------------------------------

class _BaseMerger:
    """Base class: concatenate reduced feature arrays along channel axis."""

    def merge(self, features: list) -> np.ndarray:
        features = [self._reduce(f) for f in features]
        return np.concatenate(features, axis=1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features: np.ndarray) -> np.ndarray:
        return features.reshape(len(features), -1)


# ---------------------------------------------------------------------------
# Dimensionality reduction modules
# ---------------------------------------------------------------------------

class Preprocessing(torch.nn.Module):
    """Map each feature layer to a common embedding dimension."""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList(
            [MeanMapper(output_dim) for _ in input_dims]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        out = [m(f) for m, f in zip(self.preprocessing_modules, features)]
        return torch.stack(out, dim=1)


class MeanMapper(torch.nn.Module):
    """Adaptive average pool to a fixed channel dimension."""

    def __init__(self, preprocessing_dim: int):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    """Pool multi-layer features to a single target dimension."""

    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


# ---------------------------------------------------------------------------
# Segmentation (patch score → image heatmap)
# ---------------------------------------------------------------------------

class RescaleSegmentor:
    """Upsample patch-level scores to image size with fixed Gaussian smoothing.

    Args:
        device: Torch device.
        target_size: Output (H, W) in pixels.
        smoothing: Gaussian sigma applied after upsampling.
    """

    def __init__(self, device: torch.device, target_size=224, smoothing: float = 4.0):
        self.device = device
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores: np.ndarray) -> List[np.ndarray]:
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device).unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            patch_scores = _scores.squeeze(1).cpu().numpy()
        return [
            ndimage.gaussian_filter(s, sigma=self.smoothing) for s in patch_scores
        ]


class AdaptiveRescaleSegmentor(RescaleSegmentor):
    """Our improvement: adaptive Gaussian sigma based on feature map resolution.

    Instead of a fixed ``sigma=4``, the smoothing strength scales with the
    spatial size of the patch feature map:

        sigma = clamp(feature_h / 14, min=2.0, max=6.0)

    This preserves fine-grained localisation for high-resolution feature maps
    while still smoothing over noise for coarser ones.

    Args:
        device: Torch device.
        target_size: Output (H, W) in pixels.
    """

    def __init__(self, device: torch.device, target_size=224):
        super().__init__(device=device, target_size=target_size, smoothing=4.0)
        self._patch_h: int = 28  # default; updated in convert_to_segmentation

    def convert_to_segmentation(self, patch_scores: np.ndarray) -> List[np.ndarray]:
        if isinstance(patch_scores, np.ndarray):
            h = patch_scores.shape[-2] if patch_scores.ndim >= 2 else 28
        else:
            h = patch_scores.shape[-2] if patch_scores.ndim >= 2 else 28
        # Adaptive sigma: scales with feature resolution
        sigma = float(np.clip(h / 14.0, 2.0, 6.0))

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device).unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            patch_scores = _scores.squeeze(1).cpu().numpy()
        return [
            ndimage.gaussian_filter(s, sigma=sigma) for s in patch_scores
        ]


# ---------------------------------------------------------------------------
# Forward-hook based feature aggregation
# ---------------------------------------------------------------------------

class LastLayerToExtractReachedException(Exception):
    """Raised to short-circuit backbone forward pass after target layers."""


class ForwardHook:
    """Captures intermediate activations via PyTorch forward hooks."""

    def __init__(self, hook_dict: dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()


class NetworkFeatureAggregator(torch.nn.Module):
    """Extract features from specified intermediate layers of a backbone.

    Registers forward hooks on all requested layers; raises an exception
    to stop the forward pass early once the deepest required layer is reached.
    """

    def __init__(self, backbone, layers_to_extract_from: List[str], device: torch.device):
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    network_layer = network_layer[int(extract_idx)]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

        self.to(self.device)

    def forward(self, images: torch.Tensor) -> dict:
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape: List[int]) -> List[int]:
        """Compute feature channel sizes for each target layer."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        dims = []
        for layer in self.layers_to_extract_from:
            feat = _output[layer]
            if feat.ndim == 3:
                # ViT token layout: (B, N_tokens, C) → channel is last dim
                dims.append(feat.shape[2])
            else:
                # CNN layout: (B, C, H, W) → channel is dim 1
                dims.append(feat.shape[1])
        return dims


# ---------------------------------------------------------------------------
# Nearest-neighbour anomaly scorers
# ---------------------------------------------------------------------------

class NearestNeighbourScorer:
    """Score test patches by mean distance to their k nearest training neighbours.

    This is the baseline scoring method from the original paper.

    Args:
        n_nearest_neighbours: Number of neighbours used for scoring.
        nn_method: Underlying FAISS wrapper instance.
    """

    def __init__(self, n_nearest_neighbours: int, nn_method=None) -> None:
        if nn_method is None:
            nn_method = FaissNN(False, 4)
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method
        self.imagelevel_nn = lambda q: self.nn_method.run(n_nearest_neighbours, q)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder: str, prepend: str = "") -> str:
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder: str, prepend: str = "") -> str:
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename: str, features) -> None:
        if features is None:
            return
        with open(filename, "wb") as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        det_file = self._detection_file(load_folder, prepend)
        if os.path.exists(det_file):
            self.detection_features = self._load(det_file)


class SoftmaxNNScorer(NearestNeighbourScorer):
    """Our improvement: softmax-weighted nearest-neighbour scoring.

    Instead of averaging k nearest-neighbour distances equally, this scorer
    applies a softmax weighting that emphasises the closest (most anomalous)
    neighbours.  This makes the per-patch score more sensitive to genuine
    anomalies and less influenced by distant, noisy neighbours.

    The score for a query patch q with distances d_1 ≤ d_2 ≤ ... ≤ d_k is:

        w_i = softmax(-d_i / T)_i
        score(q) = sum(w_i * d_i)

    where T is a temperature parameter controlling the sharpness of weighting.
    At T → ∞ this reduces to the baseline mean; at T → 0 it selects only d_1.

    Args:
        n_nearest_neighbours: Number of neighbours used for scoring.
        nn_method: Underlying FAISS wrapper instance.
        temperature: Softmax temperature (default 1.0).
    """

    def __init__(
        self,
        n_nearest_neighbours: int,
        nn_method=None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(n_nearest_neighbours, nn_method)
        self.temperature = temperature

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)

        # Softmax-weighted aggregation
        logits = -query_distances / max(self.temperature, 1e-8)  # (N, k)
        logits -= logits.max(axis=-1, keepdims=True)             # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True)

        anomaly_scores = np.sum(weights * query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns
