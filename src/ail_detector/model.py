"""Core anomaly detection model.

Implements ``AnomalyDetector`` – a patch-based memory-bank method that:
1. Extracts multi-scale features from a pretrained backbone
2. Compresses them with coreset subsampling (AdaptiveCoreset by default)
3. Scores test images via softmax-weighted nearest-neighbour search
4. Produces pixel-level anomaly heatmaps with adaptive Gaussian smoothing

Also provides the spatial ``PatchMaker`` helper.
"""
import logging
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import ail_detector.backbones as backbones_lib
import ail_detector.feature_extractor as fe
import ail_detector.coreset as coreset_lib

LOGGER = logging.getLogger(__name__)


class AnomalyDetector(torch.nn.Module):
    """Patch-based industrial anomaly detector.

    Training: fills a memory bank with patch embeddings from normal images.
    Inference: scores query images by the distance to their nearest neighbours
    in the memory bank, and produces spatial heatmaps for anomaly localisation.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device: torch.device,
        input_shape,
        pretrain_embed_dimension: int,
        target_embed_dimension: int,
        patchsize: int = 3,
        patchstride: int = 1,
        anomaly_score_num_nn: int = 5,
        featuresampler=None,
        nn_method=None,
        scorer_temperature: float = 1.0,
        **kwargs,
    ):
        """Initialise all sub-modules.

        Args:
            backbone: Pretrained feature extractor (torchvision / timm / DINOv2).
            layers_to_extract_from: Layer names to hook into.
            device: Compute device.
            input_shape: (C, H, W) of input images.
            pretrain_embed_dimension: Common dimension after per-layer projection.
            target_embed_dimension: Final patch embedding size.
            patchsize: Side length of the patch kernel.
            patchstride: Stride for patch unfolding.
            anomaly_score_num_nn: Neighbours used during scoring.
            featuresampler: Coreset sampler; defaults to AdaptiveCoreset.
            nn_method: FAISS wrapper; defaults to FaissNN(cpu, 4 workers).
            scorer_temperature: Temperature for SoftmaxNNScorer.
        """
        if featuresampler is None:
            featuresampler = coreset_lib.AdaptiveCoreset(device=device)
        if nn_method is None:
            nn_method = fe.FaissNN(False, 4)

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = fe.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)

        # DINOv2 ViT-B always outputs 768-dimensional patch tokens.
        backbone_name = getattr(self.backbone, "name", "")
        if "dinov2" in backbone_name.lower():
            feature_dimensions = [768 for _ in self.layers_to_extract_from]

        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.forward_modules["preprocessing"] = fe.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = fe.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # Use our improved SoftmaxNNScorer
        self.anomaly_scorer = fe.SoftmaxNNScorer(
            n_nearest_neighbours=anomaly_score_num_nn,
            nn_method=nn_method,
            temperature=scorer_temperature,
        )
        # Use our improved AdaptiveRescaleSegmentor
        self.anomaly_segmentor = fe.AdaptiveRescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        self.featuresampler = featuresampler

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, data):
        """Compute patch embeddings for a dataloader or a single batch."""
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    features.append(self._embed(image.to(torch.float).to(self.device)))
            return features
        return self._embed(data)

    def _embed(self, images: torch.Tensor, detach: bool = True, provide_patch_shapes: bool = False):
        """Extract and patchify features from a batch of images."""

        def _detach(feats):
            if detach:
                return [x.detach().cpu().numpy() for x in feats]
            return feats

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        backbone_name = getattr(self.backbone, "name", "")

        # Handle DINOv2 token layout: (B, N_tokens, C) → (B, C, H, W)
        if "dinov2" in backbone_name.lower():
            spatial_features = []
            for feat in features:
                B, N, C = feat.shape
                # Drop CLS + 4 register tokens (DINOv2-reg has 5 prefix tokens)
                patch_tokens = feat[:, 5:, :]
                H = W = int(math.sqrt(patch_tokens.shape[1]))
                feat_2d = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
                spatial_features.append(feat_2d)
            features = spatial_features

        # Handle EfficientNet block output: may be (B, C, H, W) already
        # timm EfficientNet blocks return a plain tensor – no special handling needed.

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        # Align all layers to the spatial resolution of the first layer.
        for i in range(1, len(features)):
            _f = features[i]
            patch_dims = patch_shapes[i]
            _f = _f.reshape(_f.shape[0], patch_dims[0], patch_dims[1], *_f.shape[2:])
            _f = _f.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _f.shape
            _f = _f.reshape(-1, *_f.shape[-2:])
            _f = F.interpolate(
                _f.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            _f = _f.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _f = _f.permute(0, -2, -1, 1, 2, 3)
            _f = _f.reshape(len(_f), -1, *_f.shape[-3:])
            features[i] = _f

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, training_data):
        """Build the memory bank from normal training images."""
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        _ = self.forward_modules.eval()

        def _to_features(img):
            with torch.no_grad():
                return self._embed(img.to(torch.float).to(self.device))

        features = []
        with tqdm.tqdm(
            input_data, desc="Building memory bank...", position=1, leave=False
        ) as it:
            for image in it:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, data):
        """Return anomaly scores and segmentation heatmaps.

        Returns:
            scores: List[float] – per-image anomaly score (raw, unnormalised).
            segmentations: List[np.ndarray] – per-image heatmaps (H x W).
            labels_gt: List[int] – ground-truth image labels.
            masks_gt: List[np.ndarray] – ground-truth pixel masks.
        """
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        _ = self.forward_modules.eval()
        scores, masks, labels_gt, masks_gt = [], [], [], []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as it:
            for image in it:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images: torch.Tensor):
        """Score a single batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
        return list(image_scores), list(masks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _params_file(filepath: str, prepend: str = "") -> str:
        return os.path.join(filepath, prepend + "detector_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving AnomalyDetector to %s …", save_path)
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": getattr(self.backbone, "name", ""),
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method=None,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading AnomalyDetector from %s …", load_path)
        if nn_method is None:
            nn_method = fe.FaissNN(False, 4)
        with open(self._params_file(load_path, prepend), "rb") as f:
            params = pickle.load(f)
        bb_name = params.pop("backbone.name")
        backbone = backbones_lib.load(bb_name)
        backbone.name = bb_name
        params["backbone"] = backbone
        self.load(**params, device=device, nn_method=nn_method)
        self.anomaly_scorer.load(load_path, prepend)


# ---------------------------------------------------------------------------
# Patch helper
# ---------------------------------------------------------------------------

class PatchMaker:
    """Convert convolutional feature maps into overlapping patch tensors."""

    def __init__(self, patchsize: int, stride: int = None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        """Unfold [B x C x H x W] into [B*nH*nW x C x p x p] patches."""
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded = unfolder(features)
        n_patches = []
        for s in features.shape[-2:]:
            np_ = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            n_patches.append(int(np_))
        unfolded = unfolded.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        ).permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded, n_patches
        return unfolded

    def unpatch_scores(self, x: np.ndarray, batchsize: int) -> np.ndarray:
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = isinstance(x, np.ndarray)
        if was_numpy:
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        return x.numpy() if was_numpy else x
