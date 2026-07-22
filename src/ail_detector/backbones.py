"""Backbone feature extractor registry.

Supports WideResNet-50, EfficientNet-B3 (lightweight default), and DINOv2 ViT-B/14.
EfficientNet-B3 is our primary backbone – ~5x fewer parameters than WideResNet-50
while maintaining competitive anomaly detection performance.
"""
import timm
import torch
import torchvision.models as models  # noqa


_BACKBONES = {
    # --- Lightweight (recommended for most use-cases) ---
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',

    # --- Standard ResNet family ---
    "resnet50":     "models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)",
    "wideresnet50": "models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)",

    # --- Vision Transformer (DINOv2) ---
    "dinov2_vitb14": "torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')",
}

# Default layer names per backbone (used when --layers is not specified)
BACKBONE_LAYER_DEFAULTS = {
    "efficientnet_b3": ["blocks.5", "blocks.6"],
    "resnet50":        ["layer2", "layer3"],
    "wideresnet50":    ["layer2", "layer3"],
    "dinov2_vitb14":   ["blocks.9", "blocks.11"],
}


def load(name: str):
    """Load a pretrained backbone by registry name.

    Args:
        name: Key in ``_BACKBONES``.

    Returns:
        Loaded PyTorch model (not yet moved to device).

    Raises:
        ValueError: If ``name`` is not a registered backbone.
    """
    if name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(_BACKBONES.keys())}"
        )
    return eval(_BACKBONES[name])  # noqa: S307


def default_layers(name: str):
    """Return the default layer names to hook for a given backbone."""
    if name in BACKBONE_LAYER_DEFAULTS:
        return BACKBONE_LAYER_DEFAULTS[name]
    # Fall back: DINOv2 uses ViT layout, others use ResNet layout
    if "dinov2" in name.lower():
        return ["blocks.9", "blocks.11"]
    return ["layer2", "layer3"]
