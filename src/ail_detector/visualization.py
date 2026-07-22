"""Visualisation utilities for anomaly segmentation results."""
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tqdm


def plot_segmentation_images(
    savefolder: str,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth: int = 4,
):
    """Save side-by-side panels of original image, ground-truth mask, and heatmap.

    Args:
        savefolder: Directory where images are saved.
        image_paths: List of paths to test images.
        segmentations: Predicted anomaly heatmaps [N x H x W].
        anomaly_scores: Per-image anomaly scores (optional).
        mask_paths: Paths to ground-truth binary masks (optional).
        image_transform: Transform applied before displaying the image.
        mask_transform: Transform applied before displaying the mask.
        save_depth: Number of trailing path components used as save filename.
    """
    if mask_paths is None:
        mask_paths = [None] * len(image_paths)
    masks_provided = any(p is not None for p in mask_paths)
    if anomaly_scores is None:
        anomaly_scores = [None] * len(image_paths)

    os.makedirs(savefolder, exist_ok=True)
    n_panels = 2 + int(masks_provided)

    for image_path, mask_path, score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Saving segmentation images…",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        f, axes = plt.subplots(1, n_panels)
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title("Input")

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
            axes[1].imshow(mask.transpose(1, 2, 0))
            axes[1].set_title("Ground Truth")
            axes[2].imshow(segmentation, cmap="hot")
            axes[2].set_title(f"Score: {score:.4f}" if score is not None else "Heatmap")
        else:
            axes[1].imshow(segmentation, cmap="hot")
            axes[1].set_title(f"Score: {score:.4f}" if score is not None else "Heatmap")

        for ax in axes:
            ax.axis("off")

        # Build save filename from trailing path components.
        parts = image_path.replace("\\", "/").split("/")
        savename = "_".join(parts[-save_depth:])
        f.set_size_inches(3 * n_panels, 3)
        f.tight_layout()
        f.savefig(os.path.join(savefolder, savename))
        plt.close(f)
