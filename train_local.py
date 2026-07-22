"""Train and evaluate anomaly detectors on MVTec-format data.

Supports multiple backbones in a single run with side-by-side AUROC comparison.
Outputs: per-image anomaly scores (JSON), segmentation heatmaps (PNG), CSV results.

Usage (from project root):
    # Default: lightweight EfficientNet-B3 backbone
    python train_local.py

    # WideResNet-50 (paper-style baseline)
    python train_local.py --backbone wideresnet50

    # Compare both backbones
    python train_local.py --backbone efficientnet_b3,wideresnet50 --compare

    # Quick test on one class with all outputs
    python train_local.py --classes bottle --save_images --output_scores

Options:
    --data_dir       Path to data folder [default: ./data]
    --results_dir    Where to save results [default: ./results]
    --backbone       Backbone name(s), comma-separated [default: efficientnet_b3]
    --classes        Comma-separated class names (empty = all) [default: all]
    --compare        Print a side-by-side comparison table [flag]
    --layers         Layer names to hook (leave empty for backbone defaults)
    --coreset_mode   'auto' (AdaptiveCoreset) or float pct, e.g. 0.1 [default: auto]
    --target_dim     Memory bank embedding dimension [default: 1024]
    --patchsize      Patch kernel size [default: 3]
    --num_nn         Nearest neighbours for scoring [default: 5]
    --temperature    Softmax scorer temperature [default: 1.0]
    --batch_size     DataLoader batch size [default: 2]
    --num_workers    DataLoader workers [default: 0]
    --imagesize      Centre-crop size [default: 224]
    --resize         Resize size [default: 256]
    --gpu            GPU index (-1 for CPU) [default: 0]
    --seed           Random seed [default: 0]
    --save_models    Save trained model to disk [flag]
    --save_images    Save segmentation heatmap images [flag]
    --output_scores  Save per-image anomaly_scores.json [flag]
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch

import ail_detector.backbones as backbones_lib
import ail_detector.coreset as coreset_lib
import ail_detector.feature_extractor as fe
import ail_detector.metrics as metrics_lib
import ail_detector.utils as utils_lib
import ail_detector.visualization as vis_lib
from ail_detector.model import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
LOGGER = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _collect_classes(data_dir: str, filter_classes=None):
    """Return sorted list of class names present in data_dir."""
    classes = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        if filter_classes and entry not in filter_classes:
            continue
        classes.append(entry)
    return classes


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AIL Anomaly Detector – train and evaluate"
    )
    parser.add_argument("--data_dir",     default="data",             help="Root data folder")
    parser.add_argument("--results_dir",  default="results",          help="Output folder")
    parser.add_argument(
        "--backbone",
        default="efficientnet_b3",
        help="Backbone name(s), comma-separated. Options: efficientnet_b3, wideresnet50, dinov2_vitb14",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class names to train (empty = all).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison table of all backbone results after training.",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="Override layer names (comma-separated). Leave empty for backbone defaults.",
    )
    parser.add_argument(
        "--coreset_mode",
        default="auto",
        help="Coreset strategy: 'auto' (AdaptiveCoreset) or a float like 0.1 (fixed rate).",
    )
    parser.add_argument("--target_dim",   type=int,   default=1024)
    parser.add_argument("--patchsize",    type=int,   default=3)
    parser.add_argument("--num_nn",       type=int,   default=5)
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Softmax scorer temperature (1.0 = mild weighting).")
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--imagesize",    type=int,   default=224)
    parser.add_argument("--resize",       type=int,   default=256)
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--save_models",  action="store_true")
    parser.add_argument("--save_images",  action="store_true")
    parser.add_argument("--output_scores", action="store_true",
                        help="Save per-image anomaly scores to JSON.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Coreset builder
# ---------------------------------------------------------------------------

def build_coreset_sampler(coreset_mode: str, device: torch.device):
    """Build a coreset sampler from the --coreset_mode argument.

    'auto' → AdaptiveCoreset (our improvement: automatically selects rate)
    float  → ApproximateGreedyCoreset with that fixed rate
    '0'    → IdentityCoreset (keep all)
    """
    if coreset_mode.lower() == "auto":
        LOGGER.info("Coreset: AdaptiveCoreset (auto rate selection)")
        return coreset_lib.AdaptiveCoreset(device=device)
    try:
        pct = float(coreset_mode)
    except ValueError:
        raise ValueError(
            f"--coreset_mode must be 'auto' or a float, got '{coreset_mode}'"
        )
    if pct <= 0:
        LOGGER.info("Coreset: IdentityCoreset (keep all)")
        return coreset_lib.IdentityCoreset()
    LOGGER.info("Coreset: ApproximateGreedyCoreset (pct=%.2f)", pct)
    return coreset_lib.ApproximateGreedyCoreset(
        percentage=pct, device=device, number_of_starting_points=10,
    )


# ---------------------------------------------------------------------------
# Train + evaluate one (backbone, class) combination
# ---------------------------------------------------------------------------

COL_NAMES = ["instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"]


def train_one_class(
    classname: str,
    data_dir: str,
    backbone_name: str,
    layers,
    args,
    device: torch.device,
    run_save_path: str,
):
    """Train and evaluate on one class.  Returns a result dict or None."""
    import torch.utils.data
    from ail_detector.datasets.mvtec import MVTecDataset, DatasetSplit

    LOGGER.info("=" * 60)
    LOGGER.info("Class: %-20s | Backbone: %s", classname, backbone_name)
    LOGGER.info("Layers : %s", layers)
    LOGGER.info("=" * 60)

    # Detect nested dataset layout (data/cls/cls/train vs data/cls/train)
    dataset_source = data_dir
    if not os.path.exists(os.path.join(data_dir, classname, "train")):
        if os.path.exists(os.path.join(data_dir, classname, classname, "train")):
            dataset_source = os.path.join(data_dir, classname)
            LOGGER.info("Detected nested dataset structure: %s", dataset_source)

    train_dataset = MVTecDataset(
        source=dataset_source,
        classname=classname,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TRAIN,
    )
    test_dataset = MVTecDataset(
        source=dataset_source,
        classname=classname,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TEST,
    )

    LOGGER.info("Train: %d images  |  Test: %d images",
                len(train_dataset), len(test_dataset))

    if len(train_dataset) == 0:
        LOGGER.warning("No training images for '%s' – skipping.", classname)
        return None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader.name = classname

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Build backbone + detector ----
    backbone = backbones_lib.load(backbone_name)
    backbone.name = backbone_name

    sampler   = build_coreset_sampler(args.coreset_mode, device)
    nn_method = fe.FaissNN(on_gpu=False, num_workers=4)

    input_shape = train_dataset.imagesize  # (3, H, W)
    detector    = AnomalyDetector(device)
    detector.load(
        backbone=backbone,
        layers_to_extract_from=layers,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=args.target_dim,
        target_embed_dimension=args.target_dim,
        patchsize=args.patchsize,
        patchstride=1,
        anomaly_score_num_nn=args.num_nn,
        featuresampler=sampler,
        nn_method=nn_method,
        scorer_temperature=args.temperature,
    )

    # ---- Train ----
    LOGGER.info("Training …")
    utils_lib.fix_seeds(args.seed, with_torch=True, with_cuda=True)
    detector.fit(train_loader)

    if len(test_dataset) == 0:
        LOGGER.warning("No test images for '%s'.", classname)
        return None

    # ---- Evaluate ----
    LOGGER.info("Evaluating …")
    scores, segmentations, labels_gt, masks_gt = detector.predict(test_loader)

    # Normalise image-level scores to [0, 1]
    scores_arr = np.array(scores)
    s_min, s_max = scores_arr.min(), scores_arr.max()
    if s_max > s_min:
        scores_norm = (scores_arr - s_min) / (s_max - s_min)
    else:
        scores_norm = np.zeros_like(scores_arr)

    anomaly_labels = [x[1] != "good" for x in test_dataset.data_to_iterate]

    auroc = metrics_lib.compute_imagewise_retrieval_metrics(
        scores_norm, anomaly_labels
    )["auroc"]

    # Normalise segmentation maps
    seg_arr = np.stack(segmentations)
    seg_min, seg_max = seg_arr.min(), seg_arr.max()
    if seg_max > seg_min:
        seg_arr_norm = (seg_arr - seg_min) / (seg_max - seg_min)
    else:
        seg_arr_norm = np.zeros_like(seg_arr)

    pixel_scores = metrics_lib.compute_pixelwise_retrieval_metrics(
        seg_arr_norm.tolist(), masks_gt
    )
    full_pixel_auroc = pixel_scores["auroc"]

    sel_idxs = [i for i, m in enumerate(masks_gt) if np.sum(m) > 0]
    if sel_idxs:
        anom_pixel_auroc = metrics_lib.compute_pixelwise_retrieval_metrics(
            [seg_arr_norm[i] for i in sel_idxs],
            [masks_gt[i] for i in sel_idxs],
        )["auroc"]
    else:
        anom_pixel_auroc = 0.0

    LOGGER.info("  Instance AUROC      : %.4f", auroc)
    LOGGER.info("  Full Pixel AUROC    : %.4f", full_pixel_auroc)
    LOGGER.info("  Anomaly Pixel AUROC : %.4f", anom_pixel_auroc)

    # ---- Output per-image anomaly scores JSON ----
    if args.output_scores:
        image_paths = [x[2] for x in test_dataset.data_to_iterate]
        threshold   = float(np.median(scores_norm[np.array(anomaly_labels) == 0]) + 2 * np.std(scores_norm))
        threshold   = min(threshold, 0.5)

        score_records = []
        for img_path, score, label in zip(image_paths, scores_norm, anomaly_labels):
            score_records.append({
                "image_path":       img_path.replace("\\", "/"),
                "anomaly_score":    round(float(score), 6),
                "is_anomaly_gt":    int(label),
                "is_anomaly_pred":  int(score >= threshold),
                "threshold":        round(threshold, 6),
            })

        scores_dir = os.path.join(run_save_path, "anomaly_scores", backbone_name)
        os.makedirs(scores_dir, exist_ok=True)
        json_path = os.path.join(scores_dir, f"{classname}_scores.json")
        with open(json_path, "w") as f:
            json.dump({"class": classname, "backbone": backbone_name,
                       "records": score_records}, f, indent=2)
        LOGGER.info("  Scores saved → %s", json_path)

    # ---- Save segmentation images ----
    if args.save_images:
        image_paths = [x[2] for x in test_dataset.data_to_iterate]
        mask_paths  = [x[3] for x in test_dataset.data_to_iterate]
        img_save_dir = os.path.join(
            run_save_path, "segmentation_images", backbone_name, classname
        )
        os.makedirs(img_save_dir, exist_ok=True)

        transform_std  = np.array(train_dataset.transform_std).reshape(-1, 1, 1)
        transform_mean = np.array(train_dataset.transform_mean).reshape(-1, 1, 1)

        def image_transform(img):
            t = train_dataset.transform_img(img)
            return np.clip(
                (t.numpy() * transform_std + transform_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return train_dataset.transform_mask(mask).numpy()

        vis_lib.plot_segmentation_images(
            img_save_dir, image_paths, list(seg_arr_norm),
            scores_norm, mask_paths, image_transform, mask_transform,
        )
        LOGGER.info("  Segmentation images → %s", img_save_dir)

    # ---- Save model ----
    if args.save_models:
        model_dir = os.path.join(run_save_path, "models", backbone_name, classname)
        os.makedirs(model_dir, exist_ok=True)
        detector.save_to_path(model_dir)
        LOGGER.info("  Model saved → %s", model_dir)

    return {
        "backbone":            backbone_name,
        "dataset_name":        classname,
        "instance_auroc":      auroc,
        "full_pixel_auroc":    full_pixel_auroc,
        "anomaly_pixel_auroc": anom_pixel_auroc,
    }


# ---------------------------------------------------------------------------
# Comparison table helpers
# ---------------------------------------------------------------------------

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def print_comparison_table(all_results: dict):
    """Print a side-by-side comparison of every backbone's results."""
    backbones   = sorted(all_results.keys())
    all_classes = sorted({r["dataset_name"] for rs in all_results.values() for r in rs})

    header_bb = "  ".join(f"{bb:>22}" for bb in backbones)
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("BACKBONE COMPARISON  (Instance AUROC)")
    LOGGER.info("=" * 80)
    LOGGER.info("%-22s  %s", "Class", header_bb)
    LOGGER.info("-" * 80)

    for cls in all_classes:
        row_vals = []
        for bb in backbones:
            match = [r for r in all_results[bb] if r["dataset_name"] == cls]
            val = match[0]["instance_auroc"] if match else float("nan")
            row_vals.append(f"{val:>22.4f}")
        LOGGER.info("%-22s  %s", cls, "  ".join(row_vals))

    LOGGER.info("-" * 80)
    for bb in backbones:
        means = [r["instance_auroc"] for r in all_results[bb]]
        LOGGER.info("Mean %-17s  %s", bb, f"{_mean(means):.4f}")
    LOGGER.info("=" * 80)


def save_comparison_csv(all_results: dict, save_path: str):
    """Write one CSV per backbone plus a combined comparison CSV."""
    combined_rows = []
    for bb, results in all_results.items():
        # Per-backbone CSV
        bb_csv = os.path.join(save_path, f"results_{bb}.csv")
        with open(bb_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_name"] + COL_NAMES)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["dataset_name"] + COL_NAMES})
            means = {c: _mean([r[c] for r in results]) for c in COL_NAMES}
            means["dataset_name"] = "MEAN"
            writer.writerow(means)
        LOGGER.info("Saved %s", bb_csv)
        combined_rows.extend(results)

    combined_csv = os.path.join(save_path, "results_comparison.csv")
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backbone", "dataset_name"] + COL_NAMES)
        writer.writeheader()
        writer.writerows(combined_rows)
    LOGGER.info("Saved combined comparison: %s", combined_csv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir    = os.path.abspath(args.data_dir)
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(data_dir):
        LOGGER.error("data_dir '%s' does not exist.", data_dir)
        sys.exit(1)

    gpu_list = [args.gpu] if args.gpu >= 0 else []
    device   = utils_lib.set_torch_device(gpu_list)
    LOGGER.info("Using device: %s", device)

    backbone_names = [b.strip() for b in args.backbone.split(",") if b.strip()]
    LOGGER.info("Backbones: %s", backbone_names)

    filter_classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes else None
    )
    classes = _collect_classes(data_dir, filter_classes)
    if not classes:
        LOGGER.error("No classes found in '%s'.", data_dir)
        sys.exit(1)
    LOGGER.info("Classes: %s", classes)

    run_save_path = utils_lib.create_storage_folder(
        results_dir, "run", "run", mode="iterate"
    )
    LOGGER.info("Results → %s", run_save_path)

    all_results = {}

    for bb_name in backbone_names:
        layers = (
            [l.strip() for l in args.layers.split(",")]
            if args.layers
            else backbones_lib.default_layers(bb_name)
        )

        LOGGER.info("")
        LOGGER.info("▶ Backbone: %s  (layers: %s)", bb_name, layers)
        LOGGER.info("")

        bb_results = []
        for classname in classes:
            utils_lib.fix_seeds(args.seed)
            result = train_one_class(
                classname=classname,
                data_dir=data_dir,
                backbone_name=bb_name,
                layers=layers,
                args=args,
                device=device,
                run_save_path=run_save_path,
            )
            if result is not None:
                bb_results.append(result)

        all_results[bb_name] = bb_results

        if bb_results:
            means = {c: _mean([r[c] for r in bb_results]) for c in COL_NAMES}
            LOGGER.info("")
            LOGGER.info("── %s summary ──────────────────────────────────", bb_name)
            LOGGER.info("  Mean Instance AUROC      : %.4f", means["instance_auroc"])
            LOGGER.info("  Mean Full Pixel AUROC    : %.4f", means["full_pixel_auroc"])
            LOGGER.info("  Mean Anomaly Pixel AUROC : %.4f", means["anomaly_pixel_auroc"])

        row_names = [r["dataset_name"] for r in bb_results]
        scores    = [[r[c] for c in COL_NAMES] for r in bb_results]
        bb_save   = os.path.join(run_save_path, bb_name)
        os.makedirs(bb_save, exist_ok=True)
        utils_lib.compute_and_store_final_results(
            bb_save, scores, row_names=row_names, column_names=COL_NAMES,
        )

    save_comparison_csv(all_results, run_save_path)

    if args.compare and len(backbone_names) > 1:
        print_comparison_table(all_results)

    LOGGER.info("")
    LOGGER.info("Done. All results → %s", run_save_path)


if __name__ == "__main__":
    LOGGER.info("Command: %s", " ".join(sys.argv))
    main()
