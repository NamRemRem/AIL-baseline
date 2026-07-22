"""Anomaly detection metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """Compute image-level AUROC and ROC curve.

    Args:
        anomaly_prediction_weights: [N] float anomaly scores per image.
        anomaly_ground_truth_labels: [N] binary labels (1 = anomaly).

    Returns:
        dict with keys: auroc, fpr, tpr, threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """Compute pixel-level AUROC and optimal F1 threshold.

    Args:
        anomaly_segmentations: [N x H x W] predicted heatmaps.
        ground_truth_masks: [N x H x W] binary ground-truth masks.

    Returns:
        dict with keys: auroc, fpr, tpr, optimal_threshold,
        optimal_fpr, optimal_fnr.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_pred = anomaly_segmentations.ravel()
    flat_gt = ground_truth_masks.ravel().astype(int)

    fpr, tpr, _ = metrics.roc_curve(flat_gt, flat_pred)
    auroc = metrics.roc_auc_score(flat_gt, flat_pred)

    precision, recall, thresholds = metrics.precision_recall_curve(flat_gt, flat_pred)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(f1)]
    predictions = (flat_pred >= optimal_threshold).astype(int)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": float(np.mean(predictions > flat_gt)),
        "optimal_fnr": float(np.mean(predictions < flat_gt)),
    }
