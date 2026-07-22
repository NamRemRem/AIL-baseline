"""General-purpose utilities: device setup, seed fixing, result storage."""
import csv
import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_torch_device(gpu_ids) -> torch.device:
    """Return the appropriate torch device.

    Args:
        gpu_ids: List of integer GPU indices. If empty or CUDA unavailable, CPU is used.
    """
    if len(gpu_ids) and torch.cuda.is_available():
        return torch.device("cuda:{}".format(gpu_ids[0]))
    LOGGER.warning("CUDA unavailable or no GPU ids provided – running on CPU.")
    return torch.device("cpu")


def fix_seeds(seed: int, with_torch: bool = True, with_cuda: bool = True) -> None:
    """Fix random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        with_torch: Also seed PyTorch.
        with_cuda: Also seed CUDA RNGs and enable deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_storage_folder(
    main_folder_path: str,
    project_folder: str,
    group_folder: str,
    mode: str = "iterate",
) -> str:
    """Create and return a uniquely-named output directory.

    Args:
        main_folder_path: Top-level output root.
        project_folder: Project sub-directory.
        group_folder: Run name; incremented with a numeric suffix if ``mode='iterate'``.
        mode: ``'iterate'`` (avoid overwrite) or ``'overwrite'``.
    """
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, f"{group_folder}_{counter}")
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)
    return save_path


def compute_and_store_final_results(
    results_path: str,
    results,
    row_names=None,
    column_names=None,
) -> dict:
    """Write per-dataset metrics to a CSV and log mean scores.

    Args:
        results_path: Directory where ``results.csv`` is saved.
        results: List of per-dataset metric lists.
        row_names: Optional dataset name labels.
        column_names: Metric column names.

    Returns:
        Dict of ``mean_<metric>`` values.
    """
    if column_names is None:
        column_names = ["instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc"]

    if row_names is not None:
        assert len(row_names) == len(results), "#row_names != #results"

    mean_metrics = {}
    for i, key in enumerate(column_names):
        mean_metrics[key] = np.mean([x[i] for x in results])
        LOGGER.info("%s: %.3f", key, mean_metrics[key])

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = (["dataset"] + column_names) if row_names is not None else column_names
        writer.writerow(header)
        for i, row in enumerate(results):
            writer.writerow(([row_names[i]] + row) if row_names is not None else row)
        mean_row = list(mean_metrics.values())
        writer.writerow((["mean"] + mean_row) if row_names is not None else mean_row)

    return {f"mean_{k}": v for k, v in mean_metrics.items()}
