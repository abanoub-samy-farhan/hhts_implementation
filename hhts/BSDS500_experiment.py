import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import glob
from pathlib import Path
from scipy.io import loadmat
from skimage.segmentation import slic, find_boundaries, mark_boundaries
from tqdm.auto import tqdm
from hhts import hhts_python, HHTS_CONFIG


def _gt_class_map(gt_mask: np.ndarray) -> np.ndarray:
    """
    Convert a ground-truth mask (RGB color or grayscale) to an integer class map.
    Each unique color / intensity value becomes a unique class label.
    """
    if gt_mask.ndim == 3:
        # pack RGB into a single integer for unique-color comparison
        flat = (gt_mask[:,:,0].astype(np.int32) * 256 * 256
              + gt_mask[:,:,1].astype(np.int32) * 256
              + gt_mask[:,:,2].astype(np.int32))
    else:
        flat = gt_mask.astype(np.int32)

    # remap to contiguous integers starting at 0
    unique_vals = np.unique(flat)
    lut = np.zeros(int(flat.max()) + 1, dtype=np.int32)
    for new_id, val in enumerate(unique_vals):
        lut[val] = new_id
    return lut[flat]


# Evaluation Metrics
def boundary_recall(gt_mask: np.ndarray, sp_labels: np.ndarray,
                    tolerance: int = 2) -> float:
    """
    Boundary Recall (BR): fraction of GT boundary pixels covered by
    superpixel boundaries (within `tolerance` pixels).
    Higher is better.
    """
    gt_class = _gt_class_map(gt_mask)
    gt_bound = find_boundaries(gt_class, mode="outer")
    sp_bound = find_boundaries(sp_labels, mode="outer")
    if not gt_bound.any():
        return float('nan')
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    sp_dil = cv2.dilate(sp_bound.astype(np.uint8), kernel).astype(bool)
    return float((gt_bound & sp_dil).sum() / gt_bound.sum())


def undersegmentation_error(gt_mask: np.ndarray,
                             sp_labels: np.ndarray) -> float:
    """
    Undersegmentation Error (UE): measures how much superpixels bleed
    across GT segment boundaries.
    UE = (1/N) * sum_k  min_i |sp_k ∩ gt_i| ... implemented via standard formula.
    Lower is better.
    """
    gt_class = _gt_class_map(gt_mask)
    N = gt_class.size
    gt_ids = np.unique(gt_class)
    sp_ids = np.unique(sp_labels)

    # build a contingency table: rows = sp labels, cols = gt labels
    sp_flat   = sp_labels.ravel()
    gt_flat   = gt_class.ravel()

    # map to 0-based
    sp_map = {v: i for i, v in enumerate(sp_ids)}
    gt_map = {v: i for i, v in enumerate(gt_ids)}

    table = np.zeros((len(sp_ids), len(gt_ids)), dtype=np.int64)
    for sp_v, gt_v in zip(sp_flat, gt_flat):
        table[sp_map[sp_v], gt_map[gt_v]] += 1

    # UE per GT segment: sum over SP of (size of sp NOT in this GT segment)
    # = sum_k [|sp_k| - max_i |sp_k ∩ gt_i|]  /  N
    sp_sizes = table.sum(axis=1)   # per superpixel
    overlap  = table.max(axis=1)   # best GT match per SP
    return float((sp_sizes - overlap).sum() / N)


def achievable_segmentation_accuracy(gt_mask: np.ndarray,
                                      sp_labels: np.ndarray) -> float:
    """
    Achievable Segmentation Accuracy (ASA): upper bound on classification
    accuracy if each superpixel is assigned its majority GT class.
    Higher is better.
    """
    gt_class = _gt_class_map(gt_mask)
    N = gt_class.size
    sp_ids = np.unique(sp_labels)
    correct = 0
    for sp_id in sp_ids:
        sp_mask = sp_labels == sp_id
        gt_vals = gt_class[sp_mask]
        _, counts = np.unique(gt_vals, return_counts=True)
        correct  += int(counts.max())
    return float(correct / N)


def explained_variation(image_rgb: np.ndarray,
                         sp_labels: np.ndarray) -> float:
    """
    Explained Variation (EV): fraction of image color variance explained
    by replacing each superpixel with its mean color.
    Higher is better.
    """
    img = image_rgb.astype(np.float64)
    N = img.shape[0] * img.shape[1]
    mu = img.mean(axis=(0,1), keepdims=True)
    var_total = float(np.sum((img - mu) ** 2))
    if var_total == 0:
        return 1.0

    var_within = 0.0
    for sp_id in np.unique(sp_labels):
        sp_mask = sp_labels == sp_id
        pix = img[sp_mask]          # shape (k, 3)
        sp_mu = pix.mean(axis=0)
        var_within += float(np.sum((pix - sp_mu) ** 2))

    return float(1.0 - var_within / var_total)


def intra_cluster_variation(image_rgb: np.ndarray,
                             sp_labels: np.ndarray) -> float:
    """
    Intra-Cluster Variation (ICV): mean per-channel std within superpixels.
    Lower is better.
    """
    img  = image_rgb.astype(np.float64)
    vals = []
    for sp_id in np.unique(sp_labels):
        pix = img[sp_labels == sp_id]
        if pix.shape[0] > 1:
            vals.append(float(np.mean(np.std(pix, axis=0))))
    return float(np.mean(vals)) if vals else float('nan')


def compactness(sp_labels: np.ndarray) -> float:
    """
    Compactness (CO): ratio of superpixel area to its bounding-circle area.
    Mean over all superpixels. Higher = more compact (rounder).
    """
    scores = []
    for sp_id in np.unique(sp_labels):
        sp_mask = (sp_labels == sp_id).astype(np.uint8)
        area = int(sp_mask.sum())
        if area < 4:
            continue
        contours, _ = cv2.findContours(sp_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter > 0:
            scores.append(4.0 * math.pi * area / (perimeter ** 2))
    return float(np.mean(scores)) if scores else float('nan')


def calculate_bp(gt_mask, sp_labels, tolerance=2):
    """Boundary Precision (BP)"""
    gt_bound = find_boundaries(gt_mask, mode="outer")
    sp_bound = find_boundaries(sp_labels, mode="outer")
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    gt_dil = cv2.dilate(gt_bound.astype(np.uint8), kernel).astype(bool)
    return float((sp_bound & gt_dil).sum() / sp_bound.sum()) if sp_bound.any() else 0.0


def calculate_mde(gt_mask, sp_labels):
    """Mean Distance to Edge (MDE)"""
    gt_bound = find_boundaries(gt_mask, mode="outer")
    sp_bound = find_boundaries(sp_labels, mode="outer")
    dist_map = cv2.distanceTransform((1 - sp_bound).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.mean(dist_map[gt_bound])) if gt_bound.any() else 0.0


# GT Helper Functions 
def load_bsds_gt(mat_path):
    """Extracts all human segmentations from a BSDS500 .mat file."""
    data = loadmat(str(mat_path))
    gt_list = []
    # Structure: data['groundTruth'][0, i][0, 0][0]
    for i in range(data['groundTruth'].shape[1]):
        gt_list.append(data['groundTruth'][0, i][0, 0][0])
    return gt_list


def evaluate_metrics(image, sp_labels, mat_path, runtime):
    """Calculates all 8 metrics, averaging GT-dependent ones across annotators."""
    gts = load_bsds_gt(mat_path)
    
    # GT-Independent (ICV, EV, CO) - logic from your notebook
    res = {
        "num_segments": len(np.unique(sp_labels)),
        "runtime": runtime,
        "ICV": intra_cluster_variation(image, sp_labels),
        "EV":  explained_variation(image, sp_labels),
        "CO":  compactness(sp_labels),
    }

    # GT-Dependent (Averaged across humans)
    br, ue, asa, bp, mde = [], [], [], [], []
    for gt in gts:
        br.append(boundary_recall(gt, sp_labels))
        ue.append(undersegmentation_error(gt, sp_labels))
        asa.append(achievable_segmentation_accuracy(gt, sp_labels))
        bp.append(calculate_bp(gt, sp_labels))
        mde.append(calculate_mde(gt, sp_labels))

    res.update({
        "BR": np.mean(br), "UE": np.mean(ue), "ASA": np.mean(asa),
        "BP": np.mean(bp), "MDE": np.mean(mde)
    })
    return res


def run_experiment(image_dir, gt_dir, hhts_config):
    milestones = [250, 500, 750, 1000]
    data_rows = []
    
    img_files = sorted(list(Path(image_dir).glob("*.jpg")))
    
    for img_p in tqdm(img_files, desc="Benchmarking BSDS500"):
        image = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        mat_p = Path(gt_dir) / (img_p.stem + ".mat")
        if not mat_p.exists(): continue

        # 1. RUN HHTS (Auto-terminate mode with milestones)
        cfg = {**hhts_config, "superpixels": milestones + [-1]}
        t0 = time.time()
        maps, counts, _ = hhts_python(image, **cfg)
        h_time = (time.time() - t0) / len(maps) # Average time per level

        for lmap, lcount in zip(maps, counts):
            m_target = min(milestones, key=lambda x: abs(x-lcount))
            row = evaluate_metrics(image, lmap, mat_p, h_time)
            row.update({"method": "HHTS", "milestone": m_target})
            data_rows.append(row)

        # 2. RUN SLIC (Comparison Baseline)
        for m in milestones:
            t0 = time.time()
            s_map = slic(image, n_segments=m, compactness=10, start_label=1)
            s_time = time.time() - t0
            row = evaluate_metrics(image, s_map, mat_p, s_time)
            row.update({"method": "SLIC", "milestone": m})
            data_rows.append(row)

    return pd.DataFrame(data_rows)


def plot_results(df):
    summary = df.groupby(["method", "milestone"]).mean().reset_index()
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    metrics = [
        ("BR", "Boundary Recall", axes[0,0]), ("UE", "Undersegmentation Error", axes[0,1]),
        ("MDE", "Mean Distance to Edge", axes[0,2]), ("ASA", "Achievable Seg. Accuracy", axes[1,0]),
        ("EV", "Explained Variation", axes[1,1]), ("ICV", "Intra-cluster Variation", axes[1,2]),
        ("CO", "Compactness", axes[2,0]), ("BP", "Boundary Precision", axes[2,1])
    ]

    for col, name, ax in metrics:
        for meth, color, marker in [("HHTS", "orange", "s"), ("SLIC", "olive", "x")]:
            subset = summary[summary["method"] == meth]
            ax.plot(subset["milestone"], subset[col], marker=marker, color=color, label=meth)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 9th Graph: BP vs BR Curve
    ax = axes[2,2]
    for meth, color in [("HHTS", "orange"), ("SLIC", "olive")]:
        subset = summary[summary["method"] == meth].sort_values("BR")
        ax.plot(subset["BR"], subset["BP"], 'o-', color=color, label=meth)
    ax.set_title("Boundary Precision vs Recall")
    ax.set_xlabel("BR"), ax.set_ylabel("BP"), ax.grid(True, alpha=0.3), ax.legend()

    plt.tight_layout()
    plt.show()


# Replace these with your local paths to the BSDS500 dataset
IMG_DIR = "path/to/BSDS500/images/test"
GT_DIR = "path/to/BSDS500/ground_truth/test"

# Running Experiment and Plotting Results
df = run_experiment(IMG_DIR, GT_DIR, HHTS_CONFIG)
plot_results(df)