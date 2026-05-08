import os
import glob
import time
import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from skimage.segmentation import slic, mark_boundaries, find_boundaries
from skimage.measure import label as cc_label
from skimage.color import label2rgb
from skimage import img_as_ubyte


# Configs

HHTS_CONFIG = {
    "superpixels": [1000],      # target number of superpixels; use -1 for auto-termination mode
    "homogeneity": 0.0,  # stop if best priority drops below this
    "split_threshold": 0.0,
    "histogram_bins": 32,
    "min_segment_size": 64,
    "use_rgb": True,
    "use_hsv": True,
    "use_lab": True,
    "apply_blur": False,
}


@dataclass(order=True)
class PrioritizedSegment:
    # heapq is min-heap, so store negative priority.
    neg_priority: float
    id: int = field(compare=False)
    mask: np.ndarray = field(compare=False)
    size: int = field(compare=False)
    channel_infos: list = field(compare=False, default_factory=list)
    split_channel: int = field(compare=False, default=-1)
    split_criteria: float = field(compare=False, default=-1.0)

@dataclass
class ChannelInfo:
    min_val: int
    max_val: int
    width: int
    split_criteria: float
    
    @property
    def is_exhausted(self) -> bool:
        """True when this channel holds no more splittable information."""
        return self.split_criteria < 0.0

def get_channels_rgb_hsv_lab(image_rgb: np.ndarray,
                             use_rgb=True,
                             use_hsv=True,
                             use_lab=True,
                             apply_blur=False) -> Tuple[List[np.ndarray], List[str]]:
    """
    Return uint8 single-channel images.

    Mirrors the C++ getChannels:
    RGB channels, HSV channels, LAB channels; optional 3x3 Gaussian blur.
    """
    channels, names = [], []
    img = image_rgb.copy()
    if apply_blur:
        img = cv2.GaussianBlur(img, (3, 3), 0, 0)

    if use_rgb:
        # image is RGB in Python.
        for idx, name in enumerate(["R", "G", "B"]):
            channels.append(img[:, :, idx].astype(np.uint8))
            names.append(name)

    if use_hsv:
        src = img if apply_blur else image_rgb
        hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        for idx, name in enumerate(["H", "S", "V"]):
            channels.append(hsv[:, :, idx].astype(np.uint8))
            names.append("HSV_" + name)

    if use_lab:
        src = img if apply_blur else image_rgb
        lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        for idx, name in enumerate(["L", "a", "b"]):
            channels.append(lab[:, :, idx].astype(np.uint8))
            names.append("LAB_" + name)

    return channels, names

def channel_info(channel: np.ndarray, mask: np.ndarray, size: int) -> ChannelInfo:
    values = channel[mask]
    if values.size == 0:
        return ChannelInfo(0, 0, 0, -1.0)
    min_val = int(values.min())
    max_val = int(values.max())
    width = max_val - min_val

    # C++: prevent small channels -> auto-termination if width < 2.
    if width < 2:
        return ChannelInfo(min_val, max_val, width, -1.0)

    std = float(values.std())
    split_criteria = std * size * size
    return ChannelInfo(min_val, max_val, width, split_criteria)

def build_segment(seg_id: int,
                  mask: np.ndarray,
                  channels: List[np.ndarray],
                  min_segment_size: int,
                  split_threshold: float,
                  existing_infos: Optional[list] = None) -> Optional[PrioritizedSegment]:
    
    size = int(mask.sum())
    if size // 2 < min_segment_size:   # C++ isSizeSplittable
        return None

    if existing_infos is not None:
        infos = existing_infos          # already has blacklisted channels
    else:
        infos = [channel_info(ch, mask, size) for ch in channels]

    best_idx   = int(np.argmax([ci.split_criteria for ci in infos]))
    best_score = infos[best_idx].split_criteria

    if best_score <= split_threshold:   # all channels exhausted or blacklisted
        return None

    return PrioritizedSegment(
        neg_priority  = -best_score,
        id            = seg_id,
        mask          = mask,
        size          = size,
        channel_infos = infos,
        split_channel = best_idx,
        split_criteria= best_score,
    )

def interrupt_split(segment: PrioritizedSegment,
                    failed_channel_idx: int,
                    heap: list,
                    split_threshold: float) -> None:
    """
    Exact equivalent of C++ Label::interruptSplit.
    Permanently blacklists the failed channel on the segment object
    (sets split_criteria = -1.0), then finds the next best channel.
    If the segment is still splittable, re-inserts it into the heap.
    If not, the segment becomes final (dropped silently).
    """
    # 1. Permanently invalidate the failed channel
    segment.channel_infos[failed_channel_idx].split_criteria = -1.0

    # 2. Find new best channel
    best_score = -1.0
    best_idx   = -1
    for i, ci in enumerate(segment.channel_infos):
        if ci.split_criteria > best_score:
            best_score = ci.split_criteria
            best_idx   = i

    # 3. Re-insert if still splittable
    if best_idx >= 0 and best_score > split_threshold:
        updated = PrioritizedSegment(
            neg_priority = -best_score,
            id = segment.id,
            mask = segment.mask,
            size = segment.size,
            channel_infos = segment.channel_infos,
            split_channel = best_idx,
            split_criteria= best_score,
        )
        heapq.heappush(heap, updated)
    # else: segment is final — not pushed, heap shrinks → auto-termination

def histogram_bin_to_threshold(bin_idx: int, channel_bins: int, min_val: int, max_val: int) -> int:
    """
    C++ formula:
    threshold = minVal + 0.5 * (((maxVal - minVal + 1) * (2 * bin + 1) / channelBins) - 1)
    """
    threshold = min_val + 0.5 * (((max_val - min_val + 1) * (2 * bin_idx + 1) / channel_bins) - 1)
    return int(threshold)

def get_channel_threshold(channel: np.ndarray,
                          ci: ChannelInfo,
                          mask: np.ndarray,
                          histogram_bins: int) -> int:
    channel_bins = max(2, int(min(histogram_bins, ci.width)))

    values = channel[mask]
    hist, _ = np.histogram(values, bins=channel_bins, range=(ci.min_val, ci.max_val + 1))
    hist = hist.astype(np.float32).reshape(1, -1)

    # 1D Laplacian with border replicate.
    padded = np.pad(hist[0], (1, 1), mode="edge")
    response = padded[:-2] - 2 * padded[1:-1] + padded[2:]
    response = response.astype(np.float32)

    # Balanced partition weights following the public C++ implementation.
    cdf = np.cumsum(hist[0]).astype(np.float32)
    total = float(cdf[-1])
    mean = total / 2.0
    if mean <= 0:
        return int((ci.min_val + ci.max_val) / 2)

    d = 2.0
    weights = 1.0 / ((((mean - cdf) / mean * d) ** 4) + 1.0)
    applicability = response * weights

    threshold_bin = int(np.argmax(applicability))
    return histogram_bin_to_threshold(threshold_bin, channel_bins, ci.min_val, ci.max_val)

def connected_components_bool(mask: np.ndarray, connectivity: int = 4):
    conn = 1 if connectivity == 4 else 2
    return cc_label(mask.astype(np.uint8), connectivity=conn, return_num=True)

def _absorb_tiny_fragments(seeds: List[np.ndarray],
                            tiny: np.ndarray,
                            parent_mask: np.ndarray,
                            kernel: np.ndarray) -> List[np.ndarray]:
    """
    Absorb tiny fragments into neighboring valid seed regions.

    Uses multi-source BFS on the bounding box of the parent mask only,
    not the full image — this is the key performance fix over the previous
    iterative dilation on the full 512x512 array.

    Mirrors C++ floodFill behavior: each seed expands outward and claims
    unclaimed tiny pixels; largest seeds have priority (processed first).
    """
    if not seeds:
        return []

    # Work only within the bounding box of the parent mask (major speedup)
    rows, cols = np.where(parent_mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    # Crop to bounding box
    pm_crop    = parent_mask[r0:r1, c0:c1]
    tiny_crop  = tiny[r0:r1, c0:c1]

    H, W = pm_crop.shape

    # ownership map: 0=unclaimed, -1=tiny/unclaimed, i+1=owned by seed i
    ownership = np.zeros((H, W), dtype=np.int32)
    ownership[tiny_crop] = -1   # tiny pixels start as unclaimed

    # sort seeds largest first (mirrors C++ priority)
    order = sorted(range(len(seeds)), key=lambda i: int(seeds[i].sum()), reverse=True)

    # queues for BFS — one deque per seed
    from collections import deque
    queues = [deque() for _ in seeds]

    for rank, si in enumerate(order):
        seed_crop = seeds[si][r0:r1, c0:c1]
        ownership[seed_crop] = si + 1
        # seed border pixels adjacent to tiny are the BFS frontier
        dil = cv2.dilate(seed_crop.astype(np.uint8), kernel)
        border = (dil.astype(bool)) & tiny_crop & (ownership == -1)
        for r, c in zip(*np.where(border)):
            queues[si].append((int(r), int(c)))

    # 4-connected neighbor offsets
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    # Round-robin BFS — each seed expands one pixel at a time
    # (mirrors C++ floodFill expanding from seed outward)
    active = list(range(len(seeds)))
    while active:
        still_active = []
        for si in active:
            q = queues[si]
            next_q = deque()
            # process current frontier
            while q:
                r, c = q.popleft()
                if ownership[r, c] != -1:
                    continue   # already claimed by someone else
                ownership[r, c] = si + 1
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if ownership[nr, nc] == -1 and pm_crop[nr, nc]:
                            next_q.append((nr, nc))
            if next_q:
                queues[si] = next_q
                still_active.append(si)
        active = still_active

    # Any remaining unclaimed pixels (isolated) → give to largest seed
    unclaimed = (ownership == -1) & pm_crop
    if unclaimed.any():
        largest = order[0]
        ownership[unclaimed] = largest + 1

    # Reconstruct full-size child masks
    child_masks = []
    for si in range(len(seeds)):
        full = np.zeros_like(parent_mask)
        crop_mask = (ownership == si + 1)
        full[r0:r1, c0:c1] = crop_mask
        if full.any():
            child_masks.append(full)

    return child_masks

def split_segment_hhts_like(segment: PrioritizedSegment,
                            channels: List[np.ndarray],
                            labels: np.ndarray,
                            next_label: int,
                            heap: list,
                            min_segment_size: int,
                            split_threshold: float,
                            histogram_bins: int,
                            connectivity: int = 4,
                            debug: bool = False) -> int:
    """
    Split one segment. Mirrors C++ Label::split exactly.
    Uses the segment's stored split_channel — no channel search loop needed.
    On failure: calls interrupt_split (permanent blacklist + re-insert if viable).
    On success: absorb tiny fragments, assign labels, push children.
    """
    kernel = (np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
              if connectivity == 4 else np.ones((3,3), dtype=np.uint8))

    ch_idx  = segment.split_channel
    ci      = segment.channel_infos[ch_idx]
    channel = channels[ch_idx]

    # Work on the stored mask — no full-image label scan needed here
    mask = segment.mask

    threshold = get_channel_threshold(channel, ci, mask, histogram_bins)

    low_mask  = mask & (channel <= threshold)
    high_mask = mask & (channel >  threshold)

    seeds, tiny = [], np.zeros(mask.shape, dtype=bool)
    has_low = has_high = False

    low_cc, low_n = connected_components_bool(low_mask, connectivity)
    for comp_id in range(1, low_n + 1):
        comp = low_cc == comp_id
        if int(comp.sum()) < min_segment_size:
            tiny |= comp
        else:
            seeds.append(comp)
            has_low = True

    if not has_low:
        if debug:
            print(f"  seg {segment.id}: ch {ch_idx} no low. Blacklisting.")
        interrupt_split(segment, ch_idx, heap, split_threshold)
        return next_label

    high_cc, high_n = connected_components_bool(high_mask, connectivity)
    for comp_id in range(1, high_n + 1):
        comp = high_cc == comp_id
        if int(comp.sum()) < min_segment_size:
            tiny |= comp
        else:
            seeds.append(comp)
            has_high = True

    if not has_high:
        if debug:
            print(f"  seg {segment.id}: ch {ch_idx} no high. Blacklisting.")
        interrupt_split(segment, ch_idx, heap, split_threshold)
        return next_label

    child_masks = _absorb_tiny_fragments(seeds, tiny, mask, kernel)

    if len(child_masks) < 2:
        interrupt_split(segment, ch_idx, heap, split_threshold)
        return next_label

    first = True
    for child_mask in child_masks:
        if not child_mask.any():
            continue
        child_id = segment.id if first else next_label
        if not first:
            next_label += 1
        first = False

        labels[child_mask] = child_id

        child_seg = build_segment(child_id, child_mask, channels,
                                   min_segment_size, split_threshold)
        if child_seg is not None:
            heapq.heappush(heap, child_seg)

    return next_label


def hhts_python(image_rgb: np.ndarray,
                superpixels,
                homogeneity    : float = 0.0,
                split_threshold: float = 0.0,
                histogram_bins : int   = 32,
                min_segment_size: int  = 64,
                use_rgb        : bool  = True,
                use_hsv        : bool  = True,
                use_lab        : bool  = True,
                apply_blur     : bool  = False,
                prelabels      : Optional[np.ndarray] = None,
                debug          : bool  = False):
    """
    HHTS — mirrors C++ hhts(vector<int> superpixels, ...).

    superpixels:
        [500]         single-level, stop at ~500
        [100,250,500] multi-level snapshots
        [-1]          pure auto-termination
        [500,-1]      snapshot at 500, then auto-terminate
    Returns (label_maps, label_counts, info).
    """
    if isinstance(superpixels, int):
        sp_queue = [superpixels]
    else:
        sp_queue = list(superpixels)

    h, w = image_rgb.shape[:2]
    channels, channel_names = get_channels_rgb_hsv_lab(
        image_rgb, use_rgb=use_rgb, use_hsv=use_hsv,
        use_lab=use_lab, apply_blur=apply_blur
    )

    labels     = np.zeros((h, w), dtype=np.int32)
    heap       = []
    next_label = 1

    if prelabels is None:
        prelabels = np.ones((h, w), dtype=np.int32)

    for pre_id in sorted(np.unique(prelabels), reverse=True):
        if pre_id == 0:
            continue
        mask = (prelabels == pre_id)
        if not mask.any():
            continue
        seg_id       = next_label
        next_label  += 1
        labels[mask] = seg_id
        seg = build_segment(seg_id, mask, channels, min_segment_size, split_threshold)
        if seg is not None:
            heapq.heappush(heap, seg)

    iterations   = 0
    output_maps  = []
    output_counts= []
    term_reason  = "auto_termination"

    # ── FIX: correct while condition mirroring C++ exactly ────────────
    # C++: while ((superpixels.size() > 0 || superpixels[0] < 0) && splittable.size() > 0)
    # In practice: keep going while there are pending levels OR we are in auto-term mode
    while heap:
        # ── Homogeneity check ─────────────────────────────────────────
        if homogeneity > 0.0 and (-heap[0].neg_priority) < homogeneity:
            term_reason = "homogeneity"
            while sp_queue:
                sp_queue.pop(0)
                output_maps.append(labels.copy())
                output_counts.append(next_label)
            break

        # ── Snapshot check (C++ inner while) ─────────────────────────
        while sp_queue and sp_queue[0] >= 0 and next_label > sp_queue[0]:
            output_maps.append(labels.copy())
            output_counts.append(next_label)
            sp_queue.pop(0)

        # ── Termination: all count-based levels done, no auto-term ───
        if not sp_queue:
            term_reason = "superpixel_count"
            break
        if sp_queue and sp_queue[0] >= 0 and next_label > sp_queue[0]:
            # still more to snapshot, keep going
            pass

        segment = heapq.heappop(heap)

        # The mask on the heap entry is always valid because:
        # - children are built from freshly sliced masks
        # - interrupt_split reuses the same mask object
        # We only need to verify it hasn't been claimed by a sibling split.
        # A fast check: if label at the first True pixel matches segment.id, mask is valid.
        mask_rows, mask_cols = np.where(segment.mask)
        if len(mask_rows) == 0:
            continue
        sample_r, sample_c = int(mask_rows[0]), int(mask_cols[0])
        if labels[sample_r, sample_c] != segment.id:
            # mask is stale — rebuild from label map (rare edge case)
            current_mask = (labels == segment.id)
            if not current_mask.any():
                continue
            segment = build_segment(segment.id, current_mask, channels,
                                     min_segment_size, split_threshold,
                                     existing_infos=segment.channel_infos)
            if segment is None:
                continue

        next_label = split_segment_hhts_like(
            segment, channels, labels, next_label, heap,
            min_segment_size=min_segment_size,
            split_threshold =split_threshold,
            histogram_bins  =histogram_bins,
            connectivity    =4,
            debug           =debug
        )
        iterations += 1

        # Auto-termination: if sp_queue only has -1, just keep running
        # until heap empties naturally (no break needed — while heap handles it)
    while sp_queue:
        val = sp_queue.pop(0)
        if val >= 0 or val == -1:   # include auto-term final snapshot
            output_maps.append(labels.copy())
            output_counts.append(next_label)

    if not output_maps:
        output_maps.append(labels.copy())
        output_counts.append(next_label)

    return output_maps, output_counts, {
        "iterations"    : iterations,
        "num_labels"    : int(labels.max()),
        "term_reason"   : term_reason,
        "channel_names" : channel_names,
        "heap_remaining": len(heap),
    }
