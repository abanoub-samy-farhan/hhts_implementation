import cv2
import numpy as np

from pathlib import Path
from skimage import img_as_ubyte

from typing import List
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")



def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] >= 3:
        # OpenCV loads BGR/BGRA. Keep first 3 channels and convert.
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img_as_ubyte(img)


def resize_max_side(img: np.ndarray, max_side: int = 512, interpolation=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interpolation)

def list_files(root: Path, patterns=IMAGE_EXTS) -> List[Path]:
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return sorted(files)

