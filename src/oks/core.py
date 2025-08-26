from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Tuple

import numpy as np


def _edit_keypoints(kpts: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape flat keypoints [x, y, v, ...] into (N,2) coords and (N,) visibility."""
    arr = np.array(kpts, dtype=float).reshape(-1, 3)
    vi = arr[:, 2]
    coords = arr[:, 0:2]
    return coords, vi


def oks(
    kpts1: Iterable[float], kpts2: Iterable[float], sigma: Iterable[float], area: float
) -> float:
    """
    Calculate OKS (Object Keypoint Similarity) between two keypoint sets.

    Parameters
    ----------
    kpts1, kpts2 : iterable of float
        Flattened keypoints like [x0, y0, v0, x1, y1, v1, ...]. Shapes must match.
    sigma : iterable of float
        Per-keypoint standard deviation(s), typically dataset-specific.
        Can be scalar-like (broadcastable) or per-joint array-like.
    area : float
        Object area in pixels.

    Returns
    -------
    float
        OKS value in [0, 1], rounded to 6 decimals (HALF_UP).
    """
    p1, vi1 = _edit_keypoints(kpts1)
    p2, vi2 = _edit_keypoints(kpts2)

    if p1.shape != p2.shape:
        raise ValueError(f"keypoints shape mismatch: {p1.shape} vs {p2.shape}")

    k = 2 * np.array(sigma, dtype=float)
    s = float(area)

    d = np.linalg.norm(p1 - p2, ord=2, axis=1)
    v = np.ones(len(d))

    for i in range(len(d)):
        if vi1[i] == 0 or vi2[i] == 0:
            d[i] = 0
            v[i] = 0

    if np.sum(v) != 0:
        oks_value = float(np.sum(np.exp(-(d**2) / (2 * s * (k**2))) * v) / np.sum(v))
    else:
        oks_value = 0.0

    oks_value = float(
        Decimal(str(oks_value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    )
    return oks_value


def edit_keypoints(kpts: Iterable[float]):
    """Kept for backward compatibility; prefer internal _edit_keypoints."""
    return _edit_keypoints(kpts)


def OKS(kpts1, kpts2, sigma, area) -> float:  # noqa: N802
    """Backward-compatible wrapper. Prefer `oks`."""
    return oks(kpts1, kpts2, sigma, area)
