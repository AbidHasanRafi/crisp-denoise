"""Sub-pixel ray / arc sampling of an image - fast path.

``arc_sample`` returns, for every pixel, the arc-averaged intensity at a fixed
``radius`` along each of ``K`` directions.

All sampling offsets are fixed at call time, so bilinear interpolation is just
four integer-shifted views of a once-padded image combined with scalar weights -
no per-pixel coordinate arrays, no interpolation machinery.  This is the
"precomputed weights + static gather" the design doc calls for; it makes the
whole live path ~100x faster than the ``map_coordinates`` prototype.
"""

from __future__ import annotations

import numpy as np


def _bilinear_shift(padded: np.ndarray, pad: int, dy: float, dx: float,
                    hw: tuple[int, int]) -> np.ndarray:
    """Bilinear sample of the original image at offset ``(dy, dx)``.

    ``padded`` is the reflect-padded image, ``pad`` the pad width, ``hw`` the
    original ``(H, W)``.  All four taps are contiguous slices (views).
    """
    h, w = hw
    yb, xb = pad + dy, pad + dx
    y0, x0 = int(np.floor(yb)), int(np.floor(xb))
    fy, fx = yb - y0, xb - x0

    a = padded[y0:y0 + h,         x0:x0 + w]
    b = padded[y0 + 1:y0 + 1 + h, x0:x0 + w]
    c = padded[y0:y0 + h,         x0 + 1:x0 + 1 + w]
    d = padded[y0 + 1:y0 + 1 + h, x0 + 1:x0 + 1 + w]
    return ((1 - fy) * (1 - fx) * a + fy * (1 - fx) * b
            + (1 - fy) * fx * c + fy * fx * d)


def arc_sample(img: np.ndarray, radius: float, angles: np.ndarray,
               taps: int = 3) -> np.ndarray:
    """Arc-averaged samples at ``radius`` along each angle -> ``(H, W, K)``."""
    h, w = img.shape
    k = len(angles)
    pad = int(np.ceil(radius)) + 2
    padded = np.pad(img, pad, mode="reflect")

    if taps <= 1:
        d_phi = np.array([0.0])
    else:
        step = 1.0 / max(radius, 1e-6)          # +-1 sample tangentially
        d_phi = np.linspace(-step, step, taps)

    out = np.empty((h, w, k), dtype=np.float64)
    inv_t = 1.0 / len(d_phi)
    for i, theta in enumerate(angles):
        acc = np.zeros((h, w), dtype=np.float64)
        for dp in d_phi:
            a = theta + dp
            acc += _bilinear_shift(padded, pad, radius * np.sin(a),
                                   radius * np.cos(a), (h, w))
        out[..., i] = acc * inv_t
    return out
