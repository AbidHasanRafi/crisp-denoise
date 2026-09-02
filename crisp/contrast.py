"""Noise-aware local-contrast term for the fused confidence field.

E4/E8: the scale-stability statistic ``Sigma`` misses sharp low-noise edges that a
plain gradient catches easily; a plain gradient misses low-contrast structure that
``Sigma`` catches in heavy noise.  This module supplies the gradient side, with an
**analytic white-noise floor** subtracted so that noisy-flat regions score ~0:
the floor is the mean gradient energy of synthetic noise at the estimated sigma
run through the identical operator.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def _grad_energy(img: np.ndarray, rho: float) -> np.ndarray:
    gx = sobel(img, axis=1) / 4.0
    gy = sobel(img, axis=0) / 4.0
    return gaussian_filter(gx * gx, rho) + gaussian_filter(gy * gy, rho)


def _mad_sigma(img: np.ndarray) -> float:
    """Diagonal-difference MAD noise estimate (fallback for skimage.estimate_sigma)."""
    d = (np.asarray(img, float)[1:, 1:] - np.asarray(img, float)[:-1, :-1]).ravel()
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))


def contrast_probability(noisy: np.ndarray, rho: float = 2.0, seed: int = 0):
    """Return ``(p_contrast, noise_sigma)``.

    ``p_contrast`` in [0, 1): local gradient energy above the white-noise floor,
    squashed by its own 90th percentile.
    """
    noisy = np.asarray(noisy, dtype=np.float64)
    try:
        from skimage.restoration import estimate_sigma
        s = float(estimate_sigma(noisy))
    except Exception:
        s = _mad_sigma(noisy)

    energy = _grad_energy(noisy, rho)
    ref = np.random.default_rng(seed).normal(0.0, max(s, 1e-6), noisy.shape)
    floor = float(_grad_energy(ref, rho).mean())

    sig = np.clip(energy - floor, 0.0, None)
    scale = np.percentile(sig, 90) + 1e-8
    return sig / (sig + scale), s


def fuse_confidence(p_stability: np.ndarray, p_contrast: np.ndarray,
                    noise_sigma: float) -> np.ndarray:
    """Noise-weighted blend: more stability-weight as noise grows (E8)."""
    w = float(np.clip((noise_sigma - 8.0) / 44.0, 0.15, 0.85))
    return np.clip(w * p_stability + (1.0 - w) * p_contrast, 0.0, 1.0)
