"""Synthetic phantom + noise models for the theory-validation experiments.

The phantom carries three kinds of region:
    label  1  -> structure  (step edge band, sinusoidal grating)
    label  0  -> noise-only  (flat reference patch)
    label -1  -> ignore      (ramp, transitions, image border)
"""

from __future__ import annotations

import numpy as np


def make_phantom(size: int = 256):
    img = np.full((size, size), 128.0, dtype=np.float64)
    lab = np.full((size, size), -1, dtype=np.int8)
    u = size // 6
    yy, xx = np.mgrid[0:size, 0:size]

    # --- vertical step edge (left) ------------------------------------------------
    img[:, :u] = 70.0
    img[:, u:2 * u] = 190.0
    lab[:, u - 2:u + 2] = 1                       # edge band = structure

    # --- smooth ramp (centre) : ambiguous, left as ignore ----------------------
    img[:, 2 * u:4 * u] = np.linspace(60.0, 200.0, 2 * u)[None, :]

    # --- sinusoidal grating patch (top right) = structure ----------------------
    grating = 128.0 + 45.0 * np.sin(2.0 * np.pi * xx / 9.0)
    g_reg = (xx >= 4 * u) & (yy < size // 2 - 4)
    img[g_reg] = grating[g_reg]
    lab[g_reg] = 1

    # --- flat reference patch (bottom right) = noise-only ----------------------
    f_reg = (xx >= 4 * u + 4) & (yy > size // 2 + 4)
    lab[f_reg] = 0

    # --- border margin: exclude (max radius ~6 + arc taps) ----------------------
    m = 12
    lab[:m, :] = lab[-m:, :] = lab[:, :m] = lab[:, -m:] = -1
    return img, lab


def add_noise(img, kind: str = "gaussian", sigma: float = 25.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    img = np.asarray(img, dtype=np.float64)
    if kind == "gaussian":
        return img + rng.normal(0.0, sigma, img.shape)
    if kind == "poisson":
        # sigma read as electrons-per-unit gain; larger gain -> less relative noise
        gain = max(sigma, 1e-3)
        return rng.poisson(np.clip(img, 0, None) / gain) * gain
    if kind == "saltpepper":
        out = img.copy()
        p = sigma / 255.0
        r = rng.random(img.shape)
        out[r < p / 2] = 0.0
        out[r > 1.0 - p / 2] = 255.0
        return out
    raise ValueError(f"unknown noise kind: {kind!r}")
