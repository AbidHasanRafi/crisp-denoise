"""Harmonic shrinkage of the base-scale relation field.

Each base-scale harmonic coefficient is multiplied by ``psi(w_m)`` where ``w_m``
is its stability x phase-coherence weight.  Coherent harmonics (real structure)
pass through; incoherent harmonics (noise) are suppressed.  The shrunk angular
profile is then evaluated at the reconstruction stencil directions (the 8
integer-pixel neighbours), scaled from the measurement radius ``r0`` to each
neighbour radius by the local first-order relation ``dI ~ (rho / r0) * P(phi)``.
"""

from __future__ import annotations

import numpy as np

from .config import CrispConfig


def _psi(w: np.ndarray, cfg: CrispConfig) -> np.ndarray:
    if cfg.shrink == "power":
        return np.clip(w, 0.0, 1.0) ** cfg.gamma
    if cfg.shrink == "soft":
        t = cfg.soft_thresh
        return np.clip((w - t) / (1.0 - t), 0.0, 1.0)
    raise ValueError(f"unknown shrink mode: {cfg.shrink!r}")


def shrink_field(coeffs, w: np.ndarray, angles_recon: np.ndarray,
                 radii_recon: np.ndarray, r0: float, cfg: CrispConfig):
    """Return ``(profile, c_hat)``.

    ``profile`` : ``(H, W, Krec)`` real - target relation differences for the solve.
    ``c_hat``   : ``(H, W, K//2)`` complex - the shrunk harmonic field (for figures).
    """
    c_hat = coeffs[0] * _psi(w, cfg)                       # (H, W, K//2)

    m = np.arange(1, c_hat.shape[-1] + 1)
    inv_basis = np.exp(1j * np.outer(angles_recon, m))     # (Krec, K//2)
    profile = (2.0 / cfg.K) * np.real(c_hat @ inv_basis.T) # (H, W, Krec)
    profile = profile * (np.asarray(radii_recon) / r0)[None, None, :]
    return profile, c_hat
