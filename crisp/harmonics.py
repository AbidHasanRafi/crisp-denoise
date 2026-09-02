"""Angular harmonic spectrum of the relation profiles.

The length-``K`` DFT of a centred profile turns the directional difference
pattern into ``K/2`` complex coefficients.  ``|c_m|`` is the energy of angular
harmonic ``m`` (m=1 edge/ramp, m=2 ridge/line, flat spectrum = noise); ``arg(c_m)``
is the *orientation* of that harmonic, which the phase-coherence primitive tracks
across scale.

The coefficients are optionally smoothed spatially *as complex numbers* before
any nonlinearity - the relational analogue of covariance smoothing in the
structure tensor.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import CrispConfig


def harmonic_transform(profiles, angles: np.ndarray, cfg: CrispConfig):
    """Return ``(coeffs, m)``.

    ``coeffs`` is a list of ``S`` complex arrays of shape ``(H, W, K//2)``.
    ``m`` is the vector of harmonic orders ``[1, 2, ..., K//2]``.
    """
    m = np.arange(1, cfg.n_harmonics + 1)
    basis = np.exp(-1j * np.outer(angles, m))          # (K, K//2)

    coeffs = []
    for g in profiles:
        c = g.astype(np.float64) @ basis              # (H, W, K//2) complex
        if cfg.phasor_smooth > 0:
            s = (cfg.phasor_smooth, cfg.phasor_smooth, 0.0)
            c = gaussian_filter(c.real, s) + 1j * gaussian_filter(c.imag, s)
        coeffs.append(c)
    return coeffs, m
