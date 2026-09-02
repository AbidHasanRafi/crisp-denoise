"""The CRISP structural statistics, plus the composite confidence map.

Sigma(p)   scale stability - spatially-aggregated, cross-scale cosine agreement of
           the *normalised* relation profile.  This is the primary statistic
           (E1b: d > 4, AUC > 0.99 at sigma up to 50).
           Per-pixel null (flat + white noise, before aggregation):
               N(0, 1/((K-1)(S-1)))
           spatial aggregation over ~N_eff independent pixels tightens the null
           by ~1/sqrt(N_eff), sharpening the structure/noise separation.

Phi_m(p)   relational phase coherence of angular harmonic m across scales.
           Secondary: too weak on texture to headline (E1), but gives per-
           orientation selectivity that the scalar Sigma cannot, so it is used
           to modulate the harmonic shrinkage.
           Null:  Rayleigh, E ~ sqrt(pi/(4S)),  P(Phi_m >= x) ~ exp(-S x^2).

C(p)       structural confidence in [0, 1]  ->  ~1 structured, ~0 noise-dominated.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def scale_stability(profiles, eps: float = 1e-8, agg: float = 2.5) -> np.ndarray:
    hats = [g / (np.linalg.norm(g, axis=-1, keepdims=True) + eps) for g in profiles]
    sims = np.mean(
        [np.sum(hats[s] * hats[s + 1], axis=-1) for s in range(len(hats) - 1)],
        axis=0,
    )
    if agg > 0:
        sims = gaussian_filter(sims, agg)
    return sims                                        # (H, W)


def phase_coherence(coeffs, eps: float = 1e-8) -> np.ndarray:
    num = np.abs(np.sum(coeffs, axis=0))               # |sum_s c_m^s|
    den = np.sum([np.abs(c) for c in coeffs], axis=0)  # sum_s |c_m^s|
    return num / (den + eps)                           # (H, W, K//2)


def confidence_map(sigma: np.ndarray, phi: np.ndarray, coeffs, eps: float = 1e-8):
    """Return ``(C, w)``.

    ``C``  scalar confidence, driven by the (strong) Sigma statistic.
    ``w``  per-harmonic keep-weight for shrinkage: confidence gated by Sigma,
           with a gentle per-orientation boost from phase coherence.
    """
    c = np.clip(sigma, 0.0, 1.0)                        # (H, W)
    phi_rel = phi / (phi.max(axis=-1, keepdims=True) + eps)
    w = c[..., None] * (0.5 + 0.5 * phi_rel)            # (H, W, K//2)
    return c, w
