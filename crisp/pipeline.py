"""End-to-end CRISP pipeline."""

from __future__ import annotations

import numpy as np

from .config import CrispConfig
from .contrast import contrast_probability, fuse_confidence
from .harmonics import harmonic_transform
from .primitives import confidence_map, phase_coherence, scale_stability
from .profiles import relation_profiles
from .reconstruct import confidence_gate


def analyze(img, cfg: CrispConfig | None = None, with_harmonics: bool = True) -> dict:
    """Compute the relational representation and the structural confidence field.

    Returns a dict with ``sigma`` (H,W), ``C`` (H,W) and, if ``with_harmonics``,
    also ``coeffs``, ``phi`` (H,W,K//2), ``w`` and ``profiles`` for figures /
    ablations.  The denoiser only needs ``C``.
    """
    cfg = cfg or CrispConfig()
    img = np.asarray(img, dtype=np.float64)

    profiles, angles = relation_profiles(img, cfg)
    sigma = scale_stability(profiles, cfg.eps, cfg.stability_agg)
    c_map = np.clip(sigma, 0.0, 1.0)
    out = dict(sigma=sigma, C=c_map, angles=angles)

    if cfg.c_mode == "fused":
        p_con, noise_sigma = contrast_probability(img, cfg.contrast_rho)
        out["contrast"] = p_con
        out["noise_sigma"] = noise_sigma
        out["C"] = fuse_confidence(c_map, p_con, noise_sigma)
    elif cfg.c_mode != "stability":
        raise ValueError(f"unknown c_mode: {cfg.c_mode!r}")

    if with_harmonics:
        coeffs, _ = harmonic_transform(profiles, angles, cfg)
        phi = phase_coherence(coeffs, cfg.eps)
        _, w = confidence_map(sigma, phi, coeffs, cfg.eps)   # w for shrinkage ablation
        out.update(profiles=profiles, coeffs=coeffs, phi=phi, w=w)
    return out


def denoise(img, cfg: CrispConfig | None = None,
            heavy: np.ndarray | None = None, return_aux: bool = False):
    """Denoise ``img``.

    ``cfg.mode``:
      "rnlm" (default) - relational non-local means, a standalone denoiser.
      "gate"           - confidence-gate wrapper around a pre-computed ``heavy``.
      "self"           - confidence gate with a built-in gaussian heavy stage.

    Returns ``J`` or ``(J, aux)``.
    """
    cfg = cfg or CrispConfig()
    img = np.asarray(img, dtype=np.float64)
    is_color = img.ndim == 3
    ref = img @ np.array([0.2989, 0.5870, 0.1140]) if is_color else img

    aux = analyze(ref, cfg, with_harmonics=return_aux and not is_color)

    if cfg.mode == "rnlm":
        from .relational import relational_nlm
        j = relational_nlm(img, cfg, c_map=aux["C"])
    elif is_color:
        raise ValueError('only mode="rnlm" supports colour input')
    elif cfg.mode in ("gate", "self"):
        if cfg.mode == "gate" and heavy is None:
            raise ValueError('mode="gate" needs a pre-computed `heavy` array')
        j = confidence_gate(img, aux["C"], cfg, heavy=heavy)
    else:
        raise ValueError(f"unknown mode: {cfg.mode!r}")

    if return_aux:
        aux["output"] = j
        return j, aux
    return j
