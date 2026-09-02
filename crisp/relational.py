"""Relational non-local means - the CRISP denoiser.

A non-local weighted average of intensities, but the similarity between two
pixels is measured in **relation space**: the distance between their feature
vectors

    F(p) = [ w_I * smoothed_I(p) ,  centred multi-scale directional relation profiles ]

Each column is scaled to unit white-noise variance (measured by running the same
pipeline on synthetic noise), so the noise contribution to ``||F(p) - F(q)||^2``
is a known constant ``bias`` that we subtract before the exponential.

Why the relation profile is a better matching feature than a raw patch (E10/E11):
  * arc-averaged and multi-scale -> partially denoised;
  * centring removes the common-mode noise term;
  * captures oriented structure at several radii a small patch misses;
  * per-pixel feature distance already integrates a neighbourhood -> no spatial
    patch loop, so it is a touch *cheaper* than classic NLM.

The confidence field ``C`` modulates the selectivity ``h`` per pixel: smooth
harder where ``C`` says "noise".

Colour: the feature stack matches on all channels' smoothed intensity plus the
luminance relation profiles; every channel is averaged with the shared weights.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import CrispConfig
from .profiles import relation_profiles

_LUMA = np.array([0.2989, 0.5870, 0.1140])

try:                                                    # optional compiled inner loop
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _nl_gray_numba(target, F, bias, inv_h2, radius):
        h, w, c = F.shape
        out = np.empty((h, w), np.float64)
        for i in prange(h):
            for j in range(w):
                num = 0.0; den = 0.0
                inv = inv_h2[i, j]
                for ii in range(max(0, i - radius), min(h, i + radius + 1)):
                    for jj in range(max(0, j - radius), min(w, j + radius + 1)):
                        d2 = 0.0
                        for k in range(c):
                            df = F[i, j, k] - F[ii, jj, k]
                            d2 += df * df
                        e = d2 - bias
                        weight = np.exp(-e * inv) if e > 0.0 else 1.0
                        num += weight * target[ii, jj]; den += weight
                out[i, j] = num / den if den > 0.0 else target[i, j]
        return out

    @njit(parallel=True, cache=True, fastmath=True)
    def _nl_rgb_numba(target, F, bias, inv_h2, radius):
        h, w, c = F.shape
        out = np.empty((h, w, 3), np.float64)
        for i in prange(h):
            for j in range(w):
                n0 = 0.0; n1 = 0.0; n2 = 0.0; den = 0.0
                inv = inv_h2[i, j]
                for ii in range(max(0, i - radius), min(h, i + radius + 1)):
                    for jj in range(max(0, j - radius), min(w, j + radius + 1)):
                        d2 = 0.0
                        for k in range(c):
                            df = F[i, j, k] - F[ii, jj, k]
                            d2 += df * df
                        e = d2 - bias
                        weight = np.exp(-e * inv) if e > 0.0 else 1.0
                        n0 += weight * target[ii, jj, 0]
                        n1 += weight * target[ii, jj, 1]
                        n2 += weight * target[ii, jj, 2]
                        den += weight
                if den > 0.0:
                    out[i, j, 0] = n0 / den; out[i, j, 1] = n1 / den; out[i, j, 2] = n2 / den
                else:
                    out[i, j, 0] = target[i, j, 0]
                    out[i, j, 1] = target[i, j, 1]
                    out[i, j, 2] = target[i, j, 2]
        return out

    _HAVE_NUMBA = True
except Exception:                                       # pragma: no cover
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------

def _feature_blocks(img: np.ndarray, cfg: CrispConfig):
    """Feature blocks (each (H,W,c)) for a grayscale or colour image.

    gray  -> [ smoothed I , centred profiles(I) ... ]
    colour-> [ smoothed R, G, B , centred profiles(luminance) ... ]
    """
    if img.ndim == 3:
        lum = img @ _LUMA
        smoothed = np.stack([gaussian_filter(img[..., c], cfg.light_sigma)
                             for c in range(img.shape[-1])], axis=-1)
        profiles, _ = relation_profiles(lum, cfg)
        return [smoothed, *profiles]
    profiles, _ = relation_profiles(img, cfg)
    return [gaussian_filter(img, cfg.light_sigma)[..., None], *profiles]


_COLSD_CACHE: dict = {}


def _unit_noise_colsd(cfg: CrispConfig, channels: int) -> np.ndarray:
    """Per-column feature std for unit-variance white noise (linear in sigma)."""
    key = (cfg.K, cfg.scales, cfg.arc_taps, cfg.light_sigma, channels)
    hit = _COLSD_CACHE.get(key)
    if hit is None:
        shape = (160, 160, channels) if channels > 1 else (160, 160)
        ref = np.random.default_rng(0).normal(0.0, 1.0, shape)
        blk = np.concatenate(_feature_blocks(ref, cfg), axis=-1)
        hit = blk.reshape(-1, blk.shape[-1]).std(axis=0) + 1e-6
        _COLSD_CACHE[key] = hit
    return hit


def build_feature(img: np.ndarray, cfg: CrispConfig, noise_sigma: float):
    """Return ``(F, bias)``: normalised feature stack and its white-noise offset."""
    blocks = _feature_blocks(img, cfg)
    n_int = blocks[0].shape[-1]                          # 1 (gray) or 3 (colour)
    F = np.concatenate(blocks, axis=-1).astype(np.float64)

    channels = img.shape[-1] if img.ndim == 3 else 1
    F = F / (_unit_noise_colsd(cfg, channels) * max(noise_sigma, 1e-6))
    F[..., :n_int] *= cfg.rnlm_intensity_w               # up-weight the intensity anchor

    n_prof = F.shape[-1] - n_int
    bias = 2.0 * (n_prof + n_int * cfg.rnlm_intensity_w ** 2)
    return F.astype(np.float32), float(bias)


# ---------------------------------------------------------------------------
# weighted average
# ---------------------------------------------------------------------------

def _nl_average(target: np.ndarray, F: np.ndarray, bias: float,
                inv_h2: np.ndarray, radius: int) -> np.ndarray:
    """Windowed weighted average of ``target`` (H,W or H,W,3) with feature weights."""
    Fc = np.ascontiguousarray(F, np.float32)          # half the memory traffic
    inv = np.ascontiguousarray(inv_h2, np.float64)
    if _HAVE_NUMBA:
        if target.ndim == 2:
            return _nl_gray_numba(np.ascontiguousarray(target, np.float64),
                                  Fc, float(bias), inv, int(radius))
        return _nl_rgb_numba(np.ascontiguousarray(target, np.float64),
                             Fc, float(bias), inv, int(radius))

    tgt = target[..., None] if target.ndim == 2 else target
    h, w, t = tgt.shape
    num = np.zeros((h, w, t)); den = np.zeros((h, w, 1))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dF = F - np.roll(F, (-dy, -dx), axis=(0, 1))
            d2 = np.einsum("ijc,ijc->ij", dF, dF)
            weight = np.exp(-np.clip(d2 - bias, 0.0, None) * inv_h2)[..., None]
            num += weight * np.roll(tgt, (-dy, -dx), axis=(0, 1))
            den += weight
    out = num / np.maximum(den, 1e-8)
    return out[..., 0] if target.ndim == 2 else out


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def relational_nlm(noisy: np.ndarray, cfg: CrispConfig,
                   c_map: np.ndarray | None = None,
                   noise_sigma: float | None = None) -> np.ndarray:
    noisy = np.asarray(noisy, dtype=np.float64)
    is_color = noisy.ndim == 3
    lum = noisy @ _LUMA if is_color else noisy

    if noise_sigma is None:                              # per-channel level (colour too)
        try:
            from skimage.restoration import estimate_sigma
            if is_color:
                noise_sigma = float(np.mean(estimate_sigma(noisy, channel_axis=-1)))
            else:
                noise_sigma = float(estimate_sigma(noisy))
        except Exception:
            from .contrast import _mad_sigma
            noise_sigma = _mad_sigma(noisy[..., 1] if is_color else noisy)

    if c_map is None:
        from .pipeline import analyze
        c_map = analyze(lum, cfg, with_harmonics=False)["C"]
    c_map = np.clip(c_map, 0.0, 1.0)

    F, bias = build_feature(noisy, cfg, noise_sigma)
    h_map = cfg.rnlm_h * np.sqrt(bias) * (1.0 + cfg.rnlm_hmod_k * (1.0 - c_map))
    inv_h2 = (1.0 / (h_map * h_map)).astype(np.float64)

    out = _nl_average(noisy, F, bias, inv_h2, cfg.rnlm_radius)

    for _ in range(max(cfg.rnlm_passes - 1, 0)):         # E12: 2-pass did not help
        F2, bias2 = build_feature(out, cfg, noise_sigma * 0.5)
        h2 = cfg.rnlm_h * np.sqrt(bias2) * (1.0 + cfg.rnlm_hmod_k * (1.0 - c_map))
        out = _nl_average(noisy, F2, bias2, (1.0 / (h2 * h2)).astype(np.float64),
                          cfg.rnlm_radius)
    return out
