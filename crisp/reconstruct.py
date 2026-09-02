"""Earlier reconstruction variants, kept for ablation only.

The current method is ``crisp.relational`` (relational non-local means). Before
that, the confidence field ``C(p)`` was used as a **confidence gate**:

    J(p) = C(p)**p * light(I)(p)  +  (1 - C(p)**p) * heavy(I)(p)

``light``  a tiny blur - keeps edges/texture where C says "structure";
``heavy``  any aggressive denoiser - applied where C says "noise".

This is still reachable via ``CrispConfig(mode="gate")`` (wrap an external
denoiser) or ``mode="self"`` (built-in Gaussian heavy stage).

``poisson_reconstruct`` is the relational gradient-domain solve; it plateaus
below the gate, so it is kept for ablation only.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import CrispConfig

# ---------------------------------------------------------------------------
# confidence gate  (the method)
# ---------------------------------------------------------------------------

def _heavy_denoise(img: np.ndarray, cfg: CrispConfig) -> np.ndarray:
    kind = cfg.heavy
    if kind == "gaussian":
        return gaussian_filter(img, cfg.heavy_sigma)
    if kind == "tv":
        from skimage.restoration import denoise_tv_chambolle
        rng = float(img.max() - img.min()) or 1.0
        return denoise_tv_chambolle(img / rng, weight=cfg.tv_weight) * rng
    if kind == "nlm":
        from skimage.restoration import denoise_nl_means, estimate_sigma
        s = float(estimate_sigma(img))
        return denoise_nl_means(img, h=cfg.nlm_h * s, sigma=s, patch_size=5,
                                patch_distance=6, fast_mode=True)
    raise ValueError(f"unknown heavy denoiser: {kind!r}")


def confidence_gate(img: np.ndarray, c_map: np.ndarray, cfg: CrispConfig,
                    heavy: np.ndarray | None = None) -> np.ndarray:
    """Blend a light and a heavy denoise of ``img`` by the confidence field.

    ``light`` mixes a tiny blur of the input with ``heavy`` itself
    (``cfg.light_mix``), so that in structure regions the gate *adds back* raw
    detail rather than *replacing* ``heavy`` — this bounds the damage when
    ``heavy`` is already an excellent structure preserver (E7: BM3D).
    """
    if heavy is None:
        heavy = _heavy_denoise(img, cfg)
    blur = gaussian_filter(img, cfg.light_sigma) if cfg.light_sigma > 0 else img
    light = (1.0 - cfg.light_mix) * blur + cfg.light_mix * heavy
    a = np.clip(c_map, 0.0, 1.0) ** cfg.gate_p
    return a * light + (1.0 - a) * heavy


# ---------------------------------------------------------------------------
# relational Poisson solve  (ablation only - see experiments/e3)
# ---------------------------------------------------------------------------

_OFFSETS = np.array(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
    dtype=np.int64,
)


def stencil():
    offs = _OFFSETS
    angles = np.arctan2(offs[:, 0], offs[:, 1])
    radii = np.hypot(offs[:, 0], offs[:, 1])
    opp = np.array([(i + 4) % 8 for i in range(8)])
    return offs, angles, radii, opp


def _gather(a: np.ndarray, off) -> np.ndarray:
    return np.roll(a, shift=(-int(off[0]), -int(off[1])), axis=(0, 1))


def poisson_reconstruct(img: np.ndarray, profile: np.ndarray, cfg: CrispConfig,
                        beta: np.ndarray | None = None) -> np.ndarray:
    offs, _, _, opp = stencil()
    n = len(offs)
    if beta is None:
        beta = np.ones_like(img)
    lam = cfg.lam
    b_sum = sum(_gather(beta, offs[k]) for k in range(n))
    denom = lam + beta * n + b_sum
    bt = [beta * profile[..., opp[k]] for k in range(n)]
    j = img.copy()
    for _ in range(cfg.n_iter):
        term1 = beta * sum(_gather(j, offs[k]) - profile[..., k] for k in range(n))
        term2 = sum(_gather(beta * j + bt[k], offs[k]) for k in range(n))
        j = (lam * img + term1 + term2) / denom
    return j
