"""Evaluation metrics: image quality + statistical separation."""

from __future__ import annotations

import numpy as np


def psnr(a, b, peak: float = 255.0) -> float:
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)


def ssim(a, b, peak: float = 255.0) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # pragma: no cover
        return float("nan")
    a, b = np.asarray(a, float), np.asarray(b, float)
    kw = {"channel_axis": 2} if a.ndim == 3 else {}
    return float(structural_similarity(a, b, data_range=peak, **kw))


def epi(clean, restored) -> float:
    """Edge-Preservation Index: correlation of high-pass (Laplacian) responses.

    1.0 = edges perfectly preserved; lower = edges lost or spurious edges added.
    """
    from scipy.ndimage import laplace

    a = np.asarray(clean, float)
    b = np.asarray(restored, float)
    if a.ndim == 3:
        a, b = a.mean(-1), b.mean(-1)
    da, db = laplace(a), laplace(b)
    da -= da.mean(); db -= db.mean()
    denom = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2)) + 1e-12
    return float(np.sum(da * db) / denom)


def cohens_d(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
                 / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / (sp + 1e-12))


def auc_mwu(pos, neg):
    """Return ``(AUC, p_value)`` via the Mann-Whitney U statistic."""
    from scipy.stats import mannwhitneyu

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg))), float(p)
