"""CRISP - Cross-scale Relational Image denoising by Self-similarity of Profiles.

A training-free denoiser at the computational cost of non-local means. It works
in *relation space*:

    image -> multi-scale directional relation profiles  F(p)
          -> structural confidence field  C(p)
             (spatially aggregated cross-scale stability Sigma, which has a
              closed-form white-noise null, fused with a noise-floored contrast)
          -> relational non-local means: weighted average of the noisy
             intensities with weights exp(-(||F(p)-F(q)||^2 - bias) / h(p)^2),
             where the bandwidth h(p) is widened on flat regions by C(p)

``harmonics``/``shrink``/``reconstruct`` hold the earlier phase-coherence and
Poisson-reconstruction variants, kept for ablation only.
"""

from .config import CrispConfig
from .pipeline import analyze, denoise
from .relational import relational_nlm

__all__ = ["CrispConfig", "analyze", "denoise", "relational_nlm"]
__version__ = "0.1.0"
