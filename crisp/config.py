"""Configuration for the CRISP pipeline.

Every knob the method exposes lives here so experiments and ablations can sweep a
single object.  The defaults are the frozen configuration used for every result
reported in the method article.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrispConfig:
    # -- relation profile sampling -------------------------------------------------
    K: int = 8                                    # angular sampling directions
                                                 # (E: K=16 gives ~0.07 dB more at 2x cost)
    scales: tuple[float, ...] = (1.5, 3.0, 6.0)   # geometric radius ladder r_s
    arc_taps: int = 3                             # tangential taps averaged per sample
                                                 # (E1b: 5 is marginally better, 3x slower)

    # -- structural statistics ----------------------------------------------------
    stability_agg: float = 2.5                    # gaussian sigma for spatial
                                                 # aggregation of the Sigma field
                                                 # (E1b: the dominant SNR lever)
    c_mode: str = "fused"                         # "fused" -> noise-weighted blend of
                                                 # Sigma and a noise-floored contrast
                                                 # term (E8: AUC 0.92/0.88/0.81 vs
                                                 # 0.80/0.81/0.78; +0.1-0.2 dB on
                                                 # gaussian/TV bases, ~tie on NLM).
                                                 # "stability" -> C = clip(Sigma) only
                                                 # (no skimage dep, best vs NLM at low sigma)
    contrast_rho: float = 2.0
    phasor_smooth: float = 1.0                    # gaussian sigma applied to the
                                                 # complex coefficients before Phi

    # -- reconstruction mode ----------------------------------------------------
    mode: str = "rnlm"                            # "rnlm"  -> relational non-local means
                                                 #            (standalone denoiser, E10/E11)
                                                 # "gate"  -> confidence gate wrapper (needs
                                                 #            `heavy`); "self" -> gate w/ a
                                                 #            built-in gaussian heavy stage

    # -- relational non-local means -------------------------------------------------
    rnlm_radius: int = 7                          # half-size of the square search window
    rnlm_h: float = 0.60                          # selectivity, in units of sqrt(bias)
    rnlm_hmod_k: float = 0.8                      # C-modulation: h(p) = h*(1 + k*(1-C(p)))
    rnlm_intensity_w: float = 2.2                 # weight of the intensity feature block
                                                 # relative to the profile blocks (E11a3:
                                                 # plateau at 2.2-2.5; the knobs interact
                                                 # - do not also lower h/k)
    rnlm_passes: int = 1                          # >1: recompute features on the denoised
                                                 # image and average the noisy input again

    # -- reconstruction: confidence gate ("gate" / "self" modes) ---------------
    # J = C**gate_p * light(I)  +  (1 - C**gate_p) * heavy(I)
    # light = structure-preserving (tiny blur); heavy = any aggressive denoiser.
    gate_p: float = 1.5
    light_sigma: float = 0.7
    light_mix: float = 0.0                        # light = (1-mix)*blur(I) + mix*heavy
                                                 # 0    -> largest gain vs weak/mid
                                                 #         denoisers (the intended use)
                                                 # 0.35 -> ~half the gain, but near-
                                                 #         neutral even vs BM3D (E7)
    heavy: str = "gaussian"                       # "gaussian" | "tv" | "nlm"
    heavy_sigma: float = 2.6                      # for heavy == "gaussian"
    tv_weight: float = 0.16                       # for heavy == "tv"
    nlm_h: float = 1.1                            # for heavy == "nlm" (x noise sigma)

    # -- reconstruction: relational Poisson solve (ablation only; see E3) --------
    lam: float = 0.08
    n_iter: int = 80

    # -- harmonic shrinkage (ablation only; see E1 - Phi is weak on texture) ----
    shrink: str = "power"                         # "power" | "soft"
    gamma: float = 2.0
    soft_thresh: float = 0.30

    # -- numerics -----------------------------------------------------------------
    eps: float = 1e-8

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def n_harmonics(self) -> int:
        return self.K // 2
