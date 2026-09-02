"""Multi-scale directional relation profiles.

For pixel ``p`` and radius ``r`` the raw profile is the vector of intensity
differences ``I(p + r u_k) - I(p)`` over the ``K`` sampling directions.  We then
*centre* it (subtract its own angular mean).  Centring removes the shared
``-I(p)`` term, so profiles at different radii become statistically independent
under the white-noise null - the property that gives the primitives in
``primitives.py`` their closed-form null distributions.
"""

from __future__ import annotations

import numpy as np

from .config import CrispConfig
from .sampling import arc_sample


def relation_profiles(img: np.ndarray, cfg: CrispConfig):
    """Return ``(profiles, angles)``.

    ``profiles`` is a list of ``S`` arrays of shape ``(H, W, K)`` - the centred
    relation profile at each scale.  ``angles`` is the length-``K`` vector of
    sampling directions (radians).
    """
    angles = 2.0 * np.pi * np.arange(cfg.K) / cfg.K
    profiles = []
    for r in cfg.scales:
        samples = arc_sample(img, r, angles, cfg.arc_taps)
        g = samples - img[..., None]
        g = g - g.mean(axis=-1, keepdims=True)      # centre: kill common-mode
        profiles.append(g)
    return profiles, angles
