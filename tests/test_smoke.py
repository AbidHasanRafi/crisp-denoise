import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crisp.config import CrispConfig
from crisp.metrics import psnr, ssim
from crisp.phantom import add_noise, make_phantom
from crisp.pipeline import analyze, denoise


def test_shapes():
    img, lab = make_phantom(96)
    cfg = CrispConfig(K=8, scales=(1.5, 3.0))
    a = analyze(img, cfg)
    assert a["sigma"].shape == img.shape
    assert a["phi"].shape == (*img.shape, 4)
    assert a["C"].shape == img.shape
    assert np.all(np.isfinite(a["C"]))


def test_rnlm_improves_psnr():
    img, _ = make_phantom(112)
    noisy = add_noise(img, "gaussian", 25.0, seed=1)
    cfg = CrispConfig(K=8, scales=(1.5, 3.0, 6.0), rnlm_radius=5)  # default mode="rnlm"
    out = denoise(noisy, cfg)
    assert psnr(img, out) > psnr(img, noisy) + 3.0
    assert np.all(np.isfinite(out))


def test_rnlm_beats_plain_gaussian():
    from scipy.ndimage import gaussian_filter
    img, _ = make_phantom(140)
    noisy = add_noise(img, "gaussian", 20.0, seed=4)
    cfg = CrispConfig(rnlm_radius=6)
    out = denoise(noisy, cfg)
    g = gaussian_filter(noisy, 1.4)
    assert psnr(img, out) > psnr(img, g)
    assert ssim(img, out) > ssim(img, g)


def test_gate_mode_still_works():
    from scipy.ndimage import gaussian_filter
    img, _ = make_phantom(120)
    noisy = add_noise(img, "gaussian", 25.0, seed=2)
    heavy = gaussian_filter(noisy, 2.6)
    cfg = CrispConfig(mode="gate")
    out = denoise(noisy, cfg, heavy=heavy)
    assert psnr(img, out) >= psnr(img, heavy) - 0.1


def test_fused_c_mode():
    img, _ = make_phantom(120)
    noisy = add_noise(img, "gaussian", 25.0, seed=1)
    cfg = CrispConfig(c_mode="fused")
    aux = analyze(noisy, cfg, with_harmonics=False)
    assert aux["C"].shape == img.shape
    assert 0.0 <= float(aux["C"].min()) and float(aux["C"].max()) <= 1.0
    assert "contrast" in aux


def test_null_sigma_mean_near_zero():
    rng = np.random.default_rng(0)
    noise = rng.normal(128, 20, (200, 200))
    cfg = CrispConfig(K=16, scales=(1.5, 3.0, 6.0), arc_taps=1, phasor_smooth=0.0,
                      stability_agg=0.0)
    a = analyze(noise, cfg)
    s = a["sigma"][20:-20, 20:-20]
    assert abs(float(s.mean())) < 0.03
