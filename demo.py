"""End-to-end demo for the CRISP / relational non-local means denoiser.

Usage:
    python demo.py                                   # cameraman, sigma = 25
    python demo.py samples/color/kodim23_parrots.png # a color sample
    python demo.py samples/grayscale/house.png 40    # custom noise level

Loads an image, adds white Gaussian noise, runs the denoiser with the frozen
default configuration, prints PSNR / SSIM, and writes a
noisy | denoised | clean strip next to the input image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage import io

from crisp import CrispConfig, analyze, denoise
from crisp.metrics import psnr, ssim

DEFAULT_IMAGE = Path(__file__).parent / "samples" / "grayscale" / "cameraman.png"


def add_gaussian(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    return img + np.random.default_rng(seed).normal(0.0, sigma, img.shape)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0

    clean = io.imread(path).astype(np.float64)
    if clean.ndim == 3 and clean.shape[2] == 4:      # drop alpha
        clean = clean[..., :3]
    noisy = add_gaussian(clean, sigma)

    out = denoise(noisy, CrispConfig())              # mode="rnlm" by default

    print(f"image        {path.name}  {clean.shape}")
    print(f"noise sigma  {sigma:.0f}")
    print(f"PSNR  noisy {psnr(clean, noisy):6.2f} dB   ->  denoised {psnr(clean, out):6.2f} dB")
    print(f"SSIM  noisy {ssim(clean, noisy):6.3f}      ->  denoised {ssim(clean, out):6.3f}")

    strip = np.clip(np.concatenate([noisy, out, clean], axis=1), 0, 255).astype(np.uint8)
    dst = path.with_name(path.stem + f"_demo_sigma{sigma:.0f}.png")
    io.imsave(dst, strip)
    print(f"wrote        {dst}   (noisy | denoised | clean)")

    # the structural confidence field on its own
    c_field = analyze(noisy if noisy.ndim == 2 else noisy @ [0.2989, 0.5870, 0.1140],
                      CrispConfig(), with_harmonics=False)["C"]
    io.imsave(path.with_name(path.stem + "_confidence.png"),
              (np.clip(c_field, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
