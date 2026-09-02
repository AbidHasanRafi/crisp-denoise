# Sample images

A small set of standard test images so the denoiser can be tried without
downloading a full benchmark set. Add synthetic white Gaussian noise yourself
(the demo does this) or feed your own noisy image.

```bash
python demo.py samples/grayscale/house.png 25
python demo.py samples/color/kodim23_parrots.png 25
```

## Contents

| File | Size | Source |
|------|------|--------|
| `grayscale/cameraman.png` | 256 x 256 | Set12 (`cszn/DnCNN` test set), classic test image |
| `grayscale/house.png`     | 256 x 256 | Set12 |
| `grayscale/starfish.png`  | 256 x 256 | Set12 |
| `grayscale/monarch.png`   | 256 x 256 | Set12 |
| `color/kodim05_bikes.png`      | 768 x 512 | Kodak image suite, image 05 |
| `color/kodim19_lighthouse.png` | 768 x 512 | Kodak image suite, image 19 |
| `color/kodim23_parrots.png`    | 768 x 512 | Kodak image suite, image 23 |

## Provenance and terms

- **Set12** grayscale images are the 12-image test set distributed with
  Zhang et al., *Beyond a Gaussian Denoiser* (DnCNN):
  <https://github.com/cszn/DnCNN/tree/master/testsets/Set12>. They are long-standing
  community benchmark images redistributed for research use.
- **Kodak** images are from the Kodak Lossless True Color Image Suite
  (<https://r0k.us/graphics/kodak/>), released by Eastman Kodak for unrestricted
  research use.

These files are included only to make the method reproducible. All rights remain
with the original owners. If you are the rights holder and want an image removed,
open an issue.

## Full benchmark sets

The results in the README and the method article use the complete sets:

- CBSD68  (68 images)
- Set12   (12 images), from `cszn/DnCNN`
- Kodak24 (24 images), from the Kodak suite linked above

They are standard public benchmarks and are not redistributed here.
