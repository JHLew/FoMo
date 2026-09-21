---
license: apache-2.0
tags:
- image-quality-assessment
- perceptual-metric
- perceptual-loss
- pytorch
library_name: pytorch
---

# FoMo: Perceptual Image Distance Models

Reference-based perceptual distance metrics trained on diffusion-generated (FLUX.1-dev) image pairs — no human labels. Given a reference and a candidate image, a FoMo model returns a scalar score (lower = more similar), usable as an image-quality metric or, with `output='loss'`, as an LPIPS-style perceptual loss.

Code: https://github.com/JHLew/FoMo · Training data: [JHLew/fomo_480k](https://huggingface.co/datasets/JHLew/fomo_480k)

## Usage

```bash
pip install fomo-iqa
```

```python
import fomo

# A model is defined by its backbone alone; `symmetric` and `output` are chosen
# per call. Images are float tensors [B, 3, H, W] in [0, 1], no pre-normalization.

# a CNN model — symmetric by construction
lpips_alex = fomo.FoMo('lpips_alex')
d = lpips_alex(img_ref, img_test)                   # [B]; lower = more similar

# a prediction-head model — `symmetric` averages both argument orders
dinov3 = fomo.FoMo('dinov3')
d = dinov3(img_ref, img_test, symmetric=True)       # 'raw' (default): what evaluation
                                                    # uses; may be negative

# as a perceptual loss — hinged at the identity level, so optimization settles
# once the images are as close as identical
loss = dinov3(prediction, target, output='loss').mean()
loss.backward()
```

**Output modes** (`output=`, chosen per call):

| `output` | what it returns | use it for |
|---|---|---|
| `'raw'` (default) | the model output, unchanged | evaluation, and every published number |
| `'loss'` | `relu( d(x, y) − max(d(x, x), d(y, y)) )` | a perceptual loss / image optimization |

## Files

| File | Model | Trained part | Size |
|---|---|---|---|
| `lpips_alex.pth` | LPIPS, AlexNet | linear calibration weights | 8 KB |
| `lpips_vgg.pth`  | LPIPS, VGG-16  | linear calibration weights | 9 KB |
| `dists.pth`      | DISTS, VGG-16  | α / β weights | 14 KB |
| `dinov3.pth`     | DINOv3 ViT-S/16+ + head | 3-layer transformer head | 29 MB |
| `clip.pth`       | CLIP ViT-B/32 + head    | 3-layer transformer head | 30 MB |
| `mae.pth`        | MAE ViT-B/16 + head     | 3-layer transformer head | 30 MB |
| `dreamsim.pth`   | DreamSim ensemble       | LoRA adapters (r=16) | 7 MB |

Every file is a plain `state_dict` of the **trained parameters only**, as EMA weights (decay 0.999) — the weights all reported numbers use. The frozen backbone is rebuilt from its public source at load time (torchvision, HuggingFace, or the DreamSim release), so the files are small and independent of the `transformers` version.

All seven were trained identically: 480k FoMo samples, ranked-BCE objective, 3 epochs (22.5k updates) at batch 64.

## Results

SROCC / KROCC / PLCC at native resolution, `output='raw'`; `python evaluate.py --model <name>` in the [code repository](https://github.com/JHLew/FoMo) reproduces every cell.

| Model | PIPAL | TID2013 | CSIQ | LIVE |
|---|---|---|---|---|
| `lpips_alex` | 0.739 / 0.541 / 0.773 | 0.782 / 0.583 / 0.813 | 0.940 / 0.784 / 0.946 | 0.949 / 0.794 / 0.943 |
| `lpips_vgg`  | 0.691 / 0.503 / 0.739 | 0.659 / 0.486 / 0.746 | 0.852 / 0.665 / 0.880 | 0.918 / 0.740 / 0.918 |
| `dists`      | 0.616 / 0.437 / 0.657 | 0.692 / 0.513 / 0.771 | 0.919 / 0.750 / 0.931 | 0.954 / 0.804 / 0.949 |
| `dinov3`     | 0.705 / 0.504 / 0.709 | 0.711 / 0.524 / 0.769 | 0.810 / 0.626 / 0.874 | 0.897 / 0.721 / 0.913 |
| `clip`       | 0.668 / 0.479 / 0.679 | 0.744 / 0.556 / 0.800 | 0.911 / 0.735 / 0.928 | 0.935 / 0.770 / 0.936 |
| `mae`        | 0.644 / 0.453 / 0.634 | 0.674 / 0.495 / 0.681 | 0.817 / 0.625 / 0.830 | 0.919 / 0.752 / 0.926 |
| `dreamsim`   | 0.790 / 0.585 / 0.779 | 0.806 / 0.610 / 0.836 | 0.894 / 0.710 / 0.905 | 0.933 / 0.773 / 0.936 |

## Notes

- Inputs: float tensors `[B, 3, H, W]` in `[0, 1]`; do **not** normalize (each model does so internally). Both images of a pair share a spatial size; any size works.
- `output=` picks the score presentation per call: `'raw'` (default — what evaluation and the numbers above use; lower = more similar) or `'loss'` (hinged at the identity pair, for use as a perceptual loss). `symmetric=` (default `True`) averages both argument orders for `dinov3` / `clip` / `mae`.
- `dinov3`, `clip` and `mae` return negative values. To report a non-negative figure, average the raw scores over the dataset and apply a monotone non-negative function such as `softplus` to that average — `softplus(scores.mean())`.
- **DINOv3 is gated**: accept the license of `facebook/dinov3-vits16plus-pretrain-lvd1689m` and `huggingface-cli login` before using `dinov3`.
- `dreamsim` downloads DreamSim's three base ViTs (~1.5 GB) on first use into `~/.cache/fomo/dreamsim`.

## License

Apache 2.0 (backbones and the `lpips` / `DISTS-pytorch` / `dreamsim` packages retain their upstream licenses).
