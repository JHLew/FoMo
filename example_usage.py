"""
example_usage.py — the two ways to use a released FoMo model.

  metric   score how far a candidate image is from a reference (lower = closer)
  loss     use the distance as a perceptual training loss, LPIPS-style

    python example_usage.py --model dinov3 --mode metric --ref a.png --dist b.png
    python example_usage.py --model lpips_alex --mode loss

Inputs are float tensors [B, 3, H, W] in [0, 1] (no normalization — every
model normalizes internally); both images of a pair share a spatial size.

The metric demo prints both output presentations side by side: 'raw'
(default, what evaluation uses; lower = more similar) and 'loss' (hinged at
the identity level, for optimization).
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

import fomo


def load_image(path: str, device) -> torch.Tensor:
    return to_tensor(Image.open(path).convert('RGB')).unsqueeze(0).to(device)


def synthetic_image(device, seed: int, size: int = 256) -> torch.Tensor:
    """Smooth low-frequency image in [0, 1] (white noise is far off the
    natural-image distribution the models were trained on)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, 8, 8, generator=g).to(device)
    return F.interpolate(low, size=(size, size), mode='bicubic', align_corners=False).clamp(0, 1)


@torch.inference_mode()
def demo_metric(metric, device, ref_path, dist_path):
    if ref_path and dist_path:
        ref, dist = load_image(ref_path, device), load_image(dist_path, device)
        vals = {m: metric(ref, dist, output=m).item() for m in ('raw', 'loss')}
        print('score(ref, dist); lower = more similar')
        for m, v in vals.items():
            tag = '  <- default' if m == 'raw' else ''
            print(f"  output={m!r:12s}{v:12.4f}{tag}")
        return
    ref = synthetic_image(device, seed=0)
    torch.manual_seed(0)
    mild   = (ref + 0.05 * torch.randn_like(ref)).clamp(0, 1)
    severe = (ref + 0.30 * torch.randn_like(ref)).clamp(0, 1)
    refs, cands = torch.cat([ref, ref, ref]), torch.cat([ref, mild, severe])
    cols = {m: metric(refs, cands, output=m).tolist() for m in ('raw', 'loss')}
    print(f'{"":32s}' + ''.join(f'{m:>14s}' for m in cols))
    for i, label in enumerate(['score(ref, ref)', 'score(ref, mildly noised)',
                               'score(ref, severely noised)']):
        print(f'{label:32s}' + ''.join(f'{cols[m][i]:14.4f}' for m in cols))
    d_self, d_mild, d_severe = cols['raw']
    print('OK: severe > mild > identical' if d_severe > d_mild > d_self else 'unexpected ordering')


def demo_loss(metric, device, target_path, steps: int = 60):
    target = load_image(target_path, device) if target_path else synthetic_image(device, seed=1)
    logit = torch.zeros_like(target, requires_grad=True)   # sigmoid keeps pixels in [0, 1]
    opt = torch.optim.Adam([logit], lr=0.05)
    for step in range(steps + 1):
        img = torch.sigmoid(logit)
        # output='loss' is hinged at the identity level; the default 'raw'
        # score has no floor, so optimizing it does not settle.
        loss = metric(img, target, output='loss').mean()
        if step % 10 == 0:
            print(f'step {step:3d}  perceptual={loss.item():.4f}  pixel_L1={F.l1_loss(img, target).item():.4f}')
        if step < steps:
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dinov3', choices=list(fomo.MODELS))
    ap.add_argument('--mode', default='metric', choices=['metric', 'loss'])
    ap.add_argument('--ref', default=None, help='reference image (metric mode)')
    ap.add_argument('--dist', default=None, help='candidate image (metric mode)')
    ap.add_argument('--target', default=None, help='target image (loss mode)')
    ap.add_argument('--checkpoint', default=None, help='local .pth (default: download the released weights)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    metric = fomo.FoMo(args.model, device=args.device, checkpoint=args.checkpoint)
    print(f'[{args.model}]')
    if args.mode == 'metric':
        demo_metric(metric, args.device, args.ref, args.dist)
    else:
        demo_loss(metric, args.device, args.target)


if __name__ == '__main__':
    main()
