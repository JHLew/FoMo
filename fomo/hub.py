"""
hub.py — one-line loading of the released FoMo models.

    import fomo
    metric = fomo.FoMo('dinov3')          # downloads + caches the checkpoint
    d = metric(img_ref, img_test)         # [B] scores; lower = more similar

    model = fomo.load('dinov3')           # the underlying DistanceModel

A model is defined by its backbone alone. How the score is presented is chosen
per call — `symmetric=` (default True: average d(x, y) and d(y, x) for the
prediction-head models) and `output=`:

    'raw'       (default) the model output; what evaluation uses and what the
                published numbers are computed from. Lower = more similar;
                dinov3 / clip / mae return negative values.
    'loss'      hinged at the identity pair, for use as a perceptual loss.

See fomo/model.py for the full rationale.

Released checkpoints (HuggingFace Hub, repo `JHLew/FoMo`, one .pth per model):

    lpips_alex.pth  lpips_vgg.pth  dists.pth  dinov3.pth  clip.pth  mae.pth  dreamsim.pth

Each file holds only the TRAINED parameters (LPIPS calibration weights, DISTS
alpha/beta, the prediction head, or the DreamSim LoRA adapters) as EMA
weights — the weights every reported number was computed with. The frozen
backbone is rebuilt from its public pretrained source at construction, which
keeps the files small (8 KB – 30 MB) and makes loading independent of the
`transformers` version's module layout.

`fomo.load(..., checkpoint=path)` also accepts a train.py checkpoint
(ckpt_latest.pt; its EMA weights are used) or a full state_dict.
Set FOMO_HF_REPO to download from a different Hub repo.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

from fomo.model import DistanceModel

FOMO_HF_REPO = os.environ.get('FOMO_HF_REPO', 'JHLew/FoMo')

# released name -> (DistanceModel backbone key, checkpoint file)
MODELS = {
    'lpips_alex': ('lpips_alex', 'lpips_alex.pth'),
    'lpips_vgg':  ('lpips_vgg',  'lpips_vgg.pth'),
    'dists':      ('dists',      'dists.pth'),
    'dinov3':     ('dino',       'dinov3.pth'),
    'clip':       ('clip',       'clip.pth'),
    'mae':        ('mae',        'mae.pth'),
    'dreamsim':   ('dreamsim',   'dreamsim.pth'),
}


def trained_keys(model: DistanceModel) -> set[str]:
    """Names of the parameters that training updates (requires_grad=True at
    construction). This is exactly what a released checkpoint contains."""
    return {n for n, p in model.named_parameters() if p.requires_grad}


def trained_state_dict(model: DistanceModel) -> dict:
    keys = trained_keys(model)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in keys}


def load_trained_weights(model: DistanceModel, state: dict) -> None:
    """Load the trained parameters from `state` (a released checkpoint, a
    train.py checkpoint, or a full state_dict) into `model`, verifying that
    they cover the trainable parameters exactly."""
    if 'ema' in state and isinstance(state['ema'], dict):   # train.py checkpoint
        state = state['ema']
    keys = trained_keys(model)
    state = {k: v for k, v in state.items() if k in keys}
    if set(state) != keys:
        missing = sorted(keys - set(state))[:5]
        raise RuntimeError(f'Checkpoint does not match the {model.backbone_name!r} '
                           f'architecture; missing (first 5): {missing}')
    model.load_state_dict(state, strict=False)   # the rest is the frozen backbone


def load(name: str, device: str | torch.device | None = None,
         checkpoint: str | os.PathLike | None = None) -> DistanceModel:
    """Build a DistanceModel and load its released weights.

    name        one of MODELS ('lpips_alex', 'lpips_vgg', 'dists', 'dinov3',
                'clip', 'mae', 'dreamsim').
    device      target device (default: cuda if available).
    checkpoint  local .pth / .pt path; if omitted the released file is
                downloaded from the Hub (FOMO_HF_REPO) and cached.

    Returns the model in eval mode; use
    model.eval_distance(img1, img2, symmetric=..., output=...).
    """
    if name not in MODELS:
        raise ValueError(f'Unknown model {name!r}. Choose from {list(MODELS)}')
    backbone, filename = MODELS[name]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(repo_id=FOMO_HF_REPO, filename=filename)
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)

    model = DistanceModel(backbone, device=device)
    load_trained_weights(model, state)
    return model.to(device).eval()


class FoMo(nn.Module):
    """LPIPS-style callable around a released model.

        metric = FoMo('dinov3')                            # backbone only
        d = metric(img_ref, img_test)                      # [B]; lower = more similar
        d = metric(pred, target, output='loss')            # presentation per call

    forward() is DistanceModel.eval_distance(): differentiable w.r.t. the
    inputs with every parameter frozen. For use as a perceptual loss pass
    output='loss', which is hinged at the identity level.
    """

    def __init__(self, name: str = 'dinov3', device=None, checkpoint=None):
        super().__init__()
        self.name = name
        self.model = load(name, device=device, checkpoint=checkpoint)
        self.model.requires_grad_(False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor,
                symmetric: bool = True, output: str = 'raw') -> torch.Tensor:
        return self.model.eval_distance(img1, img2, symmetric=symmetric,
                                        output=output)
