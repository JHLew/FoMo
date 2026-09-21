"""
model.py — the seven FoMo distance models.

    DistanceModel(name)   name ∈ {'lpips_alex', 'lpips_vgg', 'dists',
                                  'dino', 'clip', 'mae', 'dreamsim'}

Every model takes two images and returns one scalar per pair. Inputs are
float tensors [B, 3, H, W] in [0, 1] — do NOT pre-normalize, each backbone
applies its own preprocessing internally. Both images of a pair must share a
spatial size (any size; the transformer models zero-pad to their patch grid).

Two outputs:

    forward(img1, img2)        raw training output, single argument order.
    eval_distance(img1, img2, symmetric=True, output='raw')
                               the inference score (lower = more similar).
                               A model is defined by its backbone alone; how the
                               score is presented is chosen per call:

        symmetric  (default True; dino / clip / mae only) — the head reads the
                   two images as one concatenated sequence, so s(x, y) ≠ s(y, x)
                   in general; the score is ½·[s(x, y) + s(y, x)], both orders
                   from one backbone pass. symmetric=False returns s(x, y).
                   lpips_* / dists / dreamsim are symmetric by construction.
        output     (default 'raw') — which of three presentations to return.

`output` in detail.

    'raw'       the model output, unchanged — what evaluation uses and what the
                published numbers are computed from. Lower = more similar;
                dino / clip / mae return negative values.

    'loss'      relu( d(x, y) - max(d(x, x), d(y, y)).detach() ), for use as a
                perceptual loss. A hinge whose floor sits at the per-pair
                identity level: full gradient while the images differ, and zero
                once they are as close as identical, so optimization settles
                there rather than pushing past it. The anchor is detached
                because it depends on the image being optimized: with gradients
                flowing through it, the optimizer could lower the loss by
                raising d(y, y) instead of matching the reference.

Architecture of the prediction-head models (dino / clip / mae):

    img1, img2 ─► normalize ─► pad to patch grid ─► frozen backbone (one batched
    pass) ─► patch tokens ─► proj_patch (Linear) + per-image frame_tag
    ─► concat along the sequence [A | B] ─► DINOViTModel head
    (3 × {2D-RoPE attention, SwiGLU}, CLS readout) ─► scalar

Variable-resolution training batches are zero-padded to a common (H, W) and a
boolean patch mask [B, h*w] (True = real patch) is passed through so padding
never takes part in attention.
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from fomo.dinov3 import DINOViTModel

OUTPUT_MODES = ('raw', 'loss')

HEAD_EMBED_DIM = 384
HEAD_DEPTH = 3
HEAD_NUM_HEADS = 6
HEAD_NUM_REGISTERS = 4


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def _resolve_output(output: str) -> str:
    if output not in OUTPUT_MODES:
        raise ValueError(f'output must be one of {OUTPUT_MODES}, got {output!r}')
    return output


class DistanceModel(nn.Module):
    BACKBONES = ('lpips_alex', 'lpips_vgg', 'dists', 'dino', 'clip', 'mae', 'dreamsim')

    # Models whose output comes from an asymmetric prediction head. Only these
    # get the argument-order averaging (`symmetric`) and need explicit
    # self-scores for output='loss'; for the rest the self-score is 0, so the
    # anchor can be skipped.
    HEAD_BACKBONES = ('dino', 'clip', 'mae')

    def __init__(self, name: str, device: str | torch.device | None = None):
        """A model is defined by its backbone alone. `device` only matters for
        'dreamsim' (its ensemble caches tensors on the device it is built on);
        pass the device you will run on. How the score is presented
        (`symmetric`, `output`) is chosen per call, in eval_distance()."""
        super().__init__()
        if name not in self.BACKBONES:
            raise ValueError(f"Unknown model {name!r}. Choose from {self.BACKBONES}")
        self.backbone_name = name
        if name == 'lpips_alex':
            self.model = LPIPS('alex')
        elif name == 'lpips_vgg':
            self.model = LPIPS('vgg')
        elif name == 'dists':
            self.model = DISTS()
        elif name == 'dino':
            self.model = DINOBackbone()
        elif name == 'clip':
            self.model = CLIPBackbone()
        elif name == 'mae':
            self.model = MAEBackbone()
        elif name == 'dreamsim':
            self.model = DreamSimBackbone(device=device)

    @property
    def patch_size(self) -> int:
        return getattr(self.model, 'patch_size', 16)

    def forward(self, img1, img2, mask=None):
        """Raw training output, [B]. See the module docstring."""
        return self.model(img1, img2, mask=mask)

    def eval_distance(self, img1, img2, mask=None,
                      symmetric: bool = True, output: str = 'raw'):
        """Perceptual score, [B]; lower = more similar.

        Averaged over both argument orders for the prediction-head models
        (`symmetric`, default True) and presented as `output` — 'raw'
        (default, what evaluation uses) or 'loss' (hinged at the identity
        level, for optimization). Both are chosen per call; the model itself
        is defined by its backbone alone. See the module docstring.

        Differentiable w.r.t. the inputs. To use it as a perceptual loss pass
        output='loss': pairs at or below the identity level then get exactly
        zero gradient, so optimization settles there.
        """
        symmetric = bool(symmetric)
        output = _resolve_output(output)

        if self.backbone_name in self.HEAD_BACKBONES:
            d_xy, d_xx, d_yy = self.model.pair_scores(img1, img2, mask=mask,
                                                       symmetric=symmetric,
                                                       self_scores=(output == 'loss'))
            if output == 'raw':
                return d_xy
            # 'loss': hinge at the per-pair identity level. The anchor is
            # detached so that optimizing img2 cannot cheat by raising d(y, y).
            return F.relu(d_xy - torch.maximum(d_xx, d_yy).detach())

        # Symmetric by construction, with a self-score of 0, so the anchor is
        # 0 and 'loss' is a plain relu.
        d_xy = self.model(img1, img2, mask=mask)
        if self.backbone_name == 'dreamsim':
            d_xy = d_xy / self.model.logit_scale
        return F.relu(d_xy) if output == 'loss' else d_xy



# ---------------------------------------------------------------------------
# Shared helpers for the CNN models.
# ---------------------------------------------------------------------------

def _keyed_sequential(children: list, start: int, end: int) -> nn.Sequential:
    """Slice of a torchvision `features` list keyed by the ORIGINAL indices,
    which is how the lpips / DISTS packages name their sub-modules."""
    seq = nn.Sequential()
    for i in range(start, end):
        seq.add_module(str(i), children[i])
    return seq


def _torchvision_features(net: str) -> list:
    import torchvision.models as tv
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if net == 'alex':
            m = tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1)
        else:
            m = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1)
    return list(m.features.children())


# ---------------------------------------------------------------------------
# LPIPS — Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
# Perceptual Metric". Frozen ImageNet AlexNet / VGG-16 features + learned
# per-channel linear weights (lin0–lin4) on squared feature differences.
#
# The backbone is reset to torchvision's ImageNet weights and the calibration
# layers are re-initialized (Kaiming): FoMo trains the lin weights from scratch.
# ---------------------------------------------------------------------------

_LPIPS_SLICES = {
    'alex': [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)],
    'vgg':  [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)],
}


class LPIPS(nn.Module):
    def __init__(self, net: str = 'alex'):
        super().__init__()
        from lpips import LPIPS as _LPIPS
        self.lpips_model = _LPIPS(net=net, lpips=True, pretrained=True,
                                   pnet_rand=False, pnet_tune=True, verbose=False)

        children = _torchvision_features(net)
        slices = [getattr(self.lpips_model.net, f'slice{i}') for i in range(1, 6)]
        for (start, end), slc in zip(_LPIPS_SLICES[net], slices):
            slc.load_state_dict(_keyed_sequential(children, start, end).state_dict())
        for p in self.lpips_model.net.parameters():
            p.requires_grad_(False)

        for i in range(5):
            for m in getattr(self.lpips_model, f'lin{i}').modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, img1, img2, mask=None):
        # The lpips package expects [-1, 1].
        return self.lpips_model(img1 * 2.0 - 1.0, img2 * 2.0 - 1.0).view(img1.shape[0])


# ---------------------------------------------------------------------------
# DISTS — Ding et al., "Image Quality Assessment: Unifying Structure and
# Texture Similarity". Frozen VGG-16 stages + learned alpha/beta weights.
# Stages are reset to torchvision's ImageNet weights; alpha/beta are
# re-initialized to N(0.1, 0.01) and trained from scratch.
# ---------------------------------------------------------------------------

_DISTS_STAGES = [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]


class DISTS(nn.Module):
    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS as _DISTS
        # load_weights=False: the package's bundled alpha/beta are replaced below
        # (and its lookup of that file relative to sys.prefix fails in some
        # virtualenv layouts). VGG stages are frozen by the package.
        self.dists_model = _DISTS(load_weights=False)

        children = _torchvision_features('vgg')
        stages = [getattr(self.dists_model, f'stage{i}') for i in range(1, 6)]
        for (start, end), stage in zip(_DISTS_STAGES, stages):
            # strict=False: the stages also hold L2-pooling filter buffers.
            stage.load_state_dict(_keyed_sequential(children, start, end).state_dict(),
                                  strict=False)
        nn.init.normal_(self.dists_model.alpha, mean=0.1, std=0.01)
        nn.init.normal_(self.dists_model.beta,  mean=0.1, std=0.01)

    def forward(self, img1, img2, mask=None):
        return self.dists_model(img1, img2, require_grad=self.training).view(img1.shape[0])


# ---------------------------------------------------------------------------
# Frozen ViT backbone + DINOViTModel prediction head (dino / clip / mae).
#
# Subclasses set, before calling _build_head():
#   self.backbone      frozen HF vision model
#   self.backbone_dim  token width
#   self.patch_size    patch stride in pixels
#   self._num_prefix   number of non-spatial tokens the backbone prepends
#                      (CLS + its own registers) — stripped before the head
#   self._mean/_std    per-channel normalization constants
# ---------------------------------------------------------------------------

class _FrozenBackboneWithHead(nn.Module):
    backbone: nn.Module
    backbone_dim: int
    patch_size: int
    _num_prefix: int
    _mean: list
    _std: list

    def _build_head(self):
        self.proj_patch = nn.Linear(self.backbone_dim, HEAD_EMBED_DIM)
        # One learned offset per position in the pair, so the head can tell
        # which tokens came from which image. [1, 2, D]: 0 → first, 1 → second.
        self.frame_tag = nn.Parameter(torch.zeros(1, 2, HEAD_EMBED_DIM))
        nn.init.normal_(self.frame_tag, std=0.02)
        self.head = DINOViTModel(embed_dim=HEAD_EMBED_DIM, depth=HEAD_DEPTH,
                                 num_heads=HEAD_NUM_HEADS,
                                 num_registers=HEAD_NUM_REGISTERS)

    def _normalize(self, img):
        mean = torch.as_tensor(self._mean, device=img.device, dtype=img.dtype).view(1, -1, 1, 1)
        std  = torch.as_tensor(self._std,  device=img.device, dtype=img.dtype).view(1, -1, 1, 1)
        return (img - mean) / std

    def _pad_to_patch(self, img):
        _, _, H, W = img.shape
        pH = (self.patch_size - H % self.patch_size) % self.patch_size
        pW = (self.patch_size - W % self.patch_size) % self.patch_size
        return F.pad(img, (0, pW, 0, pH)) if (pH or pW) else img

    def _backbone_forward(self, img):
        return self.backbone(pixel_values=img, interpolate_pos_encoding=True).last_hidden_state.float()

    def _encode(self, img1, img2):
        """Both images through the frozen backbone in one batched pass.
        Returns projected tokens (p1, p2) of shape [B, h*w, D] and (h, w)."""
        img1 = self._pad_to_patch(self._normalize(img1))
        img2 = self._pad_to_patch(self._normalize(img2))
        assert img1.shape == img2.shape, 'both images of a pair must share a spatial size'
        h, w = img1.shape[-2] // self.patch_size, img1.shape[-1] // self.patch_size

        cat = torch.cat([img1, img2], dim=0)
        # The backbone is frozen; build its autograd graph only when the input
        # itself needs gradients (perceptual-loss use). During head training
        # the inputs are plain data, so this costs no backbone memory.
        with torch.set_grad_enabled(cat.requires_grad):
            tokens = self._backbone_forward(cat)[:, self._num_prefix:]   # [2B, h*w, C]
        tokens = self.proj_patch(tokens)
        B = img1.shape[0]
        return tokens[:B], tokens[B:], h, w

    def _run_head(self, first, second, h, w, mask):
        seq = torch.cat([first + self.frame_tag[:, 0:1],
                         second + self.frame_tag[:, 1:2]], dim=1)      # [B, 2*h*w, D]
        key_mask = torch.cat([mask, mask], dim=1) if mask is not None else None
        return self.head(seq, h, w, key_mask=key_mask)                 # [B]

    def forward(self, img1, img2, mask=None):
        p1, p2, h, w = self._encode(img1, img2)
        return self._run_head(p1, p2, h, w, mask)

    def pair_scores(self, img1, img2, mask=None, symmetric: bool = True,
                    self_scores: bool = False):
        """(d_xy, d_xx, d_yy), each [B], from ONE backbone pass; every head
        pass needed is batched into a single call.

            d_xy = ½·[s(x, y) + s(y, x)]  if symmetric else  s(x, y)
            d_xx = s(x, x), d_yy = s(y, y)  if self_scores else  None
        """
        p1, p2, h, w = self._encode(img1, img2)
        firsts, seconds = [p1], [p2]                  # s(x, y)
        if symmetric:
            firsts.append(p2); seconds.append(p1)     # s(y, x)
        if self_scores:
            firsts += [p1, p2]; seconds += [p1, p2]   # s(x, x), s(y, y)
        n = len(firsts)
        m = None if mask is None else mask.repeat(n, 1)
        out = self._run_head(torch.cat(firsts), torch.cat(seconds), h, w, m).view(n, -1)
        d_xy = 0.5 * (out[0] + out[1]) if symmetric else out[0]
        if self_scores:
            return d_xy, out[-2], out[-1]
        return d_xy, None, None


class DINOBackbone(_FrozenBackboneWithHead):
    """DINOv3 ViT-S/16+ (facebook/dinov3-vits16plus-pretrain-lvd1689m), bf16.
    Gated on the HuggingFace Hub: accept its license and `huggingface-cli login`."""
    MODEL_ID = 'facebook/dinov3-vits16plus-pretrain-lvd1689m'

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID, dtype=torch.bfloat16)
        self.backbone.requires_grad_(False)
        cfg = self.backbone.config
        self.backbone_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self._num_prefix = 1 + int(getattr(cfg, 'num_register_tokens', 0))   # CLS + registers
        self._mean = [0.485, 0.456, 0.406]
        self._std  = [0.229, 0.224, 0.225]
        self._build_head()


class CLIPBackbone(_FrozenBackboneWithHead):
    """CLIP ViT-B/32 vision tower (openai/clip-vit-base-patch32)."""
    MODEL_ID = 'openai/clip-vit-base-patch32'

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID).vision_model
        self.backbone.requires_grad_(False)
        cfg = self.backbone.config
        self.backbone_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self._num_prefix = 1   # CLS
        self._mean = [0.48145466, 0.4578275,  0.40821073]
        self._std  = [0.26862954, 0.26130258, 0.27577711]
        self._build_head()


class MAEBackbone(_FrozenBackboneWithHead):
    """MAE ViT-B/16 encoder (facebook/vit-mae-base), run without masking."""
    MODEL_ID = 'facebook/vit-mae-base'

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(self.MODEL_ID)
        self.backbone.requires_grad_(False)
        # Keep every patch: mask_ratio 0 (and zero noise below, so no token is dropped).
        self.backbone.config.mask_ratio = 0.0
        if hasattr(self.backbone.embeddings, 'config'):
            self.backbone.embeddings.config.mask_ratio = 0.0
        cfg = self.backbone.config
        self.backbone_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self._num_prefix = 1   # CLS
        self._mean = [0.485, 0.456, 0.406]
        self._std  = [0.229, 0.224, 0.225]
        self._build_head()

    def _backbone_forward(self, img):
        n = (img.shape[-2] // self.patch_size) * (img.shape[-1] // self.patch_size)
        noise = torch.zeros(img.shape[0], n, device=img.device)
        return self.backbone(pixel_values=img, noise=noise,
                             interpolate_pos_encoding=True).last_hidden_state.float()


# ---------------------------------------------------------------------------
# DreamSim — Fu et al. The released 3-ViT ensemble (DINO ViT-B/16 + CLIP
# ViT-B/16 + OpenCLIP ViT-B/16) with LoRA adapters (r=16, alpha=1, dropout
# 0.3 on qkv) and a cosine distance between the concatenated embeddings. FoMo
# keeps the architecture and trains fresh adapters on the FoMo data.
#
# The ensemble is fixed at 224×224, so inputs are resized (bicubic, the
# official DreamSim preprocessing) rather than padded; variable-resolution
# padded batches are rejected.
#
# forward() multiplies the cosine distance by `logit_scale` (20) so the ranking
# loss sees logit-sized differences; DistanceModel.eval_distance divides it back
# out.
# ---------------------------------------------------------------------------

DREAMSIM_CACHE_DIR = os.environ.get('FOMO_DREAMSIM_CACHE',
                                    os.path.expanduser('~/.cache/fomo/dreamsim'))


class DreamSimBackbone(nn.Module):
    RESOLUTION = 224

    def __init__(self, device=None, logit_scale: float = 20.0,
                 lora_rank: int = 16, lora_alpha: int = 1, lora_dropout: float = 0.3):
        super().__init__()
        from dreamsim.model import PerceptualModel, download_weights
        from peft import LoraConfig, get_peft_model

        # Base ViT weights (~1.2 GB, from the DreamSim GitHub release) are
        # cached under DREAMSIM_CACHE_DIR ($FOMO_DREAMSIM_CACHE).
        download_weights(cache_dir=DREAMSIM_CACHE_DIR, dreamsim_type='ensemble')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base = PerceptualModel(
            feat_type='cls,embedding,embedding',
            model_type='dino_vitb16,clip_vitb16,open_clip_vitb16',
            stride='16,16,16', lora=True, normalize_embeds=True,
            load_dir=DREAMSIM_CACHE_DIR, device=str(device),
        )
        self.base = get_peft_model(base, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias='none', target_modules=['qkv']))
        self.logit_scale = logit_scale
        self.patch_size = 16

    def _resize(self, img):
        if img.shape[-2:] == (self.RESOLUTION, self.RESOLUTION):
            return img
        return F.interpolate(img, size=(self.RESOLUTION, self.RESOLUTION),
                             mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)

    def forward(self, img1, img2, mask=None):
        if mask is not None and not bool(mask.all()):
            raise ValueError('DreamSim runs at a fixed 224x224 and cannot take '
                             'zero-padded variable-resolution batches; train it '
                             'with --min_crop_size 224 --max_crop_size 224.')
        return self.base(self._resize(img1), self._resize(img2)) * self.logit_scale
