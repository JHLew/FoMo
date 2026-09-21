"""
dinov3.py — transformer prediction head used by DINO / CLIP / MAE models.

Trimmed port of ../fomo/dino.py. Absolute-coord RoPE only.

Pipeline:
    [B, N, D] token sequence (concatenated patch features for the two images,
              padded to a common spatial size with key_mask marking real tokens)
        │
        prepend CLS token + `num_registers` learnable register tokens
        │
        depth × DINOv3Block (pre-norm RoPE-attention + SwiGLU FFN, LayerScale)
        │   key_mask propagated so padding patches are never attended to
        │
        LayerNorm → CLS token → Linear(embed_dim, 1) → [B, 1] scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D Rotary Positional Embedding (absolute integer-coord variant).
# Returns (sin, cos) tables of shape [1, 1, H*W, dim] that broadcast over
# (batch, heads) when applied to spatial Q/K in attention.
# ---------------------------------------------------------------------------
class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Each axis (H, W) gets dim//2 features. We sample inv_freq at stride 2
        # so the same frequency is paired with its sin/cos counterpart later.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def _freqs_2d(self, seq_h: torch.Tensor, seq_w: torch.Tensor) -> torch.Tensor:
        H, W = seq_h.shape[0], seq_w.shape[0]
        fh = torch.einsum('i,j->ij', seq_h, self.inv_freq)                # [H, F]
        fw = torch.einsum('i,j->ij', seq_w, self.inv_freq)                # [W, F]
        emb_h = fh.unsqueeze(1).repeat(1, W, 1).reshape(-1, fh.shape[1])  # [H*W, F]
        emb_w = fw.unsqueeze(0).repeat(H, 1, 1).reshape(-1, fw.shape[1])  # [H*W, F]
        return torch.cat((emb_h, emb_w), dim=-1)                          # [H*W, 2*F]

    def forward(self, H: int, W: int, device, dtype):
        seq_h = torch.arange(H, device=device, dtype=self.inv_freq.dtype)
        seq_w = torch.arange(W, device=device, dtype=self.inv_freq.dtype)
        freqs = self._freqs_2d(seq_h, seq_w)               # [H*W, dim//2]
        freqs = torch.cat((freqs, freqs), dim=-1)          # [H*W, dim]
        sin = freqs.sin().to(dtype)[None, None]            # [1, 1, H*W, dim]
        cos = freqs.cos().to(dtype)[None, None]
        return sin, cos


def apply_rope(x, sin, cos):
    """Rotate the last dim of x by (sin, cos).
    x: [B, heads, N, head_dim]; sin/cos: [..., N, head_dim].
    Standard rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin].
    """
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Multi-head self-attention with 2D RoPE applied to Q and K on spatial tokens
# only (registers / aux tokens are untouched). Variable-resolution batches
# are handled via an optional boolean key mask (True = real, False = padding).
#
# `n_aux_tokens` is the count of leading non-spatial tokens (e.g., registers,
# CLS). When the spatial portion has length 2 * H*W, we treat it as the
# (A | B) concatenation and apply the same RoPE table independently to each
# half — this matches the source repo's convention.
# ---------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, sin, cos, n_aux_tokens: int, attn_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]                          # [B, H, N, d]

        q_aux, q_spatial = q[:, :, :n_aux_tokens], q[:, :, n_aux_tokens:]
        k_aux, k_spatial = k[:, :, :n_aux_tokens], k[:, :, n_aux_tokens:]

        n_spatial = q_spatial.shape[2]
        n_rope = sin.shape[2]
        if n_spatial == n_rope:
            q_spatial = apply_rope(q_spatial, sin, cos)
            k_spatial = apply_rope(k_spatial, sin, cos)
        elif n_spatial == 2 * n_rope:
            # Concatenated (A | B): apply the same RoPE table to each half.
            q1, q2 = q_spatial.chunk(2, dim=2)
            k1, k2 = k_spatial.chunk(2, dim=2)
            q_spatial = torch.cat([apply_rope(q1, sin, cos), apply_rope(q2, sin, cos)], dim=2)
            k_spatial = torch.cat([apply_rope(k1, sin, cos), apply_rope(k2, sin, cos)], dim=2)
        else:
            raise RuntimeError(
                f"spatial seq length ({n_spatial}) is neither equal to nor 2x the "
                f"RoPE table length ({n_rope})."
            )

        q = torch.cat((q_aux, q_spatial), dim=2)
        k = torch.cat((k_aux, k_spatial), dim=2)

        if attn_mask is not None:
            # scaled_dot_product_attention expects an additive float mask.
            float_mask = torch.zeros(attn_mask.shape, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill(~attn_mask, float('-inf'))
        else:
            float_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN: w3(silu(w1 x) * w2 x).
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# Pre-norm transformer block with LayerScale (DINOv3 convention).
# ---------------------------------------------------------------------------
class DINOv3Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.ls1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, sin, cos, n_aux_tokens: int, attn_mask=None):
        x = x + self.ls1 * self.attn(self.norm1(x), sin, cos, n_aux_tokens, attn_mask=attn_mask)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Stacked transformer head.
#
# Input  : x — [B, N, D] token sequence. N is either H*W (single image) or
#              2*H*W (concatenated A|B). The caller is responsible for the
#              concat strategy and for adding any per-image frame tag.
# Output : [B] scalar distances. Spatial tokens (post-stack, excluding
#          registers) are passed through a small MLP and then averaged.
# ---------------------------------------------------------------------------
class DINOViTModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 3,
        num_heads: int = 12,
        num_registers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_registers = num_registers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.rope = RotaryEmbedding2D(embed_dim // num_heads)
        self.blocks = nn.ModuleList([
            DINOv3Block(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, h: int, w: int, key_mask=None):
        """
        x        : [B, N, D] — N is h*w (single image) or 2*h*w (A|B concat).
                   Images in the batch may have different real spatial extents;
                   all are zero-padded to the same (h, w) grid.
        h, w     : padded patch-grid height/width per image.
        key_mask : [B, N] bool (True = real patch, False = padding).
                   Ensures padding patches are never attended to, making
                   variable-resolution batches safe. Not needed for CLS/registers
                   (always real) — only covers the N spatial input tokens.
        """
        B, N, _ = x.shape
        if N != h * w and N != 2 * h * w:
            raise ValueError(
                f"DINOViTModel: input length {N} doesn't match h*w={h*w} or 2*h*w={2*h*w}."
            )

        # Prepend CLS then registers. Layout: [CLS | regs | spatial...].
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls, regs, x), dim=1)

        # Build per-key boolean attention mask so that padding spatial tokens
        # produce -inf before softmax and therefore receive zero attention weight.
        # CLS and register tokens are always real — prepend True for them.
        n_special = 1 + self.num_registers  # CLS + registers
        if key_mask is not None:
            special_mask = torch.ones(B, n_special, dtype=torch.bool, device=key_mask.device)
            full_mask = torch.cat([special_mask, key_mask], dim=1)  # [B, N_total]
            # Expand to [B, 1, 1, N_total] for key-dimension broadcast over all queries.
            attn_mask = full_mask[:, None, None, :]
        else:
            attn_mask = None

        # RoPE table for the padded (h, w) grid — shape [1, 1, h*w, head_dim].
        # Padding patches get sequential positions but are never attended to
        # (masked out above), so their RoPE values don't affect outputs.
        sin, cos = self.rope(h, w, x.device, x.dtype)

        for blk in self.blocks:
            x = blk(x, sin, cos, n_special, attn_mask=attn_mask)

        # CLS token carries the aggregated representation.
        cls_out = self.norm(x[:, 0])        # [B, D]
        return self.head(cls_out).squeeze(-1)  # [B]
