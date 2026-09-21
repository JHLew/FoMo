"""
evaluation.py — benchmark evaluation (PIPAL, TID2013, CSIQ, LIVE).

One function per benchmark; each takes a DistanceModel and the dataset root
and returns {'srocc', 'krocc', 'plcc'} against the human scores:
    SROCC — Spearman rank correlation
    KROCC — Kendall rank correlation
    PLCC  — Pearson correlation after a 5-parameter logistic fit

Protocol (the one every number in the paper uses):
    * images are scored at their NATIVE resolution — no resizing;
    * the score is model.eval_distance(ref, dist) with the default options:
      output='raw', averaged over both argument orders for the prediction-head
      models (symmetric=True).
"""

import numpy as np
import torch
from PIL import Image
from scipy.optimize import curve_fit
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr

from dataset import load_csiq, load_live, load_pipal, load_tid2013


def _load_rgb(path: str) -> torch.Tensor:
    """uint8 [3, H, W]."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'))
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.inference_mode()
def predict_distances(model, samples: list[tuple], batch_size: int = 8,
                      device: str = 'cuda', symmetric: bool = True,
                      output: str = 'raw') -> np.ndarray:
    """One distance per (ref_path, dist_path, score) sample, at native
    resolution. Only equally-sized pairs share a mini-batch (PIPAL, TID2013
    and CSIQ are uniform; LIVE splits into a few size groups)."""
    model.eval()
    images = {}
    for r, d, _ in samples:
        for p in (r, d):
            if p not in images:
                images[p] = _load_rgb(p)

    groups: dict[tuple, list[int]] = {}
    for i, (r, d, _) in enumerate(samples):
        groups.setdefault((tuple(images[r].shape), tuple(images[d].shape)), []).append(i)

    preds = np.empty(len(samples), dtype=np.float64)
    for idx in groups.values():
        for s in range(0, len(idx), batch_size):
            chunk = idx[s:s + batch_size]
            refs  = torch.stack([images[samples[i][0]] for i in chunk]).float().div_(255).to(device)
            dists = torch.stack([images[samples[i][1]] for i in chunk]).float().div_(255).to(device)
            preds[chunk] = model.eval_distance(
                refs, dists, symmetric=symmetric, output=output).float().cpu().numpy()
    return preds


def logistic_func(x, b1, b2, b3, b4, b5):
    return b1 * (expit(b2 * (x - b3)) - 0.5) + b4 * x + b5


def fit_and_map(preds: np.ndarray, mos: np.ndarray) -> np.ndarray:
    """Fit the logistic to (preds → mos) and return the mapped predictions.

    The fit is non-convex, so the predictions are standardised first (the fit
    is invariant to affine changes of the score, which keeps PLCC comparable
    across models with different output scales), several initial guesses are
    tried, and the fit with the smallest squared residual is kept. This is the
    fit behind every PLCC number in the paper."""
    preds = np.asarray(preds, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)
    s, m = preds.std(), preds.mean()
    x = (preds - m) / (s if s > 0 else 1.0)
    rng = float(np.ptp(mos)) or 1.0
    starts = ([rng, 1.0, 0.0, 0.1, float(np.mean(mos))],
              [rng, 0.1, float(np.median(x)), 0.0, float(np.mean(mos))],
              [-rng, 1.0, 0.0, -0.1, float(np.mean(mos))],
              [rng, 5.0, 0.0, 0.0, float(np.mean(mos))],
              [rng, 0.5, 0.0, 0.0, float(np.mean(mos))])
    best, best_sse = None, np.inf
    for p0 in starts:
        try:
            popt, _ = curve_fit(logistic_func, x, mos, p0=p0, maxfev=40000)
        except Exception:
            continue
        mapped = logistic_func(x, *popt)
        sse = float(np.sum((mapped - mos) ** 2))
        if np.isfinite(sse) and sse < best_sse:
            best, best_sse = mapped, sse
    return best if best is not None else preds


def rank_metrics(preds: np.ndarray, mos: np.ndarray) -> dict:
    srocc, _ = spearmanr(preds, mos)
    krocc, _ = kendalltau(preds, mos)
    plcc, _  = pearsonr(fit_and_map(preds, mos), mos)
    return {'srocc': float(srocc), 'krocc': float(krocc), 'plcc': float(plcc)}


def _evaluate(model, samples, sign: float, batch_size: int, device: str,
              symmetric: bool, output: str) -> dict:
    preds = predict_distances(model, samples, batch_size=batch_size, device=device,
                              symmetric=symmetric, output=output)
    # Predictions are distances (higher = worse), so quality scores (PIPAL Elo,
    # TID2013 MOS) are negated; CSIQ / LIVE DMOS already point the same way.
    mos = sign * np.array([s[2] for s in samples])
    return rank_metrics(preds, mos)


def evaluate_pipal(model, root: str, split: str = 'val', batch_size: int = 8,
                   device: str = 'cuda', symmetric: bool = True,
                   output: str = 'raw') -> dict:
    return _evaluate(model, load_pipal(root, split=split), -1.0, batch_size,
                     device, symmetric, output)


def evaluate_tid2013(model, root: str, batch_size: int = 8, device: str = 'cuda',
                     symmetric: bool = True, output: str = 'raw') -> dict:
    return _evaluate(model, load_tid2013(root), -1.0, batch_size, device,
                     symmetric, output)


def evaluate_csiq(model, root: str, batch_size: int = 8, device: str = 'cuda',
                  symmetric: bool = True, output: str = 'raw') -> dict:
    return _evaluate(model, load_csiq(root), 1.0, batch_size, device,
                     symmetric, output)


def evaluate_live(model, root: str, batch_size: int = 8, device: str = 'cuda',
                  symmetric: bool = True, output: str = 'raw') -> dict:
    return _evaluate(model, load_live(root), 1.0, batch_size, device,
                     symmetric, output)


BENCHMARKS = {
    'pipal':   evaluate_pipal,
    'tid2013': evaluate_tid2013,
    'csiq':    evaluate_csiq,
    'live':    evaluate_live,
}
