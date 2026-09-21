"""
FoMo — perceptual image distance metrics trained on diffusion-generated
(FLUX.1-dev) image pairs.

    import fomo
    metric = fomo.FoMo('dinov3')            # downloads + caches the weights
    d = metric(img_ref, img_test)           # [B]; lower = more similar

Models: lpips_alex, lpips_vgg, dists, dinov3, clip, mae, dreamsim.
Inputs: float tensors [B, 3, H, W] in [0, 1], not pre-normalized.
A model is defined by its backbone alone; the score presentation is chosen per
call. symmetric (default True) averages both argument orders for the
prediction-head models; output selects one of three presentations —

    'raw'       (default) the model output; what evaluation uses and what the
                published numbers are computed from. Lower = more similar;
                dinov3 / clip / mae return negative values.
    'loss'      hinged at the identity pair, for use as a perceptual loss.
"""

from fomo.hub import MODELS, FoMo, load
from fomo.model import DistanceModel

__version__ = '0.1.1'
__all__ = ['FoMo', 'DistanceModel', 'load', 'MODELS', '__version__']
