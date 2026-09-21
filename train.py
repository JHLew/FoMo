"""
train.py — train a FoMo distance model with the ranked-BCE objective.

Single GPU, bf16 autocast, EMA of the trainable weights, native-resolution
benchmark evaluation after every epoch, resume-by-experiment-name.

Loss (ranked BCE): a batch of (img1, img2, label) pairs — label is the FLUX-
space value of the pair's timestep s (0 for an identical pair; higher = more
different; this s corresponds to S − s in the paper, S = 50). All
B×B ordered pairs of samples are formed and BCE-with-logits is applied to the
difference of predicted scores against the ground-truth ordering, so only the
ORDER of the model's outputs is supervised.

Configuration of the paper (scripts/train.sh):
    lpips_alex / lpips_vgg / dists : lr 2e-4, fixed 256×256 crops
    dino / clip / mae              : lr 1e-4, random 64–256 crops (zero-padded
                                     + attention-masked to batch them)
    dreamsim                       : lr 1e-4, fixed 224×224 crops
    3 epochs, batch 64, constant lr, weight decay 1e-8, EMA 0.999, bf16.

Outputs go to <base_dir>/<exp_name>/: ckpt_latest.pt (raw + EMA weights,
optimizer, epoch), metrics.json (per-epoch benchmark metrics), train.log,
args.json, tb/ (TensorBoard). Relaunching the same command resumes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.utils.data
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from dataset import FoMoDataset
from evaluation import BENCHMARKS
from fomo.model import DistanceModel


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True)
    p.add_argument('--base_dir', default='./experiments')
    p.add_argument('--backbone', required=True, choices=DistanceModel.BACKBONES)

    # Data.
    p.add_argument('--dataroot', default='./data/fomo')
    p.add_argument('--train_index_file', default='./data/fomo/index.json')
    p.add_argument('--min_crop_size', type=int, default=64)
    p.add_argument('--max_crop_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)

    # Optimisation.
    p.add_argument('--epochs',       type=int,   default=3)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-8)
    p.add_argument('--ema_decay',    type=float, default=0.999)
    p.add_argument('--amp', default='bf16', choices=['no', 'bf16'])
    p.add_argument('--seed', type=int, default=0)

    # Validation (native resolution, after every epoch).
    p.add_argument('--val', nargs='*', default=['pipal', 'tid2013', 'csiq', 'live'],
                   choices=list(BENCHMARKS), help='Benchmarks to evaluate.')
    p.add_argument('--benchmark_dir', default='./benchmarks',
                   help='Contains <benchmark_dir>/{pipal,tid2013,csiq,live}.')
    p.add_argument('--eval_batch_size', type=int, default=8)
    return p.parse_args()


def ranked_bce_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """pred, label: [B]. GT(i, j) = 1 if label_i > label_j, 0.5 if equal, else 0;
    loss = BCE_with_logits(pred_i - pred_j, GT(i, j)) over all i != j."""
    pred_matrix = pred.unsqueeze(1) - pred.unsqueeze(0)
    gt = (label.unsqueeze(1) > label.unsqueeze(0)).float() \
        + 0.5 * (label.unsqueeze(1) == label.unsqueeze(0)).float()
    off_diag = ~torch.eye(pred.shape[0], device=pred.device, dtype=torch.bool)
    return binary_cross_entropy_with_logits(pred_matrix[off_diag], gt[off_diag])


def train_epoch(model, loader, optimizer, ema, device, args, epoch: int) -> float:
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}', unit='batch', dynamic_ncols=True)
    for batch in pbar:
        img1  = batch['img1'].to(device, non_blocking=True)
        img2  = batch['img2'].to(device, non_blocking=True)
        label = batch['label'].to(device).float()
        mask  = batch['mask'].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=args.amp == 'bf16'):
            loss = ranked_bce_loss(model(img1, img2, mask=mask), label)
        if torch.isnan(loss):
            raise RuntimeError('Loss is NaN.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update()
        running += loss.item()
        n += 1
        pbar.set_postfix(loss=f'{running / n:.4f}')
    return running / max(n, 1)


def evaluate(model, args, device) -> dict:
    return {name: BENCHMARKS[name](model, f'{args.benchmark_dir}/{name}',
                                   batch_size=args.eval_batch_size, device=str(device))
            for name in args.val}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = Path(args.base_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(exp_dir / 'train.log')])
    log = logging.getLogger('train')
    writer = SummaryWriter(log_dir=str(exp_dir / 'tb'))
    (exp_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    model = DistanceModel(args.backbone, device=device).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f'{args.backbone}: {sum(p.numel() for p in trainable):,} trainable parameters')
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ema = ExponentialMovingAverage(trainable, decay=args.ema_decay)

    dataset = FoMoDataset(args.dataroot, args.train_index_file,
                          min_crop_size=args.min_crop_size,
                          max_crop_size=args.max_crop_size,
                          patch_size=model.patch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True,
                                         drop_last=True)

    # Resume.
    ckpt_path = exp_dir / 'ckpt_latest.pt'
    metrics_path = exp_dir / 'metrics.json'
    start_epoch = 0
    history = []
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        with ema.average_parameters():
            model.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        if metrics_path.exists():
            history = json.loads(metrics_path.read_text())
        log.info(f'Resumed after epoch {ckpt["epoch"]}')

    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, loader, optimizer, ema, device, args, epoch)
        log.info(f'Epoch {epoch}  loss={loss:.4f}')
        writer.add_scalar('train/loss', loss, epoch)

        with ema.average_parameters():          # all reported numbers use EMA weights
            metrics = evaluate(model, args, device)
            ema_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for bench, m in metrics.items():
            log.info(f'  {bench:8s}  ' + '  '.join(f'{k}={v:.4f}' for k, v in m.items()))
            for k, v in m.items():
                writer.add_scalar(f'val/{bench}/{k}', v, epoch)
        history.append({'epoch': epoch, 'loss': loss, **metrics})
        metrics_path.write_text(json.dumps(history, indent=2))

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema': ema_state,
                    'optimizer': optimizer.state_dict(), 'args': vars(args)}, ckpt_path)
    writer.close()


if __name__ == '__main__':
    main()
