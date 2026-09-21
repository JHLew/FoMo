"""
evaluate.py — score a FoMo model on PIPAL / TID2013 / CSIQ / LIVE.

    # released checkpoint (downloaded + cached automatically)
    python evaluate.py --model dinov3

    # a checkpoint you trained (EMA weights are used)
    python evaluate.py --model dinov3 \
        --checkpoint experiments/dino/ckpt_latest.pt \
        --benchmarks pipal

Benchmark roots default to ./benchmarks/<name> (scripts/setup_benchmarks.sh).

Images are scored at native resolution with the model's default options;
`--output` and `--asymmetric` switch them (see fomo/model.py).
"""

import argparse
import json

import torch

import fomo
from evaluation import BENCHMARKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(fomo.MODELS))
    p.add_argument('--checkpoint', default=None,
                   help='Local .pth / .pt; default: download the released weights.')
    p.add_argument('--benchmarks', nargs='+', default=list(BENCHMARKS), choices=list(BENCHMARKS))
    p.add_argument('--benchmark_dir', default='./benchmarks')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='raw', choices=['raw', 'loss'],
                   help="Score presentation: 'raw' (default, what the published "
                        "numbers use) or 'loss' (hinged at the identity level, "
                        "for use as a perceptual loss).")
    p.add_argument('--asymmetric', action='store_true',
                   help='Single argument order d(ref, dist) for dinov3/clip/mae '
                        'instead of averaging both orders.')
    args = p.parse_args()

    model = fomo.load(args.model, device=args.device, checkpoint=args.checkpoint)
    print(f'protocol: symmetric={not args.asymmetric} output={args.output}')
    results = {}
    for name in args.benchmarks:
        results[name] = BENCHMARKS[name](model, f'{args.benchmark_dir}/{name}',
                                         batch_size=args.batch_size, device=args.device,
                                         symmetric=not args.asymmetric, output=args.output)
        print(f'{name:8s}  ' + '  '.join(f'{k}={v:.4f}' for k, v in results[name].items()))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
