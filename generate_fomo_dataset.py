"""
generate_fomo_dataset.py — produce FLUX image pairs for the FoMo training set.

Two modes:
  1. Scratch mode (default): generate a base image from a blank canvas
     (strength=1.0), then produce N variants by img2img at random timestep
     strengths.

Timestep convention: a variant at timestep s (its filename, s in [1, S] with
S = --num-steps = 50) is generated with img2img strength s / S, i.e. it
re-enters the denoising trajectory s steps before the end — larger s forks
earlier and lands farther from the base. NOTE: this s corresponds to S − s in
the paper.

  2. Reference mode (--ref-images DIR): load reference images from a directory
     (e.g., ImageNet) and produce N variants from each reference via img2img.

The released 480k dataset is the union of both modes (240k scratch samples,
tag FLUX, and 240k ImageNet-referenced samples, tag IN1K), each sample with
2 variants.

In both modes each sample is saved as:
    <output_dir>/<split>/<TAG>_<idx:07d>/
        base.<fmt>          — base / reference image
        <timestep>.<fmt>    — variant at that denoising timestep

After each sample the JSON index file is updated (appended) at:
    <output_dir>/<index-file>

Index schema:
    [
      {
        "base":     "<split>/<TAG>_0000000/base.png",
        "variants": ["<split>/<TAG>_0000000/12.png", ...]
      },
      ...
    ]

Paths in the index are relative to <output_dir> so the dataset root can be
relocated without invalidating the index.
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from torchvision import transforms

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_CFG_SCALE = 3.5


def _round16(v: int) -> int:
    return round(v / 16) * 16


# ---------------------------------------------------------------------------
# Pipeline loading.
# ---------------------------------------------------------------------------
def load_pipeline(model_id: str = DEFAULT_MODEL_ID):
    from diffusers import FluxImg2ImgPipeline

    print(f"Loading FLUX pipeline from {model_id}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to("cuda")
    return pipe


# ---------------------------------------------------------------------------
# Single pipeline call.
# ---------------------------------------------------------------------------
def run_pipeline(pipe, init_image: Image.Image, pipeline_args, generator=None,
                 size=(512, 512)) -> Image.Image:
    w, h = _round16(size[0]), _round16(size[1])
    init_image = init_image.resize((w, h))

    return pipe(
        prompt=pipeline_args.prompt,
        image=init_image,
        num_inference_steps=pipeline_args.num_inference_steps,
        strength=pipeline_args.strength,
        guidance_scale=pipeline_args.guidance_scale,
        width=w,
        height=h,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# ImageNet-style dataset loader.
#
# Expects the standard ImageNet layout:
#   <root>/<split>/<class_dir>/<image_file>
# with .JPEG / .jpg / .png extensions.
#
# train split: RandomResizedCrop + RandomHorizontalFlip (data augmentation).
# val   split: Resize short-side to resolution+64, then CenterCrop.
# Both return a PIL image ready to pass to the diffusion pipeline.
# ---------------------------------------------------------------------------
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


class _ImageNetRawDataset(torch.utils.data.Dataset):
    """Flat list of image paths under <root>/<split>/."""

    def __init__(self, root: str, split: str, transform=None):
        from pathlib import Path
        self.transform = transform
        split_dir = os.path.join(root, split)
        self.paths: list[str] = []
        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for fn in sorted(os.listdir(class_path)):
                if Path(fn).suffix in _IMG_EXTS:
                    self.paths.append(os.path.join(class_path, fn))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image.Image:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageNetDataset:
    """
    ImageNet loader that returns PIL images at `resolution`.

    train split: RandomResizedCrop(resolution, scale=(0.8,1.0)) + RandomHorizontalFlip
    val   split: Resize to resolution+64 then CenterCrop to resolution
    """

    def __init__(self, root: str, split: str = "train", resolution: tuple[int, int] = (512, 512)):
        w, h = resolution
        if split == "train":
            tfm = transforms.Compose([
                transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            tfm = transforms.Compose([
                transforms.Resize((h + 64, w + 64)),
                transforms.CenterCrop((h, w)),
            ])
        self._dataset = _ImageNetRawDataset(root, split, transform=tfm)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Image.Image:
        return self._dataset[idx]


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate FLUX image pairs for training.")
    p.add_argument("--outputs",      required=True, help="Output directory.")
    p.add_argument("--split",        default="train", choices=["train", "val"])
    p.add_argument("--tag",          default="FLUX",
                   help="Sample directory prefix (e.g. FLUX for scratch mode, "
                        "IN1K for reference mode).")
    p.add_argument("--resolution",   type=int, nargs=2, default=[512, 512],
                   metavar=("W", "H"))
    p.add_argument("--num-steps",    type=int, default=50,
                   help="Total denoising steps for generation.")
    p.add_argument("--n-per-sample", type=int, default=2,
                   help="Number of variants to generate per base image.")
    p.add_argument("--img-format",   default="png", choices=["png", "jpg", "webp"])
    p.add_argument("--idx-from",     type=int, default=0)
    p.add_argument("--idx-to",       type=int, default=9999,
                   help="Inclusive last sample index.")
    p.add_argument("--seed",         type=int, default=None,
                   help="Global random seed for reproducibility.")

    # Reference-image mode.
    p.add_argument("--ref-images",   default=None,
                   help="Path to a directory of reference images (ImageNet layout). "
                        "If omitted, generate base images from scratch "
                        "(blank canvas, strength=1.0).")

    # Model.
    p.add_argument("--model-id",       default=DEFAULT_MODEL_ID,
                   help="HuggingFace model ID of the FLUX img2img pipeline.")
    p.add_argument("--guidance-scale", type=float, default=DEFAULT_CFG_SCALE,
                   help="CFG guidance scale.")

    # Index file.
    p.add_argument("--index-file",    default="index.json",
                   help="Name of the JSON index file written inside --outputs.")
    p.add_argument("--overwrite-index", action="store_true",
                   help="Overwrite any existing index file instead of appending.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    resolution = tuple(args.resolution)

    # Validate ref-image mode.
    ref_dataset: ImageNetDataset | None = None
    if args.ref_images:
        ref_dataset = ImageNetDataset(args.ref_images, split=args.split, resolution=resolution)
        if len(ref_dataset) == 0:
            raise RuntimeError(f"No images found under --ref-images {args.ref_images!r}.")
        print(f"Found {len(ref_dataset)} reference images ({args.split} split).")

    pipeline = load_pipeline(model_id=args.model_id)

    split_dir = os.path.join(args.outputs, args.split)
    os.makedirs(split_dir, exist_ok=True)

    # Load existing index so we can append rather than wipe it by default.
    index_path = os.path.join(args.outputs, args.index_file)
    if not args.overwrite_index and os.path.isfile(index_path):
        with open(index_path) as f:
            index: list[dict] = json.load(f)
    else:
        index = []

    pipeline_args = argparse.Namespace(
        prompt="",
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        strength=1.0,
    )

    # Build a shuffled index list for reference mode so images aren't drawn in
    # sorted (class-sequential) order. Seeded if --seed was given.
    ref_indices: list[int] | None = None
    if ref_dataset is not None:
        ref_indices = list(range(len(ref_dataset)))
        random.shuffle(ref_indices)

    for i in range(args.idx_from, args.idx_to + 1):
        sample_dir = os.path.join(split_dir, f"{args.tag}_{i:07d}")
        os.makedirs(sample_dir, exist_ok=True)
        # Clear stale files so a re-run of the same index range is idempotent.
        for fn in os.listdir(sample_dir):
            os.remove(os.path.join(sample_dir, fn))

        fmt = args.img_format
        base_path_abs = os.path.join(sample_dir, f"base.{fmt}")

        # ---- Obtain base image ------------------------------------------------
        if ref_dataset is not None:
            # Reference mode: draw from the shuffled index, cycling if idx-to > dataset size.
            base_img = ref_dataset[ref_indices[i % len(ref_indices)]]
            base_img.save(base_path_abs)
        else:
            # Scratch mode: generate base from a blank canvas (strength=1.0).
            blank = Image.new("RGB", resolution)
            base_seed = random.randint(0, 2**32 - 1)
            pipeline_args.strength = 1.0
            base_img = run_pipeline(
                pipeline, blank, pipeline_args,
                generator=torch.Generator("cpu").manual_seed(base_seed),
                size=resolution,
            )
            base_img.save(base_path_abs)

        # ---- Generate variants ------------------------------------------------
        # Sample unique timesteps s in [1, num_steps]; strength = s / num_steps.
        # (This s corresponds to S − s in the paper, S = num_steps.)
        timesteps = random.sample(range(1, args.num_steps + 1), args.n_per_sample)
        variant_rel_paths: list[str] = []

        for timestep in timesteps:
            variant_path_abs = os.path.join(sample_dir, f"{timestep}.{fmt}")
            strength = timestep / args.num_steps
            seed = random.randint(0, 2**32 - 1)
            pipeline_args.strength = strength
            variant = run_pipeline(
                pipeline, base_img, pipeline_args,
                generator=torch.Generator("cpu").manual_seed(seed),
                size=resolution,
            )
            variant.save(variant_path_abs)

            variant_rel_paths.append(
                os.path.relpath(variant_path_abs, args.outputs)
            )

        base_rel_path = os.path.relpath(base_path_abs, args.outputs)
        index.append({"base": base_rel_path, "variants": variant_rel_paths})

        # Write the index after every sample so an interrupted run keeps its progress.
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"Sample {i} complete → {sample_dir}")

    print(f"\nDone. {len(index)} entries written to {index_path}.")


if __name__ == "__main__":
    main()
