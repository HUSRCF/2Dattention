"""Run forward-only smoke checks on real images without labels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention2d import (  # noqa: E402
    ConvOnlyClassifier,
    TinyAnchorPrefillLatticeAttnRes,
    TinyGraphPrefillLatticeAttnRes,
    TinyPrefillLatticeAttnRes,
    TinyViTClassifier,
    TinyXAttnResClassifier,
    get_best_device,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(
            "conv_only",
            "tiny_vit",
            "xattnres_style",
            "prefill_lattice_attnres",
            "anchor_prefill_attnres",
            "graph_prefill_attnres",
        ),
        default="prefill_lattice_attnres",
    )
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_images(args.image_root, max_images=args.max_images)
    if not image_paths:
        raise ValueError(f"no images found under {args.image_root}")

    device = get_best_device()
    model = build_model(
        name=args.model,
        embed_dim=args.embed_dim,
        image_size=args.image_size,
        num_classes=args.num_classes,
    ).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch_paths in batched(image_paths, args.batch_size):
            images = torch.stack([load_image(path, transform) for path in batch_paths])
            images = images.to(device)
            logits = logits_from_output(model(images))
            total += int(logits.shape[0])

    elapsed = time.perf_counter() - start
    print("device:", device)
    print("image_root:", args.image_root)
    print("model:", args.model)
    print("images_seen:", total)
    print("logits_shape_last_batch:", tuple(logits.shape))
    print("elapsed_sec:", f"{elapsed:.3f}")
    print("images_per_sec:", f"{total / max(elapsed, 1e-9):.2f}")


def find_images(root: Path, max_images: int) -> list[Path]:
    images = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:max_images]


def batched(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def load_image(path: Path, transform: transforms.Compose) -> Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def build_model(
    name: str,
    embed_dim: int,
    image_size: int,
    num_classes: int,
) -> nn.Module:
    builders = {
        "conv_only": lambda: ConvOnlyClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "tiny_vit": lambda: TinyViTClassifier(
            embed_dim=embed_dim,
            image_size=image_size,
            num_classes=num_classes,
        ),
        "xattnres_style": lambda: TinyXAttnResClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "prefill_lattice_attnres": lambda: TinyPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "anchor_prefill_attnres": lambda: TinyAnchorPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
        "graph_prefill_attnres": lambda: TinyGraphPrefillLatticeAttnRes(
            embed_dim=embed_dim,
            num_classes=num_classes,
        ),
    }
    return builders[name]()


def logits_from_output(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
        return logits
    return output


if __name__ == "__main__":
    main()
