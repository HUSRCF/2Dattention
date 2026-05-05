"""Run a small real-image smoke trial from a local torchvision ImageFolder."""

from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_best_device()
    dataset = build_dataset(args.data_root, args.image_size)
    if len(dataset.classes) < 2:
        raise ValueError("ImageFolder must contain at least two class folders")
    if len(dataset) < 4:
        raise ValueError("ImageFolder must contain at least four images")

    train_size = max(2, int(len(dataset) * 0.8))
    eval_size = len(dataset) - train_size
    if eval_size == 0:
        train_size -= 1
        eval_size = 1

    train_set, eval_set = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        name=args.model,
        embed_dim=args.embed_dim,
        image_size=args.image_size,
        num_classes=len(dataset.classes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    print("device:", device)
    print("data_root:", args.data_root)
    print("classes:", dataset.classes)
    print("model:", args.model)
    print("steps:", args.steps)

    model.train()
    loader_iter = cycle(train_loader)
    for step in range(1, args.steps + 1):
        images, labels = next(loader_iter)
        images = images.to(device)
        labels = labels.to(device)
        logits = logits_from_output(model(images))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == args.steps or step % 10 == 0:
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            print(
                f"step={step:03d}",
                f"loss={loss.item():.4f}",
                f"train_acc={accuracy:.3f}",
            )

    eval_loss, eval_acc = evaluate(model, eval_loader, device)
    print("eval_loss:", f"{eval_loss:.4f}")
    print("eval_acc:", f"{eval_acc:.3f}")


def build_dataset(data_root: Path, image_size: int) -> datasets.ImageFolder:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return datasets.ImageFolder(data_root, transform=transform)


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = logits_from_output(model(images))
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.numel())
    model.train()
    return total_loss / total_examples, total_correct / total_examples


def logits_from_output(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
        return logits
    return output


if __name__ == "__main__":
    main()
