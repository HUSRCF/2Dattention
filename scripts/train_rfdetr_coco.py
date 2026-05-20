"""Train RF-DETR on a prepared COCO directory."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=tuple(MODEL_CLASSES), default="nano")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrain-weights", type=Path, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_rfdetr_dataset(args.dataset_dir)
    model = build_rfdetr_model(args.model_size, pretrain_weights=args.pretrain_weights)
    train_kwargs = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "eval_interval": args.eval_interval,
        "run_test": args.run_test,
        "checkpoint_interval": args.checkpoint_interval,
        "tensorboard": args.tensorboard,
        "wandb": args.wandb,
    }
    if args.lr_encoder is not None:
        train_kwargs["lr_encoder"] = args.lr_encoder
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume)
    print(f"rfdetr_model: {MODEL_CLASSES[args.model_size]}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"output_dir: {args.output_dir}")
    model.train(**train_kwargs)


def build_rfdetr_model(model_size: str, pretrain_weights: Path | None = None) -> Any:
    try:
        module = importlib.import_module("rfdetr")
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed in this environment. Install it with `pip install rfdetr` "
            "inside the AIAA environment, then rerun this script."
        ) from exc
    class_name = MODEL_CLASSES[model_size]
    if not hasattr(module, class_name):
        available = ", ".join(name for name in MODEL_CLASSES.values() if hasattr(module, name))
        raise RuntimeError(f"installed rfdetr does not expose {class_name}; available known classes: {available}")
    model_cls = getattr(module, class_name)
    if pretrain_weights is not None:
        return model_cls(pretrain_weights=str(pretrain_weights))
    return model_cls()


def validate_rfdetr_dataset(dataset_dir: Path) -> None:
    for split in ("train", "valid", "test"):
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"missing RF-DETR split annotation: {annotation_path}")


if __name__ == "__main__":
    main()
