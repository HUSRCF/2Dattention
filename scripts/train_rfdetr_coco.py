"""Train RF-DETR on a prepared COCO directory."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
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
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override RF-DETR class count. Use this when fine-tuning on a non-COCO category set.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrain-weights", type=Path, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-only", action="store_true", help="Validate imports and dataset layout without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_rfdetr_dataset(args.dataset_dir)
    dataset_num_classes = detect_rfdetr_dataset_num_classes(args.dataset_dir)
    if args.num_classes is not None and args.num_classes != dataset_num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match train split category count "
            f"{dataset_num_classes}. Fix the dataset or pass --num-classes {dataset_num_classes}."
        )
    if args.check_only:
        report = rfdetr_availability_report()
        print(f"dataset_ok: {args.dataset_dir}")
        print(f"dataset_num_classes: {dataset_num_classes}")
        print(f"requested_num_classes: {args.num_classes if args.num_classes is not None else 'auto'}")
        print(f"rfdetr_classes: {', '.join(report['classes'])}")
        return
    model = build_rfdetr_model(args.model_size, pretrain_weights=args.pretrain_weights, num_classes=args.num_classes)
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
    if args.device != "auto":
        train_kwargs["device"] = args.device
    if args.resolution is not None:
        train_kwargs["resolution"] = args.resolution
    if args.lr_encoder is not None:
        train_kwargs["lr_encoder"] = args.lr_encoder
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume)
    print(f"rfdetr_model: {MODEL_CLASSES[args.model_size]}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"output_dir: {args.output_dir}")
    model.train(**train_kwargs)


def build_rfdetr_model(
    model_size: str,
    pretrain_weights: Path | None = None,
    num_classes: int | None = None,
    device: str | None = None,
) -> Any:
    module = import_rfdetr_module()
    class_name = MODEL_CLASSES[model_size]
    if not hasattr(module, class_name):
        available = ", ".join(name for name in MODEL_CLASSES.values() if hasattr(module, name))
        raise RuntimeError(f"installed rfdetr does not expose {class_name}; available known classes: {available}")
    model_cls = getattr(module, class_name)
    model_kwargs: dict[str, Any] = {}
    if pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = str(pretrain_weights)
    if num_classes is not None:
        model_kwargs["num_classes"] = num_classes
    if device is not None:
        model_kwargs["device"] = device
    return model_cls(**model_kwargs)


def import_rfdetr_module() -> Any:
    if importlib.util.find_spec("rfdetr") is None:
        raise RuntimeError(
            "RF-DETR is not installed in this environment. Install it with `pip install rfdetr` "
            "inside the AIAA environment, then rerun this script."
        )
    try:
        return importlib.import_module("rfdetr")
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is installed but failed to import, likely because a transitive dependency is missing or "
            f"incompatible: {exc}"
        ) from exc


def rfdetr_availability_report() -> dict[str, list[str]]:
    module = import_rfdetr_module()
    return {
        "classes": [class_name for class_name in MODEL_CLASSES.values() if hasattr(module, class_name)],
    }


def validate_rfdetr_dataset(dataset_dir: Path) -> None:
    for split in ("train", "valid", "test"):
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"missing RF-DETR split annotation: {annotation_path}")


def detect_rfdetr_dataset_num_classes(dataset_dir: Path) -> int:
    import json

    annotation_path = dataset_dir / "train" / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = data.get("categories", [])
    if not categories:
        raise ValueError(f"RF-DETR train split has no categories: {annotation_path}")
    return len({int(category["id"]) for category in categories})


if __name__ == "__main__":
    main()
