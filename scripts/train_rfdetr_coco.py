"""Train RF-DETR on a prepared COCO directory."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import random
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
    parser.add_argument("--lr-drop", type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=("step", "cosine"), default=None)
    parser.add_argument("--lr-min-factor", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
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
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "class-head", "query-class-head", "decoder-class-head"),
        default="all",
        help=(
            "Restrict trainable RF-DETR parameters. `class-head` freezes the "
            "backbone/decoder/box heads and trains only detector classification heads. "
            "`query-class-head` also trains query features and reference points. "
            "`decoder-class-head` trains decoder/query representation parameters and "
            "classification heads, but keeps backbone and box heads frozen."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed Python, NumPy, Torch, and Lightning if available.")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-only", action="store_true", help="Validate imports and dataset layout without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
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
    install_rfdetr_trainable_scope_patch(args.trainable_scope)
    model = build_rfdetr_model(args.model_size, pretrain_weights=args.pretrain_weights, num_classes=args.num_classes)
    trainable_report = configure_trainable_scope(model, args.trainable_scope)
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
    if args.lr_drop is not None:
        train_kwargs["lr_drop"] = args.lr_drop
    if args.lr_scheduler is not None:
        train_kwargs["lr_scheduler"] = args.lr_scheduler
    if args.lr_min_factor is not None:
        train_kwargs["lr_min_factor"] = args.lr_min_factor
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume)
    print(f"rfdetr_model: {MODEL_CLASSES[args.model_size]}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"output_dir: {args.output_dir}")
    if args.seed is not None:
        print(f"seed: {args.seed}")
    print(f"trainable_scope: {args.trainable_scope}")
    print(
        "trainable_parameters: "
        f"{trainable_report['trainable_tensors']}/{trainable_report['total_tensors']} tensors, "
        f"{trainable_report['trainable_elements']}/{trainable_report['total_elements']} elements"
    )
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


def install_rfdetr_trainable_scope_patch(scope: str) -> None:
    if scope == "all":
        return
    try:
        from rfdetr.training.module_model import RFDETRModelModule
    except ImportError:
        # Let RF-DETR raise its normal training-dependency error later.
        return
    if getattr(RFDETRModelModule, "_attention2d_trainable_scope_patch", None) == scope:
        return
    original_init = RFDETRModelModule.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        report = configure_trainable_scope(self.model, scope)
        print(
            "inner_trainable_parameters: "
            f"{report['trainable_tensors']}/{report['total_tensors']} tensors, "
            f"{report['trainable_elements']}/{report['total_elements']} elements"
        )

    RFDETRModelModule.__init__ = patched_init
    RFDETRModelModule._attention2d_trainable_scope_patch = scope


def configure_trainable_scope(model: Any, scope: str) -> dict[str, int]:
    inner = locate_torch_module(model)
    if inner is None:
        raise RuntimeError("Unable to locate RF-DETR torch module for trainable-scope configuration")
    if scope == "all":
        for _name, parameter in inner.named_parameters():
            parameter.requires_grad_(True)
    elif scope == "class-head":
        for name, parameter in inner.named_parameters():
            parameter.requires_grad_(is_class_head_parameter(name))
    elif scope == "query-class-head":
        for name, parameter in inner.named_parameters():
            parameter.requires_grad_(is_class_head_parameter(name) or is_query_parameter(name))
    elif scope == "decoder-class-head":
        for name, parameter in inner.named_parameters():
            parameter.requires_grad_(
                is_class_head_parameter(name)
                or is_query_parameter(name)
                or is_decoder_representation_parameter(name)
            )
    else:  # pragma: no cover - argparse constrains this.
        raise ValueError(f"unknown trainable scope: {scope}")
    return trainable_parameter_report(inner)


def locate_torch_module(model: Any) -> Any | None:
    if hasattr(model, "named_parameters"):
        return model
    context_model = getattr(model, "model", None)
    if hasattr(context_model, "named_parameters"):
        return context_model
    nested_model = getattr(context_model, "model", None)
    if hasattr(nested_model, "named_parameters"):
        return nested_model
    return None


def is_class_head_parameter(name: str) -> bool:
    return name == "class_embed.weight" or name == "class_embed.bias" or name.startswith(
        "transformer.enc_out_class_embed."
    )


def is_query_parameter(name: str) -> bool:
    return name in {"query_feat.weight", "refpoint_embed.weight"}


def is_decoder_representation_parameter(name: str) -> bool:
    if not name.startswith("transformer.decoder."):
        return False
    return not name.startswith("transformer.decoder.ref_point_head.")


def trainable_parameter_report(module: Any) -> dict[str, int]:
    total_tensors = 0
    trainable_tensors = 0
    total_elements = 0
    trainable_elements = 0
    for _name, parameter in module.named_parameters():
        total_tensors += 1
        total_elements += parameter.numel()
        if parameter.requires_grad:
            trainable_tensors += 1
            trainable_elements += parameter.numel()
    return {
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "total_elements": total_elements,
        "trainable_elements": trainable_elements,
    }


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import pytorch_lightning as pl

        pl.seed_everything(seed, workers=True)
    except Exception:
        pass


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
