"""Compare small models on DET-derived spatial bbox probe tasks."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from attention2d import get_best_device  # noqa: E402
from compare_imagefolder_models import (  # noqa: E402
    MODEL_NAMES,
    MODEL_SEED_OFFSETS,
    build_model,
    collect_mechanism_stats,
    count_parameters,
    get_mps_memory_mb,
    split_dataset,
    train_model,
)


@dataclass(frozen=True)
class BBoxSample:
    image_id: str
    image_path: Path
    width: int
    height: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class ProbeResultRow:
    dataset: str
    device: str
    run_seed: int
    model: str
    seed: int
    steps: int
    params: int
    train_loss: float
    train_acc: float
    eval_loss: float
    eval_acc: float
    balanced_acc: float
    macro_f1: float
    majority_baseline: float
    balanced_random_baseline: float
    best_eval_loss: float
    best_eval_acc: float
    best_step: int
    images_per_sec: float
    mps_current_mem_mb: float
    mps_driver_mem_mb: float
    gate_mean: float
    gate_max_abs: float
    read_norm_mean: float
    state_norm_mean: float
    scaled_read_ratio_mean: float
    per_class_recall_json: str
    per_class_precision_json: str
    corner_recall: float
    edge_recall: float
    center_recall: float
    confusion_json: str
    block_stats_json: str


class BBoxProbeDataset(Dataset[tuple[Tensor, int]]):
    """One largest-object bbox probe label per DET validation image."""

    def __init__(
        self,
        samples: list[BBoxSample],
        task: str,
        image_size: int,
    ) -> None:
        self.samples = samples
        self.task = task
        self.labels = [bbox_probe_label(sample, task) for sample in samples]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("data/ILSVRC2013_DET_bbox_val/ILSVRC2013_DET_bbox_val"),
    )
    parser.add_argument(
        "--task",
        choices=("quadrant4", "grid9", "size3"),
        default="quadrant4",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--reference-model", choices=MODEL_NAMES, default="no_prefill_local_mix")
    parser.add_argument(
        "--reference-models",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="optional additional paired references; defaults to --reference-model",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--balanced-train",
        action="store_true",
        help="use class-balanced weighted sampling for the training split",
    )
    parser.add_argument(
        "--balanced-eval",
        action="store_true",
        help="evaluate on a class-balanced subset of the eval split",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/bbox_probe_compare.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_device()
    samples = load_largest_bbox_samples(args.anno_root, args.image_root)
    dataset = BBoxProbeDataset(samples=samples, task=args.task, image_size=args.image_size)
    num_classes = num_probe_classes(args.task)
    label_counts = Counter(bbox_probe_label(sample, args.task) for sample in samples)

    print("device:", device)
    print("task:", args.task)
    print("images:", len(dataset))
    print("label_counts:", dict(sorted(label_counts.items())))
    print("majority_baseline_all:", max(label_counts.values()) / sum(label_counts.values()))
    print("balanced_random_baseline:", 1.0 / num_classes)
    print("balanced_train:", args.balanced_train)
    print("balanced_eval:", args.balanced_eval)
    print("steps:", args.steps)
    print("seeds:", args.seeds)
    print("model,seed,params,train_loss,train_acc,eval_loss,raw_acc,balanced_acc,macro_f1,best_eval_acc,best_step,images_per_sec")

    rows: list[ProbeResultRow] = []
    for seed_idx in range(args.seeds):
        split_seed = args.seed + seed_idx
        train_set, eval_set = split_dataset(dataset, args.train_frac, split_seed)
        if args.balanced_eval:
            eval_set = make_balanced_subset(eval_set, num_classes=num_classes, seed=20_000_000 + split_seed)
        eval_counts = label_counts_for_dataset(eval_set, num_classes=num_classes)
        majority_baseline = max(eval_counts) / max(1, sum(eval_counts))
        eval_loader = DataLoader(
            eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        for model_name in args.models:
            train_loader = build_probe_train_loader(
                train_set=train_set,
                batch_size=args.batch_size,
                seed=10_000_000 + split_seed,
                num_classes=num_classes,
                balanced=args.balanced_train,
            )
            model_seed = split_seed + MODEL_SEED_OFFSETS[model_name]
            torch.manual_seed(model_seed)
            model = build_model(
                name=model_name,
                embed_dim=args.embed_dim,
                image_size=args.image_size,
                num_classes=num_classes,
            ).to(device)
            params = count_parameters(model)
            train_result = train_model(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
            )
            eval_metrics = evaluate_probe(model, eval_loader, device, num_classes=num_classes)
            mechanism_stats = collect_mechanism_stats(model, eval_loader, device)
            mps_current_mem_mb, mps_driver_mem_mb = get_mps_memory_mb()
            row = ProbeResultRow(
                dataset=f"bbox_probe:{args.task}",
                device=str(device),
                run_seed=split_seed,
                model=model_name,
                seed=model_seed,
                steps=args.steps,
                params=params,
                train_loss=train_result["train_loss"],
                train_acc=train_result["train_acc"],
                eval_loss=eval_metrics["loss"],
                eval_acc=eval_metrics["raw_acc"],
                balanced_acc=eval_metrics["balanced_acc"],
                macro_f1=eval_metrics["macro_f1"],
                majority_baseline=majority_baseline,
                balanced_random_baseline=1.0 / num_classes,
                best_eval_loss=train_result["best_eval_loss"],
                best_eval_acc=train_result["best_eval_acc"],
                best_step=train_result["best_step"],
                images_per_sec=train_result["images_per_sec"],
                mps_current_mem_mb=mps_current_mem_mb,
                mps_driver_mem_mb=mps_driver_mem_mb,
                gate_mean=mechanism_stats["gate_mean"],
                gate_max_abs=mechanism_stats["gate_max_abs"],
                read_norm_mean=mechanism_stats["read_norm_mean"],
                state_norm_mean=mechanism_stats["state_norm_mean"],
                scaled_read_ratio_mean=mechanism_stats["scaled_read_ratio_mean"],
                per_class_recall_json=eval_metrics["per_class_recall_json"],
                per_class_precision_json=eval_metrics["per_class_precision_json"],
                corner_recall=eval_metrics["corner_recall"],
                edge_recall=eval_metrics["edge_recall"],
                center_recall=eval_metrics["center_recall"],
                confusion_json=eval_metrics["confusion_json"],
                block_stats_json=mechanism_stats["block_stats_json"],
            )
            rows.append(row)
            print(
                f"{model_name},{model_seed},{params},"
                f"{train_result['train_loss']:.4f},{train_result['train_acc']:.3f},"
                f"{eval_metrics['loss']:.4f},{eval_metrics['raw_acc']:.3f},"
                f"{eval_metrics['balanced_acc']:.3f},{eval_metrics['macro_f1']:.3f},"
                f"{train_result['best_eval_acc']:.3f},{train_result['best_step']},"
                f"{train_result['images_per_sec']:.2f},"
                f"gate={mechanism_stats['gate_mean']:.5f},"
                f"ratio={mechanism_stats['scaled_read_ratio_mean']:.5f}"
            )

    write_probe_csv(args.out, rows)
    print("saved_csv:", args.out)
    print_probe_summary(rows)
    reference_models = args.reference_models or [args.reference_model]
    for reference_model in dict.fromkeys(reference_models):
        print_probe_paired_summary(rows, reference_model=reference_model)


def load_largest_bbox_samples(anno_root: Path, image_root: Path) -> list[BBoxSample]:
    samples: list[BBoxSample] = []
    for xml_path in sorted(anno_root.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_id = required_text(root, "filename")
        image_path = image_root / f"{image_id}.JPEG"
        if not image_path.exists():
            continue
        size = root.find("size")
        if size is None:
            raise ValueError(f"missing size in {xml_path}")
        width = int(required_text(size, "width"))
        height = int(required_text(size, "height"))
        boxes = []
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = int(required_text(box, "xmin"))
            ymin = int(required_text(box, "ymin"))
            xmax = int(required_text(box, "xmax"))
            ymax = int(required_text(box, "ymax"))
            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            boxes.append((area, xmin, ymin, xmax, ymax))
        if not boxes:
            continue
        _, xmin, ymin, xmax, ymax = max(boxes, key=lambda item: item[0])
        samples.append(
            BBoxSample(
                image_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )
    if not samples:
        raise ValueError(f"no bbox samples found under {anno_root}")
    return samples


def build_probe_train_loader(
    train_set: Dataset[tuple[Tensor, int]],
    batch_size: int,
    seed: int,
    num_classes: int,
    balanced: bool,
) -> DataLoader:
    if not balanced:
        return DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )

    labels = labels_for_dataset(train_set)
    counts = Counter(labels)
    weights = [
        1.0 / max(1, counts[label])
        for label in labels
    ]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if len(counts) < num_classes:
        missing = sorted(set(range(num_classes)) - set(counts))
        print("warning_missing_train_classes:", missing)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def make_balanced_subset(
    dataset: Dataset[tuple[Tensor, int]],
    num_classes: int,
    seed: int,
) -> Subset[tuple[Tensor, int]]:
    labels = labels_for_dataset(dataset)
    grouped: dict[int, list[int]] = {label: [] for label in range(num_classes)}
    for idx, label in enumerate(labels):
        grouped[label].append(idx)
    min_count = min((len(indices) for indices in grouped.values()), default=0)
    if min_count == 0:
        missing = [label for label, indices in grouped.items() if not indices]
        raise ValueError(f"cannot build balanced eval subset, missing classes: {missing}")

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in range(num_classes):
        indices = grouped[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[idx] for idx in order[:min_count])
    selected_order = torch.randperm(len(selected), generator=generator).tolist()
    selected = [selected[idx] for idx in selected_order]
    return Subset(dataset, selected)


def labels_for_dataset(dataset: Dataset[tuple[Tensor, int]]) -> list[int]:
    if isinstance(dataset, Subset):
        parent_labels = labels_for_dataset(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]
    if isinstance(dataset, BBoxProbeDataset):
        return list(dataset.labels)
    return [int(dataset[idx][1]) for idx in range(len(dataset))]


def label_counts_for_dataset(
    dataset: Dataset[tuple[Tensor, int]],
    num_classes: int,
) -> list[int]:
    counts = Counter(labels_for_dataset(dataset))
    return [counts[label] for label in range(num_classes)]


def evaluate_probe(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float | str]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = logits_from_output(model(images))
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(dim=1)
            for target, prediction in zip(labels.cpu(), predictions.cpu()):
                confusion[int(target), int(prediction)] += 1
            total_examples += int(labels.numel())

    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    recall = true_positive / support.clamp_min(1.0)
    precision = true_positive / predicted.clamp_min(1.0)
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-8)
    raw_acc = float(true_positive.sum().item() / max(1, total_examples))
    balanced_acc = float(recall.mean().item())
    macro_f1 = float(f1.mean().item())
    return {
        "loss": total_loss / max(1, total_examples),
        "raw_acc": raw_acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "per_class_recall_json": json_dumps_float_list(recall.tolist()),
        "per_class_precision_json": json_dumps_float_list(precision.tolist()),
        "corner_recall": group_recall(recall, [0, 2, 6, 8], num_classes),
        "edge_recall": group_recall(recall, [1, 3, 5, 7], num_classes),
        "center_recall": group_recall(recall, [4], num_classes),
        "confusion_json": str(confusion.tolist()).replace(" ", ""),
    }


def logits_from_output(output: Tensor | dict[str, Tensor | list[Tensor]]) -> Tensor:
    if isinstance(output, dict):
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output['logits'] must be a tensor")
        return logits
    return output


def write_probe_csv(path: Path, rows: list[ProbeResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ProbeResultRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_probe_summary(rows: list[ProbeResultRow]) -> None:
    models = sorted({row.model for row in rows})
    print(
        "summary_model,params_mean,raw_acc_mean,raw_acc_std,"
        "balanced_acc_mean,macro_f1_mean,best_eval_acc_mean,"
        "corner_recall_mean,edge_recall_mean,center_recall_mean,"
        "majority_baseline_mean,images_per_sec_mean,gate_mean,scaled_read_ratio_mean"
    )
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        raw_accs = [row.eval_acc for row in model_rows]
        balanced_accs = [row.balanced_acc for row in model_rows]
        macro_f1s = [row.macro_f1 for row in model_rows]
        best_accs = [row.best_eval_acc for row in model_rows]
        corner_recalls = [row.corner_recall for row in model_rows if not torch.isnan(torch.tensor(row.corner_recall))]
        edge_recalls = [row.edge_recall for row in model_rows if not torch.isnan(torch.tensor(row.edge_recall))]
        center_recalls = [row.center_recall for row in model_rows if not torch.isnan(torch.tensor(row.center_recall))]
        params = [row.params for row in model_rows]
        majorities = [row.majority_baseline for row in model_rows]
        speeds = [row.images_per_sec for row in model_rows]
        gates = [row.gate_mean for row in model_rows if not torch.isnan(torch.tensor(row.gate_mean))]
        ratios = [
            row.scaled_read_ratio_mean
            for row in model_rows
            if not torch.isnan(torch.tensor(row.scaled_read_ratio_mean))
        ]
        gate_mean = mean_or_nan(gates)
        ratio_mean = mean_or_nan(ratios)
        print(
            f"{model},{sum(params) / len(params):.0f},"
            f"{mean(raw_accs):.3f},{pstdev(raw_accs):.3f},"
            f"{mean(balanced_accs):.3f},{mean(macro_f1s):.3f},"
            f"{mean(best_accs):.3f},"
            f"{mean_or_nan(corner_recalls):.3f},{mean_or_nan(edge_recalls):.3f},"
            f"{mean_or_nan(center_recalls):.3f},{mean(majorities):.3f},"
            f"{mean(speeds):.2f},{gate_mean:.5f},{ratio_mean:.5f}"
        )


def print_probe_paired_summary(rows: list[ProbeResultRow], reference_model: str) -> None:
    if reference_model not in {row.model for row in rows}:
        return
    print(f"paired_balanced_vs,{reference_model}")
    print(
        "paired_model,raw_delta_mean,raw_delta_std,raw_wins,"
        "balanced_delta_mean,balanced_delta_std,balanced_wins,"
        "macro_f1_delta_mean,macro_f1_delta_std,macro_f1_wins"
    )
    run_seeds = sorted({row.run_seed for row in rows})
    for model in sorted({row.model for row in rows}):
        if model == reference_model:
            continue
        raw_deltas = []
        balanced_deltas = []
        f1_deltas = []
        for run_seed in run_seeds:
            ref = find_probe_row(rows, run_seed=run_seed, model=reference_model)
            cur = find_probe_row(rows, run_seed=run_seed, model=model)
            if ref is None or cur is None:
                continue
            raw_deltas.append(cur.eval_acc - ref.eval_acc)
            balanced_deltas.append(cur.balanced_acc - ref.balanced_acc)
            f1_deltas.append(cur.macro_f1 - ref.macro_f1)
        if not raw_deltas:
            continue
        print(
            f"{model},"
            f"{mean(raw_deltas):.3f},{pstdev_or_zero(raw_deltas):.3f},{wins(raw_deltas)},"
            f"{mean(balanced_deltas):.3f},{pstdev_or_zero(balanced_deltas):.3f},{wins(balanced_deltas)},"
            f"{mean(f1_deltas):.3f},{pstdev_or_zero(f1_deltas):.3f},{wins(f1_deltas)}"
        )


def find_probe_row(
    rows: list[ProbeResultRow],
    run_seed: int,
    model: str,
) -> ProbeResultRow | None:
    for row in rows:
        if row.run_seed == run_seed and row.model == model:
            return row
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


def pstdev_or_zero(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def wins(values: list[float]) -> str:
    return f"{sum(value > 0 for value in values)}/{len(values)}"


def group_recall(recall: Tensor, indices: list[int], num_classes: int) -> float:
    valid = [index for index in indices if index < num_classes]
    if not valid:
        return float("nan")
    return float(recall[valid].mean().item())


def json_dumps_float_list(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"


def required_text(root: ET.Element, tag: str) -> str:
    child = root.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"missing XML tag: {tag}")
    return child.text.strip()


def bbox_probe_label(sample: BBoxSample, task: str) -> int:
    center_x = ((sample.xmin + sample.xmax) * 0.5) / max(1, sample.width)
    center_y = ((sample.ymin + sample.ymax) * 0.5) / max(1, sample.height)
    center_x = min(max(center_x, 0.0), 0.999999)
    center_y = min(max(center_y, 0.0), 0.999999)

    if task == "quadrant4":
        return int(center_y >= 0.5) * 2 + int(center_x >= 0.5)
    if task == "grid9":
        return int(center_y * 3) * 3 + int(center_x * 3)
    if task == "size3":
        box_area = max(0, sample.xmax - sample.xmin) * max(0, sample.ymax - sample.ymin)
        ratio = box_area / max(1, sample.width * sample.height)
        if ratio < 0.15:
            return 0
        if ratio < 0.45:
            return 1
        return 2
    raise ValueError(f"unknown bbox probe task: {task}")


def num_probe_classes(task: str) -> int:
    if task == "quadrant4":
        return 4
    if task == "grid9":
        return 9
    if task == "size3":
        return 3
    raise ValueError(f"unknown bbox probe task: {task}")


if __name__ == "__main__":
    main()
