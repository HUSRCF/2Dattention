"""Export torchvision RPN proposals as COCO-format class-agnostic predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection.rpn import concat_box_prediction_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_torchvision_proposal_recall import set_rpn_top_n  # noqa: E402
from scripts.train_torchvision_coco_detector import (  # noqa: E402
    CocoDetectionLite,
    build_model,
    collate_detection,
    load_checkpoint,
    max_category_id,
    scale_xyxy_to_original,
    select_device,
    xyxy_to_xywh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/ILSVRC2013_DET_val"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-eval-images", type=int, default=0)
    parser.add_argument("--max-proposals", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--weights", choices=("none", "coco"), default="none")
    parser.add_argument("--weights-file", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category id assigned to all proposals; use class-agnostic eval for localization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset: Dataset[Any] = CocoDetectionLite(args.eval_json, args.image_root, image_size=args.image_size)
    if args.max_eval_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_eval_images, len(dataset)))))
    num_classes = max_category_id(dataset) + 1
    model = build_model(
        num_classes=num_classes,
        image_size=args.image_size,
        weights=args.weights,
        weights_file=args.weights_file,
    ).to(device)
    if args.resume_checkpoint is not None:
        load_checkpoint(args.resume_checkpoint, model, optimizer=None)
    model.eval()
    set_rpn_top_n(model, args.max_proposals)
    records = export_rpn_predictions(
        model,
        dataset,
        device=device,
        category_id=args.category_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records), encoding="utf-8")
    print(f"saved_rpn_predictions: {args.out}")
    print(f"prediction_count: {len(records)}")


@torch.no_grad()
def export_rpn_predictions(
    model: torch.nn.Module,
    dataset: Dataset[Any],
    device: torch.device,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for images, targets in DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_detection):
        image = images[0].to(device)
        target = targets[0]
        boxes, scores = rpn_proposals_with_scores(model, image)
        records.extend(
            proposal_records(
                image_id=int(target["image_id"].item()),
                boxes=boxes,
                scores=scores,
                orig_size=target["orig_size"],
                resized_size=target["resized_size"],
                category_id=category_id,
            )
        )
    return records


def proposal_records(
    image_id: int,
    boxes: Tensor,
    scores: Tensor,
    orig_size: Tensor,
    resized_size: Tensor,
    category_id: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for box, score in zip(boxes.cpu(), scores.cpu(), strict=False):
        original_box = scale_xyxy_to_original(box, orig_size=orig_size, resized_size=resized_size)
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": xyxy_to_xywh(original_box),
                "score": float(score.item()),
            }
        )
    return records


def rpn_proposals_with_scores(model: torch.nn.Module, image: Tensor) -> tuple[Tensor, Tensor]:
    transformed, _ = model.transform([image], None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, Tensor):
        features = OrderedDict([("0", features)])
    feature_values = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(feature_values)
    anchors = model.rpn.anchor_generator(transformed, feature_values)
    num_images = len(anchors)
    num_anchors_per_level = [shape[0] * shape[1] * shape[2] for shape in (item[0].shape for item in objectness)]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = model.rpn.filter_proposals(
        proposals,
        objectness,
        transformed.image_sizes,
        num_anchors_per_level,
    )
    return boxes[0].detach().cpu(), scores[0].detach().cpu()


if __name__ == "__main__":
    main()
