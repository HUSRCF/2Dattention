"""Visualize RF-DETR deformable cross-attention sampling locations.

This is a diagnostic, not a benchmark. It aggregates the last captured
MSDeformAttn layer over queries, heads, feature levels, and sampling points,
then overlays the resulting sampling-density heatmap on selected COCO images.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as tvF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_rfdetr_coco import load_rgb_image
from scripts.train_rfdetr_coco import build_rfdetr_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("nano", "small", "medium", "large", "base"), default="small")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="mps")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument(
        "--query-overlays",
        type=int,
        default=0,
        help="Also save per-query attention overlays for the top K predictions per image.",
    )
    parser.add_argument(
        "--prefer-slice",
        choices=("all", "offcenter", "small"),
        default="offcenter",
        help="Prefer images from a diagnostic slice when selecting examples.",
    )
    parser.add_argument("--heatmap-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coco = json.loads(args.annotation_json.read_text(encoding="utf-8"))
    annotations_by_image = group_annotations(coco)
    image_records = select_images(coco, annotations_by_image, args.prefer_slice, args.max_images)

    capture = AttentionCapture()
    capture.install()
    try:
        model = build_rfdetr_model(
            args.model_size,
            pretrain_weights=args.weights,
            num_classes=args.num_classes,
            device=None if args.device == "auto" else args.device,
        )
        shape = (args.resolution, args.resolution)
        manifest_rows: list[dict[str, Any]] = []
        for image in image_records:
            image_path = args.image_root / image["file_name"]
            pil = load_rgb_image(image_path)
            capture.clear()
            detections = predict_with_query_indices(model, pil, threshold=args.threshold, shape=shape)
            if not capture.records:
                raise RuntimeError("No MSDeformAttn records were captured; RF-DETR internals may have changed.")
            last_record = capture.records[-1]
            heatmap = aggregate_sampling_heatmap(last_record, args.heatmap_size)
            image_annotations = annotations_by_image[int(image["id"])]
            overlay = render_overlay(pil, heatmap, image_annotations, detections)
            out_path = args.out_dir / f"{int(image['id']):012d}_{Path(image['file_name']).stem}_attn.jpg"
            overlay.save(out_path, quality=92)
            attention_stats = summarize_attention_alignment(heatmap, image, image_annotations, detections)
            query_rows = save_query_overlays(
                args.out_dir,
                pil,
                image,
                image_annotations,
                detections,
                last_record,
                args.heatmap_size,
                args.query_overlays,
            )
            manifest_rows.append(
                {
                    "image_id": int(image["id"]),
                    "file_name": image["file_name"],
                    "out": str(out_path),
                    "gt_boxes": len(image_annotations),
                    "predictions": int(len(getattr(detections, "xyxy", []))),
                    "captured_layers": len(capture.records),
                    "query_overlays": query_rows,
                    **attention_stats,
                }
            )
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")
        make_contact_sheet([Path(row["out"]) for row in manifest_rows], args.out_dir / "contact_sheet.jpg")
        query_paths = [Path(query["out"]) for row in manifest_rows for query in row.get("query_overlays", [])]
        if query_paths:
            make_contact_sheet(query_paths, args.out_dir / "query_contact_sheet.jpg")
        print(f"saved_attention_overlays: {args.out_dir}")
        print(f"images: {len(manifest_rows)}")
    finally:
        capture.uninstall()


def predict_with_query_indices(
    model: Any,
    image: Image.Image,
    threshold: float,
    shape: tuple[int, int],
) -> SimpleNamespace:
    """Run RF-DETR forward while preserving the query index for each top-k detection."""
    from rfdetr.detr import _ensure_model_on_device

    _ensure_model_on_device(model.model)
    model.model.model.eval()
    orig_h, orig_w = image.height, image.width
    tensor = tvF.to_tensor(image)
    if (tensor > 1).any() or (tensor < 0).any():
        raise ValueError("Expected image tensor values in [0, 1].")
    tensor = tensor.to(model.model.device)
    tensor = tvF.resize(tensor, list(shape))
    tensor = tvF.normalize(tensor, model.means, model.stds)
    batch = tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = model.model.model(batch)
    if isinstance(predictions, tuple):
        predictions = {"pred_logits": predictions[1], "pred_boxes": predictions[0]}
    logits = predictions["pred_logits"]
    boxes_cxcywh = predictions["pred_boxes"]
    prob = logits.sigmoid()
    num_select = int(getattr(model.model.postprocess, "num_select", 300))
    topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), num_select, dim=1)
    query_indices = topk_indexes // logits.shape[2]
    labels = topk_indexes % logits.shape[2]
    boxes = cxcywh_to_xyxy_tensor(boxes_cxcywh)
    boxes = torch.gather(boxes, 1, query_indices.unsqueeze(-1).repeat(1, 1, 4))
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=boxes.device, dtype=boxes.dtype)
    boxes = boxes * scale[None, None, :]
    scores = topk_values[0]
    keep = scores > threshold
    return SimpleNamespace(
        xyxy=boxes[0][keep].float().cpu().numpy(),
        confidence=scores[keep].float().cpu().numpy(),
        class_id=labels[0][keep].cpu().numpy(),
        query_index=query_indices[0][keep].cpu().numpy(),
    )


def cxcywh_to_xyxy_tensor(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        [
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        ],
        dim=-1,
    )


class AttentionCapture:
    def __init__(self) -> None:
        self.records: list[dict[str, torch.Tensor]] = []
        self._original_forward: Any = None

    def install(self) -> None:
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        self._original_forward = MSDeformAttn.forward
        capture = self

        def wrapped_forward(module_self: Any, *args: Any, **kwargs: Any) -> Any:
            query = args[0] if len(args) > 0 else kwargs["query"]
            reference_points = args[1] if len(args) > 1 else kwargs["reference_points"]
            input_spatial_shapes = args[3] if len(args) > 3 else kwargs["input_spatial_shapes"]
            with torch.no_grad():
                sampling_offsets = module_self.sampling_offsets(query).view(
                    query.shape[0],
                    query.shape[1],
                    module_self.n_heads,
                    module_self.n_levels,
                    module_self.n_points,
                    2,
                )
                attention_weights = module_self.attention_weights(query).view(
                    query.shape[0],
                    query.shape[1],
                    module_self.n_heads,
                    module_self.n_levels * module_self.n_points,
                )
                if reference_points.shape[-1] == 2:
                    offset_normalizer = torch.stack(
                        [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
                    )
                    sampling_locations = (
                        reference_points[:, :, None, :, None, :]
                        + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                    )
                elif reference_points.shape[-1] == 4:
                    sampling_locations = (
                        reference_points[:, :, None, :, None, :2]
                        + sampling_offsets / module_self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
                    )
                else:
                    sampling_locations = None
                if sampling_locations is not None:
                    capture.records.append(
                        {
                            "sampling_locations": sampling_locations.detach().float().cpu(),
                            "attention_weights": torch.softmax(attention_weights, -1).detach().float().cpu(),
                        }
                    )
            return capture._original_forward(module_self, *args, **kwargs)

        MSDeformAttn.forward = wrapped_forward

    def clear(self) -> None:
        self.records.clear()

    def uninstall(self) -> None:
        if self._original_forward is None:
            return
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        MSDeformAttn.forward = self._original_forward
        self._original_forward = None


def group_annotations(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        out[int(ann["image_id"])].append(ann)
    return out


def select_images(
    coco: dict[str, Any],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    prefer_slice: str,
    max_images: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for image in coco.get("images", []):
        anns = annotations_by_image[int(image["id"])]
        if not anns:
            continue
        score = 0.0
        if prefer_slice == "offcenter":
            score = max(center_distance(ann, image) for ann in anns)
        elif prefer_slice == "small":
            score = -min(area_ratio(ann, image) for ann in anns)
        else:
            score = len(anns)
        scored.append((score, image))
    scored.sort(key=lambda item: (-item[0], int(item[1]["id"])))
    return [image for _, image in scored[:max_images]]


def center_distance(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    x, y, w, h = [float(v) for v in annotation["bbox"]]
    cx = (x + 0.5 * w) / float(image["width"])
    cy = (y + 0.5 * h) / float(image["height"])
    return float(((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5)


def area_ratio(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    _, _, w, h = [float(v) for v in annotation["bbox"]]
    return float((w * h) / (float(image["width"]) * float(image["height"])))


def aggregate_sampling_heatmap(
    record: dict[str, torch.Tensor],
    size: int,
    query_index: int | None = None,
) -> np.ndarray:
    locations = record["sampling_locations"].numpy()
    weights = record["attention_weights"].numpy()
    # locations: B, Q, H, L, P, 2. weights: B, Q, H, L*P.
    b, q, heads, levels, points, _ = locations.shape
    weights = weights.reshape(b, q, heads, levels, points)
    if query_index is not None:
        if query_index < 0 or query_index >= q:
            raise ValueError(f"query_index={query_index} outside captured query range [0, {q})")
        locations = locations[:, query_index : query_index + 1]
        weights = weights[:, query_index : query_index + 1]
    heat = np.zeros((size, size), dtype=np.float32)
    xs = np.clip(locations[..., 0], 0.0, 0.999999)
    ys = np.clip(locations[..., 1], 0.0, 0.999999)
    xi = (xs * size).astype(np.int64)
    yi = (ys * size).astype(np.int64)
    np.add.at(heat, (yi.ravel(), xi.ravel()), weights.ravel())
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def render_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    annotations: list[dict[str, Any]],
    detections: Any,
    selected_detection_index: int | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    heat_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(heat_img.point(lambda value: int(value * 0.55)))
    out = Image.alpha_composite(base, red)
    draw = ImageDraw.Draw(out)
    for ann in annotations:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0, 255), width=3)
    boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
    scores = np.asarray(getattr(detections, "confidence", np.empty((0,))), dtype=float)
    order = np.argsort(-scores)[:5] if scores.size else []
    for idx in order:
        x1, y1, x2, y2 = boxes[idx]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 128, 255, 255), width=2)
    if selected_detection_index is not None and 0 <= selected_detection_index < len(boxes):
        x1, y1, x2, y2 = boxes[selected_detection_index]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 230, 0, 255), width=4)
    return out.convert("RGB")


def save_query_overlays(
    out_dir: Path,
    image: Image.Image,
    image_record: dict[str, Any],
    annotations: list[dict[str, Any]],
    detections: Any,
    attention_record: dict[str, torch.Tensor],
    heatmap_size: int,
    topk: int,
) -> list[dict[str, Any]]:
    if topk <= 0:
        return []
    boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
    scores = np.asarray(getattr(detections, "confidence", np.empty((0,))), dtype=float)
    labels = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=int)), dtype=int)
    query_indices = np.asarray(getattr(detections, "query_index", np.empty((0,), dtype=int)), dtype=int)
    if not scores.size:
        return []
    query_dir = out_dir / "query_overlays"
    query_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-scores)[:topk]
    rows: list[dict[str, Any]] = []
    for rank, det_idx in enumerate(order, start=1):
        query_index = int(query_indices[det_idx])
        heatmap = aggregate_sampling_heatmap(attention_record, heatmap_size, query_index=query_index)
        overlay = render_overlay(image, heatmap, annotations, detections, selected_detection_index=int(det_idx))
        out_path = query_dir / (
            f"{int(image_record['id']):012d}_{Path(image_record['file_name']).stem}"
            f"_rank{rank:02d}_q{query_index:03d}_attn.jpg"
        )
        overlay.save(out_path, quality=92)
        stats = summarize_attention_alignment(heatmap, image_record, annotations, detection_subset(detections, det_idx))
        rows.append(
            {
                "rank": rank,
                "query_index": query_index,
                "class_id": int(labels[det_idx]) if labels.size else -1,
                "score": float(scores[det_idx]),
                "box_xyxy": [float(v) for v in boxes[det_idx].tolist()],
                "out": str(out_path),
                **{f"query_{key}": value for key, value in stats.items()},
            }
        )
    return rows


def detection_subset(detections: Any, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)[index : index + 1],
        confidence=np.asarray(getattr(detections, "confidence", np.empty((0,))), dtype=float)[index : index + 1],
        class_id=np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=int)), dtype=int)[index : index + 1],
        query_index=np.asarray(getattr(detections, "query_index", np.empty((0,), dtype=int)), dtype=int)[index : index + 1],
    )


def summarize_attention_alignment(
    heatmap: np.ndarray,
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    detections: Any,
    topk: int = 5,
) -> dict[str, float | int]:
    heat = np.maximum(heatmap.astype(np.float64), 0.0)
    total_mass = float(heat.sum())
    if total_mass <= 0:
        return {
            "gt_attention_mass": 0.0,
            "top_pred_attention_mass": 0.0,
            "attention_entropy": 0.0,
            "attention_peak_x": -1.0,
            "attention_peak_y": -1.0,
        }
    gt_mask = boxes_to_heatmap_mask([xywh_to_xyxy(ann["bbox"]) for ann in annotations], image, heat.shape)
    boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4))), dtype=float)
    scores = np.asarray(getattr(detections, "confidence", np.empty((0,))), dtype=float)
    order = np.argsort(-scores)[:topk] if scores.size else []
    pred_mask = boxes_to_heatmap_mask([boxes[idx].tolist() for idx in order], image, heat.shape)
    prob = heat / total_mass
    entropy = float(-(prob[prob > 0] * np.log(prob[prob > 0])).sum() / np.log(prob.size))
    peak_y, peak_x = np.unravel_index(int(np.argmax(heat)), heat.shape)
    return {
        "gt_attention_mass": float(heat[gt_mask].sum() / total_mass),
        "top_pred_attention_mass": float(heat[pred_mask].sum() / total_mass),
        "attention_entropy": entropy,
        "attention_peak_x": float((peak_x + 0.5) / heat.shape[1]),
        "attention_peak_y": float((peak_y + 0.5) / heat.shape[0]),
    }


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def boxes_to_heatmap_mask(
    boxes_xyxy: list[list[float]],
    image: dict[str, Any],
    heatmap_shape: tuple[int, int],
) -> np.ndarray:
    height, width = heatmap_shape
    mask = np.zeros((height, width), dtype=bool)
    image_width = float(image["width"])
    image_height = float(image["height"])
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in box]
        left = int(np.floor(np.clip(x1 / image_width, 0.0, 1.0) * width))
        right = int(np.ceil(np.clip(x2 / image_width, 0.0, 1.0) * width))
        top = int(np.floor(np.clip(y1 / image_height, 0.0, 1.0) * height))
        bottom = int(np.ceil(np.clip(y2 / image_height, 0.0, 1.0) * height))
        if right <= left or bottom <= top:
            continue
        mask[top:bottom, left:right] = True
    return mask


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_width: int = 320) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumbs.append(img.resize((thumb_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS))
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    for idx, img in enumerate(thumbs):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * cell_h
        sheet.paste(img, (x, y))
    sheet.save(out_path, quality=92)


if __name__ == "__main__":
    main()
