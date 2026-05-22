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
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

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
            detections = model.predict(pil, threshold=args.threshold, shape=shape)
            if not capture.records:
                raise RuntimeError("No MSDeformAttn records were captured; RF-DETR internals may have changed.")
            heatmap = aggregate_sampling_heatmap(capture.records[-1], args.heatmap_size)
            overlay = render_overlay(pil, heatmap, annotations_by_image[int(image["id"])], detections)
            out_path = args.out_dir / f"{int(image['id']):012d}_{Path(image['file_name']).stem}_attn.jpg"
            overlay.save(out_path, quality=92)
            manifest_rows.append(
                {
                    "image_id": int(image["id"]),
                    "file_name": image["file_name"],
                    "out": str(out_path),
                    "gt_boxes": len(annotations_by_image[int(image["id"])]),
                    "predictions": int(len(getattr(detections, "xyxy", []))),
                    "captured_layers": len(capture.records),
                }
            )
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")
        make_contact_sheet([Path(row["out"]) for row in manifest_rows], args.out_dir / "contact_sheet.jpg")
        print(f"saved_attention_overlays: {args.out_dir}")
        print(f"images: {len(manifest_rows)}")
    finally:
        capture.uninstall()


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


def aggregate_sampling_heatmap(record: dict[str, torch.Tensor], size: int) -> np.ndarray:
    locations = record["sampling_locations"].numpy()
    weights = record["attention_weights"].numpy()
    # locations: B, Q, H, L, P, 2. weights: B, Q, H, L*P.
    b, q, heads, levels, points, _ = locations.shape
    weights = weights.reshape(b, q, heads, levels, points)
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
    return out.convert("RGB")


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
