from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from scripts.train_det_real import (
    RealBox,
    RealDetDataset,
    RealDetSample,
    box_slice_masks,
    build_label_map,
    build_splits,
    build_train_slice_sampler,
    binary_auc,
    calibration_ece,
    center_distance_score_multiplier,
    det_collate,
    detector_restore_metric,
    duplicate_predictions_per_gt,
    filter_samples,
    fixed_quality_score_multiplier,
    load_semantic_hard_negative_loss_map,
    load_real_det_samples,
    matcher_aware_quality_classification_loss,
    objectness_logits,
    objectness_ap50_for_image,
    oracle_iou_score_multiplier,
    oracle_query_mask_logits,
    pearson_corr,
    proposal_gap_closure,
    quality_score_multipliers,
    quality_head_loss_scale,
    query_mask_center_slice_metrics,
    query_quality_head_loss,
    query_ranking_diagnostics,
    ranking_gap_closure,
    sample_matches_slice,
    score_iou_calibration_loss,
    semantic_hard_negative_dataset_coverage,
    select_semantic_hard_negative_samples,
    set_quality_head_only_trainable,
    summarize_slice_ap,
    summarize_slice_ranking_diagnostics,
)
from scripts.export_det_manifest_to_coco import (
    build_coco_datasets,
    filter_rows,
    limit_rows_by_images_per_split,
    read_manifest,
    read_split_map,
)
from attention2d.detection import DetectionCriterion
from attention2d.detection import TinyAnchorRegionDETR


def test_real_det_dataset_parses_xml_and_normalizes_boxes(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    anno_root = tmp_path / "annos"
    image_root.mkdir()
    anno_root.mkdir()
    Image.new("RGB", (100, 50), "white").save(image_root / "sample.JPEG")
    (anno_root / "sample.xml").write_text(
        """
        <annotation>
          <filename>sample</filename>
          <size><width>100</width><height>50</height></size>
          <object>
            <name>class_a</name>
            <bndbox><xmin>10</xmin><ymin>5</ymin><xmax>60</xmax><ymax>25</ymax></bndbox>
          </object>
          <object>
            <name>class_b</name>
            <bndbox><xmin>70</xmin><ymin>10</ymin><xmax>90</xmax><ymax>40</ymax></bndbox>
          </object>
        </annotation>
        """,
        encoding="utf-8",
    )

    samples = load_real_det_samples(anno_root, image_root)
    label_to_id = build_label_map(samples, top_classes=0)
    filtered = filter_samples(samples, set(label_to_id), max_samples=0)
    dataset = RealDetDataset(filtered, label_to_id, image_size=32, max_objects=2)
    image, target = dataset[0]

    assert image.shape == (3, 32, 32)
    assert target["labels"].shape == (2,)
    assert target["boxes"].shape == (2, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([0.35, 0.30, 0.50, 0.40]))

    images, targets = det_collate([dataset[0]])
    assert images.shape == (1, 3, 32, 32)
    assert len(targets) == 1


def test_export_det_manifest_to_coco_can_split_by_seed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,image_path,width,height,label,xmin,ymin,xmax,ymax",
                "img_a,images/img_a.JPEG,100,80,class_b,10,20,50,60",
                "img_a,images/img_a.JPEG,100,80,class_a,60,10,90,30",
                "img_b,images/img_b.JPEG,50,50,class_b,5,5,25,25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split_path = tmp_path / "split.csv"
    split_path.write_text(
        "\n".join(
            [
                "run_seed,split,index,image_id,object_count",
                "41,train,0,img_a,2",
                "41,eval,1,img_b,1",
                "42,eval,0,img_a,2",
                "42,train,1,img_b,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = filter_rows(read_manifest(manifest_path), top_classes=0, max_images=0)
    split_map = read_split_map(split_path, run_seed=41)
    datasets = build_coco_datasets(
        rows,
        split_map=split_map,
        file_name_mode="relative",
        image_root=Path("images"),
    )

    assert sorted(datasets) == ["eval", "train"]
    assert [image["file_name"] for image in datasets["train"]["images"]] == ["img_a.JPEG"]
    assert [image["file_name"] for image in datasets["eval"]["images"]] == ["img_b.JPEG"]
    assert len(datasets["train"]["annotations"]) == 2
    assert datasets["train"]["annotations"][0]["bbox"] == [10.0, 20.0, 40.0, 40.0]
    assert datasets["train"]["annotations"][0]["area"] == 1600.0
    assert [category["name"] for category in datasets["train"]["categories"]] == ["class_a", "class_b"]


def test_export_det_manifest_to_coco_limits_each_split_independently(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,image_path,width,height,label,xmin,ymin,xmax,ymax",
                "train_a,train_a.JPEG,20,20,cls,1,1,5,5",
                "train_b,train_b.JPEG,20,20,cls,1,1,5,5",
                "eval_a,eval_a.JPEG,20,20,cls,1,1,5,5",
                "eval_b,eval_b.JPEG,20,20,cls,1,1,5,5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = read_manifest(manifest_path)
    split_map = {
        "train_a": "train",
        "train_b": "train",
        "eval_a": "eval",
        "eval_b": "eval",
    }

    limited = limit_rows_by_images_per_split(rows, max_images=1, split_map=split_map)
    datasets = build_coco_datasets(limited, split_map=split_map, file_name_mode="basename", image_root=Path("."))

    assert [image["file_name"] for image in datasets["train"]["images"]] == ["train_a.JPEG"]
    assert [image["file_name"] for image in datasets["eval"]["images"]] == ["eval_a.JPEG"]


def test_export_det_manifest_to_coco_filters_top_classes_and_images(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "image_id,image_path,width,height,label,xmin,ymin,xmax,ymax",
                "img_a,img_a.JPEG,20,20,keep,1,1,5,5",
                "img_b,img_b.JPEG,20,20,drop,1,1,5,5",
                "img_c,img_c.JPEG,20,20,keep,2,2,6,6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = filter_rows(read_manifest(manifest_path), top_classes=1, max_images=1)
    datasets = build_coco_datasets(rows, split_map=None, file_name_mode="basename", image_root=Path("."))

    assert list(datasets) == ["annotations"]
    assert len(datasets["annotations"]["images"]) == 1
    assert datasets["annotations"]["images"][0]["file_name"] == "img_a.JPEG"
    assert [category["name"] for category in datasets["annotations"]["categories"]] == ["keep"]


def test_real_det_build_splits_can_hold_out_calibration() -> None:
    samples = [
        RealDetSample(
            image_id=f"sample_{idx}",
            image_path=Path(f"sample_{idx}.JPEG"),
            width=64,
            height=64,
            boxes=(RealBox("class_a", 4, 4, 32, 32),),
        )
        for idx in range(10)
    ]
    train_set, calibration_set, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=1,
        train_frac=0.6,
        calibration_frac=0.5,
        seed=123,
    )

    assert len(train_set) == 6
    assert calibration_set is not None
    assert len(calibration_set) == 2
    assert len(eval_set) == 2
    assert {str(row["split"]) for row in split_rows} == {"train", "calibration", "eval"}


def test_real_det_load_semantic_hard_negative_loss_map_uses_class_indices(tmp_path: Path) -> None:
    loss_map_path = tmp_path / "loss_map.json"
    loss_map_path.write_text(
        """
        {
          "task": "semantic_hard_negative_loss_map",
          "entries": [
            {
              "positive_category_id": 3,
              "positive_class_index": 2,
              "hard_negatives": [
                {
                  "negative_category_id": 1,
                  "negative_class_index": 0,
                  "weight": 2.5
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    hard_negatives = load_semantic_hard_negative_loss_map(loss_map_path)

    assert hard_negatives == {2: [(0, 2.5)]}


def test_real_det_load_semantic_hard_negative_loss_map_can_remap_by_name(tmp_path: Path) -> None:
    loss_map_path = tmp_path / "loss_map.json"
    loss_map_path.write_text(
        """
        {
          "entries": [
            {
              "positive_category_name": "target",
              "positive_class_index": 99,
              "hard_negatives": [
                {
                  "negative_category_name": "wrong",
                  "negative_class_index": 98,
                  "weight": 1.5
                },
                {
                  "negative_category_name": "missing",
                  "negative_class_index": 97,
                  "weight": 3.0
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    hard_negatives = load_semantic_hard_negative_loss_map(
        loss_map_path,
        class_name_to_index={"wrong": 0, "target": 4},
    )

    assert hard_negatives == {4: [(0, 1.5)]}


def test_real_det_semantic_hard_negative_dataset_coverage_counts_ranked_targets(tmp_path: Path) -> None:
    image_path = tmp_path / "img.JPEG"
    Image.new("RGB", (64, 64), "white").save(image_path)
    samples = [
        RealDetSample(
            image_id="sample",
            image_path=image_path,
            width=64,
            height=64,
            boxes=(
                RealBox("target", 0, 0, 32, 32),
                RealBox("other", 0, 0, 8, 8),
                RealBox("target", 0, 0, 4, 4),
            ),
        )
    ]
    dataset = RealDetDataset(samples, {"other": 0, "target": 1}, image_size=32, max_objects=2)
    subset = torch.utils.data.Subset(dataset, [0])

    images, boxes = semantic_hard_negative_dataset_coverage(subset, {1: [(0, 1.0)]})

    assert images == 1
    assert boxes == 1


def test_real_det_select_semantic_hard_negative_samples_can_prioritize_targets(tmp_path: Path) -> None:
    image_path = tmp_path / "img.JPEG"
    Image.new("RGB", (64, 64), "white").save(image_path)
    samples = [
        RealDetSample(
            image_id="other",
            image_path=image_path,
            width=64,
            height=64,
            boxes=(RealBox("other", 0, 0, 20, 20),),
        ),
        RealDetSample(
            image_id="target",
            image_path=image_path,
            width=64,
            height=64,
            boxes=(RealBox("target", 0, 0, 20, 20),),
        ),
    ]

    selected = select_semantic_hard_negative_samples(
        samples,
        {"other": 0, "target": 1},
        hard_negatives={1: [(0, 1.0)]},
        max_objects=1,
        mode="prioritize",
        max_samples=1,
    )

    assert [sample.image_id for sample in selected] == ["target"]


def test_real_det_select_semantic_hard_negative_samples_can_keep_only_targets(tmp_path: Path) -> None:
    image_path = tmp_path / "img.JPEG"
    Image.new("RGB", (64, 64), "white").save(image_path)
    samples = [
        RealDetSample(
            image_id="target",
            image_path=image_path,
            width=64,
            height=64,
            boxes=(RealBox("target", 0, 0, 20, 20),),
        ),
        RealDetSample(
            image_id="other",
            image_path=image_path,
            width=64,
            height=64,
            boxes=(RealBox("other", 0, 0, 20, 20),),
        ),
    ]

    selected = select_semantic_hard_negative_samples(
        samples,
        {"other": 0, "target": 1},
        hard_negatives={1: [(0, 1.0)]},
        max_objects=1,
        mode="only",
        max_samples=0,
    )

    assert [sample.image_id for sample in selected] == ["target"]


def test_real_det_build_splits_can_use_train_calibration() -> None:
    samples = [
        RealDetSample(
            image_id=f"sample_{idx}",
            image_path=Path(f"sample_{idx}.JPEG"),
            width=64,
            height=64,
            boxes=(RealBox("class_a", 4, 4, 32, 32),),
        )
        for idx in range(10)
    ]
    train_set, calibration_set, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=1,
        train_frac=0.6,
        calibration_frac=0.5,
        calibration_source="train",
        seed=123,
    )

    assert len(train_set) == 3
    assert calibration_set is not None
    assert len(calibration_set) == 3
    assert len(eval_set) == 4
    split_counts = {str(row["split"]): 0 for row in split_rows}
    for row in split_rows:
        split_counts[str(row["split"])] += 1
    assert split_counts == {"train": 3, "calibration": 3, "eval": 4}


def test_real_det_slice_masks_and_ap_summary() -> None:
    boxes = torch.tensor(
        [
            [0.50, 0.50, 0.10, 0.10],
            [0.80, 0.80, 0.30, 0.30],
        ]
    )
    masks = box_slice_masks(boxes)

    assert masks["small"].tolist() == [True, False]
    assert masks["large"].tolist() == [False, True]
    assert masks["center"].tolist() == [True, False]
    assert masks["offcenter"].tolist() == [False, True]

    summary = summarize_slice_ap({"small": [torch.tensor(0.25), torch.tensor(0.75)], "large": []})
    assert summary["small"] == 0.5
    assert summary["large"] == 0.0


def test_real_det_build_splits_can_filter_eval_by_slice() -> None:
    samples = [
        RealDetSample(
            image_id="center_large",
            image_path=Path("center_large.JPEG"),
            width=100,
            height=100,
            boxes=(RealBox("class_a", 25, 25, 75, 75),),
        ),
        RealDetSample(
            image_id="offcenter_small",
            image_path=Path("offcenter_small.JPEG"),
            width=100,
            height=100,
            boxes=(RealBox("class_a", 80, 80, 90, 90),),
        ),
        RealDetSample(
            image_id="mixed_center_offcenter",
            image_path=Path("mixed_center_offcenter.JPEG"),
            width=100,
            height=100,
            boxes=(
                RealBox("class_a", 25, 25, 75, 75),
                RealBox("class_a", 80, 80, 90, 90),
            ),
        ),
    ]

    assert sample_matches_slice(samples[0], max_objects=1, slice_name="center")
    assert not sample_matches_slice(samples[0], max_objects=1, slice_name="offcenter")
    assert sample_matches_slice(samples[1], max_objects=1, slice_name="offcenter")
    assert sample_matches_slice(samples[1], max_objects=1, slice_name="offcenter_only")
    assert sample_matches_slice(samples[1], max_objects=1, slice_name="small")
    assert sample_matches_slice(samples[2], max_objects=2, slice_name="offcenter")
    assert not sample_matches_slice(samples[2], max_objects=2, slice_name="offcenter_only")

    _, _, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=1,
        train_frac=0.5,
        calibration_frac=0.0,
        eval_slice_filter="offcenter",
        seed=0,
    )

    assert len(eval_set) == 1
    assert [row["image_id"] for row in split_rows if row["split"] == "eval"] == ["offcenter_small"]

    _, _, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=2,
        train_frac=0.5,
        calibration_frac=0.0,
        eval_slice_filter="offcenter_only",
        seed=0,
    )

    assert len(eval_set) == 1
    assert [row["image_id"] for row in split_rows if row["split"] == "eval"] == ["offcenter_small"]


def test_real_det_build_splits_can_filter_calibration_by_slice() -> None:
    samples = []
    for idx in range(10):
        if idx % 2 == 0:
            box = RealBox("class_a", 25, 25, 75, 75)
            image_id = f"center_{idx}"
        else:
            box = RealBox("class_a", 80, 80, 90, 90)
            image_id = f"offcenter_{idx}"
        samples.append(
            RealDetSample(
                image_id=image_id,
                image_path=Path(f"{image_id}.JPEG"),
                width=100,
                height=100,
                boxes=(box,),
            )
        )

    train_set, calibration_set, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=1,
        train_frac=0.8,
        calibration_frac=0.5,
        calibration_source="train",
        calibration_slice_filter="offcenter",
        seed=0,
    )

    assert len(train_set) > 0
    assert calibration_set is not None
    assert len(calibration_set) > 0
    assert len(eval_set) > 0
    calibration_ids = [str(row["image_id"]) for row in split_rows if row["split"] == "calibration"]
    assert calibration_ids
    assert all(image_id.startswith("offcenter_") for image_id in calibration_ids)


def test_real_det_build_splits_can_filter_train_by_slice() -> None:
    samples = []
    for idx in range(10):
        if idx % 2 == 0:
            box = RealBox("class_a", 25, 25, 75, 75)
            image_id = f"center_{idx}"
        else:
            box = RealBox("class_a", 80, 80, 90, 90)
            image_id = f"offcenter_{idx}"
        samples.append(
            RealDetSample(
                image_id=image_id,
                image_path=Path(f"{image_id}.JPEG"),
                width=100,
                height=100,
                boxes=(box,),
            )
        )

    train_set, calibration_set, eval_set, split_rows = build_splits(
        samples=samples,
        label_to_id={"class_a": 0},
        image_size=64,
        max_objects=1,
        train_frac=0.8,
        calibration_frac=0.0,
        train_slice_filter="offcenter",
        seed=0,
    )

    assert len(train_set) > 0
    assert calibration_set is None
    assert len(eval_set) > 0
    train_ids = [str(row["image_id"]) for row in split_rows if row["split"] == "train"]
    assert train_ids
    assert all(image_id.startswith("offcenter_") for image_id in train_ids)


def test_real_det_build_train_slice_sampler_oversamples_slice() -> None:
    image_root = Path("/unused")
    samples = []
    for idx in range(6):
        if idx < 2:
            box = RealBox("class_a", 80, 80, 90, 90)
            image_id = f"offcenter_{idx}"
        else:
            box = RealBox("class_a", 25, 25, 75, 75)
            image_id = f"center_{idx}"
        samples.append(
            RealDetSample(
                image_id=image_id,
                image_path=image_root / f"{image_id}.JPEG",
                width=100,
                height=100,
                boxes=(box,),
            )
        )
    dataset = RealDetDataset(samples, {"class_a": 0}, image_size=64, max_objects=1)
    train_set = torch.utils.data.Subset(dataset, list(range(len(samples))))

    sampler = build_train_slice_sampler(
        train_set,
        slice_name="offcenter",
        factor=4.0,
        max_objects=1,
        seed=123,
    )

    assert sampler is not None
    assert sampler.weights.tolist() == [4.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    assert sampler.num_samples == len(samples)


def test_real_det_build_train_slice_sampler_oversamples_semantic_targets() -> None:
    image_root = Path("/unused")
    samples = [
        RealDetSample(
            image_id="target",
            image_path=image_root / "target.JPEG",
            width=100,
            height=100,
            boxes=(RealBox("target", 25, 25, 75, 75),),
        ),
        RealDetSample(
            image_id="other",
            image_path=image_root / "other.JPEG",
            width=100,
            height=100,
            boxes=(RealBox("other", 25, 25, 75, 75),),
        ),
    ]
    dataset = RealDetDataset(samples, {"other": 0, "target": 1}, image_size=64, max_objects=1)
    train_set = torch.utils.data.Subset(dataset, [0, 1])

    sampler = build_train_slice_sampler(
        train_set,
        slice_name="none",
        factor=1.0,
        max_objects=1,
        seed=123,
        semantic_hard_negatives={1: [(0, 1.0)]},
        semantic_factor=5.0,
    )

    assert sampler is not None
    assert sampler.weights.tolist() == [5.0, 1.0]


def test_real_det_ranking_diagnostics_capture_high_score_false_positive() -> None:
    pred_logits = torch.tensor(
        [
            [4.0, -2.0, -3.0],
            [3.0, -2.0, -3.0],
            [-2.0, 4.0, -3.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.40, 0.40],
            [0.52, 0.50, 0.40, 0.40],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    target_labels = torch.tensor([0])

    diagnostics = query_ranking_diagnostics(pred_logits, pred_boxes, target_boxes, target_labels)

    assert torch.allclose(diagnostics["matched_assignment_class_correct"], torch.tensor([1.0]))
    assert torch.allclose(diagnostics["tp50_class_correct"], torch.tensor([1.0]))
    assert float(diagnostics["objectness_auc"]) < 0.5
    assert "objectness_ece50" in diagnostics
    assert "combined_ece50" in diagnostics
    assert float(diagnostics["topk_fp_rate"]) == 0.0
    assert float(diagnostics["combined_topk_fp_rate"]) == 0.0
    assert float(diagnostics["topk_center_distance"]) < 0.05
    assert float(diagnostics["combined_topk_center_distance"]) < 0.05
    assert float(diagnostics["duplicate_per_gt"]) == 1.0
    assert diagnostics["matched_query_counts"].shape == (3,)


def test_real_det_slice_diagnostics_split_non_slice_gt_matches() -> None:
    pred_logits = torch.tensor(
        [
            [5.0, -2.0, -3.0],
            [4.0, -2.0, -3.0],
            [3.0, -2.0, -3.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.80, 0.50, 0.20, 0.20],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    all_target_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.80, 0.50, 0.20, 0.20],
        ]
    )
    offcenter_target_boxes = all_target_boxes[1:]
    target_labels = torch.tensor([0])

    diagnostics = query_ranking_diagnostics(
        pred_logits,
        pred_boxes,
        offcenter_target_boxes,
        target_labels,
        combined_quality_scores=torch.tensor([0.01, 1.0, 0.01]),
        all_target_boxes=all_target_boxes,
    )

    assert float(diagnostics["topk_fp_rate"]) == 1.0
    assert float(diagnostics["topk_slice_match_rate"]) == 0.0
    assert float(diagnostics["topk_non_slice_match_rate"]) == 1.0
    assert float(diagnostics["topk_no_gt_match_rate"]) == 0.0
    assert float(diagnostics["combined_topk_fp_rate"]) == 0.0
    assert float(diagnostics["combined_topk_slice_match_rate"]) == 1.0
    assert float(diagnostics["combined_topk_non_slice_match_rate"]) == 0.0
    assert float(diagnostics["combined_topk_no_gt_match_rate"]) == 0.0


def test_real_det_combined_diagnostics_use_fixed_quality_multiplier() -> None:
    pred_logits = torch.tensor(
        [
            [4.0, -2.0, -3.0],
            [3.0, -2.0, -3.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.40, 0.40],
            [0.10, 0.10, 0.10, 0.10],
        ]
    )
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    target_labels = torch.tensor([0])

    q1_diagnostics = query_ranking_diagnostics(
        pred_logits,
        pred_boxes,
        target_boxes,
        target_labels,
        quality_logits=torch.tensor([0.0, 0.0]),
    )
    suppressed_diagnostics = query_ranking_diagnostics(
        pred_logits,
        pred_boxes,
        target_boxes,
        target_labels,
        quality_logits=torch.tensor([0.0, 0.0]),
        combined_quality_scores=torch.tensor([0.01, 0.99]),
    )

    assert float(q1_diagnostics["combined_auc"]) > 0.99
    assert float(suppressed_diagnostics["combined_auc"]) < 0.01
    assert float(q1_diagnostics["combined_topk_fp_rate"]) == 0.0
    assert float(suppressed_diagnostics["combined_topk_fp_rate"]) == 1.0
    assert float(q1_diagnostics["combined_topk_center_distance"]) < 0.05
    assert float(suppressed_diagnostics["combined_topk_center_distance"]) > 0.5


def test_real_det_center_distance_multiplier_downranks_center_boxes() -> None:
    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [1.00, 1.00, 0.20, 0.20],
        ]
    )
    scores = center_distance_score_multiplier(pred_boxes)

    assert float(scores[0]) == 0.0
    assert abs(float(scores[1]) - 1.0) < 1e-6


def test_real_det_scalar_diagnostics() -> None:
    assert torch.allclose(
        pearson_corr(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        binary_auc(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([True, True, False])),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        calibration_ece(torch.tensor([0.9, 0.1]), torch.tensor([True, False]), bins=2),
        torch.tensor(0.1),
    )
    assert torch.allclose(
        calibration_ece(torch.tensor([0.1, 0.9]), torch.tensor([True, False]), bins=2),
        torch.tensor(0.9),
    )
    iou_matrix = torch.tensor([[0.6, 0.1], [0.7, 0.2], [0.0, 0.8]])
    assert torch.allclose(duplicate_predictions_per_gt(iou_matrix, threshold=0.5), torch.tensor(0.5))


def test_real_det_slice_ranking_diagnostics_summary_handles_empty_vectors() -> None:
    summary = summarize_slice_ranking_diagnostics(
        {
            "center": {
                "matched_assignment_class_acc": [torch.tensor([1.0, 0.0])],
                "tp50_class_acc": [torch.zeros(0)],
            },
            "offcenter": {
                "matched_assignment_class_acc": [],
                "tp50_class_acc": [torch.tensor([1.0])],
            },
        },
        {
            "center": {
                "score_iou_corr": [torch.tensor(0.25), torch.tensor(0.75)],
                "topk_fp_rate": [torch.tensor(0.0)],
                "combined_topk_fp_rate": [torch.tensor(0.5)],
                "topk_center_distance": [torch.tensor(0.1)],
                "combined_topk_center_distance": [torch.tensor(0.2)],
            },
            "offcenter": {
                "score_iou_corr": [],
                "topk_fp_rate": [],
                "combined_topk_fp_rate": [],
                "topk_center_distance": [],
                "combined_topk_center_distance": [],
            },
        },
    )

    assert summary["center_matched_assignment_class_acc"] == 0.5
    assert summary["center_tp50_class_acc"] == 0.0
    assert summary["offcenter_matched_assignment_class_acc"] == 0.0
    assert summary["offcenter_tp50_class_acc"] == 1.0
    assert summary["center_score_iou_corr"] == 0.5
    assert summary["center_topk_fp_rate"] == 0.0
    assert summary["center_combined_topk_fp_rate"] == 0.5
    assert abs(summary["center_topk_center_distance"] - 0.1) < 1e-6
    assert abs(summary["center_combined_topk_center_distance"] - 0.2) < 1e-6
    assert summary["offcenter_score_iou_corr"] == 0.0
    assert summary["offcenter_combined_topk_fp_rate"] == 0.0
    assert summary["offcenter_topk_center_distance"] == 0.0


def test_real_det_ranking_diagnostics_handle_empty_targets() -> None:
    diagnostics = query_ranking_diagnostics(
        pred_logits=torch.zeros(2, 3),
        pred_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        target_boxes=torch.zeros(0, 4),
        target_labels=torch.zeros(0, dtype=torch.long),
    )

    assert diagnostics["matched_assignment_class_correct"].numel() == 0
    assert diagnostics["tp50_class_correct"].numel() == 0
    assert float(diagnostics["topk_fp_rate"]) == 0.0
    assert float(diagnostics["combined_topk_fp_rate"]) == 0.0
    assert float(diagnostics["topk_center_distance"]) == 0.0
    assert float(diagnostics["duplicate_per_gt"]) == 0.0


def test_real_det_oracle_query_mask_logits_from_targets() -> None:
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.50, 0.50]]),
        },
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        },
    ]
    logits = oracle_query_mask_logits(
        targets=targets,
        feature_height=8,
        feature_width=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert logits.shape == (2, 8, 8)
    assert float(logits.max()) == 8.0
    assert float(logits.min()) == -8.0
    assert int((logits[0] > 0).sum().item()) > int((logits[1] > 0).sum().item())


def test_query_mask_center_slice_metrics_tracks_center_and_offcenter() -> None:
    criterion = DetectionCriterion(num_classes=1)
    pred_boxes = torch.tensor([[[0.625, 0.625, 0.20, 0.20], [0.125, 0.125, 0.20, 0.20]]])
    pred_logits = torch.tensor([[[5.0, -5.0], [5.0, -5.0]]])
    mask_logits = torch.full((1, 2, 4, 4), -8.0)
    mask_logits[0, 0, 2, 2] = 8.0
    mask_logits[0, 1, 0, 0] = 8.0
    outputs = {
        "pred_boxes": pred_boxes,
        "pred_logits": pred_logits,
        "query_mask_logits_per_query": mask_logits,
    }
    targets = [
        {
            "labels": torch.tensor([0, 0]),
            "boxes": torch.tensor([[0.625, 0.625, 0.20, 0.20], [0.125, 0.125, 0.20, 0.20]]),
        }
    ]

    metrics = query_mask_center_slice_metrics(outputs, targets, criterion)

    assert metrics["query_mask_center_l2"] < 1e-3
    assert metrics["center_query_mask_center_l2"] < 1e-3
    assert metrics["offcenter_query_mask_center_l2"] < 1e-3
    assert metrics["center_query_mask_center_pck025"] == 1.0
    assert metrics["offcenter_query_mask_center_pck025"] == 1.0


def test_real_det_score_iou_calibration_loss_backpropagates_to_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -1.0], [-1.0, 2.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    loss = score_iou_calibration_loss(
        {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        targets,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_logits.grad is not None
    assert float(pred_logits.grad.abs().sum()) > 0.0
    assert pred_boxes.grad is None


def test_real_det_objectness_logits_are_foreground_vs_background() -> None:
    logits = torch.tensor(
        [
            [3.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    values = objectness_logits(logits)

    assert values.shape == (2,)
    assert float(values[0]) > 0.0
    assert float(values[1]) < 0.0


def test_real_det_matcher_aware_quality_loss_uses_class_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -2.0, -1.0], [-2.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    criterion = DetectionCriterion(num_classes=2)
    loss = matcher_aware_quality_classification_loss(
        {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        targets,
        criterion,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_logits.grad is not None
    assert float(pred_logits.grad.abs().sum()) > 0.0
    assert pred_boxes.grad is None


def test_real_det_query_quality_head_loss_uses_quality_logits_only() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -2.0, -1.0], [-2.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.50, 0.50, 0.40, 0.40], [0.10, 0.10, 0.10, 0.10]]],
        requires_grad=True,
    )
    pred_quality_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40]]),
        }
    ]
    criterion = DetectionCriterion(num_classes=2)
    loss = query_quality_head_loss(
        {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_quality_logits": pred_quality_logits,
        },
        targets,
        criterion,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred_quality_logits.grad is not None
    assert float(pred_quality_logits.grad.abs().sum()) > 0.0
    assert pred_logits.grad is None
    assert pred_boxes.grad is None


def test_real_det_query_quality_head_loss_can_upweight_target_slice() -> None:
    pred_logits = torch.tensor(
        [[[2.0, -2.0, -1.0], [-2.0, 2.0, -1.0]]],
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [[[0.10, 0.10, 0.20, 0.20], [0.80, 0.80, 0.20, 0.20]]],
        requires_grad=True,
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.10, 0.10, 0.20, 0.20]]),
        }
    ]
    criterion = DetectionCriterion(num_classes=2)
    plain_logits = torch.zeros(1, 2, requires_grad=True)
    weighted_logits = torch.zeros(1, 2, requires_grad=True)

    plain_loss = query_quality_head_loss(
        {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_quality_logits": plain_logits,
        },
        targets,
        criterion,
    )
    weighted_loss = query_quality_head_loss(
        {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_quality_logits": weighted_logits,
        },
        targets,
        criterion,
        slice_name="offcenter",
        slice_weight=3.0,
    )

    assert weighted_loss > plain_loss


def test_real_det_quality_score_can_rescue_ap_ranking() -> None:
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    pred_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.40, 0.40],
        ]
    )
    pred_logits = torch.tensor(
        [
            [4.0, -2.0],
            [2.0, -2.0],
        ]
    )
    quality_logits = torch.tensor([-4.0, 4.0])

    base_ap = objectness_ap50_for_image(pred_logits, pred_boxes, target_boxes)
    quality_scores = quality_score_multipliers(quality_logits)
    quality_ap = objectness_ap50_for_image(
        pred_logits,
        pred_boxes,
        target_boxes,
        score_multiplier=quality_scores[1.0],
    )

    assert float(base_ap) < 0.6
    assert float(quality_ap) > 0.99


def test_real_det_quality_score_multipliers_include_calibration_sweep() -> None:
    quality_logits = torch.tensor([0.0])
    scores = quality_score_multipliers(quality_logits)

    assert set(scores) == {0.25, 0.5, 1.0, 2.0, 4.0}
    assert torch.allclose(scores[1.0], torch.tensor([0.5]))
    assert float(scores[0.25][0]) > float(scores[1.0][0]) > float(scores[4.0][0])


def test_real_det_fixed_quality_score_multiplier_supports_temperature() -> None:
    quality_logits = torch.tensor([2.0])
    standard = fixed_quality_score_multiplier(quality_logits, alpha=2.0, temperature=1.0)
    softened = fixed_quality_score_multiplier(quality_logits, alpha=2.0, temperature=2.0)
    identity = fixed_quality_score_multiplier(quality_logits, alpha=0.0, temperature=1.0)

    assert standard is not None
    assert softened is not None
    assert identity is not None
    assert float(standard[0]) > float(softened[0])
    assert torch.allclose(identity, torch.ones_like(identity))


def test_real_det_ranking_gap_closure_tracks_oracle_headroom() -> None:
    assert abs(ranking_gap_closure(base_score=0.2, quality_score=0.5, oracle_score=0.8) - 0.5) < 1e-6
    assert ranking_gap_closure(base_score=0.8, quality_score=0.7, oracle_score=0.8) == 0.0
    assert ranking_gap_closure(base_score=0.5, quality_score=0.4, oracle_score=0.8) < 0.0
    assert ranking_gap_closure(base_score=0.5, quality_score=0.9, oracle_score=0.8) > 1.0


def test_real_det_proposal_gap_closure_tracks_predicted_oracle_gap() -> None:
    assert proposal_gap_closure(base_pred=0.2, candidate=0.2, oracle=0.6) == 0.0
    assert abs(proposal_gap_closure(base_pred=0.2, candidate=0.4, oracle=0.6) - 0.5) < 1e-6
    assert proposal_gap_closure(base_pred=0.2, candidate=0.7, oracle=0.6) > 1.0
    assert proposal_gap_closure(base_pred=0.5, candidate=0.4, oracle=0.5) == 0.0
    assert proposal_gap_closure(base_pred=0.5, candidate=0.6, oracle=0.4) == 0.0


def test_real_det_oracle_iou_score_rescues_ap_ranking() -> None:
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    pred_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.40, 0.40],
        ]
    )
    pred_logits = torch.tensor(
        [
            [4.0, -2.0],
            [2.0, -2.0],
        ]
    )

    base_ap = objectness_ap50_for_image(pred_logits, pred_boxes, target_boxes)
    oracle_scores = oracle_iou_score_multiplier(pred_boxes, target_boxes)
    oracle_ap = objectness_ap50_for_image(
        pred_logits,
        pred_boxes,
        target_boxes,
        score_multiplier=oracle_scores,
    )

    assert torch.allclose(oracle_scores, torch.tensor([0.0, 1.0]))
    assert float(base_ap) < 0.6
    assert float(oracle_ap) > 0.99


def test_real_det_objectness_ap_supports_higher_iou_threshold() -> None:
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    pred_boxes = torch.tensor([[0.52, 0.50, 0.40, 0.40]])
    pred_logits = torch.tensor([[4.0, -2.0]])

    ap50 = objectness_ap50_for_image(
        pred_logits,
        pred_boxes,
        target_boxes,
        iou_threshold=0.5,
    )
    ap95 = objectness_ap50_for_image(
        pred_logits,
        pred_boxes,
        target_boxes,
        iou_threshold=0.95,
    )

    assert float(ap50) > 0.99
    assert float(ap95) == 0.0


def test_real_det_query_ranking_diagnostics_include_combined_score_corr() -> None:
    pred_logits = torch.tensor(
        [
            [4.0, -2.0],
            [2.0, -2.0],
            [3.0, -2.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.50, 0.50, 0.40, 0.40],
            [0.20, 0.20, 0.10, 0.10],
        ]
    )
    target_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
    target_labels = torch.tensor([0])
    quality_logits = torch.tensor([-4.0, 4.0, -4.0])

    diagnostics = query_ranking_diagnostics(
        pred_logits,
        pred_boxes,
        target_boxes,
        target_labels,
        quality_logits=quality_logits,
    )

    assert float(diagnostics["combined_iou_corr"]) > float(diagnostics["score_iou_corr"])
    assert float(diagnostics["combined_auc"]) >= float(diagnostics["objectness_auc"])


def test_real_det_quality_head_loss_scale_supports_late_start_and_warmup() -> None:
    assert quality_head_loss_scale(step=50, start_step=100, warmup_steps=0) == 0.0
    assert quality_head_loss_scale(step=100, start_step=100, warmup_steps=0) == 1.0
    assert quality_head_loss_scale(step=100, start_step=100, warmup_steps=50) == 0.02
    assert quality_head_loss_scale(step=124, start_step=100, warmup_steps=50) == 0.5
    assert quality_head_loss_scale(step=200, start_step=100, warmup_steps=50) == 1.0


def test_real_det_detector_restore_metric_selects_requested_score() -> None:
    metrics = {
        "iou": 0.4,
        "ap50": 0.3,
        "ap50_class": 0.2,
        "ap75": 0.1,
        "center_ap50": 0.6,
        "offcenter_ap50": 0.7,
    }

    assert detector_restore_metric(metrics, "iou") == 0.4
    assert detector_restore_metric(metrics, "ap50") == 0.3
    assert detector_restore_metric(metrics, "ap50_class") == 0.2
    assert detector_restore_metric(metrics, "ap75") == 0.1
    assert detector_restore_metric(metrics, "center_ap50") == 0.6
    assert detector_restore_metric(metrics, "offcenter_ap50") == 0.7


def test_real_det_set_quality_head_only_trainable_freezes_detector() -> None:
    model = TinyAnchorRegionDETR(embed_dim=16, num_classes=2, num_queries=4)
    set_quality_head_only_trainable(model)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("head.quality_head.") for name in trainable)
