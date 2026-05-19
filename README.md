# 2Dattention

Tiny torch prototype for **2D spatial prefill memory + lattice Attention Residual reads**.

The goal is not to train a competitive model yet. This repository is a compact scaffold for validating the information flow:

1. Patch an image into a 2D feature lattice.
2. Build a full-image memory pool with local 2D prefill updates.
3. Read from `memory depth x 2D offsets` with AttnRes-style learned routing.
4. Confirm shapes, routing normalization, and gradient flow.

## Structure

```text
src/attention2d/      Core torch modules and tiny model
scripts/              Runnable smoke demos
docs/                 Research notes and TODO roadmap
tests/                Shape and backward-pass tests
```

## Environment

Use the existing conda environment:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/run_shape_demo.py
```

The `AIAA` environment has torch 2.8.0 available.

On Apple Silicon, the demo automatically prefers PyTorch `mps` when available and falls back to CPU otherwise. PyTorch does not normally expose the M-series Neural Engine/NPU as a standard training backend; `mps` uses the Apple GPU through Metal.

## Expected Demo Output

The demo prints:

- input and logits shape
- memory map shapes
- routing tensor shapes
- max error from routing weights summing to 1
- dummy loss and backward-pass status

## Synthetic Toy Task

The first non-random task is generated online, with no dataset download:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/run_toy_task.py --steps 30
```

By default, each image contains a red marker and a green marker on an 8 by 8 patch grid. The label is `1` if the green marker is one cell to the right of the red marker, and `0` if it is one cell below. This is intentionally simple, but it tests whether the model can learn oriented 2D offset relations.

To compare against simple baselines:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/compare_toy_models.py --steps 40
```

For a more stable multi-seed result:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/compare_toy_models.py --task aligned_pair --steps 100 --seeds 3
```

The comparison script writes a CSV to `results/toy_compare.csv` by default. The `results/` folder is ignored because it is generated output.

This compares:

- `conv_only`: compact 2D convolutional baseline
- `seq_transformer`: flattened-token Transformer baseline
- `tiny_vit`: CLS-token ViT-style baseline with learned position embedding
- `xattnres_style`: cross-stage/history attention residual without 2D offset reads
- `prefill_lattice_attnres`: the proposed memory-read prototype
- `anchor_prefill_attnres`: proposed prototype plus row/column/global anchor memory
- `graph_prefill_attnres`: proposed prototype plus semantic top-k graph memory

A harder relation task is also available:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/run_toy_task.py --task aligned_pair --steps 100
```

For `aligned_pair`, the label is `1` if the markers share the same row or column, otherwise `0`.

The hardest current synthetic task adds clutter:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/compare_toy_models.py --task distractor_aligned_pair --steps 100 --seeds 3
```

This should be treated as a stress test, not evidence that any method is better.

See `docs/review.md` for the current evidence boundary and why these toy tasks do not justify broad claims.

## Local Real Images

For real pictures, provide a local folder in torchvision `ImageFolder` format:

```text
your_data/
  class_a/
    image1.jpg
  class_b/
    image2.jpg
```

Then run:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/run_imagefolder_smoke.py --data-root /path/to/your_data --model prefill_lattice_attnres
```

This script does not download datasets and should only be used as a smoke trial unless the dataset split and budget are controlled.

To compare multiple models, including ViT, on the same fixed split:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/compare_imagefolder_models.py \
  --data-root data/ILSVRC2013_DET_val_supervised/single_label_imagefolder \
  --steps 100 \
  --seeds 3
```

This writes `results/imagefolder_compare.csv` and prints per-model mean/std accuracy.

If you only have unlabeled images, such as an extracted ImageNet detection validation tar, run a forward-only smoke check:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/run_unlabeled_image_smoke.py --image-root /path/to/extracted/images --model prefill_lattice_attnres
```

For `ILSVRC2013_DET_val.tar`, the image tar alone is useful for real-image forward/speed checks. Supervised classification or detection needs the corresponding labels/annotations/devkit.

## ILSVRC2013 DET Supervised Helpers

After downloading `ILSVRC2013_DET_bbox_val.tgz`, prepare detection and classification helper files:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/prepare_ilsvrc_det_val_supervised.py
```

This creates:

- `data/ILSVRC2013_DET_val_supervised/det_val_manifest.csv`: one row per object box
- `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder/`: symlinked ImageFolder subset for classification demos
- `data/ILSVRC2013_DET_val_supervised/single_label_classes.csv`: selected class counts

The ImageFolder subset uses only images whose annotation contains a single class label, because DET images can contain multiple object categories.

To hand the same detection data to an external detector pipeline, export COCO-style annotations:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/export_det_manifest_to_coco.py \
  --manifest data/ILSVRC2013_DET_val_supervised/det_val_manifest.csv \
  --out-dir data/ILSVRC2013_DET_val_supervised/coco
```

If you want the exact train/eval split from a `train_det_real.py` run, pass its split CSV:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/export_det_manifest_to_coco.py \
  --manifest data/ILSVRC2013_DET_val_supervised/det_val_manifest.csv \
  --split-csv results/det_real_no_object_w05_anchor_refbox_dab_offcenter_only_1000img_500step_3seed_split.csv \
  --split-run-seed 41 \
  --out-dir data/ILSVRC2013_DET_val_supervised/coco_seed41
```

You can run a tiny external-detector smoke test on the exported JSON with torchvision:

```bash
/opt/anaconda3/bin/conda run -n AIAA python scripts/train_torchvision_coco_detector.py \
  --train-json data/ILSVRC2013_DET_val_supervised/coco_seed41/train.json \
  --eval-json data/ILSVRC2013_DET_val_supervised/coco_seed41/eval.json \
  --image-root data/ILSVRC2013_DET_val \
  --device mps \
  --steps 1 \
  --batch-size 1 \
  --image-size 64
```

This runner is a handoff baseline, not the project's main tiny detector. Use it to validate real detector data plumbing before attaching RF-DETR-style training or teacher outputs.
By default it uses random detector weights. If COCO weights are cached or downloads are allowed, add `--weights coco`; the script loads the COCO-pretrained detector and replaces the box predictor for the exported dataset's category count.
If weights have already been downloaded manually, pass them with `--weights-file /path/to/checkpoint.pth`.
Use `--steps 0` for an eval-only plumbing check.
For transfer smokes, `--trainable-parts box_predictor` trains only the replaced Fast R-CNN predictor, while `--trainable-parts roi_heads` freezes the backbone and RPN.
For faster evaluation during smoke tests, reduce `--detections-per-img` or raise `--score-threshold`.

## Tests

```bash
/opt/anaconda3/bin/conda run -n AIAA python -m pytest tests
```

## Design Notes

This v1 includes lattice, anchor, and semantic graph memory prototypes. 2D-SSM/Mamba-style prefill remains documented in `docs/TODO.md` as a later branch.
