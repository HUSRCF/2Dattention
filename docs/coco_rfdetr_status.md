# COCO and RF-DETR Status

Updated 2026-09-08.

## COCO validation subset

Downloaded from the official COCO 2017 endpoints into `data/coco/`:

- `val2017.zip` (815,585,330 bytes), extracted to `data/coco/val2017/`
- `annotations_trainval2017.zip` (252,907,541 bytes), extracted to `data/coco/annotations/`

Integrity checks passed with `unzip -tq`. The validation annotations contain
5,000 images, 36,781 instances, and 80 categories. `train2017` has not been
downloaded yet; the official RF-DETR setup lists it as approximately 18 GB,
compared with approximately 1 GB for `val2017` and 241 MB for annotations.

The temporary `data/coco/rfdetr_val_smoke/` layout reuses val images for all
three required directories so the dataset validator can be tested. It is a
plumbing smoke fixture, not a train/validation split and must not be used for
accuracy claims.

## RF-DETR package

The AIAA conda environment now has `rfdetr==1.10.1`, with the package's current
dependencies (`transformers==5.16.1`, `supervision==0.30.2`, and
`pyDeprecate==0.9.0`). The repository's availability check reports:

```text
RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase
```

The current package is version 1.x. `RF-DETR-2XL` is a model size name, not a
software 2.0 release. Do not describe it as RF-DETR v2 without a separate
official release announcement.

## Next gate

Before downloading `train2017`, run a real pretrained RF-DETR forward/eval
smoke on `val2017`, then finish the matched grouped-expert controls in the tiny
prototype. Only after those pass should COCO train fine-tuning begin.
