from __future__ import annotations

import torch

from scripts.export_torchvision_rpn_predictions import proposal_records


def test_proposal_records_exports_original_coco_boxes() -> None:
    records = proposal_records(
        image_id=3,
        boxes=torch.tensor([[6.4, 6.4, 32.0, 32.0]]),
        scores=torch.tensor([0.75]),
        orig_size=torch.tensor([50.0, 100.0]),
        resized_size=torch.tensor([64.0, 64.0]),
        category_id=1,
    )

    assert records == [
        {
            "image_id": 3,
            "category_id": 1,
            "bbox": [10.0, 5.0, 40.0, 20.0],
            "score": 0.75,
        }
    ]
