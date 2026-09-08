"""Small 2D attention-memory prototype modules."""

from .baselines import (
    ConvOnlyClassifier,
    MultiCLSViTClassifier,
    SequenceTransformerClassifier,
    TinyViTClassifier,
)
from .modules import (
    AxisAnchorMemoryRead,
    Coordinate2DEncoding,
    GroupedLatticeMemoryRead,
    LatticeMemoryRead,
    PatchEmbed2D,
    SemanticGraphMemoryRead,
    SpatialPrefill2D,
    cross_group_decorrelation_loss,
    variance_floor_loss,
)
from .model import (
    TinyAnchorOnlyAttnResClassifier,
    TinyAnchorPrefillLatticeAttnRes,
    TinyDelayedXAttnResClassifier,
    TinyFPNLiteClassifier,
    TinyGraphPrefillLatticeAttnRes,
    TinyPrefillLocalMixClassifier,
    TinyPrefillLatticeAttnRes,
    TinyRegionPoolMixerClassifier,
    TinyStageRefreshRegionSlotsClassifier,
    TinyXAttnResClassifier,
)
from .device import get_best_device
from .toy import (
    sample_aligned_pair_batch,
    sample_distractor_aligned_pair_batch,
    sample_oriented_pair_batch,
)

__all__ = [
    "ConvOnlyClassifier",
    "AxisAnchorMemoryRead",
    "get_best_device",
    "Coordinate2DEncoding",
    "GroupedLatticeMemoryRead",
    "LatticeMemoryRead",
    "MultiCLSViTClassifier",
    "PatchEmbed2D",
    "sample_aligned_pair_batch",
    "sample_distractor_aligned_pair_batch",
    "sample_oriented_pair_batch",
    "SemanticGraphMemoryRead",
    "SequenceTransformerClassifier",
    "SpatialPrefill2D",
    "cross_group_decorrelation_loss",
    "variance_floor_loss",
    "TinyAnchorOnlyAttnResClassifier",
    "TinyAnchorPrefillLatticeAttnRes",
    "TinyDelayedXAttnResClassifier",
    "TinyFPNLiteClassifier",
    "TinyGraphPrefillLatticeAttnRes",
    "TinyPrefillLocalMixClassifier",
    "TinyPrefillLatticeAttnRes",
    "TinyRegionPoolMixerClassifier",
    "TinyStageRefreshRegionSlotsClassifier",
    "TinyViTClassifier",
    "TinyXAttnResClassifier",
]
