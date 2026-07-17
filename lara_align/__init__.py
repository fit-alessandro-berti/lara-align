"""Learned Alignment with Replay Assurance (LARA)."""

from lara_align.certifier import CertifyingAlignmentSystem, LARAMode
from lara_align.data import AlignmentSample
from lara_align.decode import GreedyCandidateDecoder
from lara_align.families import (
    BehaviorFamily,
    BehaviorFamilyConfig,
    BehaviorSpec,
    ModelVariant,
    ObservedTrace,
    TraceCorruptor,
    TraceEdit,
    generate_behavior_family,
)
from lara_align.model import LARANeuralModel
from lara_align.types import (
    Alignment,
    AlignmentMove,
    CandidateResult,
    CertificationResult,
    CostModel,
    LARAResult,
)
from lara_align.verify import verify_alignment

__all__ = [
    "Alignment",
    "AlignmentMove",
    "AlignmentSample",
    "BehaviorFamily",
    "BehaviorFamilyConfig",
    "BehaviorSpec",
    "CertifyingAlignmentSystem",
    "CandidateResult",
    "CertificationResult",
    "CostModel",
    "GreedyCandidateDecoder",
    "LARAMode",
    "LARANeuralModel",
    "LARAResult",
    "ModelVariant",
    "ObservedTrace",
    "TraceCorruptor",
    "TraceEdit",
    "generate_behavior_family",
    "verify_alignment",
]
