from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

SKIP = ">>"


class MoveKind(str, Enum):
    SYNCHRONOUS = "sync"
    LOG = "log"
    MODEL = "model"


@dataclass(frozen=True)
class CostModel:
    """Move-cost configuration used by both LARA and pm4py.

    Costs are intentionally small integer values by default. pm4py accepts the
    same scale through its cost-function parameters, so exact and neural
    candidates can be compared without post-hoc normalization.
    """

    sync: int = 0
    log: int = 1
    model: int = 1
    silent_model: int = 0
    trace_costs: Sequence[int] | None = None
    transition_model_costs: Mapping[str, int] = field(default_factory=dict)
    transition_sync_costs: Mapping[str, int] = field(default_factory=dict)

    def log_cost(self, trace_index: int) -> int:
        if self.trace_costs is None:
            return self.log
        return int(self.trace_costs[trace_index])

    def model_cost(self, transition_name: str, transition_label: str | None) -> int:
        if transition_name in self.transition_model_costs:
            return int(self.transition_model_costs[transition_name])
        return self.silent_model if transition_label is None else self.model

    def sync_cost(self, transition_name: str) -> int:
        return int(self.transition_sync_costs.get(transition_name, self.sync))


@dataclass(frozen=True)
class AlignmentMove:
    """One alignment move with explicit transition identity.

    `log_label=None` represents no log consumption. `transition_name=None`
    represents no model transition. The transition label is carried separately
    because duplicate labels must still decode to concrete transitions.
    """

    log_label: str | None
    transition_name: str | None
    transition_label: str | None = None

    @property
    def kind(self) -> MoveKind:
        if self.log_label is not None and self.transition_name is not None:
            return MoveKind.SYNCHRONOUS
        if self.log_label is not None:
            return MoveKind.LOG
        return MoveKind.MODEL

    @property
    def is_silent_model_move(self) -> bool:
        return self.kind == MoveKind.MODEL and self.transition_label is None

    def as_pm4py_tuple(self) -> tuple[str, str | None]:
        left = self.log_label if self.log_label is not None else SKIP
        right = self.transition_label if self.transition_name is not None else SKIP
        return left, right


@dataclass
class Alignment:
    moves: list[AlignmentMove]
    cost: int | None = None
    source: str = "neural"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.moves)

    def __len__(self) -> int:
        return len(self.moves)

    def to_pm4py_label_alignment(self) -> list[tuple[str, str | None]]:
        return [move.as_pm4py_tuple() for move in self.moves]


@dataclass(frozen=True)
class VerificationResult:
    legal: bool
    cost: int
    consumed_events: int
    final_marking_reached: bool
    failure_index: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LowerBound:
    value: int
    source: str
    exact: bool = False


@dataclass
class LARAResult:
    alignment: Alignment | None
    legal: bool
    cost: int | None
    lower_bound: int | None
    upper_bound: int | None
    certified_optimal: bool
    mode: str
    verifier: VerificationResult | None = None
    exact_alignment: Alignment | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
