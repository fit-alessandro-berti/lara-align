from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from lara_align.verify import trace_labels
from lara_ui.input_service import ParsedLog
from lara_ui.ui_types import TraceVariant


def group_trace_variants(parsed_log: ParsedLog) -> list[TraceVariant]:
    groups: OrderedDict[tuple[str, ...], dict] = OrderedDict()
    for index, (trace, case_id) in enumerate(
        zip(parsed_log.traces, parsed_log.case_ids)
    ):
        labels = tuple(trace_labels(trace, parsed_log.activity_key))
        group = groups.setdefault(labels, {"case_ids": [], "first_index": index})
        group["case_ids"].append(case_id)

    total_cases = max(len(parsed_log.traces), 1)
    variants: list[TraceVariant] = []
    for variant_number, (labels, group) in enumerate(groups.items(), start=1):
        case_ids = list(group["case_ids"])
        variants.append(
            TraceVariant(
                variant_id=f"V{variant_number:03d}",
                labels=list(labels),
                case_ids=case_ids,
                first_index=group["first_index"],
                frequency=len(case_ids),
                coverage=len(case_ids) / total_cases,
            )
        )
    return variants


def order_variants(
    variants: Iterable[TraceVariant], order: str
) -> list[TraceVariant]:
    items = list(variants)
    if order == "Most frequent variants first":
        return sorted(items, key=lambda item: (-item.frequency, item.first_index))
    if order == "Shortest traces first":
        return sorted(items, key=lambda item: (item.length, item.first_index))
    if order == "Longest traces first":
        return sorted(items, key=lambda item: (-item.length, item.first_index))
    return sorted(items, key=lambda item: item.first_index)


def variant_table_rows(variants: Iterable[TraceVariant]) -> list[dict]:
    return [
        {
            "Selected": variant.selected,
            "Variant": variant.variant_id,
            "Frequency": variant.frequency,
            "Coverage": variant.coverage,
            "Length": variant.length,
            "Sequence": " → ".join(variant.labels),
            "Cases": ", ".join(variant.case_ids[:5]),
        }
        for variant in variants
    ]
