from __future__ import annotations

from statistics import mean, median
from typing import Iterable

from lara_align.verify import trace_labels
from lara_ui.input_service import ParsedLog, ParsedNet
from lara_ui.ui_types import TraceVariant


def visible_transition_groups(net) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for transition in net.transitions:
        if transition.label is not None:
            groups.setdefault(str(transition.label), []).append(str(transition.name))
    return {label: sorted(names) for label, names in groups.items()}


def log_summary(parsed_log: ParsedLog, model_labels: set[str] | None = None) -> dict:
    lengths = [len(trace) for trace in parsed_log.traces]
    labels = [
        label
        for trace in parsed_log.traces
        for label in trace_labels(trace, parsed_log.activity_key)
    ]
    variants = {tuple(trace_labels(trace, parsed_log.activity_key)) for trace in parsed_log.traces}
    activities = set(labels)
    return {
        "cases": len(parsed_log.traces),
        "events": len(labels),
        "unique_activities": len(activities),
        "trace_variants": len(variants),
        "mean_trace_length": mean(lengths) if lengths else 0,
        "median_trace_length": median(lengths) if lengths else 0,
        "maximum_trace_length": max(lengths, default=0),
        "empty_traces": parsed_log.empty_trace_count,
        "activities_not_in_model": len(activities - (model_labels or set())),
    }


def net_summary(parsed_net: ParsedNet) -> dict:
    net = parsed_net.net
    groups = visible_transition_groups(net)
    duplicates = {label: names for label, names in groups.items() if len(names) > 1}
    visible = [transition for transition in net.transitions if transition.label is not None]
    invisible = [transition for transition in net.transitions if transition.label is None]
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "visible_transitions": len(visible),
        "invisible_transitions": len(invisible),
        "arcs": len(net.arcs),
        "initial_tokens": sum(parsed_net.initial_marking.values()),
        "final_tokens": sum(parsed_net.final_marking.values()),
        "duplicate_label_groups": len(duplicates),
        "largest_duplicate_label_group": max(
            (len(names) for names in duplicates.values()), default=0
        ),
        "duplicate_labels": duplicates,
        "initial_marking_present": bool(parsed_net.initial_marking),
        "final_marking_present": bool(parsed_net.final_marking),
        "source": parsed_net.source_kind,
    }


def compatibility_analysis(
    parsed_log: ParsedLog,
    parsed_net: ParsedNet,
    variants: Iterable[TraceVariant],
) -> dict:
    net_groups = visible_transition_groups(parsed_net.net)
    model_labels = set(net_groups)
    log_labels = {
        label
        for trace in parsed_log.traces
        for label in trace_labels(trace, parsed_log.activity_key)
    }
    unknown = log_labels - model_labels
    variants = list(variants)
    cases_with_unknown = sum(
        variant.frequency for variant in variants if set(variant.labels) & unknown
    )
    variants_with_unknown = sum(
        1 for variant in variants if set(variant.labels) & unknown
    )
    duplicate = {label: names for label, names in net_groups.items() if len(names) > 1}
    return {
        "shared_activities": sorted(log_labels & model_labels),
        "log_only_activities": sorted(unknown),
        "model_only_labels": sorted(model_labels - log_labels),
        "duplicate_visible_labels": duplicate,
        "invisible_transition_count": sum(
            transition.label is None for transition in parsed_net.net.transitions
        ),
        "cases_with_unknown_activities": cases_with_unknown,
        "variants_with_unknown_activities": variants_with_unknown,
        "initial_marking_valid": bool(parsed_net.initial_marking),
        "final_marking_valid": bool(parsed_net.final_marking),
    }


def warnings_for_inputs(
    parsed_log: ParsedLog,
    parsed_net: ParsedNet,
    variants: Iterable[TraceVariant],
) -> list[str]:
    compatibility = compatibility_analysis(parsed_log, parsed_net, variants)
    warnings = []
    if not compatibility["initial_marking_valid"]:
        warnings.append("The PNML has no usable initial marking.")
    if not compatibility["final_marking_valid"]:
        warnings.append("The PNML has no usable final marking.")
    if not any(t.label is not None for t in parsed_net.net.transitions):
        warnings.append("The model contains no visible transitions.")
    if compatibility["log_only_activities"]:
        warnings.append(
            "Unknown log activities will require log-only moves: "
            + ", ".join(compatibility["log_only_activities"])
        )
    if compatibility["duplicate_visible_labels"]:
        warnings.append(
            f"{len(compatibility['duplicate_visible_labels'])} visible label group(s) "
            "map to multiple transition identities."
        )
    return warnings
