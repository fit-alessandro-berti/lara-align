from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from lara_ui.app_state import apply_configuration
from lara_ui.input_service import (
    checkpoint_metadata,
    content_hash,
    parse_pnml_bytes,
    parse_xes_bytes,
    save_uploaded_checkpoint,
    trusted_checkpoint_paths,
)
from lara_ui.page_utils import cached_checkpoint, prepare_page
from lara_ui.validation_service import (
    compatibility_analysis,
    log_summary,
    net_summary,
    visible_transition_groups,
    warnings_for_inputs,
)
from lara_ui.variant_service import group_trace_variants, variant_table_rows
from lara_ui.visualizations.petri_graph import petri_net_dot

prepare_page("Setup & overview", "⚙️")
st.title("Setup and data overview")
st.caption("Inputs are validated together before any alignment starts.")

root = Path(__file__).resolve().parents[1]

with st.sidebar:
    st.header("Inputs")
    log_source = st.radio("Event log source", ["Bundled example", "Upload XES"])
    if log_source == "Bundled example":
        log_name = st.selectbox(
            "Bundled log",
            ["running-example.xes", "receipt.xes", "roadtraffic100traces.xes"],
        )
        log_bytes = (root / "files" / log_name).read_bytes()
    else:
        log_upload = st.file_uploader("XES event log", type=["xes", "xml"])
        log_bytes = log_upload.getvalue() if log_upload else None
        log_name = log_upload.name if log_upload else ""

    pnml_source = st.radio("Petri-net source", ["Bundled example", "Upload PNML"])
    if pnml_source == "Bundled example":
        pnml_name = "running-example.pnml"
        pnml_bytes = (root / "files" / pnml_name).read_bytes()
    else:
        pnml_upload = st.file_uploader("PNML model", type=["pnml", "xml"])
        pnml_bytes = pnml_upload.getvalue() if pnml_upload else None
        pnml_name = pnml_upload.name if pnml_upload else ""

    st.subheader("Log settings")
    activity_key = st.text_input("Activity attribute", "concept:name")
    case_id_key = st.text_input("Case identifier attribute", "concept:name")
    max_cases_enabled = st.checkbox("Limit cases", value=False)
    max_cases = st.number_input("Maximum cases", 1, 1_000_000, 100) if max_cases_enabled else None
    ignore_empty = st.checkbox("Ignore empty traces", value=True)
    include_lifecycle = st.checkbox("Include lifecycle in activity label", value=False)

    st.subheader("Trusted checkpoint")
    checkpoint_source = st.radio("Checkpoint source", ["Trusted server checkpoint", "Upload trusted file"])
    checkpoint_path = None
    if checkpoint_source == "Trusted server checkpoint":
        checkpoints = trusted_checkpoint_paths(root / "runs")
        checkpoint_path = st.selectbox(
            "Checkpoint",
            checkpoints,
            format_func=lambda path: str(path.relative_to(root)) if path else "No checkpoint found",
        ) if checkpoints else None
        if not checkpoints:
            st.error("No .pt checkpoints were found under runs/.")
    else:
        st.warning("PyTorch checkpoint files are executable content. Upload only a file you trust.")
        trust = st.checkbox("I trust this checkpoint and its source")
        checkpoint_upload = st.file_uploader("LARA checkpoint", type=["pt", "pth"])
        if trust and checkpoint_upload:
            checkpoint_path = save_uploaded_checkpoint(
                checkpoint_upload.getvalue(), checkpoint_upload.name
            )
    device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = st.selectbox("Inference device", device_options)

loaded_log = None
loaded_net = None
input_error = None
try:
    if log_bytes:
        log_fingerprint = (
            content_hash(log_bytes), activity_key, case_id_key, max_cases,
            ignore_empty, include_lifecycle,
        )
        if st.session_state.get("log_input_fingerprint") != log_fingerprint:
            loaded_log = parse_xes_bytes(
                log_bytes,
                log_name,
                activity_key=activity_key,
                case_id_key=case_id_key,
                max_cases=max_cases,
                ignore_empty=ignore_empty,
                include_lifecycle=include_lifecycle,
            )
            st.session_state.loaded_log = loaded_log
            st.session_state.log_input_fingerprint = log_fingerprint
        else:
            loaded_log = st.session_state.loaded_log
    if pnml_bytes:
        pnml_fingerprint = content_hash(pnml_bytes)
        if st.session_state.get("pnml_input_fingerprint") != pnml_fingerprint:
            loaded_net = parse_pnml_bytes(pnml_bytes, pnml_name)
            st.session_state.loaded_net = loaded_net
            st.session_state.pnml_input_fingerprint = pnml_fingerprint
        else:
            loaded_net = st.session_state.loaded_net
except Exception as exc:
    input_error = f"{type(exc).__name__}: {exc}"
    st.error(input_error)

if checkpoint_path:
    try:
        checkpoint_path = Path(checkpoint_path)
        model, checkpoint = cached_checkpoint(
            str(checkpoint_path), device, checkpoint_path.stat().st_mtime_ns
        )
        st.session_state.loaded_checkpoint = model
        st.session_state.checkpoint_path = str(checkpoint_path)
        st.session_state.checkpoint_metadata = checkpoint_metadata(checkpoint, checkpoint_path)
    except Exception as exc:
        st.session_state.loaded_checkpoint = None
        st.error(f"Checkpoint architecture is incompatible or unreadable: {type(exc).__name__}: {exc}")

if loaded_log and loaded_net:
    variants = group_trace_variants(loaded_log)
    previous_selected = set(st.session_state.selected_variants)
    if previous_selected:
        for variant in variants:
            variant.selected = variant.variant_id in previous_selected
    st.session_state.variant_index = variants

    compatibility = compatibility_analysis(loaded_log, loaded_net, variants)
    log_stats = log_summary(loaded_log, set(visible_transition_groups(loaded_net.net)))
    model_stats = net_summary(loaded_net)

    for warning in warnings_for_inputs(loaded_log, loaded_net, variants):
        st.warning(warning)

    st.subheader("Input summary")
    columns = st.columns(6)
    for column, (label, value) in zip(
        columns,
        [
            ("Cases", log_stats["cases"]),
            ("Events", log_stats["events"]),
            ("Activities", log_stats["unique_activities"]),
            ("Variants", log_stats["trace_variants"]),
            ("Mean length", f"{log_stats['mean_trace_length']:.1f}"),
            ("Max length", log_stats["maximum_trace_length"]),
        ],
    ):
        column.metric(label, value)

    tabs = st.tabs(["Trace variants", "Compatibility", "Petri net", "Checkpoint", "Run configuration"])
    with tabs[0]:
        st.write("Identical activity sequences are computed once and retain their complete case mapping.")
        variant_df = pd.DataFrame(variant_table_rows(variants))
        edited = st.data_editor(
            variant_df,
            hide_index=True,
            width="stretch",
            disabled=["Variant", "Frequency", "Coverage", "Length", "Sequence", "Cases"],
            column_config={
                "Selected": st.column_config.CheckboxColumn(required=True),
                "Coverage": st.column_config.ProgressColumn(format="percent"),
            },
            key="variant_selector",
        )
        selection = set(edited.loc[edited["Selected"], "Variant"].tolist())
        st.session_state.selected_variants = sorted(selection)
        for variant in variants:
            variant.selected = variant.variant_id in selection
        st.caption(f"{len(selection)} of {len(variants)} variants selected.")

    with tabs[1]:
        left, right = st.columns(2)
        left.write("Activities in both log and model")
        left.dataframe(pd.DataFrame({"Activity": compatibility["shared_activities"]}), hide_index=True)
        left.write("Only in log (necessarily log moves under this mapping)")
        left.dataframe(pd.DataFrame({"Activity": compatibility["log_only_activities"]}), hide_index=True)
        right.write("Only in model")
        right.dataframe(pd.DataFrame({"Label": compatibility["model_only_labels"]}), hide_index=True)
        right.write("Duplicate visible labels")
        duplicate_rows = [
            {"Label": label, "Transition identities": ", ".join(names), "Count": len(names)}
            for label, names in compatibility["duplicate_visible_labels"].items()
        ]
        right.dataframe(pd.DataFrame(duplicate_rows), hide_index=True, width="stretch")
        metrics = st.columns(4)
        metrics[0].metric("Cases with unknowns", compatibility["cases_with_unknown_activities"])
        metrics[1].metric("Variants with unknowns", compatibility["variants_with_unknown_activities"])
        metrics[2].metric("Invisible transitions", compatibility["invisible_transition_count"])
        metrics[3].metric("Markings available", "Yes" if compatibility["initial_marking_valid"] and compatibility["final_marking_valid"] else "No")

    with tabs[2]:
        net_metrics = st.columns(6)
        for column, key, label in zip(
            net_metrics,
            ["places", "transitions", "visible_transitions", "invisible_transitions", "arcs", "duplicate_label_groups"],
            ["Places", "Transitions", "Visible", "Invisible", "Arcs", "Duplicate groups"],
        ):
            column.metric(label, model_stats[key])
        if model_stats["duplicate_label_groups"]:
            st.info("Duplicate-label transition identities are retained in every alignment and export.")
        st.graphviz_chart(
            petri_net_dot(
                loaded_net.net,
                loaded_net.initial_marking,
                loaded_net.final_marking,
            ),
            width="stretch",
        )

    with tabs[3]:
        metadata = st.session_state.checkpoint_metadata
        if metadata:
            display_metadata = {key: value for key, value in metadata.items() if key not in {"metrics", "model_config"}}
            st.json(display_metadata)
            with st.expander("Architecture configuration"):
                st.json(metadata.get("model_config", {}))
            with st.expander("Training metrics"):
                st.json(metadata.get("metrics", {}))
            st.success("The checkpoint loaded successfully with the current project architecture.")
        else:
            st.warning("Select a checkpoint in the sidebar.")

    with tabs[4]:
        left, middle, right = st.columns(3)
        with left:
            mode = st.selectbox(
                "Alignment mode",
                ["Candidate with certification", "Fast candidate", "Exact baseline"],
                help="Progressive certification shows the candidate first, then starts exact search.",
            )
            strategy = st.selectbox(
                "Execution strategy",
                ["Fast overview first", "Complete one variant at a time"],
            )
            order = st.selectbox(
                "Processing order",
                ["Most frequent variants first", "Shortest traces first", "Longest traces first", "Original log order"],
            )
        with middle:
            timeout = st.number_input("Exact timeout per variant (seconds)", 0.1, 86_400.0, 30.0)
            max_variants = st.number_input("Maximum selected variants", 1, max(len(variants), 1), min(len(variants), 100))
            max_certify = st.number_input("Maximum variants to certify", 0, max(len(variants), 1), min(len(variants), 50))
            certification_policy = st.selectbox(
                "Selective certification",
                ["All legal candidates", "Candidates with deviations", "Illegal candidates", "Cost at or above threshold", "Most frequent variants", "Longest variants", "Duplicate-label variants", "Manual selection"],
            )
            cost_threshold = st.number_input("Certification cost threshold", 0, 1_000_000, 1)
        with right:
            with st.expander("Advanced decoder", expanded=True):
                guided = st.checkbox("Guided neural decoding", value=True)
                prefix_depth = st.number_input("Maximum prefix-model search depth", 0, 1_000, 8)
                final_depth = st.number_input("Maximum final-model search depth", 0, 10_000, 32)
                retain_diagnostics = st.checkbox("Retain full selected-trace diagnostics", value=False)
            with st.expander("Cost model", expanded=False):
                sync_cost = st.number_input("Synchronous move", 0, 1_000_000, 0)
                log_cost = st.number_input("Log-only move", 0, 1_000_000, 1)
                model_cost = st.number_input("Visible model-only move", 0, 1_000_000, 1)
                silent_cost = st.number_input("Silent model-only move", 0, 1_000_000, 0)

        configuration = {
            "mode": mode,
            "strategy": strategy,
            "order": order,
            "timeout_seconds": timeout,
            "max_variants": int(max_variants),
            "max_certify": int(max_certify),
            "certification_policy": certification_policy,
            "cost_threshold": int(cost_threshold),
            "guided": guided,
            "prefix_depth": int(prefix_depth),
            "final_depth": int(final_depth),
            "retain_full_diagnostics": retain_diagnostics,
            "cost_model": {
                "sync": int(sync_cost), "log": int(log_cost),
                "model": int(model_cost), "silent_model": int(silent_cost),
            },
            "activity_key": "concept:name",
            "log_hash": content_hash(log_bytes) if log_bytes else None,
            "pnml_hash": content_hash(pnml_bytes) if pnml_bytes else None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "selected_variants": sorted(st.session_state.selected_variants),
        }
        if apply_configuration(st.session_state, configuration):
            st.warning("A computation-relevant setting changed. Previous alignment results were invalidated.")
        st.json(configuration, expanded=False)

    ready = bool(
        loaded_log
        and loaded_net
        and st.session_state.loaded_checkpoint is not None
        and st.session_state.selected_variants
        and loaded_net.initial_marking
        and loaded_net.final_marking
    )
    if ready:
        st.success("Preflight complete. The selected variants are ready for progressive alignment.")
        st.info("Continue with **Live alignment run** from the page navigation.")
    else:
        st.error("Preflight is incomplete. Resolve missing inputs, markings, checkpoint, or variant selection.")
else:
    st.info("Choose an event log and PNML model in the sidebar to begin preflight analysis.")
