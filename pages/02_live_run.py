from __future__ import annotations

from collections import Counter
from time import perf_counter

import pandas as pd
import streamlit as st

from lara_align.certifier import CertifyingAlignmentSystem
from lara_align.decode import GreedyCandidateDecoder
from lara_align.types import CostModel
from lara_ui.alignment_runner import (
    WorkbenchRunner,
    certification_subset,
    result_rows,
)
from lara_ui.export_service import case_csv, full_json, variant_csv
from lara_ui.page_utils import (
    append_feed,
    prepare_page,
    pretty_status,
    render_live_summary,
    require_setup,
)
from lara_ui.validation_service import (
    log_summary,
    net_summary,
    visible_transition_groups,
)
from lara_ui.variant_service import order_variants

prepare_page("Live alignment run", "▶️")
st.title("Live alignment run")
st.caption("Every candidate is rendered and verified before exact certification begins.")

if not require_setup():
    st.stop()

configuration = st.session_state.run_configuration
if not configuration:
    st.warning("Save a run configuration on the Setup page first.")
    st.stop()

loaded_log = st.session_state.loaded_log
loaded_net = st.session_state.loaded_net
variants = [
    variant
    for variant in st.session_state.variant_index
    if variant.variant_id in set(st.session_state.selected_variants)
]
variants = order_variants(variants, configuration.get("order", "Most frequent variants first"))
variants = variants[: configuration.get("max_variants", len(variants))]

top = st.columns([2, 2, 2, 1, 1])
top[0].metric("Mode", configuration["mode"])
top[1].metric("Strategy", configuration["strategy"])
top[2].metric("Order", configuration["order"])
top[3].metric("Variants", len(variants))
top[4].metric("Cases", sum(variant.frequency for variant in variants))

manual_ids: set[str] = set()
if configuration.get("certification_policy") == "Manual selection":
    manual_ids = set(
        st.multiselect(
            "Variants to certify",
            [variant.variant_id for variant in variants],
            default=[variant.variant_id for variant in variants],
        )
    )

controls = st.columns([1, 1, 1, 4])
start_clicked = controls[0].button(
    "Start new run",
    type="primary",
    disabled=st.session_state.run_status == "running",
)
if controls[1].button("Request cancellation", disabled=st.session_state.run_status != "running"):
    st.session_state.cancel_requested = True
    for pending_result in st.session_state.variant_results.values():
        if pending_result.state in {"queued", "candidate_running", "certifying"}:
            pending_result.state = "cancelled"
    st.session_state.run_status = "cancelled"
    append_feed("Run cancelled between variant stages.")
if controls[2].button("Clear results", disabled=st.session_state.run_status == "running"):
    st.session_state.variant_results = {}
    st.session_state.execution_feed = []
    st.session_state.replay_snapshots = {}
    st.session_state.run_status = "not_started"
    st.rerun()
controls[3].caption(
    "Cancellation is checked after every candidate and exact stage. Exact work already inside PM4Py uses its configured per-variant timeout."
)

metrics_placeholder = st.empty()
progress_placeholder = st.empty()
latest_placeholder = st.empty()
table_placeholder = st.empty()
feed_placeholder = st.empty()


def current_results():
    return [
        st.session_state.variant_results[variant.variant_id]
        for variant in variants
        if variant.variant_id in st.session_state.variant_results
    ]


def refresh():
    render_live_summary(
        current_results(),
        metrics_placeholder,
        progress_placeholder,
        table_placeholder,
        feed_placeholder,
    )


def cancelled() -> bool:
    if not st.session_state.cancel_requested:
        return False
    for result in current_results():
        if result.state in {"queued", "candidate_running", "certifying"}:
            result.state = "cancelled"
    st.session_state.run_status = "cancelled"
    append_feed("Run cancelled between variant stages.")
    refresh()
    return True


if start_clicked:
    costs = CostModel(**configuration["cost_model"])
    decoder = GreedyCandidateDecoder(
        max_prefix_model_depth=configuration["prefix_depth"],
        max_final_model_depth=configuration["final_depth"],
    )
    system = CertifyingAlignmentSystem(
        model=st.session_state.loaded_checkpoint,
        decoder=decoder,
        cost_model=costs,
        device=next(st.session_state.loaded_checkpoint.parameters()).device,
        use_guidance=configuration["guided"],
    )
    runner = WorkbenchRunner(
        system,
        loaded_net.net,
        loaded_net.initial_marking,
        loaded_net.final_marking,
        timeout_seconds=configuration["timeout_seconds"],
    )
    st.session_state.variant_results = runner.queued(variants)
    st.session_state.execution_feed = []
    st.session_state.cancel_requested = False
    st.session_state.run_status = "running"
    st.session_state.run_started_at = perf_counter()
    append_feed(
        f"Run started · {configuration['mode']} · {len(variants)} trace variants queued."
    )
    refresh()

    if configuration["mode"] == "Exact baseline":
        for variant in variants:
            if cancelled():
                break
            result = st.session_state.variant_results[variant.variant_id]
            runner.exact_only(result)
            exact = result.exact_result
            append_feed(
                f"{variant.variant_id} {pretty_status(result.status).lower()} · "
                f"exact cost {exact.exact_cost if exact else 'unavailable'} · "
                f"{exact.exact_seconds * 1000:.1f} ms" if exact else f"{variant.variant_id} failed"
            )
            refresh()
    elif configuration["strategy"] == "Complete one variant at a time":
        certified_count = 0
        for variant in variants:
            if cancelled():
                break
            result = st.session_state.variant_results[variant.variant_id]
            runner.propose(result)
            candidate = result.candidate
            append_feed(
                f"{variant.variant_id} candidate ready · "
                f"{'legal' if candidate and candidate.legal else 'illegal'} · "
                f"cost {candidate.cost if candidate else 'unavailable'} · "
                f"{candidate.total_seconds * 1000:.1f} ms" if candidate else f"{variant.variant_id} candidate failed"
            )
            with latest_placeholder.container():
                st.success(
                    f"{variant.variant_id} candidate is visible before certification · "
                    f"legal={candidate.legal if candidate else False} · cost={candidate.cost if candidate else '—'}"
                )
            refresh()
            if cancelled() or configuration["mode"] == "Fast candidate":
                continue
            eligible = certification_subset(
                [result],
                configuration["certification_policy"],
                manual_ids=manual_ids,
                cost_threshold=configuration["cost_threshold"],
                duplicate_labels=set(net_summary(loaded_net)["duplicate_labels"]),
            )
            if eligible and certified_count < configuration["max_certify"]:
                runner.certify(result)
                certified_count += 1
                exact = result.exact_result
                append_feed(
                    f"{variant.variant_id} {pretty_status(result.status).lower()} · "
                    f"candidate {candidate.cost if candidate else '—'}, "
                    f"exact {exact.exact_cost if exact else '—'}"
                )
                refresh()
    else:
        # Pass 1: every selected candidate becomes available before certification.
        for variant in variants:
            if cancelled():
                break
            result = st.session_state.variant_results[variant.variant_id]
            runner.propose(result)
            candidate = result.candidate
            append_feed(
                f"{variant.variant_id} candidate ready · "
                f"{'legal' if candidate and candidate.legal else 'illegal'} · "
                f"cost {candidate.cost if candidate else 'unavailable'} · "
                f"{candidate.total_seconds * 1000:.1f} ms" if candidate else f"{variant.variant_id} candidate failed"
            )
            with latest_placeholder.container():
                st.success(
                    f"Latest candidate: {variant.variant_id} · "
                    f"legal={candidate.legal if candidate else False} · "
                    f"cost={candidate.cost if candidate else '—'}"
                )
            refresh()

        # Pass 2: enrich existing rows; the candidate objects remain untouched.
        if not cancelled() and configuration["mode"] == "Candidate with certification":
            selected_for_exact = certification_subset(
                current_results(),
                configuration["certification_policy"],
                limit=configuration["max_certify"],
                manual_ids=manual_ids,
                cost_threshold=configuration["cost_threshold"],
                duplicate_labels=set(net_summary(loaded_net)["duplicate_labels"]),
            )
            append_feed(f"Candidate pass complete · {len(selected_for_exact)} variants selected for exact certification.")
            for result in selected_for_exact:
                if cancelled():
                    break
                runner.certify(result)
                exact = result.exact_result
                append_feed(
                    f"{result.variant.variant_id} {pretty_status(result.status).lower()} · "
                    f"candidate {result.candidate.cost if result.candidate else '—'}, "
                    f"exact {exact.exact_cost if exact else '—'}"
                )
                refresh()

    if st.session_state.run_status == "running":
        st.session_state.run_status = "completed"
        elapsed = perf_counter() - st.session_state.run_started_at
        append_feed(f"Run complete in {elapsed:.2f} seconds.")
        refresh()

if st.session_state.variant_results:
    refresh()
    results = current_results()
    st.divider()
    st.subheader("Explore and export results")

    filter_columns = st.columns(4)
    statuses = sorted({result.status for result in results})
    chosen_statuses = filter_columns[0].multiselect(
        "Status", statuses, format_func=pretty_status
    )
    deviations_only = filter_columns[1].checkbox("Deviating candidates only")
    unknown_only = filter_columns[2].checkbox("Contains unknown activity")
    activity_filter = filter_columns[3].selectbox(
        "Contains activity", ["All"] + sorted({label for variant in variants for label in variant.labels})
    )
    model_labels = set(visible_transition_groups(loaded_net.net))
    filtered = []
    for result in results:
        if chosen_statuses and result.status not in chosen_statuses:
            continue
        if deviations_only and not (result.candidate and (result.candidate.cost or 0) > 0):
            continue
        if unknown_only and not (set(result.variant.labels) - model_labels):
            continue
        if activity_filter != "All" and activity_filter not in result.variant.labels:
            continue
        filtered.append(result)

    filtered_frame = pd.DataFrame(result_rows(filtered))
    if not filtered_frame.empty:
        filtered_frame["Status"] = filtered_frame["Status"].map(pretty_status)
    st.dataframe(filtered_frame, hide_index=True, width="stretch")

    if filtered:
        charts = st.tabs(["Status", "Costs", "Runtime", "Move composition", "Deviation ranking"])
        with charts[0]:
            status_counts = Counter(result.status for result in filtered)
            st.bar_chart(pd.DataFrame({"Variants": status_counts}).T.T)
            weighted = Counter()
            for result in filtered:
                weighted[result.status] += result.variant.frequency
            st.caption("Case-frequency weighted")
            st.bar_chart(pd.DataFrame({"Cases": weighted}).T.T)
        with charts[1]:
            cost_rows = [row for row in result_rows(filtered) if row["Candidate cost"] is not None]
            cost_frame = pd.DataFrame(cost_rows)
            if not cost_frame.empty:
                st.bar_chart(cost_frame.set_index("Variant")[["Candidate cost", "Exact cost"]])
                paired = cost_frame.dropna(subset=["Candidate cost", "Exact cost"])
                if not paired.empty:
                    st.scatter_chart(paired, x="Exact cost", y="Candidate cost", size="Cases", color="Status")
                    st.caption("The equality diagonal is candidate cost = exact cost; points above it are suboptimal candidates.")
        with charts[2]:
            runtime = pd.DataFrame(result_rows(filtered))
            if not runtime.empty:
                st.scatter_chart(runtime, x="Length", y="Candidate time (s)", size="Cases", color="Status")
                paired = runtime.dropna(subset=["Candidate time (s)", "Exact time (s)"])
                if not paired.empty:
                    st.scatter_chart(paired, x="Candidate time (s)", y="Exact time (s)", size="Cases", color="Status")
        with charts[3]:
            moves = pd.DataFrame(result_rows(filtered)).set_index("Variant")
            st.bar_chart(moves[["Sync moves", "Log moves", "Model moves", "Silent moves"]])
        with charts[4]:
            ranking = pd.DataFrame(result_rows(filtered))
            if not ranking.empty:
                ranking["Case-weighted impact"] = ranking["Cases"] * ranking["Candidate cost"].fillna(0)
                st.dataframe(
                    ranking.sort_values(["Case-weighted impact", "Candidate cost"], ascending=False),
                    hide_index=True,
                    width="stretch",
                )

    export_columns = st.columns(4)
    export_columns[0].download_button(
        "Variant CSV", variant_csv(results), "lara_variant_results.csv", "text/csv"
    )
    export_columns[1].download_button(
        "Case CSV", case_csv(results), "lara_case_results.csv", "text/csv"
    )
    log_stats = log_summary(loaded_log, set(visible_transition_groups(loaded_net.net)))
    model_stats = net_summary(loaded_net)
    export_columns[2].download_button(
        "Full JSON",
        full_json(
            results,
            configuration,
            st.session_state.checkpoint_metadata,
            log_stats,
            model_stats,
            configuration["cost_model"],
        ),
        "lara_full_results.json",
        "application/json",
    )
    export_columns[3].download_button(
        "Execution log",
        "\n".join(st.session_state.execution_feed),
        "lara_execution.log",
        "text/plain",
    )

    inspectable = [result.variant.variant_id for result in results if result.candidate_alignment or result.exact_alignment]
    if inspectable:
        selected = st.selectbox("Select result for detailed inspection", inspectable)
        st.session_state.selected_result_id = selected
        st.info(f"Open **Trace inspector** from the page navigation to inspect {selected}.")
else:
    st.info("No results yet. Start a run to see candidates arrive progressively.")
