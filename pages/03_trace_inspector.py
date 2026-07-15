from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pm4py.objects.petri_net.obj import Marking

from lara_align.types import CostModel, MoveKind
from lara_ui.diagnostics_service import log_score_rows, synchronous_score_rows
from lara_ui.export_service import move_csv, move_text
from lara_ui.page_utils import prepare_page, pretty_status, require_setup
from lara_ui.replay_service import (
    alignment_differences,
    alignment_move_rows,
    replay_alignment,
)
from lara_ui.visualizations.alignment_ribbon import alignment_ribbon_html
from lara_ui.visualizations.petri_graph import petri_net_dot

prepare_page("Trace inspector", "🔎")
st.title("Trace inspector")
st.caption("The observed trace is the shared horizontal reference for candidate and exact alignments.")

if not require_setup():
    st.stop()

available = [
    result
    for result in st.session_state.variant_results.values()
    if result.candidate_alignment is not None or result.exact_alignment is not None
]
if not available:
    st.info("Run at least one candidate or exact alignment before opening the inspector.")
    st.stop()

ids = [result.variant.variant_id for result in available]
default_id = st.session_state.selected_result_id if st.session_state.selected_result_id in ids else ids[0]
selected_id = st.selectbox("Trace variant", ids, index=ids.index(default_id))
st.session_state.selected_result_id = selected_id
result = next(item for item in available if item.variant.variant_id == selected_id)
candidate = result.candidate_alignment
exact = result.exact_alignment
candidate_result = result.candidate
exact_result = result.exact_result
loaded_net = st.session_state.loaded_net
configuration = st.session_state.run_configuration
cost_model = CostModel(**configuration.get("cost_model", {}))

header = st.columns(7)
header[0].metric("Status", pretty_status(result.status))
header[1].metric("Cases", result.variant.frequency)
header[2].metric("Coverage", f"{result.variant.coverage:.1%}")
header[3].metric("Length", result.variant.length)
header[4].metric("Candidate cost", candidate_result.cost if candidate_result else "—")
header[5].metric("Exact cost", exact_result.exact_cost if exact_result else "—")
header[6].metric("Final source", result.final_alignment_source.replace("_", " ").title())
st.caption("Observed trace · " + " → ".join(result.variant.labels))

tabs = st.tabs(
    [
        "Alignment comparison",
        "Marking replay",
        "Verification",
        "Neural diagnostics",
        "Exact diagnostics",
    ]
)

with tabs[0]:
    available_modes = []
    if candidate is not None:
        available_modes.append("Candidate only")
    if exact is not None:
        available_modes.append("Exact only")
    if candidate is not None and exact is not None:
        available_modes.extend(["Candidate and exact stacked", "Difference-focused"])
    mode = st.radio(
        "Display mode",
        available_modes,
        index=available_modes.index("Candidate and exact stacked") if "Candidate and exact stacked" in available_modes else 0,
        horizontal=True,
    )
    differences = alignment_differences(candidate, exact, result.variant.length) if candidate and exact else []
    if differences:
        nav = st.columns([1, 1, 3])
        difference_choice = nav[0].selectbox(
            "Difference",
            differences,
            format_func=lambda position: "Completion" if position == result.variant.length else f"Event {position + 1}",
        )
        nav[1].metric("Different anchors", len(differences))
        nav[2].info(
            "A difference can be a transition-identity choice, a log/model move, a silent prefix, or a completion path."
        )
    else:
        difference_choice = None
        st.success("No trace-anchored differences are present between the available alignments.")
    st.html(
        alignment_ribbon_html(
            result.variant.labels,
            candidate=candidate,
            exact=exact,
            mode=mode,
            difference_positions=set(differences),
        )
    )
    st.caption(
        "✓ green = synchronous · ! orange/red = log-only · M blue = visible model-only · τ purple = silent model-only. "
        "Every card shows the transition identity and visible label separately."
    )

    move_tabs = []
    if candidate is not None:
        move_tabs.append(("Candidate moves", candidate))
    if exact is not None:
        move_tabs.append(("Exact moves", exact))
    rendered_tabs = st.tabs([name for name, _ in move_tabs])
    for tab, (name, alignment) in zip(rendered_tabs, move_tabs):
        with tab:
            rows = alignment_move_rows(
                alignment,
                loaded_net.net,
                loaded_net.initial_marking,
                loaded_net.final_marking,
                result.variant.labels,
                cost_model,
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            downloads = st.columns(3)
            stem = f"{selected_id}_{'candidate' if alignment is candidate else 'exact'}_moves"
            downloads[0].download_button("CSV", move_csv(rows), f"{stem}.csv", "text/csv", key=stem + "_csv")
            downloads[1].download_button(
                "JSON", json.dumps(rows, indent=2), f"{stem}.json", "application/json", key=stem + "_json"
            )
            downloads[2].download_button(
                "Text", move_text(rows), f"{stem}.txt", "text/plain", key=stem + "_txt"
            )

with tabs[1]:
    choices = []
    if candidate is not None:
        choices.append("Candidate")
    if exact is not None:
        choices.append("Exact")
    replay_source = st.radio("Replay source", choices, horizontal=True)
    replay_alignment_value = candidate if replay_source == "Candidate" else exact
    cache_key = f"{selected_id}:{replay_source}:{configuration.get('cost_model')}"
    if cache_key not in st.session_state.replay_snapshots:
        st.session_state.replay_snapshots[cache_key] = replay_alignment(
            replay_alignment_value,
            loaded_net.net,
            loaded_net.initial_marking,
            loaded_net.final_marking,
            result.variant.labels,
            cost_model,
        )
    snapshots = st.session_state.replay_snapshots[cache_key]
    if snapshots:
        index_key = f"replay_index_{selected_id}_{replay_source}"
        if index_key not in st.session_state:
            invalid = next((snapshot.move_index for snapshot in snapshots if snapshot.verification_state != "valid"), 0)
            st.session_state[index_key] = invalid

        deviations = [
            snapshot.move_index
            for snapshot in snapshots
            if snapshot.selected_move["kind"] != MoveKind.SYNCHRONOUS.value
        ]
        controls = st.columns(6)
        if controls[0].button("⏮ First", key=index_key + "_first"):
            st.session_state[index_key] = 0
        if controls[1].button("← Previous", key=index_key + "_prev"):
            st.session_state[index_key] = max(0, st.session_state[index_key] - 1)
        if controls[2].button("Next →", key=index_key + "_next"):
            st.session_state[index_key] = min(len(snapshots) - 1, st.session_state[index_key] + 1)
        if controls[3].button("Next deviation", key=index_key + "_dev") and deviations:
            st.session_state[index_key] = next(
                (value for value in deviations if value > st.session_state[index_key]), deviations[0]
            )
        if controls[4].button("First log move", key=index_key + "_log"):
            st.session_state[index_key] = next(
                (snapshot.move_index for snapshot in snapshots if snapshot.selected_move["kind"] == MoveKind.LOG.value),
                st.session_state[index_key],
            )
        controls[5].caption("Snapshots are precomputed; moving the slider does not replay from the start.")

        replay_index = st.slider(
            "Move",
            0,
            len(snapshots) - 1,
            key=index_key,
        )
        snapshot = snapshots[replay_index]
        summary = st.columns(6)
        summary[0].metric("Observed event", snapshot.selected_move["log_label"] or ">>")
        summary[1].metric("Move", snapshot.selected_move["kind"])
        summary[2].metric("Fired", snapshot.fired_transition or "None")
        summary[3].metric("Consumed events", snapshot.trace_index_after)
        summary[4].metric("Move cost", snapshot.move_cost if snapshot.move_cost is not None else "—")
        summary[5].metric("Cumulative", snapshot.cumulative_cost)
        if snapshot.error:
            st.error(snapshot.error)
        detail = st.columns(2)
        detail[0].markdown("**Marking before**")
        detail[0].json(snapshot.marking_before)
        detail[0].markdown("**Enabled transitions**")
        detail[0].json(snapshot.enabled_transitions)
        detail[1].markdown("**Marking after**")
        detail[1].json(snapshot.marking_after)
        detail[1].markdown("**Selected move**")
        detail[1].json(snapshot.selected_move)

        places_by_name = {str(place.name): place for place in loaded_net.net.places}
        current_marking = Marking(
            {
                places_by_name[name]: tokens
                for name, tokens in snapshot.marking_after.items()
                if name in places_by_name
            }
        )
        st.graphviz_chart(
            petri_net_dot(
                loaded_net.net,
                loaded_net.initial_marking,
                loaded_net.final_marking,
                alignment=replay_alignment_value,
                current_marking=current_marking,
                fired_transition=snapshot.fired_transition,
                enabled=set(snapshot.enabled_transitions),
            ),
            width="stretch",
        )
    else:
        st.info("This alignment contains no moves.")

with tabs[2]:
    verification = candidate_result.verification if candidate_result else None
    if verification is None:
        st.info("No neural candidate verification is available in exact-only mode.")
    else:
        fields = st.columns(6)
        fields[0].metric("Replay legal", "Yes" if verification.legal else "No")
        fields[1].metric("All events consumed", "Yes" if verification.consumed_events == len(result.variant.labels) else "No")
        fields[2].metric("Final marking reached", "Yes" if verification.final_marking_reached else "No")
        fields[3].metric("Failure move", verification.failure_index if verification.failure_index is not None else "—")
        fields[4].metric("Recomputed cost", verification.cost)
        fields[5].metric(
            "Decoder/replay cost agrees",
            "Yes" if candidate and candidate.cost == verification.cost else "No",
        )
        if verification.reason:
            st.error(verification.reason)
        else:
            st.success("Every candidate transition was enabled, every event was consumed, and the required final marking was reached.")
        st.json(
            {
                "candidate_source": candidate.source if candidate else None,
                "candidate_reported_cost": candidate.cost if candidate else None,
                "replay_computed_cost": verification.cost,
                "failure_category": verification.reason,
                "verification_is_independent_of_optimality": True,
            }
        )

with tabs[3]:
    if candidate_result is None:
        st.info("Neural diagnostics are unavailable in exact-only mode.")
    elif not candidate_result.neural_diagnostics.get("guided"):
        st.info("This candidate used unguided structural decoding, so no neural move scores exist.")
    else:
        neural_tabs = st.tabs(["Synchronous transitions", "Model moves", "Log moves", "Latent regions", "Research debug"])
        with neural_tabs[0]:
            st.write("Relative decoder scores used to rank transitions compatible with each event. These are not calibrated confidence values.")
            sync_rows = synchronous_score_rows(candidate_result, exact, loaded_net.net, loaded_net.initial_marking)
            st.dataframe(pd.DataFrame(sync_rows), hide_index=True, width="stretch")
        with neural_tabs[1]:
            st.write("Ranked model-transition scores used as search ordering guidance.")
            st.dataframe(
                pd.DataFrame(candidate_result.neural_diagnostics.get("model_move_scores", [])),
                hide_index=True,
                width="stretch",
            )
        with neural_tabs[2]:
            st.warning("Predicted deviation score; not used directly by the current greedy decoder.")
            log_rows = log_score_rows(candidate_result, exact)
            log_frame = pd.DataFrame(log_rows)
            st.dataframe(log_frame, hide_index=True, width="stretch")
            if not log_frame.empty:
                st.bar_chart(log_frame, x="Trace position", y="Predicted deviation score")
        with neural_tabs[3]:
            st.info("Region identifiers are latent and unsupervised. Region 1 or Region 2 does not have a fixed process-semantic meaning.")
            event_regions = candidate_result.neural_diagnostics.get("event_region_probs", [])
            if event_regions:
                region_frame = pd.DataFrame(
                    event_regions,
                    index=[f"{index}: {label}" for index, label in enumerate(result.variant.labels)],
                    columns=[f"Region {index + 1}" for index in range(len(event_regions[0]))],
                )
                st.dataframe(region_frame.style.background_gradient(cmap="Blues"), width="stretch")
                st.bar_chart(region_frame.sum(axis=0))
        with neural_tabs[4]:
            st.warning(candidate_result.neural_diagnostics.get("experimental_warning"))
            st.write("Experimental local uncertainty, lower-value, and sketch heads are deliberately not promoted as calibrated explanations or admissible bounds.")

with tabs[4]:
    if exact_result is None:
        st.info("Exact alignment was not requested for this result.")
    else:
        metrics = st.columns(7)
        diagnostics = exact_result.exact_diagnostics
        metrics[0].metric("Status", pretty_status(exact_result.status))
        metrics[1].metric("Exact cost", exact_result.exact_cost if exact_result.exact_cost is not None else "Unavailable")
        metrics[2].metric("Runtime", f"{exact_result.exact_seconds:.4f}s")
        metrics[3].metric("Visited states", diagnostics.get("visited_states", "—"))
        metrics[4].metric("Queued states", diagnostics.get("queued_states", "—"))
        metrics[5].metric("Traversed arcs", diagnostics.get("traversed_arcs", "—"))
        metrics[6].metric("LP solves", diagnostics.get("lp_solved", "—"))
        if exact_result.timed_out:
            st.warning("Exact search timed out. The absence of an exact cost is not evidence of infeasibility.")
        elif exact_result.error:
            st.error(exact_result.error)
        elif exact_result.exact_cost is not None:
            st.success("The exact backend returned an optimal alignment before its timeout.")
        candidate_cost = candidate_result.cost if candidate_result else None
        exact_cost = exact_result.exact_cost
        comparison = {
            "backend": diagnostics.get("backend", "pm4py_state_equation_a_star"),
            "candidate_cost": candidate_cost,
            "exact_cost": exact_cost,
            "absolute_gap": (
                candidate_cost - exact_cost
                if candidate_cost is not None and exact_cost is not None
                else None
            ),
            "relative_gap": (
                (candidate_cost - exact_cost) / exact_cost
                if candidate_cost is not None and exact_cost not in {None, 0}
                else None
            ),
            "candidate_runtime": candidate_result.total_seconds if candidate_result else None,
            "exact_runtime": exact_result.exact_seconds,
            "speed_ratio_exact_over_candidate": (
                exact_result.exact_seconds / candidate_result.total_seconds
                if candidate_result and candidate_result.total_seconds > 0
                else None
            ),
            "timeout": exact_result.timed_out,
            "error": exact_result.error,
        }
        st.json(comparison)
