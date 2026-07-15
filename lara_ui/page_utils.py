from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import streamlit as st

from lara_align.checkpoint import load_checkpoint
from lara_ui.alignment_runner import aggregate_metrics, result_rows
from lara_ui.app_state import initialize_state
from lara_ui.ui_types import TraceAlignmentResult


STATUS_LABELS = {
    "queued": "Queued",
    "candidate_running": "Candidate running",
    "candidate_ready": "Candidate ready",
    "certifying": "Certifying",
    "certified": "Certified",
    "repaired": "Repaired",
    "illegal_candidate": "Illegal candidate",
    "illegal_candidate_exact_available": "Illegal candidate · exact available",
    "exact_timeout": "Exact timeout",
    "exact_failure": "Exact failure",
    "candidate_failure": "Candidate failure",
    "exact_only": "Exact only",
    "cancelled": "Cancelled",
}


def prepare_page(title: str, icon: str = "⚖️") -> None:
    st.set_page_config(page_title=f"{title} · LARA-Align", page_icon=icon, layout="wide")
    initialize_state(st.session_state)
    st.sidebar.caption("LARA-Align conformance workbench")
    st.sidebar.caption("Use the page navigation above to move between the four workbench stages.")


@st.cache_resource(show_spinner="Loading trusted LARA checkpoint…")
def cached_checkpoint(path: str, device: str, modified_ns: int):
    _ = modified_ns
    model, checkpoint = load_checkpoint(path, device=device)
    model.eval()
    return model, checkpoint


def require_setup() -> bool:
    missing = []
    if st.session_state.loaded_log is None:
        missing.append("event log")
    if st.session_state.loaded_net is None:
        missing.append("Petri net")
    if st.session_state.loaded_checkpoint is None:
        missing.append("checkpoint")
    if not st.session_state.variant_index:
        missing.append("trace variants")
    if missing:
        st.warning("Complete Setup first. Missing: " + ", ".join(missing) + ".")
        st.info("Open **Setup & overview** from the page navigation.")
        return False
    return True


def pretty_status(status: str) -> str:
    return STATUS_LABELS.get(status, status.replace("_", " ").title())


def append_feed(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.execution_feed.append(f"{timestamp} · {message}")
    st.session_state.execution_feed = st.session_state.execution_feed[-200:]


def format_number(value, digits: int = 2, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{100 * value:.{digits}f}%"
    return f"{value:.{digits}f}"


def render_live_summary(
    results: Iterable[TraceAlignmentResult],
    metrics_placeholder,
    progress_placeholder,
    table_placeholder,
    feed_placeholder,
) -> None:
    results = list(results)
    metrics = aggregate_metrics(results)
    with metrics_placeholder.container():
        row1 = st.columns(6)
        values = [
            ("Variants queued", metrics["variants_queued"]),
            ("Candidates ready", metrics["candidates_completed"]),
            ("Certified", metrics["variants_certified"]),
            ("Cases covered", format_number(metrics["cases_covered"], percent=True)),
            ("Candidate legal", format_number(metrics["candidate_legal_rate"], percent=True)),
            ("Repair rate", format_number(metrics["repair_rate"], percent=True)),
        ]
        for column, (label, value) in zip(row1, values):
            column.metric(label, value)
        with st.expander("Variant-level and case-weighted metrics", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Level": "Variant",
                            "Legal rate": metrics["candidate_legal_rate"],
                            "Mean candidate cost": metrics["mean_candidate_cost"],
                            "Mean exact cost": metrics["mean_exact_cost"],
                            "Mean cost gap": metrics["mean_cost_gap"],
                            "Mean candidate time": metrics["mean_candidate_runtime"],
                            "Mean exact time": metrics["mean_exact_runtime"],
                        },
                        {
                            "Level": "Case-weighted",
                            "Legal rate": metrics["case_weighted_legal_rate"],
                            "Mean candidate cost": metrics["case_weighted_candidate_cost"],
                            "Mean exact cost": metrics["case_weighted_exact_cost"],
                        },
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    with progress_placeholder.container():
        total = max(len(results), 1)
        candidates = metrics["candidates_completed"]
        exact_done = sum(result.exact_result is not None for result in results)
        candidate_cases = sum(
            result.variant.frequency for result in results if result.candidate is not None
        )
        total_cases = max(sum(result.variant.frequency for result in results), 1)
        st.caption("Candidate generation")
        st.progress(candidates / total, text=f"{candidates}/{len(results)} variants")
        st.caption("Exact certification")
        st.progress(exact_done / total, text=f"{exact_done}/{len(results)} selected variants completed")
        st.caption("Case coverage")
        st.progress(candidate_cases / total_cases, text=f"{candidate_cases}/{total_cases} cases")
    rows = result_rows(results)
    display = pd.DataFrame(rows)
    if not display.empty:
        display["Status"] = display["Status"].map(pretty_status)
        table_placeholder.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Coverage": st.column_config.ProgressColumn(format="percent"),
                "Candidate time (s)": st.column_config.NumberColumn(format="%.4f"),
                "Exact time (s)": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    with feed_placeholder.container():
        st.caption("Live event feed")
        entries = st.session_state.execution_feed[-10:]
        st.code("\n".join(reversed(entries)) if entries else "Waiting for the first stage…")
