from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lara_ui.page_utils import prepare_page

prepare_page("Experiment dashboard", "📊")
st.title("Experiment dashboard")
st.caption("Optional research view for benchmark outputs, training curves, and checkpoint comparisons.")

root = Path(__file__).resolve().parents[1]
source = st.radio("Data source", ["Repository files", "Upload CSV files"], horizontal=True)

frames: dict[str, pd.DataFrame] = {}
if source == "Repository files":
    candidates = sorted(
        list((root / "runs").rglob("*.csv"))
        + list(root.glob("*results*.csv"))
        + list(root.glob("*metrics*.csv"))
    )
    selected = st.multiselect(
        "Metric/benchmark files",
        candidates,
        default=candidates[:3],
        format_func=lambda path: str(path.relative_to(root)),
    )
    for path in selected:
        try:
            frames[str(path.relative_to(root))] = pd.read_csv(path)
        except Exception as exc:
            st.error(f"Could not read {path}: {exc}")
else:
    uploads = st.file_uploader("CSV metric or benchmark files", type=["csv"], accept_multiple_files=True)
    for upload in uploads:
        try:
            frames[upload.name] = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read {upload.name}: {exc}")

if not frames:
    st.info("Choose one or more result files. Ordinary alignment work does not require this page.")
else:
    for name, frame in frames.items():
        with st.expander(name, expanded=True):
            st.dataframe(frame, hide_index=True, width="stretch")
            numeric = frame.select_dtypes(include="number")
            if numeric.empty:
                continue
            x_options = ["Row"] + list(numeric.columns)
            controls = st.columns(2)
            x = controls[0].selectbox("Horizontal axis", x_options, key=name + "_x")
            y_options = [column for column in numeric.columns if column != x]
            y = controls[1].multiselect(
                "Measurements",
                y_options,
                default=y_options[: min(3, len(y_options))],
                key=name + "_y",
            )
            if y:
                chart = numeric[y].copy()
                if x != "Row":
                    chart.index = numeric[x]
                st.line_chart(chart)

    if len(frames) > 1:
        st.subheader("Checkpoint/file comparison")
        summaries = []
        for name, frame in frames.items():
            numeric = frame.select_dtypes(include="number")
            summary = {"File": name, "Rows": len(frame)}
            for column in numeric.columns:
                summary[f"Last {column}"] = numeric[column].iloc[-1] if len(numeric) else None
                summary[f"Best/min {column}"] = numeric[column].min()
            summaries.append(summary)
        st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
