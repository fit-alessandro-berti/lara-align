import streamlit as st

from lara_ui.app_state import initialize_state

st.set_page_config(
    page_title="LARA-Align Conformance Workbench",
    page_icon="⚖️",
    layout="wide",
)
initialize_state(st.session_state)

st.title("LARA-Align Conformance Workbench")
st.caption("Candidate first. Verification separate. Exact evidence added progressively.")

st.markdown(
    """
This workbench separates the three questions that a single final alignment can hide:

1. What did the neural LARA decoder propose?
2. Can that proposal be replayed legally on the Petri net?
3. Did exact search certify it, repair it, time out, or fail?

Start with an XES log, PNML model, and trusted checkpoint. Variant-level processing is
the default, so repeated traces are aligned once and expanded back to their cases on export.
"""
)

steps = st.columns(4)
with steps[0]:
    st.subheader("1 · Setup")
    st.write("Load and validate inputs, inspect variants, and choose run settings.")
    st.page_link("pages/01_setup.py", label="Open setup", icon="⚙️")
with steps[1]:
    st.subheader("2 · Live run")
    st.write("Watch candidates appear before exact certification finishes.")
    st.page_link("pages/02_live_run.py", label="Open live run", icon="▶️")
with steps[2]:
    st.subheader("3 · Inspect")
    st.write("Compare alignments by trace position, replay markings, and inspect scores.")
    st.page_link("pages/03_trace_inspector.py", label="Open inspector", icon="🔎")
with steps[3]:
    st.subheader("4 · Experiments")
    st.write("Optionally compare benchmark and training metric files.")
    st.page_link("pages/04_experiments.py", label="Open dashboard", icon="📊")

st.divider()
ready = all(
    [
        st.session_state.loaded_log is not None,
        st.session_state.loaded_net is not None,
        st.session_state.loaded_checkpoint is not None,
        bool(st.session_state.variant_index),
    ]
)
if ready:
    st.success(
        f"Setup ready · {len(st.session_state.variant_index)} variants · "
        f"run status: {st.session_state.run_status.replace('_', ' ')}"
    )
else:
    st.info("No complete setup is loaded yet. Bundled examples are available on the Setup page.")
