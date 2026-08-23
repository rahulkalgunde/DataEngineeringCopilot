"""Human labeling UI for judge calibration rows.

Run: make label-calibration (or streamlit run data_engineering_copilot/ui/label_calibration.py)

Loads tests/evaluation/golden/judge_calibration.jsonl, shows question + contexts +
answer per row, records your faithfulness judgment (0/1), saves back to the JSONL.
Zero LLM calls. After all rows are labeled run:
    dec eval-judge-calibrate
"""

from __future__ import annotations

import json
import pathlib

import streamlit as st

DEFAULT_PATH = "tests/evaluation/golden/judge_calibration.jsonl"

st.set_page_config(page_title="Judge Calibration Labeler", page_icon=":material/fact_check:", layout="wide")


@st.cache_data
def load_rows(path: str) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(path: str, rows: list[dict]) -> None:
    pathlib.Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


dataset_path = st.sidebar.text_input("Dataset path", value=DEFAULT_PATH)
rows = load_rows(dataset_path)

if not rows:
    st.error(f"Dataset not found or empty: {dataset_path}")
    st.stop()

labeled = [r for r in rows if not r.get("needs_label") and r.get("human_faithfulness", -1) >= 0]
unlabeled_idx = [i for i, r in enumerate(rows) if r not in labeled]
progress = len(labeled) / len(rows)

st.sidebar.metric("Labeled", f"{len(labeled)}/{len(rows)}")
st.sidebar.progress(progress)
if st.sidebar.button("Jump to first unlabeled", icon=":material/fast_forward:", disabled=not unlabeled_idx):
    st.session_state.row_idx = unlabeled_idx[0]
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("After labeling all rows:\n\n`dec eval-judge-calibrate`")

if "row_idx" not in st.session_state:
    st.session_state.row_idx = unlabeled_idx[0] if unlabeled_idx else 0

idx = min(st.session_state.row_idx, len(rows) - 1)
row = rows[idx]

header_cols = st.columns([4, 1])
with header_cols[0]:
    st.subheader(f"{row['id']}", divider="gray")
with header_cols[1]:
    nav_prev, nav_next = st.columns(2)
    if nav_prev.button(":material/chevron_left:", disabled=idx == 0):
        st.session_state.row_idx = idx - 1
        st.rerun()
    if nav_next.button(":material/chevron_right:", disabled=idx >= len(rows) - 1):
        st.session_state.row_idx = idx + 1
        st.rerun()

st.markdown(f"**Question** — {row['question']}")

with st.expander(f"Contexts ({len(row.get('contexts', []))} chunks)", expanded=True):
    for j, ctx in enumerate(row.get("contexts", [])):
        st.markdown(f"**Chunk {j + 1}**")
        st.text(ctx[:2500])
        st.divider()

answer_cols = st.columns([3, 2])
with answer_cols[0]:
    st.markdown("**Answer to judge**")
    st.info(row.get("answer", "(empty)"), icon=":material/chat:")

with st.form(key=f"judge_{idx}", border=True):
    faithful = st.radio(
        "Is the answer fully supported by the context? (faithfulness)",
        options=[1, 0],
        format_func=lambda v: "1 — faithful" if v else "0 — not faithful",
        index=None,
        horizontal=True,
    )
    submitted = st.form_submit_button(
        "Save judgment & next", type="primary", icon=":material/save:", disabled=faithful is None
    )
    if submitted and faithful is not None:
        rows[idx]["human_faithfulness"] = int(faithful)
        rows[idx]["needs_label"] = False
        rows[idx]["label_note"] = "human_streamlit"
        save_rows(dataset_path, rows)
        st.cache_data.clear()
        next_unlabeled = [
            i
            for i, r in enumerate(load_rows(dataset_path))
            if r.get("needs_label") or r.get("human_faithfulness", -1) < 0
        ]
        st.session_state.row_idx = next_unlabeled[0] if next_unlabeled else min(idx + 1, len(rows) - 1)
        st.rerun()

if progress == 1.0:
    st.success("All rows labeled. Run: `dec eval-judge-calibrate`", icon=":material/check_circle:")
