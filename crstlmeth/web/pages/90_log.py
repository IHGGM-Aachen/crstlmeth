"""
crstlmeth.web.pages.90_log

view and filter global TSV log by session ID
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="crstlmeth - log", page_icon=":material/bug_report:")

ensure_web_state()

st.title("log")
render_sidebar()

# ────────────────────────────────────────────────────────────────────
# resolve log file
# ────────────────────────────────────────────────────────────────────
env = (os.getenv("CRSTLMETH_LOGFILE") or "").strip()
if not env:
    st.info(
        "No log path configured. Start the app via `crstlmeth --log-file ... web`."
    )
    st.stop()

log_path = Path(env).expanduser()

# CLI allows --log-file to be a directory; if so, use default filename inside it
if log_path.exists() and log_path.is_dir():
    log_path = log_path / "crstlmeth.log.tsv"

if not log_path.exists():
    st.info(f"no log file found at: {log_path}")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# load TSV
# ────────────────────────────────────────────────────────────────────
COLS = ["ts", "level", "session", "event", "cmd", "parameters", "message", "runtime"]

try:
    df = pd.read_csv(
        log_path,
        sep="\t",
        header=None,
        names=COLS,
        dtype=str,
        keep_default_na=False,
    )
except Exception as e:
    st.error(f"failed to read log TSV: {e}")
    st.stop()

if df.empty:
    st.info("log file is empty")
    st.stop()

# If someone manually added a header row, drop it
if df.iloc[0].tolist()[: len(COLS)] == COLS:
    df = df.iloc[1:].reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────
# filter + controls
# ────────────────────────────────────────────────────────────────────
default_sid = st.session_state.get("session_id", "")
sid = st.text_input(
    "filter by session-id (leave blank for all)",
    value=default_sid,
    key="log_filter_sid",
)

c1, c2, c3 = st.columns([0.34, 0.33, 0.33], gap="large")
with c1:
    level = st.multiselect(
        "level",
        options=sorted(df["level"].unique().tolist()),
        default=[],
        help="empty = all levels",
    )
with c2:
    event = st.text_input("event contains", value="", help="substring match")
with c3:
    tail_n = st.number_input(
        "show last N rows",
        min_value=0,
        max_value=50000,
        value=2000,
        step=100,
        help="0 = show all (can be slow on huge logs)",
    )

# apply filters
view = df

if sid.strip():
    view = view[view["session"] == sid.strip()]

if level:
    view = view[view["level"].isin(level)]

if event.strip():
    view = view[view["event"].str.contains(event.strip(), case=False, na=False)]

# show tail for speed (after filtering)
if int(tail_n) > 0 and len(view) > int(tail_n):
    view = view.tail(int(tail_n)).reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────
# show + export
# ────────────────────────────────────────────────────────────────────
st.caption(f"log file: {log_path}")
st.dataframe(view, use_container_width=True, hide_index=True)

st.download_button(
    "download as CSV",
    view.to_csv(index=False).encode(),
    file_name="crstlmeth_log_filtered.csv",
    mime="text/csv",
    use_container_width=True,
)
