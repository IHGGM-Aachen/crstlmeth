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
    st.info("No log path configured. Start the app via `crstlmeth --log-file ... web`.")
    st.stop()

log_path = Path(env).expanduser()

# CLI allows --log-file to be a directory, if then use default filename inside it
if log_path.exists() and log_path.is_dir():
    log_path = log_path / "crstlmeth.log.tsv"

if not log_path.exists():
    st.info(f"no log file found at: {log_path}")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# load TSV 
# ────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(log_path, sep="\t")
except Exception as e:
    st.error(f"failed to read log TSV: {e}")
    st.stop()

if df.empty:
    st.info("log file is empty")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# filter by session-id
# ────────────────────────────────────────────────────────────────────
default_sid = st.session_state.get("session_id", "")
sid = st.text_input(
    "filter by session-id (leave blank for all)",
    value=default_sid,
    key="log_filter_sid",
)

if sid and "session" in df.columns:
    df = df.query("session == @sid")
elif sid and "session" not in df.columns:
    st.warning("log has no 'session' column; cannot filter by session id")

# ────────────────────────────────────────────────────────────────────
# show + export
# ────────────────────────────────────────────────────────────────────
st.caption(f"log file: {log_path}")
st.dataframe(df, use_container_width=True)

st.download_button(
    "download as CSV",
    df.to_csv(index=False).encode(),
    file_name="crstlmeth_log_filtered.csv",
    mime="text/csv",
    use_container_width=True,
)
