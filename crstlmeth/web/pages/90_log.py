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

# page setup
st.set_page_config(
    page_title="crstlmeth - log", page_icon=":material/bug_report:"
)
st.title("log")
render_sidebar()

# resolve path from env var
log_path = Path(os.environ["CRSTLMETH_LOGFILE"])

if not log_path.exists():
    st.info("no log file found")
    st.stop()

# load TSV
df = pd.read_csv(log_path, sep="\t")

# filter by session-id
sid = st.text_input("filter by session-id (leave blank for all)")

if sid:
    df = df.query("session == @sid")

# show + export
st.dataframe(df, use_container_width=True)

st.download_button(
    "download as CSV",
    df.to_csv(index=False).encode(),
    file_name="crstlmeth_log_filtered.csv",
    mime="text/csv",
)
