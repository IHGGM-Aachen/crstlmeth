"""
crstlmeth.web.pages.04_kits

browse built-in MLPA kits or custom BED files
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from crstlmeth.core.regions import load_intervals
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.utils import list_builtin_kits

# page setup and sidebar
st.set_page_config(
    page_title="crstlmeth - kits", page_icon="material/arrow_range:"
)
st.title("kits")
render_sidebar()

st.subheader("browse MLPA kits or custom BED files")

# select a built-in kit or upload custom
kits = list_builtin_kits()
opt_custom = "choose BED…"
choice = st.selectbox("kit / BED", [*kits, opt_custom])

if choice == opt_custom:
    bed_path = st.file_uploader("upload a BED file", type="bed")
    if not bed_path:
        st.stop()
    kit_label = Path(bed_path.name).stem
    bed_file = bed_path
else:
    kit_label = choice
    bed_file = choice  # load_intervals accepts kit name or path

# load BED intervals
intervals, names = load_intervals(bed_file)
df = pd.DataFrame(intervals, columns=["chrom", "start", "end"])
df["name"] = names

# preview and download
st.subheader(f"{len(df):,} regions in {kit_label}")
st.dataframe(df.head(100), hide_index=True, use_container_width=True)

st.download_button(
    label="download CSV",
    data=df.to_csv(index=False).encode(),
    file_name=f"{kit_label}_regions.csv",
    mime="text/csv",
)
