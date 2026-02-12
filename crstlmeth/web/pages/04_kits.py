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
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import list_builtin_kits

# ────────────────────────────────────────────────────────────────────
# page setup and sidebar
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - kits", page_icon=":material/arrow_range:"
)

ensure_web_state()

st.title("kits")
render_sidebar()

st.subheader("browse MLPA kits or custom BED files")

session_id: str = st.session_state["session_id"]
out_dir = resolve_outdir(session_id)
default_kit: str = (st.session_state.get("default_kit") or "ME030").strip()

# discoveries
custom_beds: list[str] = st.session_state.setdefault("custom_beds", [])


def _existing_paths(xs: list[str]) -> list[Path]:
    out: list[Path] = []
    for x in xs:
        try:
            p = Path(x)
            if p.exists():
                out.append(p)
        except Exception:
            continue
    return out


# build choices are bundled kits and discovered beds and upload
kits = list_builtin_kits()
ext_beds = _existing_paths(custom_beds)

choices: list[str] = []
choices += [f"bundled kit · {k}" for k in sorted(kits.keys())]
choices += [
    f"external BED · {p.name}"
    for p in sorted(ext_beds, key=lambda pp: pp.name.lower())
]
choices += ["upload BED …"]

# default selection
default_label = f"bundled kit · {default_kit}"
default_index = choices.index(default_label) if default_label in choices else 0

choice = st.selectbox(
    "kit / BED", choices, index=default_index, key="kits_choice"
)

# resolve selection into a bed identifier usable by load_intervals
kit_label: str
bed_id: str | Path

if choice == "upload BED …":
    upl = st.file_uploader("upload a BED file", type=["bed"], key="kits_upload")
    if not upl:
        st.stop()
    tmp = out_dir / ".streamlit_tmp" / upl.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(upl.getbuffer())
    kit_label = tmp.stem
    bed_id = tmp
elif choice.startswith("bundled kit ·"):
    kit_label = choice.split("·", 1)[1].strip()
    bed_id = kit_label  # load_intervals accepts kit id
else:
    # external BED
    bed_name = choice.split("·", 1)[1].strip()
    match = next((p for p in ext_beds if p.name == bed_name), None)
    if not match:
        st.error("Selected external BED no longer exists. Re-scan on Home.")
        st.stop()
    kit_label = match.stem
    bed_id = match

# load BED intervals
try:
    intervals, names = load_intervals(bed_id)
except Exception as e:
    st.error(f"Failed to load intervals: {e}")
    st.stop()

df = pd.DataFrame(intervals, columns=["chrom", "start", "end"])
df["name"] = names

# preview and download
st.subheader(f"{len(df):,} regions in {kit_label}")
st.dataframe(df.head(200), hide_index=True, use_container_width=True)

st.download_button(
    label="download CSV",
    data=df.to_csv(index=False).encode(),
    file_name=f"{kit_label}_regions.csv",
    mime="text/csv",
    use_container_width=True,
)
