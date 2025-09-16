"""
crstlmeth.web.pages.03_calculate

streamlit interface providing different calculation methods.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from crstlmeth.core.methylation import Methylation
from crstlmeth.core.parsers import query_bedmethyl
from crstlmeth.core.regions import load_intervals
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.utils import list_builtin_kits

# ────────────────────────────────────────────────────────────────────
# page configuration
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - calculate", page_icon=":material/calculate:"
)
st.title("calculate")
render_sidebar()

# pull discoveries/config
cmeth_files: List[str] = st.session_state.get("cmeth_files", [])
bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.get(
    "bed_by_sample", {}
)
custom_beds: List[str] = st.session_state.get("custom_beds", [])
out_dir_str: str = st.session_state.get("outdir_resolved", "") or ""

# basic guards
if not out_dir_str:
    st.error("Set an **output directory** on the Home page first.")
    st.stop()
out_dir = Path(out_dir_str)

if not bed_by_sample:
    st.warning(
        "No bgzipped & indexed bedmethyl files found – configure folders on **Home** and click *scan folders*."
    )
    st.stop()


# small helpers
def _persist(upload: BytesIO) -> Path:
    """store an uploaded file under OUTDIR/.streamlit_tmp and return its path"""
    tmp = out_dir / ".streamlit_tmp" / upload.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(upload.getbuffer())
    return tmp.resolve()


def _all_parts_for_sample(sample_id: str) -> List[Path]:
    """return bedmethyl part paths for a discovered sample, ordered h1,h2,ungrouped if present"""
    parts = bed_by_sample.get(sample_id, {})
    ordered: List[Path] = []
    for k in ("1", "2", "ungrouped"):
        p = parts.get(k)
        if p:
            ordered.append(Path(p))
    # include any stray keys deterministically
    for k in sorted(set(parts) - {"1", "2", "ungrouped"}):
        if parts.get(k):
            ordered.append(Path(parts[k]))
    return ordered


# ────────────────────────────────────────────────────────────────────
# single-interval spot-check
# ────────────────────────────────────────────────────────────────────
st.subheader("single-interval methylation calculator")

sids = sorted(bed_by_sample)
sid = st.selectbox("sample", sids)
part_key = st.selectbox(
    "bedmethyl part",
    options=[
        k for k in ("1", "2", "ungrouped") if k in bed_by_sample.get(sid, {})
    ],
    index=0,
    help="Pick hap1, hap2, or ungrouped for this sample.",
)

bed_path = bed_by_sample.get(sid, {}).get(part_key)
if not bed_path:
    st.info("Selected sample has no file for this part.")
    st.stop()

c1, c2, c3 = st.columns(3)
chrom = c1.text_input("chrom", value="chr11")
start = c2.number_input("start", min_value=0, value=1_976_000, step=100)
end = c3.number_input("end", min_value=1, value=1_976_200, step=100)

if end <= start:
    st.error("end must be greater than start")
elif st.button("fetch", type="primary"):
    df = query_bedmethyl(str(bed_path), chrom, int(start), int(end))
    if df.empty:
        st.info("no records in this interval")
    else:
        total_mod = int(df["Nmod"].sum())
        total_valid = int(df["Nvalid_cov"].sum())
        pct = (total_mod / total_valid) * 100 if total_valid else 0.0
        st.metric("methylation (%)", f"{pct:.2f}")
        st.caption(f"Nmod = {total_mod:,}   |   Nvalid = {total_valid:,}")
        with st.expander("raw records"):
            st.dataframe(df, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────────────
# multi-region deviation scan
# ────────────────────────────────────────────────────────────────────
st.subheader("multi-region deviation scan")

# kit or custom bed selection from discoveries
kits = list_builtin_kits()
kit_or_bed = st.selectbox(
    "MLPA kit / BED",
    [*kits, *custom_beds],
    help="Select a built-in MLPA kit or a custom BED discovered on Home.",
)

# optional: also allow uploading a temporary custom BED (kept under OUTDIR)
with st.expander("optional: use a BED not in your configured folders"):
    upl = st.file_uploader("upload BED", type=["bed"])
    if upl is not None:
        tmp_bed = _persist(upl)
        kit_or_bed = str(tmp_bed)
        st.caption(f"using uploaded BED → {tmp_bed}")

# choose target/ref samples from discovered set
left, right = st.columns([0.5, 0.5], gap="large")
with left:
    tgt_sid = st.selectbox(
        "target sample",
        sids,
        help="One target; pooled from its available hap parts.",
    )
with right:
    ref_sids = st.multiselect(
        "reference samples",
        [s for s in sids if s != tgt_sid],
        help="Pick ≥1 reference samples for the cohort.",
    )

if not tgt_sid or not ref_sids:
    st.info(
        "Select one **target** and at least one **reference** sample to run the scan."
    )
    st.stop()

# allow supplementing target/ref with additional uploads (optional)
with st.expander(
    "optional: include extra target/reference files (not discovered)"
):
    col_t, col_r = st.columns(2)
    with col_t:
        extra_tgt = st.file_uploader(
            "extra target (.bedmethyl.gz)", type=["gz"]
        )
    with col_r:
        extra_refs = st.file_uploader(
            "extra reference(s) (.bedmethyl.gz)",
            type=["gz"],
            accept_multiple_files=True,
        )

# resolve paths
tgt_paths: List[Path] = _all_parts_for_sample(tgt_sid)
ref_paths: List[Path] = [
    p for sid_ in ref_sids for p in _all_parts_for_sample(sid_)
]

if extra_tgt:
    tgt_paths.append(_persist(extra_tgt))
if extra_refs:
    ref_paths.extend(_persist(f) for f in extra_refs)

if st.button("run deviation scan", type="primary", use_container_width=True):
    if not kit_or_bed:
        st.error("Select a **kit/BED** first.")
        st.stop()
    if not (tgt_paths and ref_paths):
        st.error("Need one target and at least one reference file.")
        st.stop()

    with st.spinner("calculating …"):
        try:
            intervals, region_names = load_intervals(kit_or_bed)
        except Exception as e:
            st.error(f"Failed to load intervals from {kit_or_bed}: {e}")
            st.stop()

        # pooled methylation levels per sample across the provided parts
        ref_lv = Methylation.get_levels([str(p) for p in ref_paths], intervals)
        tgt_lv = Methylation.get_levels([str(p) for p in tgt_paths], intervals)

        # guard zero or all-nan columns in the cohort (avoid /0 or NaNs in z)
        mean_ref = np.nanmean(ref_lv, axis=0)
        std_ref = np.nanstd(ref_lv, axis=0, ddof=1)
        std_ref = np.where(
            (~np.isfinite(std_ref)) | (std_ref == 0.0), np.nan, std_ref
        )

        # flags using the library helper
        flags = Methylation.get_deviations(ref_lv, tgt_lv, fdr_alpha=0.05)

        z_scores = (tgt_lv[0] - mean_ref) / std_ref
        table = pd.DataFrame(
            {
                "region": region_names,
                "mean_ref": mean_ref,
                "target": tgt_lv[0],
                "z_score": z_scores,
                "flag_alpha_0.05": flags[0],
            }
        )

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "download CSV",
        data=table.to_csv(index=False).encode(),
        file_name=f"{tgt_sid}_deviation_scan.csv",
        mime="text/csv",
        use_container_width=True,
    )
