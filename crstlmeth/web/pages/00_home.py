"""
crstlmeth.web.pages.00_home
---------------------------
landing page with folder setup and discovery of input files
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from crstlmeth.core.discovery import (
    scan_bedmethyl,
    scan_cmeth,
    scan_region_beds,
)
from crstlmeth.web.sidebar import render_sidebar

# ────────────────────────────────────────────────────────────────────
# page config + sidebar
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="crstlmeth", page_icon=":material/home:")
render_sidebar()

# ────────────────────────────────────────────────────────────────────
# header with logo + title
# ────────────────────────────────────────────────────────────────────
LOGO = Path(__file__).parent.parent / "assets" / "logo.svg"
col_logo, col_txt = st.columns([1, 3], gap="medium")

with col_logo:
    if LOGO.exists():
        st.image(str(LOGO), width=160)

with col_txt:
    st.title("crstlmeth")
    st.markdown(
        "crstlmeth is a toolkit for analysis, computation and visualization of **bedmethyl** data."
    )

# ────────────────────────────────────────────────────────────────────
# short workflow guidance
# ────────────────────────────────────────────────────────────────────
st.markdown(
    """
crstlmeth setup and workflow in three steps:
1. set folders below (data, cmeth, custom regions, output).
2. use the **references page** to create or view `.cmeth` cohort files.
3. use the **analyze page** to draw methylation and copy-number figures from references and targets.

tips: hover the input labels for extra hints; the sidebar shows a live overview of discovered files.
"""
)

st.divider()

# ────────────────────────────────────────────────────────────────────
# folder selectors (persisted in session_state)
# ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(
        "folders",
        help="set input and output folders; click scan to update the dashboard",
    )

    c1, c2 = st.columns([1, 1], gap="large")
    c3, c4 = st.columns([1, 1], gap="large")

    with c1:
        st.text_input(
            "data directory",
            value=st.session_state.get("data_dir", ""),
            help="root with bgzipped + indexed *.bedmethyl.gz files; subfolders are allowed",
            key="data_dir",
        )
    with c2:
        st.text_input(
            "cmeth reference folder",
            value=st.session_state.get("ref_dir", ""),
            help="folder containing *.cmeth files (full or aggregated)",
            key="ref_dir",
        )
    with c3:
        st.text_input(
            "custom regions folder",
            value=st.session_state.get("region_dir", ""),
            help="optional folder with extra *.bed files defining intervals",
            key="region_dir",
        )
    with c4:
        st.text_input(
            "output directory",
            value=st.session_state.get("outdir", ""),
            help="destination for generated figures and exports (required)",
            key="outdir",
        )

    scan_clicked = st.button(
        "scan folders", type="primary", use_container_width=True
    )


# ────────────────────────────────────────────────────────────────────
# scan on demand (or first load if nothing cached yet)
# ────────────────────────────────────────────────────────────────────
def _resolve_or_none(p: str) -> Path | None:
    p = (p or "").strip()
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return None


def _maybe_scan() -> None:
    """perform scans only if outdir is valid; update session_state"""
    data_path = _resolve_or_none(st.session_state.get("data_dir", ""))
    ref_path = _resolve_or_none(st.session_state.get("ref_dir", ""))
    region_path = _resolve_or_none(st.session_state.get("region_dir", ""))
    out_path = _resolve_or_none(st.session_state.get("outdir", ""))

    if out_path is None:
        # hard fail: outdir is required
        st.error("Please set an **output directory** before scanning.")
        # ensure we don't leave a stale resolved path in the session
        st.session_state.pop("outdir_resolved", None)
        return

    # create output folder if it doesn't exist
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create output directory: {e}")
        st.session_state.pop("outdir_resolved", None)
        return

    with st.spinner("scanning folders …"):
        bed_by_sample = scan_bedmethyl(data_path) if data_path else {}
        cmeth_files = [
            str(p) for p in (scan_cmeth(ref_path) if ref_path else [])
        ]
        custom_beds = [
            str(p)
            for p in (scan_region_beds(region_path) if region_path else [])
        ]

    st.session_state["bed_by_sample"] = bed_by_sample
    st.session_state["cmeth_files"] = cmeth_files
    st.session_state["custom_beds"] = custom_beds
    st.session_state["outdir_resolved"] = str(out_path)


# only scan when the user clicks; on first load do nothing until outdir is set
if scan_clicked:
    _maybe_scan()

# pull cached discoveries
bed_by_sample = st.session_state.get("bed_by_sample", {})
cmeth_files = st.session_state.get("cmeth_files", [])
custom_beds = st.session_state.get("custom_beds", [])
outdir_resolved = (
    Path(st.session_state.get("outdir_resolved", ""))
    if st.session_state.get("outdir_resolved")
    else None
)


# ────────────────────────────────────────────────────────────────────
# counts + completeness buckets
# ────────────────────────────────────────────────────────────────────
def _has(d: dict, key: str) -> bool:
    return key in d and d[key] is not None


n_samples = len(bed_by_sample)
n_bed_files = sum(len(h) for h in bed_by_sample.values())
n_cmeth = len(cmeth_files)
n_beds = len(custom_beds)

n_hap_resolved = sum(
    1 for d in bed_by_sample.values() if _has(d, "1") and _has(d, "2")
)
n_pooled_only = max(0, n_samples - n_hap_resolved)

# ────────────────────────────────────────────────────────────────────
# dashboard
# ────────────────────────────────────────────────────────────────────

# top status bar
with st.container(border=True):
    ok_data = n_bed_files > 0
    ok_ref = n_cmeth > 0
    ok_regions = n_beds > 0
    ok_outdir = outdir_resolved is not None

    icon_ok = ":material/check_circle:"
    icon_no = ":material/cancel:"

    s1, s2, s3, s4 = st.columns([1, 1, 1, 1], gap="large")
    with s1:
        st.markdown(
            f"**output**  \n{icon_ok if ok_outdir else icon_no} "
            f"{'ready' if ok_outdir else 'missing'}",
            help="required for writing figures and references",
        )
    with s2:
        st.markdown(
            f"**data**  \n{icon_ok if ok_data else icon_no} "
            f"{'ready' if ok_data else 'not found'}",
            help="bedmethyl inputs discovered",
        )
    with s3:
        st.markdown(
            f"**references**  \n{icon_ok if ok_ref else icon_no} "
            f"{'ready' if ok_ref else 'none'}",
            help=".cmeth files available",
        )
    with s4:
        st.markdown(
            f"**regions**  \n{icon_ok if ok_regions else icon_no} "
            f"{'custom beds' if ok_regions else 'optional'}",
            help="custom region BEDs (optional)",
        )

st.write("")  # breathing room

# row: haplotype-resolved vs pooled-only + total files
r1c1, r1c2, r1c3 = st.columns([1, 1, 1], gap="large")
with r1c1:
    box = st.container(border=True)
    box.metric("haplotype-resolved", f"{n_hap_resolved}")
    box.caption(
        "samples with both hap1 and hap2", help="ungrouped may be missing"
    )
with r1c2:
    box = st.container(border=True)
    box.metric("pooled-only", f"{n_pooled_only}")
    box.caption(
        "missing either hap1 or hap2", help="treated as pooled for analysis"
    )
with r1c3:
    box = st.container(border=True)
    box.metric("bedmethyl files", f"{n_bed_files}")
    box.caption("all discovered *.bedmethyl.gz")

st.write("")  # breathing room

# separate lines: cmeth references and custom regions
with st.container(border=True):
    st.metric("cmeth references", f"{n_cmeth}")
    st.caption(
        "full or aggregated .cmeth", help="set 'cmeth reference folder' above"
    )

with st.container(border=True):
    st.metric("custom region beds", f"{n_beds}")
    st.caption("optional extra *.bed", help="set 'custom regions folder' above")

# next steps
st.markdown(
    """
**next steps**
- go to **references** to create aggregated or full `.cmeth` files.
- go to **analyze** to produce methylation and copy-number figures using your selections.
"""
)
