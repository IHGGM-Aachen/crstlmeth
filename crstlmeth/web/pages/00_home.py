"""
crstlmeth.web.pages.00_home
---------------------------
landing page with folder setup and discovery of input files
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from crstlmeth.core.discovery import scan_bedmethyl, scan_cmeth, scan_region_beds
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state

# ────────────────────────────────────────────────────────────────────
# page config + state
# ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="crstlmeth", page_icon=":material/home:")

ensure_web_state()

# ────────────────────────────────────────────────────────────────────
# scan helpers
# ────────────────────────────────────────────────────────────────────
def _resolve_or_none(p: str) -> Path | None:
    p = (p or "").strip()
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return None


def _scan() -> None:
    data_path = _resolve_or_none(st.session_state.get("data_dir", ""))
    ref_path = _resolve_or_none(st.session_state.get("ref_dir", ""))
    region_path = _resolve_or_none(st.session_state.get("region_dir", ""))

    with st.spinner("scanning folders …"):
        bed_by_sample = scan_bedmethyl(data_path) if data_path else {}
        cmeth_files = [str(p) for p in (scan_cmeth(ref_path) if ref_path else [])]
        custom_beds = [
            str(p) for p in (scan_region_beds(region_path) if region_path else [])
        ]

    st.session_state["bed_by_sample"] = bed_by_sample
    st.session_state["cmeth_files"] = cmeth_files
    st.session_state["custom_beds"] = custom_beds


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
        "crstlmeth is a toolkit for analysis and visualization of **bedmethyl** data."
    )

st.markdown(
    """
workflow:
1. set folders (data, references, optional custom regions) and scan.
2. use **analyze** to upload targets and plot (defaults to bundled refs).
3. use **references** to create/view `.cmeth` cohort files.
"""
)

st.divider()

# ────────────────────────────────────────────────────────────────────
# folder selectors (persisted in session_state)
# ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("folders", help="set input folders; click scan to update the dashboard")

    c1, c2 = st.columns([1, 1], gap="large")
    c3 = st.columns([1])[0]

    with c1:
        st.text_input(
            "data directory",
            key="data_dir",
            help="root with bgzipped + indexed *.bedmethyl.gz files; subfolders allowed",
        )
    with c2:
        st.text_input(
            "cmeth reference folder",
            key="ref_dir",
            help="folder containing *.cmeth files",
        )
    with c3:
        st.text_input(
            "custom regions folder",
            key="region_dir",
            help="optional folder with extra *.bed defining intervals",
        )

    scan_clicked = st.button("scan folders", type="primary", use_container_width=True)

# ────────────────────────────────────────────────────────────────────
# auto-scan once when config prefilled paths
# ────────────────────────────────────────────────────────────────────
st.session_state.setdefault("_home_auto_scanned", False)

has_any_folder = any(
    (st.session_state.get(k, "") or "").strip()
    for k in ("data_dir", "ref_dir", "region_dir")
)

discoveries_empty = (
    not st.session_state.get("bed_by_sample")
    and not st.session_state.get("cmeth_files")
    and not st.session_state.get("custom_beds")
)

auto_scan = (
    st.session_state.get("_has_config", False)
    and has_any_folder
    and discoveries_empty
    and not st.session_state["_home_auto_scanned"]
)

if scan_clicked or auto_scan:
    _scan()
    st.session_state["_home_auto_scanned"] = True
    if auto_scan:
        st.toast("Scanned folders from config defaults", icon="✅")

# IMPORTANT: render sidebar AFTER scanning so it reflects updated session_state
render_sidebar()

# pull cached discoveries
bed_by_sample = st.session_state.setdefault("bed_by_sample", {})
cmeth_files = st.session_state.setdefault("cmeth_files", [])
custom_beds = st.session_state.setdefault("custom_beds", [])

# ────────────────────────────────────────────────────────────────────
# counts + dashboard
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

with st.container(border=True):
    icon_ok = ":material/check_circle:"
    icon_no = ":material/cancel:"

    s1, s2, s3 = st.columns([1, 1, 1], gap="large")
    with s1:
        st.markdown(
            f"**data**  \n{icon_ok if n_bed_files > 0 else icon_no} "
            f"{'ready' if n_bed_files > 0 else 'not found'}",
            help="bedmethyl inputs discovered",
        )
    with s2:
        st.markdown(
            f"**references**  \n{icon_ok if n_cmeth > 0 else icon_no} "
            f"{'ready' if n_cmeth > 0 else 'none'}",
            help=".cmeth files available",
        )
    with s3:
        st.markdown(
            f"**regions**  \n{icon_ok if n_beds > 0 else icon_no} "
            f"{'custom beds' if n_beds > 0 else 'optional'}",
            help="custom region BEDs (optional)",
        )

r1c1, r1c2, r1c3 = st.columns([1, 1, 1], gap="large")
with r1c1:
    box = st.container(border=True)
    box.metric("haplotype-resolved", f"{n_hap_resolved}")
    box.caption("samples with both hap1 and hap2", help="ungrouped may be missing")
with r1c2:
    box = st.container(border=True)
    box.metric("pooled-only", f"{n_pooled_only}")
    box.caption("missing either hap1 or hap2", help="treated as pooled for analysis")
with r1c3:
    box = st.container(border=True)
    box.metric("bedmethyl files", f"{n_bed_files}")
    box.caption("all discovered *.bedmethyl.gz")

with st.container(border=True):
    st.metric("cmeth references", f"{n_cmeth}")
    st.caption("full or aggregated .cmeth", help="set 'cmeth reference folder' above")

with st.container(border=True):
    st.metric("custom region beds", f"{n_beds}")
    st.caption("optional extra *.bed", help="set 'custom regions folder' above")

st.markdown(
    """
**next steps**
- go to **analyze** to upload targets and plot (bundled refs by default).
- go to **references** to create aggregated or full `.cmeth` files.
"""
)
