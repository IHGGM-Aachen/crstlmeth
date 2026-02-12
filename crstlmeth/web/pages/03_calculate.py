"""
crstlmeth.web.pages.03_calculate

streamlit interface providing different calculation methods.
"""

from __future__ import annotations

import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from crstlmeth.core.methylation import Methylation
from crstlmeth.core.parsers import query_bedmethyl
from crstlmeth.core.regions import load_intervals
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import list_builtin_kits

# ────────────────────────────────────────────────────────────────────
# page configuration
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - calculate", page_icon=":material/calculate:"
)

ensure_web_state()

st.title("calculate")
render_sidebar()

session_id: str = st.session_state["session_id"]
out_dir = resolve_outdir(session_id)

# discoveries/config
bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.setdefault("bed_by_sample", {})
custom_beds: List[str] = st.session_state.setdefault("custom_beds", [])
tabix_bin: str = (st.session_state.get("tabix_bin") or "tabix").strip()
default_kit: str = (st.session_state.get("default_kit") or "ME030").strip()

if not bed_by_sample:
    st.warning(
        "No bgzipped & indexed bedmethyl files found – configure folders on **Home** and click *scan folders*."
    )
    st.stop()


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def _persist(upload: BytesIO) -> Path:
    """Store an uploaded file under OUTDIR/.streamlit_tmp and return its path."""
    tmp = out_dir / ".streamlit_tmp" / upload.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(upload.getbuffer())
    return tmp.resolve()


def _ensure_tabix_index(bgz: Path) -> None:
    """
    Ensure *.tbi exists for a .bedmethyl.gz file using the configured tabix binary.
    No-op if index already exists or file missing.
    """
    if not bgz.exists():
        return
    tbi = Path(str(bgz) + ".tbi")
    if tbi.exists():
        return
    try:
        subprocess.run(
            [tabix_bin, "-f", "-s", "1", "-b", "2", "-e", "3", str(bgz)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        # best-effort; downstream may fail and show an error
        return


def _existing_paths(xs: List[str]) -> list[Path]:
    out: list[Path] = []
    for x in xs:
        try:
            p = Path(x)
            if p.exists():
                out.append(p)
        except Exception:
            continue
    return out


def _all_parts_for_sample(sample_id: str) -> List[Path]:
    """
    Return bedmethyl part paths for a discovered sample, ordered h1,h2,ungrouped if present.
    """
    parts = bed_by_sample.get(sample_id, {})
    ordered: List[Path] = []
    for k in ("1", "2", "ungrouped"):
        p = parts.get(k)
        if p:
            ordered.append(Path(p))
    for k in sorted(set(parts) - {"1", "2", "ungrouped"}):
        if parts.get(k):
            ordered.append(Path(parts[k]))
    return ordered


def _make_bed_choices() -> List[Tuple[str, Tuple[str, str]]]:
    """
    Regions selector entries: bundled kits + external BEDs.
    Returns list of (label, (flag, value)).
    """
    kits = list_builtin_kits()
    beds = _existing_paths(custom_beds)

    rows: List[Tuple[str, Tuple[str, str]]] = []
    for k in sorted(kits.keys()):
        rows.append((f"bundled kit · {k}", ("--kit", k)))
    for b in sorted(beds, key=lambda pp: pp.name.lower()):
        rows.append((f"external BED · {b.name}", ("--bed", str(b))))
    return rows


# ────────────────────────────────────────────────────────────────────
# single-interval spot-check
# ────────────────────────────────────────────────────────────────────
st.subheader("single-interval methylation calculator")

sids = sorted(bed_by_sample)
sid = st.selectbox("sample", sids, key="calc_spot_sid")

part_options = [k for k in ("1", "2", "ungrouped") if k in bed_by_sample.get(sid, {})]
if not part_options:
    st.warning("Selected sample has no hap1/hap2/ungrouped file entries.")
    st.stop()

part_key = st.selectbox(
    "bedmethyl part",
    options=part_options,
    index=0,
    key="calc_spot_part",
    help="Pick hap1, hap2, or ungrouped for this sample.",
)

bed_path = bed_by_sample.get(sid, {}).get(part_key)
if not bed_path:
    st.info("Selected sample has no file for this part.")
    st.stop()

c1, c2, c3 = st.columns(3)
chrom = c1.text_input("chrom", value="chr11", key="calc_spot_chr")
start = c2.number_input("start", min_value=0, value=1_976_000, step=100, key="calc_spot_start")
end = c3.number_input("end", min_value=1, value=1_976_200, step=100, key="calc_spot_end")

if end <= start:
    st.error("end must be greater than start")
elif st.button("fetch", type="primary", key="calc_spot_fetch"):
    try:
        df = query_bedmethyl(str(bed_path), chrom, int(start), int(end))
    except Exception as e:
        st.error(f"query failed: {e}")
        st.stop()

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

bed_choices = _make_bed_choices()
if not bed_choices:
    st.error("No region definitions found (bundled kits or custom BEDs).")
    st.stop()

labels_only = [lbl for lbl, _ in bed_choices]
default_label = f"bundled kit · {default_kit}"
default_index = labels_only.index(default_label) if default_label in labels_only else 0

bed_label = st.selectbox(
    "MLPA kit / BED",
    labels_only,
    index=default_index,
    key="calc_scan_bed_label",
    help="Select a built-in MLPA kit or a custom BED discovered on Home.",
)
bed_flag, bed_val = dict(bed_choices)[bed_label]
kit_or_bed = bed_val  # load_intervals accepts either kit id or a BED path

# optional: also allow uploading a temporary custom BED (kept under OUTDIR)
with st.expander("optional: use a BED not in your configured folders"):
    upl = st.file_uploader("upload BED", type=["bed"], key="calc_scan_bed_upload")
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
        key="calc_scan_tgt_sid",
        help="One target; pooled from its available parts.",
    )
with right:
    ref_sids = st.multiselect(
        "reference samples",
        [s for s in sids if s != tgt_sid],
        key="calc_scan_ref_sids",
        help="Pick ≥1 reference samples for the cohort.",
    )

if not tgt_sid or not ref_sids:
    st.info("Select one **target** and at least one **reference** sample to run the scan.")
    st.stop()

# allow supplementing target/ref with additional uploads (optional)
with st.expander("optional: include extra target/reference files (not discovered)"):
    col_t, col_r = st.columns(2)
    with col_t:
        extra_tgt = st.file_uploader(
            "extra target (.bedmethyl.gz / .tbi)",
            type=["gz", "tbi"],
            accept_multiple_files=True,
            key="calc_scan_extra_tgt",
        )
    with col_r:
        extra_refs = st.file_uploader(
            "extra reference(s) (.bedmethyl.gz / .tbi)",
            type=["gz", "tbi"],
            accept_multiple_files=True,
            key="calc_scan_extra_refs",
        )

# resolve discovered paths
tgt_paths: List[Path] = _all_parts_for_sample(tgt_sid)
ref_paths: List[Path] = [p for sid_ in ref_sids for p in _all_parts_for_sample(sid_)]

# persist extra uploads (store both gz + tbi if provided)
def _collect_extra_bedmethyl(files: list | None) -> list[Path]:
    if not files:
        return []
    persisted = [_persist(f) for f in files]
    # ensure indexing for any gz
    for p in persisted:
        if str(p).endswith(".gz"):
            _ensure_tabix_index(p)
    # return only gz files as inputs to methylation calls
    return [p for p in persisted if str(p).endswith(".gz")]

tgt_paths.extend(_collect_extra_bedmethyl(extra_tgt))
ref_paths.extend(_collect_extra_bedmethyl(extra_refs))

if st.button("run deviation scan", type="primary", use_container_width=True, key="calc_scan_run"):
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

        ref_lv = Methylation.get_levels([str(p) for p in ref_paths], intervals)
        tgt_lv = Methylation.get_levels([str(p) for p in tgt_paths], intervals)

        # pool target across its available parts (if multiple)
        tgt_pool = np.nanmean(tgt_lv, axis=0, keepdims=True)

        mean_ref = np.nanmean(ref_lv, axis=0)
        std_ref = np.nanstd(ref_lv, axis=0, ddof=1)
        std_ref = np.where((~np.isfinite(std_ref)) | (std_ref == 0.0), np.nan, std_ref)

        flags = Methylation.get_deviations(ref_lv, tgt_pool, fdr_alpha=0.05)
        z_scores = (tgt_pool[0] - mean_ref) / std_ref

        table = pd.DataFrame(
            {
                "region": region_names,
                "mean_ref": mean_ref,
                "target": tgt_pool[0],
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
