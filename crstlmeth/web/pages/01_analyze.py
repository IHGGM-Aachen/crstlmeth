"""
crstlmeth.web.pages.01_analyze

plot methylation and copy number
"""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Dict

import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.plot import plot as plot_group
from crstlmeth.core.discovery import scan_bedmethyl
from crstlmeth.core.references import read_cmeth
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.utils import list_builtin_kits

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - analyze", page_icon=":material/analytics:"
)
st.title("analyze")
render_sidebar()

# shared state (from Home/Sidebar)
cmeth_files: list[str] = st.session_state.get("cmeth_files", [])
orig_bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.get(
    "bed_by_sample", {}
)
out_dir_str = st.session_state.get("outdir_resolved", "")
default_log = Path(
    st.session_state.get("log_file", Path.cwd() / "crstlmeth.log.tsv")
)
os.environ.setdefault("CRSTLMETH_LOGFILE", str(default_log))

# basic guards
if not out_dir_str:
    st.error("Set an **output directory** on the Home page first.")
    st.stop()
out_dir = Path(out_dir_str)

if not cmeth_files:
    st.warning("No .cmeth reference discovered - set folder in the Home page.")
    st.stop()


# ────────────────────────────────────────────────────────────────────
# helper: persist uploaded files
# ────────────────────────────────────────────────────────────────────
def _save_uploads(files: list, dest_dir: Path) -> list[Path]:
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for up in files:
        fname = Path(up.name).name
        outp = dest_dir / fname
        with outp.open("wb") as fh:
            shutil.copyfileobj(up, fh)
        saved.append(outp.resolve())
    return saved


def _combine_bed_maps(
    a: Dict[str, Dict[str, Path]], b: Dict[str, Dict[str, Path]]
) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {k: dict(v) for k, v in a.items()}
    for sid, parts in b.items():
        out.setdefault(sid, {})
        out[sid].update(parts)
    return out


def _cli_plot(argv: list[str]) -> tuple[int, Path, str]:
    """Run CLI group with argv and return (exit, out_path, combined_output)."""
    res = CliRunner().invoke(plot_group, argv, catch_exceptions=True)
    out_idx = argv.index("--out") + 1 if "--out" in argv else -1
    out_png = Path(argv[out_idx]) if out_idx > 0 else out_dir / "figure.png"
    out_text = res.output or ""
    if res.exception:
        out_text += "\n" + "".join(traceback.format_exception(res.exception))
    return res.exit_code, out_png, out_text


# ────────────────────────────────────────────────────────────────────
# selectors – reference, kit, targets (+ upload)
# ────────────────────────────────────────────────────────────────────
left, right = st.columns([0.6, 0.4], gap="large")

with left:
    cm_ref = st.selectbox(
        "reference (.cmeth)",
        [str(p) for p in cmeth_files],
        format_func=lambda p: Path(p).name,
        help="Pick the cohort reference file (.cmeth).",
    )

    # detect reference mode
    try:
        _, meta = read_cmeth(Path(cm_ref))
        ref_mode = str(meta.get("mode", "aggregated")).lower()
    except Exception as e:
        st.error(f"Failed to parse reference: {e}")
        st.stop()

    builtin_kits = list_builtin_kits()
    custom_beds = st.session_state.get("custom_beds", [])
    kit_or_bed = st.selectbox(
        "MLPA kit / BED",
        list(builtin_kits) + custom_beds,
        help="Select built-in MLPA kit or a custom BED defining intervals.",
    )

with right:
    st.markdown(
        f"**reference mode:** `{ref_mode}`",
        help=(
            "full: cohort boxes from per-sample rows. "
            "aggregated: quantiles (median/IQR/95%) only; anonymized and shareable."
        ),
    )

st.divider()

# ────────────────────────────────────────────────────────────────────
# targets – discovered + uploads
# ────────────────────────────────────────────────────────────────────
st.subheader("targets")

up_col, pick_col = st.columns([0.55, 0.45], gap="large")

with up_col:
    st.markdown(
        "**upload bedMethyl**",
        help="Upload .bedmethyl.gz and its matching .tbi (index). You can upload multiple files.",
    )
    uploads = st.file_uploader(
        "drop .bedmethyl.gz and .tbi here",
        type=["gz", "tbi"],
        accept_multiple_files=True,
        help="Both the .bedmethyl.gz and .tbi must be present for each file.",
    )

    # persist uploads under OUTDIR/uploads/<session-id>/
    uploaded_map: Dict[str, Dict[str, Path]] = {}
    if uploads:
        session_id = st.session_state.get("session_id", "tmp")
        up_dir = out_dir / "uploads" / session_id
        saved = _save_uploads(uploads, up_dir)
        # scan to register
        uploaded_map = scan_bedmethyl(up_dir)

with pick_col:
    # combine discovered + uploaded
    bed_by_sample: Dict[str, Dict[str, Path]] = _combine_bed_maps(
        orig_bed_by_sample, uploaded_map
    )
    if not bed_by_sample:
        st.warning(
            "No bgzipped & indexed bedmethyl files found – adjust paths on Home, or upload above."
        )
        st.stop()

    sample_ids = sorted(bed_by_sample)
    picked = st.multiselect(
        "target samples",
        sample_ids,
        help="For haplotype series, pick exactly one sample with both _1 and _2.",
    )

st.divider()

# ────────────────────────────────────────────────────────────────────
# methylation – pooled and hap-aware series
# ────────────────────────────────────────────────────────────────────
st.subheader("methylation")

mode_choice = st.radio(
    "plot mode",
    options=["Pooled only", "Haplotype series (pooled + hap1 + hap2)"],
    index=0,
    help=(
        "Pooled only: one plot using pooled cohort/reference. "
        "Haplotype series: three plots — pooled, and each target hap "
        "mapped to the closest reference hap using robust auto-matching."
    ),
)

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")
with mcol1:
    meth_pooled_png = st.text_input(
        "pooled output",
        value="methylation_pooled.png",
        help="Filename under the output directory.",
    )
with mcol2:
    meth_h1_png = st.text_input(
        "hap1 output",
        value="methylation_hap1.png",
        help="Filename for hap1-vs-ref hap plot.",
    )
    meth_h2_png = st.text_input(
        "hap2 output",
        value="methylation_hap2.png",
        help="Filename for hap2-vs-ref hap plot.",
    )

go_meth = st.button(
    "plot methylation", type="primary", use_container_width=True
)

if go_meth:
    if not picked:
        st.error("Select at least one target sample.")
        st.stop()

    # ── pooled plot (always) --------------------------------------------------
    pooled_argv = [
        "methylation",
        "--cmeth",
        cm_ref,
        "--kit",
        str(kit_or_bed),
        "--out",
        str(out_dir / meth_pooled_png),
    ]
    for sid in picked:
        parts = bed_by_sample.get(sid, {})
        for key in ("1", "2", "ungrouped"):
            p = parts.get(key)
            if p:
                pooled_argv.append(str(p))

    with st.spinner("Rendering pooled methylation …"):
        code, out_png, stdout = _cli_plot(pooled_argv)

    if code == 0 and out_png.exists():
        st.success(f"Pooled figure written → {out_png}")
        st.image(
            str(out_png),
            use_container_width=True,
            caption="Methylation (pooled)",
        )
    else:
        st.error(f"Pooled methylation plotting failed (exit {code})")
    if stdout.strip():
        with st.expander("pooled - CLI output"):
            st.code(stdout, language="bash")

    # ── haplotype series (optional) ------------------------------------------
    if mode_choice.startswith("Haplotype"):
        if len(picked) != 1:
            st.error("Haplotype series requires exactly **one** target sample.")
            st.stop()
        sid = picked[0]
        parts = bed_by_sample.get(sid, {})
        if not (parts.get("1") and parts.get("2")):
            st.error(f"Sample `{sid}` is missing either `_1` or `_2` file.")
            st.stop()

        # hap1 plot (ref-hap=1)
        h1_argv = [
            "methylation",
            "--cmeth",
            cm_ref,
            "--kit",
            str(kit_or_bed),
            "--out",
            str(out_dir / meth_h1_png),
            "--hap-ref-plot",
            "--ref-hap",
            "1",
            "--auto-hap-match",  # ← correct flag (was --auto-map)
            str(parts["1"]),
            str(parts["2"]),
        ]
        with st.spinner("Rendering hap1 ↔ ref-hap1 …"):
            code1, out_h1, stdout1 = _cli_plot(h1_argv)

        # hap2 plot (ref-hap=2)
        h2_argv = [
            "methylation",
            "--cmeth",
            cm_ref,
            "--kit",
            str(kit_or_bed),
            "--out",
            str(out_dir / meth_h2_png),
            "--hap-ref-plot",
            "--ref-hap",
            "2",
            "--auto-hap-match",  # ← correct flag (was --auto-map)
            str(parts["1"]),
            str(parts["2"]),
        ]
        with st.spinner("Rendering hap2 ↔ ref-hap2 …"):
            code2, out_h2, stdout2 = _cli_plot(h2_argv)

        if code1 == 0 and out_h1.exists():
            st.success(f"Hap1 plot written → {out_h1}")
            st.image(
                str(out_h1),
                use_container_width=True,
                caption=f"Methylation – {sid} hap mapped to ref-hap1",
            )
        else:
            st.error(f"Hap1 plot failed (exit {code1})")
        if stdout1.strip():
            with st.expander("hap1 – CLI output"):
                st.code(stdout1, language="bash")

        if code2 == 0 and out_h2.exists():
            st.success(f"Hap2 plot written → {out_h2}")
            st.image(
                str(out_h2),
                use_container_width=True,
                caption=f"Methylation – {sid} hap mapped to ref-hap2",
            )
        else:
            st.error(f"Hap2 plot failed (exit {code2})")
        if stdout2.strip():
            with st.expander("hap2 – CLI output"):
                st.code(stdout2, language="bash")

st.divider()

# ────────────────────────────────────────────────────────────────────
# copy number
# ────────────────────────────────────────────────────────────────────
st.subheader("copy number")

c1, c2 = st.columns([0.6, 0.4], gap="large")
with c1:
    st.caption(
        "Supports full and aggregated references.",
        help="full: cohort log2 boxes; aggregated: quantile boxes (median/IQR/95%) in log2 space.",
    )
with c2:
    cn_png = st.text_input(
        "copy-number output",
        value="copy_number.png",
        key="cn_png_name",
        help="Filename under the output directory.",
    )

go_cn = st.button(
    "plot copy number", type="secondary", use_container_width=True
)

if go_cn:
    if not picked:
        st.error("Select at least one target sample.")
        st.stop()

    argv = [
        "copynumber",
        "--cmeth",
        cm_ref,
        "--kit",
        str(kit_or_bed),
        "--out",
        str(out_dir / cn_png),
    ]
    for sid in picked:
        parts = bed_by_sample.get(sid, {})
        for key in ("1", "2", "ungrouped"):
            p = parts.get(key)
            if p:
                argv.append(str(p))

    with st.spinner("Rendering copy number …"):
        res = CliRunner().invoke(plot_group, argv, catch_exceptions=True)

    out_png = out_dir / cn_png
    if res.exit_code == 0 and out_png.exists():
        st.success(f"Figure written → {out_png}")
        st.image(
            str(out_png),
            use_container_width=True,
            caption="Copy number (log2 ratio)",
        )
    else:
        st.error(f"Copy-number plotting failed (exit {res.exit_code})")

    if res.output.strip():
        with st.expander("copy number – CLI output"):
            st.code(res.output, language="bash")
    if res.exception:
        with st.expander("traceback"):
            st.code(
                "".join(traceback.format_exception(res.exception)),
                language="python",
            )
