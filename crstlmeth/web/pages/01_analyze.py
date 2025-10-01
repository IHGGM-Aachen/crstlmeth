"""
crstlmeth.web.pages.01_analyze

plot methylation and copy number
"""

from __future__ import annotations

import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.plot import plot as plot_group
from crstlmeth.core.discovery import scan_bedmethyl
from crstlmeth.core.methylation import Methylation
from crstlmeth.core.references import read_cmeth
from crstlmeth.core.regions import load_intervals
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.utils import list_builtin_kits, list_bundled_refs

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - analyze", page_icon=":material/analytics:"
)
st.title("analyze")
render_sidebar()

# session + paths
cmeth_files: list[str] = st.session_state.get("cmeth_files", [])
orig_bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.get(
    "bed_by_sample", {}
)

# stable session id + session temp outdir (under data_dir if set, else system tmp)
session_id = st.session_state.get("session_id") or uuid.uuid4().hex[:8]
st.session_state["session_id"] = session_id
base_tmp = Path(st.session_state.get("data_dir") or tempfile.gettempdir())
out_dir = base_tmp / "crstlmeth_out" / session_id
out_dir.mkdir(parents=True, exist_ok=True)
st.session_state["outdir_resolved"] = str(out_dir)  # keep other pages happy

# log file env (optional)
default_log = Path(
    st.session_state.get("log_file", Path.cwd() / "crstlmeth.log.tsv")
)
os.environ.setdefault("CRSTLMETH_LOGFILE", str(default_log))


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def _save_uploads(files: list, dest_dir: Path) -> list[Path]:
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for up in files:
        outp = dest_dir / Path(up.name).name
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


def _make_grouped_choices(
    bundled: Dict[str, Path],
    external: List[str],
    bundled_tag: str,
    external_tag: str,
) -> List[Tuple[str, Path]]:
    """
    Build a single select list with clear labels, e.g.
      'bundled · 030.cmeth' or 'external · mycohort.cmeth'
    Returns list of (label, path).
    """
    rows: List[Tuple[str, Path]] = []
    for _k, p in sorted(bundled.items(), key=lambda kv: kv[0].lower()):
        rows.append((f"{bundled_tag} · {p.name}", p))
    for p in sorted(
        (Path(x) for x in external), key=lambda pp: pp.name.lower()
    ):
        rows.append((f"{external_tag} · {p.name}", p))
    return rows


def _diagnose_hap_coverage(parts: Dict[str, Path], kit_args: List[str]) -> str:
    """
    Quick diagnostic when hap-plot fails: check if hap1/hap2 have any finite
    methylation values across the chosen intervals.
    Returns a small text table as a string.
    """
    # resolve intervals from kit_args (either --kit ID or --bed /path)
    if kit_args[0] == "--kit":
        bed_id = kit_args[1]
    else:
        bed_id = Path(kit_args[1])
    try:
        intervals, region_names = load_intervals(bed_id)
    except Exception as e:
        return f"Failed to load intervals for diagnostics: {e}"

    paths = []
    labels = []
    if parts.get("1"):
        paths.append(parts["1"])
        labels.append("hap1")
    if parts.get("2"):
        paths.append(parts["2"])
        labels.append("hap2")

    if not paths:
        return "No hap1/hap2 files available to diagnose."

    try:
        levels = Methylation.get_levels(
            paths, intervals
        )  # shape: (n_files, n_regions)
    except Exception as e:
        return f"Failed to compute methylation levels for diagnostics: {e}"

    lines = ["diagnostic (finite values per hap across regions):"]
    for i, lab in enumerate(labels):
        vals = levels[i]
        finite = np.isfinite(vals)
        lines.append(f"  {lab}: {finite.sum()} / {finite.size} regions finite")
    # show a few region names with no finite values for quick insight
    idx_all_nan = np.where(~np.isfinite(levels).any(axis=0))[0].tolist()
    if idx_all_nan:
        preview = ", ".join(region_names[j] for j in idx_all_nan[:10])
        more = " …" if len(idx_all_nan) > 10 else ""
        lines.append(
            f"  regions with no finite in any hap: {len(idx_all_nan)} ({preview}{more})"
        )
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# reference + regions (single selectors with labeled entries)
# ────────────────────────────────────────────────────────────────────
left, right = st.columns([0.6, 0.4], gap="large")

with left:
    # references: bundled (default) + external
    bundled_refs = list_bundled_refs()
    ref_choices = _make_grouped_choices(
        bundled_refs, cmeth_files, "bundled", "external"
    )
    if not ref_choices:
        st.error("No references available (bundled or external).")
        st.stop()

    ref_label_default, ref_path_default = ref_choices[0]
    ref_label = st.selectbox(
        "reference (.cmeth)",
        options=[lbl for lbl, _ in ref_choices],
        index=0,
        help="Bundled references are shipped with the package; external can be set on home page.",
    )
    # map selected label back to path
    cm_ref_path = dict(ref_choices)[ref_label]

    # parse mode (safe)
    try:
        _, meta = read_cmeth(Path(cm_ref_path))
        ref_mode = str(meta.get("mode", "aggregated")).lower()
    except Exception as e:
        ref_mode = "unknown"
        st.warning(f"Could not parse reference metadata ({e}). Proceeding.")

    # regions: bundled kits + custom beds
    builtin_kits = list_builtin_kits()
    custom_beds = st.session_state.get("custom_beds", [])
    # Build a single choice list
    bed_choices: List[Tuple[str, str | Path]] = []
    for k in sorted(builtin_kits.keys()):
        bed_choices.append((f"bundled kit · {k}", ("--kit", k)))
    for b in sorted(
        (Path(x) for x in custom_beds), key=lambda pp: pp.name.lower()
    ):
        bed_choices.append((f"external BED · {b.name}", ("--bed", b)))

    if not bed_choices:
        st.error("No region definitions found (bundled kits or custom BEDs).")
        st.stop()

    bed_label = st.selectbox(
        "regions",
        options=[lbl for lbl, _ in bed_choices],
        index=0,
        help="Choose a bundled MLPA kit or a discovered custom BED (set on home page).",
    )
    selected_flag, selected_val = dict(bed_choices)[bed_label]
    # turn into CLI args
    kit_args: List[str] = [selected_flag, str(selected_val)]
    region_label = bed_label.split("·", 1)[1].strip()

with right:
    st.markdown(
        f"**reference:** `{Path(cm_ref_path).name}`  \n**mode:** `{ref_mode}`  \n**regions:** `{region_label}`"
    )

st.divider()

# ────────────────────────────────────────────────────────────────────
# targets – discovered + uploads
# ────────────────────────────────────────────────────────────────────
st.subheader("targets")

up_col, pick_col = st.columns([0.55, 0.45], gap="large")

with up_col:
    st.markdown("**upload bedMethyl**", help="Upload .bedmethyl.gz (+ .tbi).")
    uploads = st.file_uploader(
        "drop .bedmethyl.gz and .tbi here",
        type=["gz", "tbi"],
        accept_multiple_files=True,
        help="For each .bedmethyl.gz a matching .tbi is recommended.",
    )
    uploaded_map: Dict[str, Dict[str, Path]] = {}
    if uploads:
        up_dir = out_dir / "uploads"
        _ = _save_uploads(uploads, up_dir)
        uploaded_map = scan_bedmethyl(up_dir)

with pick_col:
    bed_by_sample: Dict[str, Dict[str, Path]] = _combine_bed_maps(
        orig_bed_by_sample, uploaded_map
    )
    if not bed_by_sample:
        st.warning(
            "No bgzipped & indexed bedmethyl files found - set paths on home or upload above."
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
)

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")
with mcol1:
    meth_pooled_png = st.text_input(
        "pooled output", value="methylation_pooled.png"
    )
with mcol2:
    meth_h1_png = st.text_input("hap1 output", value="methylation_hap1.png")
    meth_h2_png = st.text_input("hap2 output", value="methylation_hap2.png")

go_meth = st.button(
    "plot methylation", type="primary", use_container_width=True
)

if go_meth:
    if not picked:
        st.error("Select at least one target sample.")
        st.stop()

    # pooled plot
    pooled_argv = [
        "methylation",
        "--cmeth",
        str(cm_ref_path),
        *kit_args,
        "--out",
        str(out_dir / meth_pooled_png),
    ]
    for sid in picked:
        parts = bed_by_sample.get(sid, {})
        for key in ("1", "2", "ungrouped"):
            p = parts.get(key)
            if p:
                pooled_argv.append(str(p))

    with st.expander("pooled – CLI argv", expanded=False):
        st.code(" ".join(map(str, pooled_argv)), language="bash")

    code, out_png, stdout = _cli_plot(pooled_argv)

    if code == 0 and out_png.exists():
        st.success(f"Pooled figure → {out_png}")
        st.image(
            str(out_png),
            use_container_width=True,
            caption="Methylation (pooled)",
        )
        st.download_button(
            "download pooled PNG",
            data=out_png.read_bytes(),
            file_name=out_png.name,
            mime="image/png",
        )
    else:
        st.error(f"Pooled methylation plotting failed (exit {code})")
    if stdout.strip():
        with st.expander("pooled – CLI stdout/stderr", expanded=False):
            st.code(stdout, language="bash")

    # hap series
    if mode_choice.startswith("Haplotype"):
        if len(picked) != 1:
            st.error("Haplotype series requires exactly **one** target sample.")
            st.stop()
        sid = picked[0]
        parts = bed_by_sample.get(sid, {})
        if not (parts.get("1") and parts.get("2")):
            st.error(f"Sample `{sid}` is missing either `_1` or `_2` file.")
            st.stop()

        # hap1
        h1_argv = [
            "methylation",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / meth_h1_png),
            "--hap-ref-plot",
            "--ref-hap",
            "1",
            "--auto-hap-match",
            str(parts["1"]),
            str(parts["2"]),
        ]
        with st.expander("hap1 – CLI argv", expanded=True):
            st.code(" ".join(map(str, h1_argv)), language="bash")
        code1, out_h1, stdout1 = _cli_plot(h1_argv)

        if code1 == 0 and out_h1.exists():
            st.success(f"Hap1 plot → {out_h1}")
            st.image(str(out_h1), use_container_width=True)
            st.download_button(
                "download hap1 PNG",
                data=out_h1.read_bytes(),
                file_name=out_h1.name,
                mime="image/png",
            )
        else:
            st.error(f"Hap1 plot failed (exit {code1})")
            diag = _diagnose_hap_coverage(parts, kit_args)
            with st.expander("hap1 – diagnostics", expanded=True):
                st.code(diag, language="text")
        if stdout1.strip():
            with st.expander("hap1 – CLI stdout/stderr", expanded=False):
                st.code(stdout1, language="bash")

        # hap2
        h2_argv = [
            "methylation",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / meth_h2_png),
            "--hap-ref-plot",
            "--ref-hap",
            "2",
            "--auto-hap-match",
            str(parts["1"]),
            str(parts["2"]),
        ]
        with st.expander("hap2 – CLI argv", expanded=True):
            st.code(" ".join(map(str, h2_argv)), language="bash")
        code2, out_h2, stdout2 = _cli_plot(h2_argv)

        if code2 == 0 and out_h2.exists():
            st.success(f"Hap2 plot → {out_h2}")
            st.image(str(out_h2), use_container_width=True)
            st.download_button(
                "download hap2 PNG",
                data=out_h2.read_bytes(),
                file_name=out_h2.name,
                mime="image/png",
            )
        else:
            st.error(f"Hap2 plot failed (exit {code2})")
            diag = _diagnose_hap_coverage(parts, kit_args)
            with st.expander("hap2 – diagnostics", expanded=True):
                st.code(diag, language="text")
        if stdout2.strip():
            with st.expander("hap2 – CLI stdout/stderr", expanded=False):
                st.code(stdout2, language="bash")

st.divider()

# ────────────────────────────────────────────────────────────────────
# copy number
# ────────────────────────────────────────────────────────────────────
st.subheader("copy number")

c1, c2 = st.columns([0.6, 0.4], gap="large")
with c1:
    st.caption("Supports full and aggregated references.")
with c2:
    cn_png = st.text_input(
        "copy-number output", value="copy_number.png", key="cn_png_name"
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
        str(cm_ref_path),
        *kit_args,
        "--out",
        str(out_dir / cn_png),
    ]
    for sid in picked:
        parts = bed_by_sample.get(sid, {})
        for key in ("1", "2", "ungrouped"):
            p = parts.get(key)
            if p:
                argv.append(str(p))

    with st.expander("copy-number – CLI argv", expanded=False):
        st.code(" ".join(map(str, argv)), language="bash")

    res = CliRunner().invoke(plot_group, argv, catch_exceptions=True)
    out_png = out_dir / cn_png
    if res.exit_code == 0 and out_png.exists():
        st.success(f"Figure → {out_png}")
        st.image(
            str(out_png),
            use_container_width=True,
            caption="Copy number (log2 ratio)",
        )
        st.download_button(
            "download CN PNG",
            data=out_png.read_bytes(),
            file_name=out_png.name,
            mime="image/png",
        )
    else:
        st.error(f"Copy-number plotting failed (exit {res.exit_code})")
    if res.output.strip():
        with st.expander("copy number – CLI stdout/stderr"):
            st.code(res.output, language="bash")
    if res.exception:
        with st.expander("traceback"):
            st.code(
                "".join(traceback.format_exception(res.exception)),
                language="python",
            )
