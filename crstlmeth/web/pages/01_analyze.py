"""
crstlmeth.web.pages.01_analyze

plot methylation and copy number
"""

from __future__ import annotations

import os
import shutil
import traceback
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
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import list_builtin_kits, list_bundled_refs

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - analyze", page_icon=":material/analytics:"
)

ensure_web_state()

st.title("analyze")
render_sidebar()

# stable session id + stable outdir
session_id: str = st.session_state["session_id"]
out_dir = resolve_outdir(session_id)

# log file env (CLI sets this; fall back to local file)
default_log = os.getenv("CRSTLMETH_LOGFILE") or str(
    Path.cwd() / "crstlmeth.log.tsv"
)
os.environ.setdefault("CRSTLMETH_LOGFILE", default_log)

# session + discoveries
cmeth_files: list[str] = st.session_state.setdefault("cmeth_files", [])
orig_bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.setdefault(
    "bed_by_sample", {}
)


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
    rows: List[Tuple[str, Path]] = []
    for _k, p in sorted(bundled.items(), key=lambda kv: kv[0].lower()):
        rows.append((f"{bundled_tag} · {p.name}", p))

    # filter stale external paths
    ext_paths: list[Path] = []
    for x in external:
        try:
            p = Path(x)
            if p.exists():
                ext_paths.append(p)
        except Exception:
            continue

    for p in sorted(ext_paths, key=lambda pp: pp.name.lower()):
        rows.append((f"{external_tag} · {p.name}", p))
    return rows


def _diagnose_hap_coverage(parts: Dict[str, Path], kit_args: List[str]) -> str:
    """
    Quick diagnostic when hap-plot fails: check how many regions have finite
    methylation values for hap1/hap2 (and overall pooled).
    """
    # resolve intervals from kit_args (either --kit ID or --bed /path)
    try:
        if kit_args[0] == "--kit":
            bed_id = kit_args[1]
        else:
            bed_id = Path(kit_args[1])
        intervals, region_names = load_intervals(bed_id)
    except Exception as e:
        return f"Failed to load intervals for diagnostics: {e}"

    # keep only known keys and non-empty paths
    hap_paths = {
        k: v for k, v in parts.items() if k in ("1", "2", "ungrouped") and v
    }
    if not hap_paths:
        return "No hap1/hap2/ungrouped files available to diagnose."

    try:
        h1, h2, overall = Methylation.get_levels_by_haplotype(
            hap_paths, intervals
        )
    except Exception as e:
        return f"Failed to compute methylation levels for diagnostics: {e}"

    lines = ["diagnostic (finite values across regions):"]

    if "1" in hap_paths:
        finite1 = np.isfinite(h1)
        lines.append(f"  hap1: {finite1.sum()} / {finite1.size} regions finite")
    else:
        finite1 = None
        lines.append("  hap1: (missing)")

    if "2" in hap_paths:
        finite2 = np.isfinite(h2)
        lines.append(f"  hap2: {finite2.sum()} / {finite2.size} regions finite")
    else:
        finite2 = None
        lines.append("  hap2: (missing)")

    finite_overall = np.isfinite(overall)
    lines.append(
        f"  overall (pooled): {finite_overall.sum()} / {finite_overall.size} regions finite"
    )

    # regions with no finite value in any available hap
    no_finite = np.ones(len(region_names), dtype=bool)
    if finite1 is not None:
        no_finite &= ~finite1
    if finite2 is not None:
        no_finite &= ~finite2

    idx_all_nan = np.where(no_finite)[0].tolist()
    if idx_all_nan:
        preview = ", ".join(region_names[j] for j in idx_all_nan[:10])
        more = " …" if len(idx_all_nan) > 10 else ""
        lines.append(
            f"  regions with no finite in any hap: {len(idx_all_nan)} ({preview}{more})"
        )

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# reference + regions
# ────────────────────────────────────────────────────────────────────
left, right = st.columns([0.6, 0.4], gap="large")

with left:
    bundled_refs = list_bundled_refs()
    ref_choices = _make_grouped_choices(
        bundled_refs, cmeth_files, "bundled", "external"
    )
    if not ref_choices:
        st.error("No references available (bundled or external).")
        st.stop()

    ref_label = st.selectbox(
        "reference (.cmeth)",
        options=[lbl for lbl, _ in ref_choices],
        index=0,
        key="an_ref_label",
        help="Bundled references ship with the package; external refs come from Home page folder scan.",
    )
    cm_ref_path = dict(ref_choices)[ref_label]

    try:
        _, meta = read_cmeth(Path(cm_ref_path))
        ref_mode = str(meta.get("mode", "aggregated")).lower()
    except Exception as e:
        ref_mode = "unknown"
        st.warning(f"Could not parse reference metadata ({e}). Proceeding.")

    # regions: bundled kits + custom beds
    builtin_kits = list_builtin_kits()
    custom_beds = st.session_state.setdefault("custom_beds", [])

    # filter stale custom beds
    ext_beds: list[Path] = []
    for x in custom_beds:
        try:
            p = Path(x)
            if p.exists():
                ext_beds.append(p)
        except Exception:
            continue

    bed_choices: List[Tuple[str, Tuple[str, str]]] = []
    for k in sorted(builtin_kits.keys()):
        bed_choices.append((f"bundled kit · {k}", ("--kit", k)))
    for b in sorted(ext_beds, key=lambda pp: pp.name.lower()):
        bed_choices.append((f"external BED · {b.name}", ("--bed", str(b))))

    if not bed_choices:
        st.error("No region definitions found (bundled kits or custom BEDs).")
        st.stop()

    # choose default kit if present
    default_kit = (st.session_state.get("default_kit") or "ME030").strip()
    default_label = f"bundled kit · {default_kit}"
    labels_only = [lbl for lbl, _ in bed_choices]
    default_index = (
        labels_only.index(default_label) if default_label in labels_only else 0
    )

    bed_label = st.selectbox(
        "regions",
        options=labels_only,
        index=default_index,
        key="an_regions_label",
        help="Choose a bundled MLPA kit or a discovered custom BED (set on Home page).",
    )
    selected_flag, selected_val = dict(bed_choices)[bed_label]
    kit_args: List[str] = [selected_flag, str(selected_val)]
    region_label = bed_label.split("·", 1)[1].strip()

with right:
    st.markdown(
        f"**reference:** `{Path(cm_ref_path).name}`  \n"
        f"**mode:** `{ref_mode}`  \n"
        f"**regions:** `{region_label}`"
    )

st.divider()

# ────────────────────────────────────────────────────────────────────
# targets – discovered + uploads
# ────────────────────────────────────────────────────────────────────
st.subheader("targets")

up_col, pick_col = st.columns([0.55, 0.45], gap="large")

upload_dir = out_dir / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)

with up_col:
    st.markdown("**upload bedMethyl**", help="Upload .bedmethyl.gz (+ .tbi).")
    uploads = st.file_uploader(
        "drop .bedmethyl.gz and .tbi here",
        type=["gz", "tbi"],
        accept_multiple_files=True,
        help="For each .bedmethyl.gz a matching .tbi is recommended.",
        key="an_uploads",
    )
    if uploads:
        _save_uploads(uploads, upload_dir)

uploaded_map: Dict[str, Dict[str, Path]] = scan_bedmethyl(upload_dir)

with pick_col:
    bed_by_sample: Dict[str, Dict[str, Path]] = _combine_bed_maps(
        orig_bed_by_sample, uploaded_map
    )
    if not bed_by_sample:
        st.warning(
            "No bgzipped & indexed bedmethyl files found — set paths on Home or upload above."
        )
        st.stop()

    sample_ids = sorted(bed_by_sample)
    picked = st.multiselect(
        "target samples",
        sample_ids,
        key="an_targets",
        help="For haplotype series, pick exactly one sample with both _1 and _2.",
    )

st.divider()

# ────────────────────────────────────────────────────────────────────
# methylation
# ────────────────────────────────────────────────────────────────────
st.subheader("methylation")

mode_choice = st.radio(
    "plot mode",
    options=["Pooled only", "Haplotype series (pooled + hap1 + hap2)"],
    index=0,
    key="an_meth_mode",
)

mcol1, mcol2 = st.columns([0.5, 0.5], gap="large")
with mcol1:
    meth_pooled_png = st.text_input(
        "pooled output",
        value="methylation_pooled.png",
        key="an_meth_pooled_name",
    )
with mcol2:
    meth_h1_png = st.text_input(
        "hap1 output", value="methylation_hap1.png", key="an_meth_h1_name"
    )
    meth_h2_png = st.text_input(
        "hap2 output", value="methylation_hap2.png", key="an_meth_h2_name"
    )

min_hap = st.slider("min hap regions", 1, 50, 10, 1, key="an_min_hap_regions")

go_meth = st.button(
    "plot methylation", type="primary", use_container_width=True
)

if go_meth:
    if not picked:
        st.error("Select at least one target sample.")
        st.stop()

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
        # Prefer ungrouped if present; otherwise fall back to hap1+hap2.
        if parts.get("ungrouped"):
            pooled_argv.append(str(parts["ungrouped"]))
        else:
            for key in ("1", "2"):
                p = parts.get(key)
                if p:
                    pooled_argv.append(str(p))

    with st.expander("pooled - CLI argv", expanded=False):
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

    if mode_choice.startswith("Haplotype"):
        if len(picked) != 1:
            st.error("Haplotype series requires exactly **one** target sample.")
            st.stop()

        sid = picked[0]
        parts = bed_by_sample.get(sid, {})
        if not (parts.get("1") and parts.get("2")):
            st.error(f"Sample `{sid}` is missing either `_1` or `_2` file.")
            st.stop()

        h1_argv = [
            "methylation",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / meth_h1_png),
            "--hap-ref-plot",
            "--min-hap-regions",
            str(min_hap),
            "--ref-hap",
            "1",
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

        h2_argv = [
            "methylation",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / meth_h2_png),
            "--hap-ref-plot",
            "--min-hap-regions",
            str(min_hap),
            "--ref-hap",
            "2",
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
        "copy-number output", value="copy_number.png", key="an_cn_png_name"
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

    with st.expander("copy-number - CLI argv", expanded=False):
        st.code(" ".join(map(str, argv)), language="bash")

    code, out_png, stdout = _cli_plot(argv)
    if code == 0 and out_png.exists():
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
        st.error(f"Copy-number plotting failed (exit {code})")

    if stdout.strip():
        with st.expander("copy number – CLI stdout/stderr", expanded=False):
            st.code(stdout, language="bash")
