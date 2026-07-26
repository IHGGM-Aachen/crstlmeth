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
import pandas as pd
import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.plot import plot as plot_group
from crstlmeth.core.cpg_profile import (
    SAMPLE_TRACK_CHOICES,
    align_sample_tracks,
    available_regions,
    build_cpg_profile_table,
    get_reference_cpg_rows,
    match_region,
    select_sample_tracks,
)
from crstlmeth.core.discovery import scan_bedmethyl
from crstlmeth.core.methylation import Methylation
from crstlmeth.core.references import read_cmeth
from crstlmeth.core.regions import load_intervals
from crstlmeth.core.samples import (
    ready_sample_ids,
    sample_status_table,
    summarize_parts,
)
from crstlmeth.viz.cpg_profile import make_cpg_profile_plotly
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import (
    list_builtin_kits,
    list_bundled_refs,
    preserve_tabix_pair_timestamp,
)

# --------------------------------------------------------------------
# page setup
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------
def _save_uploads(files: list, dest_dir: Path) -> list[Path]:
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    for up in files:
        outp = dest_dir / Path(up.name).name
        with outp.open("wb") as fh:
            shutil.copyfileobj(up, fh)
        saved.append(outp.resolve())

    # After all uploads are saved, repair .tbi mtimes for matching pairs.
    # Important: do this after the loop, because the .gz and .tbi may arrive
    # in either order.
    for p in saved:
        if p.name.endswith(".bedmethyl.gz"):
            preserve_tabix_pair_timestamp(p)

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
        rows.append((f"{bundled_tag}  -  {p.name}", p))

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
        rows.append((f"{external_tag}  -  {p.name}", p))
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
        more = " ..." if len(idx_all_nan) > 10 else ""
        lines.append(
            f"  regions with no finite in any hap: {len(idx_all_nan)} ({preview}{more})"
        )

    return "\n".join(lines)


def _cmd_text(argv: list[str]) -> str:
    return " ".join(
        f"'{str(a)}'" if any(ch.isspace() for ch in str(a)) else str(a)
        for a in argv
    )


def _part_summary(parts: Dict[str, Path]) -> str:
    return summarize_parts(parts)


def _sample_input_panel(
    *,
    key: str,
    label: str,
    multiple: bool,
    require_haps: bool = False,
    require_ungrouped: bool = False,
) -> tuple[list[str], Dict[str, Dict[str, Path]]]:
    """Each analysis block gets independent upload/select controls.

    Uploaded .tbi files are copied beside the data files and used only for
    readiness checks.  They are never passed as target arguments to the CLI.
    """
    section_upload_dir = out_dir / "uploads" / key
    section_upload_dir.mkdir(parents=True, exist_ok=True)

    with st.expander(f"{label} sample input", expanded=True):
        st.caption(
            "Use samples discovered on the Home page, or upload files for this panel only. "
            "Accepted names include SAMPLE_1, SAMPLE.1, SAMPLE-1 and the same for 2/ungrouped."
        )
        uploads = st.file_uploader(
            "optional upload .bedmethyl.gz plus matching .tbi",
            type=["gz", "tbi"],
            accept_multiple_files=True,
            key=f"an_{key}_uploads",
        )
        if uploads:
            _save_uploads(uploads, section_upload_dir)

        uploaded_map: Dict[str, Dict[str, Path]] = scan_bedmethyl(
            section_upload_dir,
            require_index=False,
        )
        bed_by_sample: Dict[str, Dict[str, Path]] = _combine_bed_maps(
            orig_bed_by_sample,
            uploaded_map,
        )

        status_df = sample_status_table(bed_by_sample)
        if status_df.empty:
            st.warning("No bedMethyl samples found for this panel.")
            return [], bed_by_sample

        visible_cols = [
            "sample_id",
            "1_file",
            "1_tbi",
            "2_file",
            "2_tbi",
            "ungrouped_file",
            "ungrouped_tbi",
            "ready_haps",
            "ready_ungrouped",
            "ready_any",
        ]
        st.dataframe(status_df[visible_cols], width="stretch", hide_index=True)

        sample_ids = ready_sample_ids(
            bed_by_sample,
            require_haps=require_haps,
            require_ungrouped=require_ungrouped,
        )
        if not sample_ids:
            need = []
            if require_haps:
                need.append("hap1 + hap2 with .tbi")
            if require_ungrouped:
                need.append("ungrouped with .tbi")
            st.warning(
                "No ready samples for this panel"
                + (f" ({', '.join(need)})" if need else ".")
            )
            return [], bed_by_sample

        if multiple:
            picked = st.multiselect(
                "target sample(s)",
                sample_ids,
                key=f"an_{key}_targets",
            )
        else:
            opts = [" -  select  - "] + sample_ids
            picked_one = st.selectbox(
                "target sample",
                opts,
                index=0,
                key=f"an_{key}_target",
            )
            picked = [] if picked_one == " -  select  - " else [picked_one]

        if picked:
            st.code(
                "\n".join(
                    f"{sid}: {_part_summary(bed_by_sample.get(sid, {}))}"
                    for sid in picked
                ),
                language="text",
            )
        return list(picked), bed_by_sample


# --------------------------------------------------------------------
# reference + regions
# --------------------------------------------------------------------
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

    ref_df = pd.DataFrame()
    try:
        ref_df, meta = read_cmeth(Path(cm_ref_path))
        ref_mode = str(
            meta.get("feature_types", meta.get("mode", "reference"))
        ).lower()
    except Exception as e:
        meta = {}
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
        bed_choices.append((f"bundled kit  -  {k}", ("--kit", k)))
    for b in sorted(ext_beds, key=lambda pp: pp.name.lower()):
        bed_choices.append((f"external BED  -  {b.name}", ("--bed", str(b))))

    if not bed_choices:
        st.error("No region definitions found (bundled kits or custom BEDs).")
        st.stop()

    # choose default kit if present
    default_kit = (st.session_state.get("default_kit") or "ME030").strip()
    default_label = f"bundled kit  -  {default_kit}"
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
    region_label = bed_label.split(" - ", 1)[1].strip()

with right:
    st.markdown(
        f"**reference:** `{Path(cm_ref_path).name}`  \n"
        f"**mode:** `{ref_mode}`  \n"
        f"**regions:** `{region_label}`"
    )

st.divider()

# --------------------------------------------------------------------
# CpG profile: interactive clinical view
# --------------------------------------------------------------------
with st.container(border=True):
    st.header("1  -  CpG profile")
    st.caption(
        "Interactive CpG-level view from CMETH CpG rows. Hover shows CpG index and BED coordinate; clicked CpGs appear in the table."
    )

    cpg_picked, cpg_bed_by_sample = _sample_input_panel(
        key="cpg",
        label="CpG profile",
        multiple=False,
        require_haps=False,
    )

    cpg_regions = available_regions(ref_df) if not ref_df.empty else []
    has_cpg_rows = (
        not ref_df.empty
        and "feature_type" in ref_df.columns
        and (ref_df["feature_type"].astype(str) == "cpg").any()
    )

    if not has_cpg_rows:
        st.info(
            "Selected CMETH has no CpG rows. Rebuild the reference with --include-cpgs to enable this view."
        )
    else:
        default_region_idx = 0
        for needle in ("SNURF:TSS-DMR", "SNRPN:alt-TSS-DMR"):
            if needle in cpg_regions:
                default_region_idx = cpg_regions.index(needle)
                break

        cpg_c1, cpg_c2, cpg_c3 = st.columns([1.6, 0.8, 0.8], gap="large")
        with cpg_c1:
            cpg_region = st.selectbox(
                "CpG region",
                options=cpg_regions,
                index=default_region_idx,
                key="an_cpg_region",
            )
        with cpg_c2:
            cpg_ref_hap = st.selectbox(
                "reference track",
                options=["pooled", "allele_low", "allele_high", "unphased"],
                index=0,
                key="an_cpg_ref_hap",
                help="CMETH cohort reference rows to use for the median/bands.",
            )
        with cpg_c3:
            cpg_x_mode = st.selectbox(
                "x-axis",
                options=["index", "genomic"],
                index=0,
                key="an_cpg_x_mode",
                help="Index is clearer clinically; genomic mode uses BED start coordinates.",
            )

        cpg_o1, cpg_o2, cpg_o3, cpg_o4 = st.columns([1, 1, 1, 1], gap="large")
        with cpg_o1:
            cpg_sample_track = st.selectbox(
                "sample track",
                options=list(SAMPLE_TRACK_CHOICES),
                index=list(SAMPLE_TRACK_CHOICES).index("pooled"),
                key="an_cpg_sample_track",
                help="both_haps shows direct hap1 and hap2 separately. both_alleles shows lower/higher methylated allele.",
            )
        with cpg_o2:
            cpg_show_outer = st.toggle(
                "show q05-q95", value=False, key="an_cpg_outer"
            )
        with cpg_o3:
            cpg_sample_line = st.toggle(
                "connect sample points", value=False, key="an_cpg_line"
            )
        with cpg_o4:
            cpg_genomic_track = st.toggle(
                "genomic ruler", value=True, key="an_cpg_genomic_track"
            )

        cpg_m1, cpg_m2 = st.columns([0.6, 0.4], gap="large")
        with cpg_m1:
            cpg_mod_mode = st.selectbox(
                "modkit modification mode",
                options=["m", "h", "mh", "any", "custom"],
                index=0,
                key="an_cpg_mod_mode",
                help="Usually m for 5mC. This is passed to the modkit-compatible parser.",
            )
        with cpg_m2:
            cpg_mod_codes_text = st.text_input(
                "custom mod codes",
                value="",
                key="an_cpg_mod_codes",
                help="Comma-separated codes for custom mode, e.g. m,h.",
            )

        go_cpg = st.button(
            "plot CpG profile", type="primary", width="stretch", key="an_cpg_go"
        )

        if go_cpg:
            if len(cpg_picked) != 1:
                st.error("CpG profile requires exactly one target sample.")
                st.stop()
            sid = cpg_picked[0]
            parts = cpg_bed_by_sample.get(sid, {})
            if not any(parts.get(k) for k in ("1", "2", "ungrouped")):
                st.error(f"Sample `{sid}` has no usable bedMethyl parts.")
                st.stop()
            if cpg_sample_track in {
                "hap1",
                "hap2",
                "both_haps",
                "allele_low",
                "allele_high",
                "both_alleles",
            }:
                if not (
                    parts.get("1")
                    and Path(str(parts["1"]) + ".tbi").exists()
                    and parts.get("2")
                    and Path(str(parts["2"]) + ".tbi").exists()
                ):
                    st.error(
                        f"Sample `{sid}` needs indexed hap1 and hap2 files for `{cpg_sample_track}`."
                    )
                    st.stop()

            try:
                region_id, chrom, cpg_start, cpg_end = match_region(
                    ref_df, cpg_region
                )
                ref_cpg_rows = get_reference_cpg_rows(
                    ref_df,
                    region_id=region_id,
                    reference_hap=cpg_ref_hap,
                )
                if ref_cpg_rows.empty:
                    st.error(
                        f"No CpG rows for `{region_id}` and hap `{cpg_ref_hap}`."
                    )
                    st.stop()
                custom_codes = [
                    x.strip()
                    for x in cpg_mod_codes_text.split(",")
                    if x.strip()
                ]
                all_tracks = align_sample_tracks(
                    ref_cpg_rows,
                    parts,
                    chrom,
                    cpg_start,
                    cpg_end,
                    mod_mode=cpg_mod_mode,
                    mod_codes=custom_codes or None,
                )
                selected_tracks = select_sample_tracks(
                    all_tracks, cpg_sample_track
                )
                labels = {
                    "pooled": f"{sid} pooled",
                    "hap1": f"{sid} hap1",
                    "hap2": f"{sid} hap2",
                    "allele_low": f"{sid} allele low",
                    "allele_high": f"{sid} allele high",
                    "unphased": f"{sid} unphased",
                }
                cpg_table = build_cpg_profile_table(
                    ref_cpg_rows, selected_tracks
                )
                fig = make_cpg_profile_plotly(
                    reference_rows=ref_cpg_rows,
                    sample_tracks=selected_tracks,
                    title=f"CpG methylation profile  -  {region_id}",
                    x_mode=cpg_x_mode,
                    sample_labels=labels,
                    show_outer_band=cpg_show_outer,
                    sample_line=cpg_sample_line,
                    genomic_track=cpg_genomic_track,
                )
            except Exception as e:
                st.error(f"CpG profile failed: {e}")
                st.stop()

            st.success(f"Interactive CpG profile for `{region_id}` / `{sid}`")
            selected_event = None
            try:
                selected_event = st.plotly_chart(
                    fig,
                    width="stretch",
                    key="an_cpg_plotly",
                    on_select="rerun",
                    selection_mode="points",
                )
            except TypeError:
                st.plotly_chart(fig, width="stretch", key="an_cpg_plotly")

            selected_rows = pd.DataFrame()
            try:
                pts = (
                    selected_event.get("selection", {}).get("points", [])
                    if selected_event
                    else []
                )
                if pts:
                    indices = sorted(
                        {
                            int(
                                p.get("customdata", [p.get("point_index", 1)])[
                                    0
                                ]
                            )
                            for p in pts
                        }
                    )
                    selected_rows = cpg_table[
                        cpg_table["cpg_index"].isin(indices)
                    ]
            except Exception:
                selected_rows = pd.DataFrame()

            if not selected_rows.empty:
                st.markdown("**clicked CpG(s)**")
                st.dataframe(selected_rows, width="stretch", hide_index=True)

            with st.expander("CpG table", expanded=True):
                st.caption(
                    "Coordinates use BED 0-based start/end. Beta = methylated reads / valid reads."
                )
                st.dataframe(cpg_table, width="stretch", hide_index=True)
                safe_region = (
                    region_id.replace("/", "_")
                    .replace(":", "_")
                    .replace(" ", "_")
                )
                st.download_button(
                    "download CpG table TSV",
                    data=cpg_table.to_csv(sep="\t", index=False).encode(),
                    file_name=f"{sid}_{safe_region}_cpg_profile.tsv",
                    mime="text/tab-separated-values",
                )
                st.download_button(
                    "download interactive HTML",
                    data=fig.to_html(include_plotlyjs="cdn").encode(),
                    file_name=f"{sid}_{safe_region}_cpg_profile.html",
                    mime="text/html",
                )


st.divider()

# --------------------------------------------------------------------
# Methylation
# --------------------------------------------------------------------
with st.container(border=True):
    st.header("2  -  Methylation")
    st.caption(
        "Region-level DMR methylation summary against the CMETH cohort reference."
    )

    meth_picked, meth_bed_by_sample = _sample_input_panel(
        key="meth",
        label="Methylation",
        multiple=True,
        require_haps=False,
    )

    mode_choice = st.radio(
        "plot mode",
        options=["Pooled only", "Haplotype series (pooled + hap1 + hap2)"],
        index=0,
        key="an_meth_mode",
        horizontal=True,
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

    min_hap = st.slider(
        "min hap regions", 1, 50, 10, 1, key="an_min_hap_regions"
    )

    go_meth = st.button(
        "plot methylation", type="primary", width="stretch", key="an_meth_go"
    )

    if go_meth:
        if not meth_picked:
            st.error("Select at least one target sample for methylation.")
            st.stop()

        pooled_argv = [
            "methylation",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / meth_pooled_png),
        ]
        for sid in meth_picked:
            parts = meth_bed_by_sample.get(sid, {})
            if parts.get("ungrouped"):
                pooled_argv.append(str(parts["ungrouped"]))
            else:
                for key in ("1", "2"):
                    p = parts.get(key)
                    if p:
                        pooled_argv.append(str(p))

        with st.expander("pooled methylation - CLI argv", expanded=False):
            st.code(_cmd_text(pooled_argv), language="bash")

        code, out_png, stdout = _cli_plot(pooled_argv)
        if code == 0 and out_png.exists():
            st.success(f"Pooled figure -> {out_png}")
            st.image(
                str(out_png), width="stretch", caption="Methylation (pooled)"
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
            with st.expander(
                "pooled methylation  -  CLI stdout/stderr", expanded=False
            ):
                st.code(stdout, language="bash")

        if mode_choice.startswith("Haplotype"):
            if len(meth_picked) != 1:
                st.error("Haplotype series requires exactly one target sample.")
                st.stop()
            sid = meth_picked[0]
            parts = meth_bed_by_sample.get(sid, {})
            if not (parts.get("1") and parts.get("2")):
                st.error(f"Sample `{sid}` is missing hap1 or hap2 file.")
                st.stop()

            for ref_hap, outfile, label in [
                ("allele_high", meth_h1_png, "higher-methylated allele"),
                ("allele_low", meth_h2_png, "lower-methylated allele"),
            ]:
                argv = [
                    "methylation",
                    "--cmeth",
                    str(cm_ref_path),
                    *kit_args,
                    "--out",
                    str(out_dir / outfile),
                    "--hap-ref-plot",
                    "--min-hap-regions",
                    str(min_hap),
                    "--ref-hap",
                    ref_hap,
                    str(parts["1"]),
                    str(parts["2"]),
                ]
                with st.expander(f"{label}  -  CLI argv", expanded=False):
                    st.code(_cmd_text(argv), language="bash")
                code_h, out_h, stdout_h = _cli_plot(argv)
                if code_h == 0 and out_h.exists():
                    st.success(f"{label} plot -> {out_h}")
                    st.image(str(out_h), width="stretch")
                    st.download_button(
                        f"download {label} PNG",
                        data=out_h.read_bytes(),
                        file_name=out_h.name,
                        mime="image/png",
                    )
                else:
                    st.error(f"{label} plot failed (exit {code_h})")
                    with st.expander(f"{label}  -  diagnostics", expanded=True):
                        st.code(
                            _diagnose_hap_coverage(parts, kit_args),
                            language="text",
                        )
                if stdout_h.strip():
                    with st.expander(
                        f"{label}  -  CLI stdout/stderr", expanded=False
                    ):
                        st.code(stdout_h, language="bash")


st.divider()

# --------------------------------------------------------------------
# Copy number
# --------------------------------------------------------------------
with st.container(border=True):
    st.header("3  -  Copy number")
    st.caption(
        "Region-level copy-number log2 ratio against the CMETH cohort reference."
    )

    cn_picked, cn_bed_by_sample = _sample_input_panel(
        key="cn",
        label="Copy number",
        multiple=True,
        require_haps=False,
    )

    cn_png = st.text_input(
        "copy-number output", value="copy_number.png", key="an_cn_png_name"
    )
    go_cn = st.button(
        "plot copy number", type="primary", width="stretch", key="an_cn_go"
    )

    if go_cn:
        if not cn_picked:
            st.error("Select at least one target sample for copy number.")
            st.stop()

        argv = [
            "copynumber",
            "--cmeth",
            str(cm_ref_path),
            *kit_args,
            "--out",
            str(out_dir / cn_png),
        ]
        for sid in cn_picked:
            parts = cn_bed_by_sample.get(sid, {})
            for key in ("1", "2", "ungrouped"):
                p = parts.get(key)
                if p:
                    argv.append(str(p))

        with st.expander("copy-number - CLI argv", expanded=False):
            st.code(_cmd_text(argv), language="bash")

        code, out_png, stdout = _cli_plot(argv)
        if code == 0 and out_png.exists():
            st.success(f"Figure -> {out_png}")
            st.image(
                str(out_png),
                width="stretch",
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
            with st.expander("copy number  -  CLI stdout/stderr", expanded=False):
                st.code(stdout, language="bash")
