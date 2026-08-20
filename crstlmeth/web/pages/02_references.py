"""
crstlmeth.web.pages.02_references

View and create CMETH references from bedMethyl files.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.reference import create as cli_create
from crstlmeth.core.references import parse_cmeth_header, read_cmeth
from crstlmeth.core.samples import ready_sample_ids, sample_status_table
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import list_builtin_kits, list_bundled_refs

st.set_page_config(
    page_title="crstlmeth - references", page_icon=":material/database:"
)
ensure_web_state()
st.title("references")
render_sidebar()

bed_by_sample: Dict[str, Dict[str, Path]] = st.session_state.setdefault(
    "bed_by_sample", {}
)
cmeth_files: list[str] = st.session_state.setdefault("cmeth_files", [])
custom_beds: list[str] = st.session_state.setdefault("custom_beds", [])
session_id: str = st.session_state["session_id"]


def _resolve_or_none(p: str) -> Path | None:
    p = (p or "").strip()
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return None


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


def _make_grouped_cmeth_choices(
    bundled: Dict[str, Path], external: List[str]
) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for _k, p in sorted(bundled.items(), key=lambda kv: kv[0].lower()):
        rows.append((f"bundled  -  {p.name}", p))
    for p in sorted(_existing_paths(external), key=lambda pp: pp.name.lower()):
        rows.append((f"external  -  {p.name}", p))
    return rows


def _eligible_samples(hap_only: bool) -> list[str]:
    return ready_sample_ids(bed_by_sample, require_haps=hap_only)


def _output_base_dir() -> Path:
    ref_dir = _resolve_or_none(st.session_state.get("ref_dir", ""))
    if ref_dir and ref_dir.exists():
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir
    return resolve_outdir(session_id)


out_base = _output_base_dir()

with st.container(border=True):
    st.subheader(
        "inspect reference",
        help="open a CMETH file, view header metadata, optionally preview rows",
    )
    bundled = list_bundled_refs()
    ref_choices = _make_grouped_cmeth_choices(bundled, cmeth_files)
    uploaded_cmeth = st.file_uploader(
        "optional upload CMETH for inspection",
        type=["cmeth", "gz"],
        key="ref_inspect_cmeth_upload",
    )
    if uploaded_cmeth is not None:
        upload_dir = out_base / "uploads" / "references"
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_path = upload_dir / Path(uploaded_cmeth.name).name
        uploaded_path.write_bytes(uploaded_cmeth.getbuffer())
        if str(uploaded_path) not in cmeth_files:
            cmeth_files.append(str(uploaded_path))
        ref_choices.append((f"uploaded  -  {uploaded_path.name}", uploaded_path))
        st.caption(f"uploaded reference: {uploaded_path}")

    if not ref_choices:
        st.warning("no CMETH files available (bundled or external)")
    else:
        labels = [" -  select  - "] + [lbl for lbl, _ in ref_choices]
        picked_label = st.selectbox(
            "choose file", labels, index=0, key="ref_inspect_pick"
        )
        if picked_label != " -  select  - ":
            path = dict(ref_choices)[picked_label]
            try:
                meta = parse_cmeth_header(path)
            except Exception as e:
                st.error(f"failed to parse header:\n{e}")
                meta = None
            if meta:
                st.markdown("**header**")
                top = [
                    "version",
                    "kind",
                    "coordinate",
                    "created",
                    "description",
                    "target_name",
                    "target_count",
                    "target_bed_count",
                    "source_sample_count",
                    "source_file_count",
                    "cn_norm",
                    "mvalue_eps",
                    "meth_hist_bins",
                    "mval_hist_bins",
                    "cn_log2_hist_bins",
                ]
                shown = set()
                for k in top:
                    if k in meta:
                        st.markdown(f"{k:>22} : {meta[k]}")
                        shown.add(k)
                for k in sorted(k for k in meta if k not in shown):
                    st.markdown(f"{k:>22} : {meta[k]}")
            with st.expander("preview rows (optional)", expanded=False):
                n_rows = st.number_input(
                    "rows",
                    min_value=5,
                    max_value=2000,
                    value=50,
                    step=5,
                    key="ref_preview_rows",
                )
                if st.button(
                    "load preview", width="stretch", key="ref_load_preview"
                ):
                    try:
                        df, _meta2 = read_cmeth(path)
                        st.dataframe(df.head(int(n_rows)), width="stretch")
                    except Exception as e:
                        st.error(f"failed to load data:\n{e}")

st.divider()

with st.container(border=True):
    st.subheader(
        "create new reference",
        help="build a CMETH cohort reference from selected bedMethyl inputs",
    )
    if not bed_by_sample:
        st.warning(
            "no bedmethyl files found - use Setup to scan folders or add upload support here"
        )
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        hap_resolved = st.toggle(
            "haplotype-resolved",
            value=True,
            key="ref_hap_resolved",
            help="when on, only samples with both hap1 and hap2 can be selected; ungrouped files are also used when present.",
        )
    with c2:
        include_cpgs = st.toggle(
            "include CpG rows",
            value=True,
            key="ref_include_cpgs",
            help="needed for CpG profile plots; increases reference size.",
        )
    with c3:
        out_file_name = st.text_input(
            "output file",
            value="reference.cmeth.gz",
            key="ref_out_name",
            help="filename written into the reference folder (if set) else session output",
        )

    m1, m2 = st.columns([0.45, 0.55], gap="large")
    with m1:
        mod_mode = st.selectbox(
            "modkit mod mode",
            options=["m", "h", "mh", "any", "custom"],
            index=0,
            key="ref_mod_mode",
            help="m is standard 5mC. mh combines m and h. custom uses the codes field.",
        )
    with m2:
        mod_codes = st.text_input(
            "custom mod codes",
            value="",
            key="ref_mod_codes",
            help="Comma-separated modkit codes used when mode=custom.",
        )

    description = st.text_area(
        "description",
        value="",
        key="ref_description",
        help="optional text written into the CMETH header",
    )

    builtin = list_builtin_kits()
    ext_beds = _existing_paths(custom_beds)
    bed_choices: List[Tuple[str, str]] = []
    for k in sorted(builtin.keys()):
        bed_choices.append((f"bundled kit  -  {k}", k))
    for b in sorted(ext_beds, key=lambda pp: pp.name.lower()):
        bed_choices.append((f"external BED  -  {b.name}", str(b)))
    if not bed_choices:
        st.error("no region definitions found (bundled kits or custom BEDs).")
        st.stop()

    default_kit = (st.session_state.get("default_kit") or "ME030").strip()
    default_label = f"bundled kit  -  {default_kit}"
    labels_only = [lbl for lbl, _ in bed_choices]
    default_index = (
        labels_only.index(default_label) if default_label in labels_only else 0
    )
    c4, _ = st.columns([2, 1], gap="large")
    with c4:
        bed_label = st.selectbox(
            "mlpa kit / bed",
            options=labels_only,
            index=default_index,
            key="ref_bed_label",
        )
        kit_value = dict(bed_choices)[bed_label]

    eligible = _eligible_samples(hap_resolved)
    status_df = sample_status_table(bed_by_sample)
    with st.expander("select samples", expanded=True):
        st.caption(
            f"{len(eligible)} selectable indexed sample(s) "
            + (
                "(hap1 + hap2 required)"
                if hap_resolved
                else "(any indexed bedMethyl part)"
            )
        )
        if not status_df.empty:
            st.dataframe(status_df, width="stretch", hide_index=True)
        selected_sids = st.multiselect(
            "samples", eligible, key="ref_selected_sids"
        )

    build = st.button(
        "build reference", type="primary", width="stretch", key="ref_build"
    )
    if build:
        if not selected_sids:
            st.error("select at least one sample")
            st.stop()
        if not out_file_name.strip():
            st.error("provide an output filename (e.g. reference.cmeth.gz)")
            st.stop()

        paths: list[str] = []
        skipped: list[str] = []
        for sid in selected_sids:
            parts = bed_by_sample.get(sid, {})
            if hap_resolved:
                if (
                    parts.get("1")
                    and Path(str(parts["1"]) + ".tbi").exists()
                    and parts.get("2")
                    and Path(str(parts["2"]) + ".tbi").exists()
                ):
                    paths.extend([str(parts["1"]), str(parts["2"])])
                    if (
                        parts.get("ungrouped")
                        and Path(str(parts["ungrouped"]) + ".tbi").exists()
                    ):
                        paths.append(str(parts["ungrouped"]))
                else:
                    skipped.append(sid)
            else:
                if (
                    parts.get("ungrouped")
                    and Path(str(parts["ungrouped"]) + ".tbi").exists()
                ):
                    paths.append(str(parts["ungrouped"]))
                elif (
                    parts.get("1")
                    and Path(str(parts["1"]) + ".tbi").exists()
                    and parts.get("2")
                    and Path(str(parts["2"]) + ".tbi").exists()
                ):
                    paths.extend([str(parts["1"]), str(parts["2"])])
                else:
                    skipped.append(sid)
        if skipped:
            st.warning("skipped incomplete samples: " + ", ".join(skipped))
        if len(paths) < 2:
            st.error(
                "need at least two input files to build a cohort reference"
            )
            st.stop()

        out_path = (out_base / out_file_name).resolve()
        args = ["--kit", str(kit_value), "-o", str(out_path)]
        args.append("--include-cpgs" if include_cpgs else "--no-include-cpgs")
        args.extend(["--mod-mode", str(mod_mode)])
        for code in [x.strip() for x in mod_codes.split(",") if x.strip()]:
            args.extend(["--mod-code", code])
        if description.strip():
            args.extend(["--description", description.strip()])
        args.extend(paths)

        with st.spinner("building reference ..."):
            runner = CliRunner()
            try:
                result = runner.invoke(cli_create, args, catch_exceptions=True)
            except Exception as exc:
                st.error("unhandled python exception during CLI run")
                st.exception(exc)
                st.stop()

        st.markdown("**cli command**")
        st.code(
            "crstlmeth reference create "
            + " ".join(f"'{a}'" if " " in a else a for a in args),
            language="bash",
        )
        if result.exit_code == 0 and out_path.exists():
            st.success(f"reference written -> {out_path}")
            pstr = str(out_path)
            if pstr not in st.session_state["cmeth_files"]:
                st.session_state["cmeth_files"].append(pstr)
            try:
                meta = parse_cmeth_header(out_path)
                with st.expander("header preview", expanded=True):
                    st.code(
                        "\n".join([f"{k:>22} : {v}" for k, v in meta.items()]),
                        language="text",
                    )
            except Exception as e:
                st.warning(f"created file, but failed to parse header: {e}")
        else:
            st.error(f"reference creation failed (exit {result.exit_code})")
        if result.output and result.output.strip():
            with st.expander("cli output", expanded=False):
                st.code(result.output, language="bash")
        if result.exception:
            with st.expander("traceback", expanded=False):
                st.code(
                    "".join(traceback.format_exception(result.exception)),
                    language="python",
                )
