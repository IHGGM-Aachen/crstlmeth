"""
crstlmeth.web.pages.02_references

view and create *.cmeth references from bedMethyl files
"""

from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.reference import create as cli_create
from crstlmeth.core.references import parse_cmeth_header, read_cmeth
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.utils import list_builtin_kits

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="crstlmeth - references", page_icon=":material/database:"
)
st.title("references")
render_sidebar()

bed_by_sample = st.session_state.get("bed_by_sample", {})
cmeth_files: list[str] = st.session_state.get("cmeth_files", [])
custom_beds: list[str] = st.session_state.get("custom_beds", [])
out_dir = Path(st.session_state.get("outdir_resolved", Path.cwd() / "output"))

# ────────────────────────────────────────────────────────────────────
# section 1 - view existing references (lazy preview)
# ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(
        "inspect reference",
        help="open a *.cmeth file, view header metadata, optionally preview rows",
    )

    if not cmeth_files:
        st.warning(
            "no .cmeth files found - set folder on the home page and scan"
        )
    else:
        # do not auto-select to avoid immediate loading
        label = "choose file"
        choices = ["— select —"] + [str(p) for p in cmeth_files]
        picked = st.selectbox(
            label, choices, index=0, help="select a *.cmeth file to inspect"
        )

        if picked != "— select —":
            path = Path(picked)

            # load just the header quickly
            try:
                meta = parse_cmeth_header(path)
            except Exception as e:
                st.error(f"failed to parse header:\n{e}")
                meta = None

            if meta:
                st.markdown("**header**")
                top = [
                    "mode",
                    "version",
                    "date",
                    "kit",
                    "md5_bed",
                    "denom_dedup",
                    "k_min",
                    "cn_norm",
                ]
                shown = set()
                for k in top:
                    if k in meta:
                        st.markdown(f"{k:>12} : {meta[k]}")
                        shown.add(k)
                for k in sorted(k for k in meta if k not in shown):
                    st.markdown(f"{k:>12} : {meta[k]}")

            # optional data preview
            with st.expander("preview rows (optional)", expanded=False):
                n_rows = st.number_input(
                    "rows",
                    min_value=5,
                    max_value=2000,
                    value=50,
                    step=5,
                    help="limit preview to avoid rendering very large tables",
                )
                if st.button("load preview", use_container_width=True):
                    try:
                        df, meta2 = read_cmeth(path)
                        st.dataframe(
                            df.head(int(n_rows)), use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"failed to load data:\n{e}")

st.divider()

# ────────────────────────────────────────────────────────────────────
# section 2 - create new reference
# ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(
        "create new reference",
        help="build a *.cmeth cohort file from selected bedMethyl inputs",
    )

    if not bed_by_sample:
        st.warning(
            "no bedmethyl files found - set folder on the home page and scan"
        )
        st.stop()

    # controls row 1
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        mode = st.selectbox(
            "reference mode",
            options=["aggregated", "full"],
            index=0,
            help="aggregated: anonymized per-region quantiles; full: per-sample rows (plot-ready).",
        )
    with c2:
        hap_resolved = st.toggle(
            "haplotype-resolved",
            value=False,
            help="when on, only samples with both hap1 and hap2 can be selected; ungrouped is ignored.",
        )
    with c3:
        default_name = (
            "reference_full.cmeth"
            if mode == "full"
            else "reference_aggregated.cmeth"
        )
        out_file_name = st.text_input(
            "output file",
            value=default_name,
            help="filename written under the output directory set on the home page",
        )

    # kit / bed row
    c4, _ = st.columns([2, 1], gap="large")
    with c4:
        builtin = list_builtin_kits()
        kit_choices = list(builtin.keys()) + custom_beds
        kit_or_bed = st.selectbox(
            "mlpa kit / bed",
            kit_choices,
            help="choose a built-in kit or a custom BED file defining intervals",
        )

    # sample picker (filtered if hap_resolved)
    def _eligible_samples(hap_only: bool) -> list[str]:
        if not hap_only:
            return sorted(bed_by_sample)
        out: list[str] = []
        for sid, parts in bed_by_sample.items():
            if ("1" in parts) and ("2" in parts):
                out.append(sid)
        return sorted(out)

    eligible = _eligible_samples(hap_resolved)
    with st.expander("select samples", expanded=True):
        st.caption(
            f"{len(eligible)} selectable "
            + (
                "(haplotype-resolved required)"
                if hap_resolved
                else "(all discovered samples)"
            ),
            help="only samples listed below will be used to build the cohort",
        )
        selected_sids = st.multiselect(
            "samples",
            eligible,
            help="pick one or more samples for the reference",
        )

    # build button
    build = st.button(
        "build reference", type="primary", use_container_width=True
    )

    if build:
        # basic validation
        if not selected_sids:
            st.error("select at least one sample")
            st.stop()
        if not kit_or_bed:
            st.error("choose a kit or BED")
            st.stop()
        if not out_file_name.strip():
            st.error(
                "provide an output filename (e.g. reference_aggregated.cmeth)"
            )
            st.stop()

        # gather file paths according to toggle
        paths: list[str] = []
        skipped: list[str] = []

        for sid in selected_sids:
            parts = bed_by_sample.get(sid, {})
            if hap_resolved:
                # strictly require both hap files; ignore ungrouped
                if "1" in parts and "2" in parts:
                    paths.extend([str(parts["1"]), str(parts["2"])])
                else:
                    skipped.append(sid)
            else:
                # prefer ungrouped if present; else include both haps if available
                if "ungrouped" in parts:
                    paths.append(str(parts["ungrouped"]))
                elif "1" in parts and "2" in parts:
                    paths.extend([str(parts["1"]), str(parts["2"])])
                else:
                    skipped.append(sid)

        if skipped:
            st.warning("skipped incomplete samples: " + ", ".join(skipped))

        if mode == "aggregated" and len(paths) < 2:
            st.error("need at least two input files for aggregated mode")
            st.stop()

        out_path = out_dir / out_file_name

        # kit value: pass built-in key as is, custom beds as paths
        kit_val = kit_or_bed if kit_or_bed in builtin else kit_or_bed

        # args for CLI
        args = [
            "--kit",
            str(kit_val),
            "--mode",
            mode,
            "-o",
            str(out_path),
            *paths,
        ]

        with st.spinner("building reference …"):
            runner = CliRunner()
            try:
                result = runner.invoke(cli_create, args, catch_exceptions=True)
            except Exception as exc:
                st.error("unhandled python exception during CLI run")
                st.exception(exc)
                st.stop()

        # show outcome
        st.markdown("**cli command**")
        st.code(
            "crstlmeth reference create "
            + " ".join(f"'{a}'" if " " in a else a for a in args),
            language="bash",
        )

        if result.exit_code == 0 and out_path.exists():
            st.success(f"reference written → {out_path}")

            # header preview
            try:
                meta = parse_cmeth_header(out_path)
                with st.expander("header preview", expanded=True):
                    st.code(
                        "\n".join([f"{k:>12} : {v}" for k, v in meta.items()]),
                        language="text",
                    )
            except Exception as e:
                st.warning(f"created file, but failed to parse header: {e}")
        else:
            st.error(f"reference creation failed (exit {result.exit_code})")

        # stdout / stderr
        if result.output and result.output.strip():
            with st.expander("cli output", expanded=False):
                st.code(result.output, language="bash")

        if result.exception:
            with st.expander("traceback", expanded=False):
                st.code(
                    "".join(traceback.format_exception(result.exception)),
                    language="python",
                )
