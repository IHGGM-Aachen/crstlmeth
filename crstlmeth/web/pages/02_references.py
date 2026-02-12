"""
crstlmeth.web.pages.02_references

view and create *.cmeth references from bedMethyl files
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from click.testing import CliRunner

from crstlmeth.cli.reference import create as cli_create
from crstlmeth.core.references import parse_cmeth_header, read_cmeth
from crstlmeth.web.sidebar import render_sidebar
from crstlmeth.web.state import ensure_web_state, resolve_outdir
from crstlmeth.web.utils import list_builtin_kits, list_bundled_refs

# ────────────────────────────────────────────────────────────────────
# page setup
# ────────────────────────────────────────────────────────────────────
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
        rows.append((f"bundled · {p.name}", p))

    for p in sorted(_existing_paths(external), key=lambda pp: pp.name.lower()):
        rows.append((f"external · {p.name}", p))

    return rows


def _eligible_samples(hap_only: bool) -> list[str]:
    if not hap_only:
        return sorted(bed_by_sample)
    out: list[str] = []
    for sid, parts in bed_by_sample.items():
        if parts.get("1") and parts.get("2"):
            out.append(sid)
    return sorted(out)


def _output_base_dir() -> Path:
    """
    Prefer writing new references into ref_dir (if set), so they persist and
    naturally show up in the references list. Otherwise write into session outdir.
    """
    ref_dir = _resolve_or_none(st.session_state.get("ref_dir", ""))
    if ref_dir and ref_dir.exists():
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir
    return resolve_outdir(session_id)


out_base = _output_base_dir()

# ────────────────────────────────────────────────────────────────────
# section 1 - view existing references (lazy preview)
# ────────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(
        "inspect reference",
        help="open a *.cmeth file, view header metadata, optionally preview rows",
    )

    bundled = list_bundled_refs()
    ref_choices = _make_grouped_cmeth_choices(bundled, cmeth_files)

    if not ref_choices:
        st.warning("no .cmeth files available (bundled or external)")
    else:
        labels = ["— select —"] + [lbl for lbl, _ in ref_choices]
        picked_label = st.selectbox(
            "choose file",
            labels,
            index=0,
            key="ref_inspect_pick",
            help="select a *.cmeth file to inspect",
        )

        if picked_label != "— select —":
            path = dict(ref_choices)[picked_label]

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

            with st.expander("preview rows (optional)", expanded=False):
                n_rows = st.number_input(
                    "rows",
                    min_value=5,
                    max_value=2000,
                    value=50,
                    step=5,
                    help="limit preview to avoid rendering very large tables",
                    key="ref_preview_rows",
                )
                if st.button(
                    "load preview",
                    use_container_width=True,
                    key="ref_load_preview",
                ):
                    try:
                        df, _meta2 = read_cmeth(path)
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

    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        mode = st.selectbox(
            "reference mode",
            options=["aggregated", "full"],
            index=0,
            key="ref_mode",
            help="aggregated: anonymized per-region quantiles; full: per-sample rows (plot-ready).",
        )
    with c2:
        hap_resolved = st.toggle(
            "haplotype-resolved",
            value=False,
            key="ref_hap_resolved",
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
            key="ref_out_name",
            help="filename written into the reference folder (if set) else session output",
        )

    # kit / bed
    builtin = list_builtin_kits()

    ext_beds = _existing_paths(custom_beds)

    bed_choices: List[Tuple[str, Tuple[str, str]]] = []
    for k in sorted(builtin.keys()):
        bed_choices.append((f"bundled kit · {k}", ("--kit", k)))
    for b in sorted(ext_beds, key=lambda pp: pp.name.lower()):
        bed_choices.append((f"external BED · {b.name}", ("--bed", str(b))))

    if not bed_choices:
        st.error("no region definitions found (bundled kits or custom BEDs).")
        st.stop()

    # default kit index
    default_kit = (st.session_state.get("default_kit") or "ME030").strip()
    default_label = f"bundled kit · {default_kit}"
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
            help="choose a built-in kit or a custom BED file defining intervals",
        )
        bed_flag, bed_val = dict(bed_choices)[bed_label]

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
            key="ref_selected_sids",
            help="pick one or more samples for the reference",
        )

    build = st.button(
        "build reference",
        type="primary",
        use_container_width=True,
        key="ref_build",
    )

    if build:
        if not selected_sids:
            st.error("select at least one sample")
            st.stop()
        if not out_file_name.strip():
            st.error(
                "provide an output filename (e.g. reference_aggregated.cmeth)"
            )
            st.stop()

        paths: list[str] = []
        skipped: list[str] = []

        for sid in selected_sids:
            parts = bed_by_sample.get(sid, {})
            if hap_resolved:
                if parts.get("1") and parts.get("2"):
                    paths.extend([str(parts["1"]), str(parts["2"])])
                else:
                    skipped.append(sid)
            else:
                if parts.get("ungrouped"):
                    paths.append(str(parts["ungrouped"]))
                elif parts.get("1") and parts.get("2"):
                    paths.extend([str(parts["1"]), str(parts["2"])])
                else:
                    skipped.append(sid)

        if skipped:
            st.warning("skipped incomplete samples: " + ", ".join(skipped))

        if mode == "aggregated" and len(paths) < 2:
            st.error("need at least two input files for aggregated mode")
            st.stop()

        out_path = (out_base / out_file_name).resolve()

        args = [
            bed_flag,
            str(bed_val),
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

        st.markdown("**cli command**")
        st.code(
            "crstlmeth reference create "
            + " ".join(f"'{a}'" if " " in a else a for a in args),
            language="bash",
        )

        if result.exit_code == 0 and out_path.exists():
            st.success(f"reference written → {out_path}")

            # add to session list so sidebar updates immediately
            pstr = str(out_path)
            if pstr not in st.session_state["cmeth_files"]:
                st.session_state["cmeth_files"].append(pstr)

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

        if result.output and result.output.strip():
            with st.expander("cli output", expanded=False):
                st.code(result.output, language="bash")

        if result.exception:
            with st.expander("traceback", expanded=False):
                st.code(
                    "".join(traceback.format_exception(result.exception)),
                    language="python",
                )
