"""
crstlmeth.web.sidebar

compact sidebar with an overview of discovered files and session id
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st


def _fmt_path(p: str | None) -> str:
    if not p:
        return "(none)"
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)


def render_sidebar() -> None:
    """
    draw a compact overview with counts, short lists, paths, and session id
    """
    # stable session id
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]

    bed_by_sample = st.session_state.get("bed_by_sample", {})
    cmeth_files = st.session_state.get("cmeth_files", [])
    custom_beds = st.session_state.get("custom_beds", [])

    data_dir = st.session_state.get("data_dir", "")
    ref_dir = st.session_state.get("ref_dir", "")
    region_dir = st.session_state.get("region_dir", "")
    outdir = st.session_state.get("outdir", "output")

    outdir_resolved = Path(outdir).expanduser().resolve()
    st.session_state["outdir_resolved"] = str(outdir_resolved)

    st.sidebar.markdown("## overview")

    n_samples = len(bed_by_sample)
    n_files = sum(len(h) for h in bed_by_sample.values())
    st.sidebar.markdown(f"**bedmethyl:** {n_samples} samples / {n_files} files")
    st.sidebar.markdown(f"**cmeth:** {len(cmeth_files)} files")
    st.sidebar.markdown(f"**regions:** {len(custom_beds)} beds")

    # spaced lists
    with st.sidebar.expander("sample ids", expanded=False):
        if n_samples:
            st.code(
                "\n".join(
                    sorted(bed_by_sample.keys())[:25]
                    + (["..."] if n_samples > 25 else [])
                ),
                language="text",
            )
        else:
            st.code("(none)", language="text")

    with st.sidebar.expander("references", expanded=False):
        if cmeth_files:
            st.code(
                "\n".join(
                    [Path(x).name for x in cmeth_files[:25]]
                    + (["..."] if len(cmeth_files) > 25 else [])
                ),
                language="text",
            )
        else:
            st.code("(none)", language="text")

    with st.sidebar.expander("custom beds", expanded=False):
        if custom_beds:
            st.code(
                "\n".join(
                    [Path(x).name for x in custom_beds[:25]]
                    + (["..."] if len(custom_beds) > 25 else [])
                ),
                language="text",
            )
        else:
            st.code("(none)", language="text")

    st.sidebar.divider()
    st.sidebar.markdown("**paths**")
    st.sidebar.code(
        "\n".join(
            [
                f"data : {_fmt_path(data_dir)}",
                f"cmeth: {_fmt_path(ref_dir)}",
                f"beds : {_fmt_path(region_dir)}",
                f"out  : {_fmt_path(str(outdir_resolved))}",
            ]
        ),
        language="text",
    )

    st.sidebar.divider()
    st.sidebar.markdown(f"*session id: `{st.session_state.session_id}`*")
