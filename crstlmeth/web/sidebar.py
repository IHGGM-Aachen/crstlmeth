"""
crstlmeth.web.sidebar

Read-only sidebar. Never mutates paths or discovery state.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st


def _fmt_path(p: str | None) -> str:
    if not p:
        return "(unset)"
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)


def render_sidebar() -> None:
    """Draw a compact overview from session_state (no side effects)."""
    # stable session id
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]

    bed_by_sample = st.session_state.get("bed_by_sample", {}) or {}
    cmeth_files = st.session_state.get("cmeth_files", []) or []
    custom_beds = st.session_state.get("custom_beds", []) or []

    data_dir = st.session_state.get("data_dir", "") or ""
    ref_dir = st.session_state.get("ref_dir", "") or ""
    region_dir = st.session_state.get("region_dir", "") or ""
    outdir_disp = (
        st.session_state.get(
            "outdir_resolved", st.session_state.get("outdir", "")
        )
        or ""
    )

    st.sidebar.markdown("## overview")

    n_samples = len(bed_by_sample)
    n_files = sum(len(h) for h in bed_by_sample.values())
    st.sidebar.markdown(f"**bedmethyl:** {n_samples} samples / {n_files} files")
    st.sidebar.markdown(f"**cmeth:** {len(cmeth_files)} files")
    st.sidebar.markdown(f"**regions:** {len(custom_beds)} beds")

    with st.sidebar.expander("sample ids", expanded=False):
        if n_samples:
            ids = sorted(bed_by_sample.keys())
            show = ids[:25] + (["..."] if len(ids) > 25 else [])
            st.code("\n".join(show), language="text")
        else:
            st.code("(none)", language="text")

    with st.sidebar.expander("references", expanded=False):
        if cmeth_files:
            show = [Path(x).name for x in cmeth_files[:25]]
            if len(cmeth_files) > 25:
                show.append("...")
            st.code("\n".join(show), language="text")
        else:
            st.code("(none)", language="text")

    with st.sidebar.expander("custom beds", expanded=False):
        if custom_beds:
            show = [Path(x).name for x in custom_beds[:25]]
            if len(custom_beds) > 25:
                show.append("...")
            st.code("\n".join(show), language="text")
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
                f"out  : {_fmt_path(outdir_disp)}",
            ]
        ),
        language="text",
    )

    st.sidebar.divider()
    st.sidebar.markdown(f"*session id: `{st.session_state.session_id}`*")
