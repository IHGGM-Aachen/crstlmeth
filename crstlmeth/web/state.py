# crstlmeth/web/state.py
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from crstlmeth.config.settings import Settings, load_settings


@st.cache_resource
def get_settings() -> Settings:
    """
    Cached settings for the Streamlit process.
    Source is resolved inside load_settings() (env var + standard locations).
    """
    return load_settings()


def _set_if_empty(key: str, value: str | None) -> None:
    if not value:
        return
    cur = (st.session_state.get(key) or "").strip()
    if not cur:
        st.session_state[key] = value


def apply_settings_defaults(s: Settings) -> None:
    """
    Apply config-derived defaults to session_state, but only into empty fields.
    Never overwrites user edits in the UI.
    """
    _set_if_empty("data_dir", str(s.data_dir) if s.data_dir else None)
    _set_if_empty("ref_dir", str(s.refs_dir) if s.refs_dir else None)
    _set_if_empty("region_dir", str(s.regions_dir) if s.regions_dir else None)
    _set_if_empty("outdir", str(s.out_dir) if s.out_dir else None)

    # Optional defaults that pages may use
    st.session_state.setdefault("tabix_bin", s.tabix)
    st.session_state.setdefault("default_kit", s.default_kit)

    # Mark whether a config contributed any path defaults (used for Home auto-scan)
    has_cfg_paths = any([s.data_dir, s.out_dir, s.refs_dir, s.regions_dir])
    st.session_state.setdefault("_has_config", bool(has_cfg_paths))


def ensure_web_state() -> None:
    """
    Initialize all web-facing state keys. Safe to call on every page.
    """
    # stable per-session id
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])

    # user-set folders
    st.session_state.setdefault("data_dir", "")
    st.session_state.setdefault("ref_dir", "")
    st.session_state.setdefault("region_dir", "")  # custom region beds folder
    st.session_state.setdefault("outdir", "")
    st.session_state.setdefault("outdir_resolved", "")

    # discoveries
    st.session_state.setdefault("bed_by_sample", {})
    st.session_state.setdefault("cmeth_files", [])
    st.session_state.setdefault("custom_beds", [])

    # apply config as defaults (only fills blanks)
    apply_settings_defaults(get_settings())

    # Ensure outdir_resolved is set early so Home + Sidebar can display it
    if not (st.session_state.get("outdir_resolved") or "").strip():
        out = resolve_outdir(st.session_state["session_id"])
        st.session_state["outdir_resolved"] = str(out)


def resolve_outdir(session_id: str) -> Path:
    """
    Compute/use a stable per-session outdir.
    Priority:
      1) outdir_resolved (if set)
      2) outdir (from config/UI) -> <outdir>/<session_id>
      3) data_dir/crstlmeth_out/<session_id>
      4) system tmp/crstlmeth_out/<session_id>
    """

    def _resolve(p: str) -> Path | None:
        p = (p or "").strip()
        if not p:
            return None
        try:
            return Path(p).expanduser().resolve()
        except Exception:
            return None

    existing = _resolve(st.session_state.get("outdir_resolved", ""))
    if existing:
        existing.mkdir(parents=True, exist_ok=True)
        return existing

    base = _resolve(st.session_state.get("outdir", ""))
    if base:
        out = base / session_id
        out.mkdir(parents=True, exist_ok=True)
        st.session_state["outdir_resolved"] = str(out)
        return out

    data = _resolve(st.session_state.get("data_dir", ""))
    if data:
        out = data / "crstlmeth_out" / session_id
        out.mkdir(parents=True, exist_ok=True)
        st.session_state["outdir_resolved"] = str(out)
        return out

    out = Path(tempfile.gettempdir()).resolve() / "crstlmeth_out" / session_id
    out.mkdir(parents=True, exist_ok=True)
    st.session_state["outdir_resolved"] = str(out)
    return out
