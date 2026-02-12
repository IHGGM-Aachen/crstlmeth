"""
crstlmeth.web.utils

Shared utilities.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from importlib.resources import files as rfiles


# ── misc helpers ─────────────────────────────────────────────────────
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _cache_root() -> Path:
    """
    Where we materialize packaged resources (kits/refs) to get real Paths.
    """
    env = os.getenv("CRSTLMETH_CACHE_DIR", "").strip()
    if env:
        root = Path(env).expanduser()
    else:
        xdg = os.getenv("XDG_CACHE_HOME", "").strip()
        root = Path(xdg).expanduser() if xdg else (Path.home() / ".cache")
        root = root / "crstlmeth"

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_tmp(base_dir: Path | None = None) -> Path:
    root = (base_dir or Path.cwd()).resolve()
    p = root / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _export_traversable_dir(trav, dst_dir: Path) -> None:
    """
    Copy a package directory (Traversable) to a real directory on disk.
    Only copies if missing (cheap). You can force refresh by deleting cache.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in trav.iterdir():
        out = dst_dir / item.name
        if item.is_dir():
            _export_traversable_dir(item, out)
        else:
            if out.exists():
                continue
            with item.open("rb") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def list_builtin_kits(kits_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    List kit BED files. If kits_dir is provided, use that directory.
    Otherwise load packaged kits via importlib.resources (materialized to cache).

    Returns:
        dict mapping kit name -> path to *_meth.bed
    """
    if kits_dir is None:
        trav = rfiles("crstlmeth").joinpath("kits")
        cache = _cache_root() / "resources" / "kits"
        _export_traversable_dir(trav, cache)
        kits_dir = cache

    kits: Dict[str, Path] = {}
    if not kits_dir.exists():
        return kits

    for bed in kits_dir.glob("*_meth.bed"):
        kits[bed.stem.replace("_meth", "")] = bed

    return kits


def list_bundled_refs(refs_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    List bundled .cmeth refs. If refs_dir is provided, use that directory.
    Otherwise load packaged refs via importlib.resources (materialized to cache).
    """
    if refs_dir is None:
        trav = rfiles("crstlmeth").joinpath("refs")
        cache = _cache_root() / "resources" / "refs"
        _export_traversable_dir(trav, cache)
        refs_dir = cache

    out: Dict[str, Path] = {}
    if refs_dir.exists():
        for p in refs_dir.glob("*.cmeth"):
            out[p.stem] = p
    return out


# ── output + indexing helpers ───────────────────────────────────
def default_output_dir_for(any_input: Path | None, session_id: str) -> Path:
    root = (any_input.parent if any_input else Path.cwd()).resolve()
    out = root / "crstlmeth_out" / session_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_tabix_index(bgz: Path, tabix_bin: str = "tabix") -> None:
    if not bgz.exists():
        return
    tbi = Path(str(bgz) + ".tbi")
    if tbi.exists():
        return

    try:
        subprocess.run(
            [tabix_bin, "-f", "-s", "1", "-b", "2", "-e", "3", str(bgz)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        pass
