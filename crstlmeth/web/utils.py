"""
crstlmeth.web.utils

Shared utilities.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict


# ── misc helpers ─────────────────────────────────────────────────────
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_tmp() -> Path:
    p = Path.cwd() / "tmp"
    p.mkdir(exist_ok=True)
    return p


def list_builtin_kits() -> Dict[str, Path]:
    pkg_root = Path(__file__).resolve().parents[1]
    kits_dir = pkg_root / "kits"
    kits: Dict[str, Path] = {}
    for bed in kits_dir.glob("*_meth.bed"):
        kits[bed.stem.replace("_meth", "")] = bed
    return kits


def list_bundled_refs() -> Dict[str, Path]:
    """
    return bundled .cmeth references shipped with the package
    as {ref_stem: <path>}. works regardless of install location.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    refs_dir = pkg_root / "refs"
    out: Dict[str, Path] = {}
    if refs_dir.exists():
        for p in refs_dir.glob("*.cmeth"):
            out[p.stem] = p
    return out


# ── output + indexing helpers ───────────────────────────────────
def default_output_dir_for(any_input: Path | None, session_id: str) -> Path:
    """
    Choose a per-session output folder near the input data if possible.
    """
    root = (any_input.parent if any_input else Path.cwd()).resolve()
    out = root / "crstlmeth_out" / session_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_tabix_index(bgz: Path) -> None:
    """
    Ensure *.tbi exists for a .bedmethyl.gz file using tabix if available.
    No-op if index already exists.
    """
    if not bgz.exists():
        return
    tbi = Path(str(bgz) + ".tbi")
    if tbi.exists():
        return
    # bgzip/tabix must be on PATH for this to work
    try:
        subprocess.run(
            ["tabix", "-f", "-s", "1", "-b", "2", "-e", "3", str(bgz)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        # best-effort: ignore errors; caller may surface a message
        pass
