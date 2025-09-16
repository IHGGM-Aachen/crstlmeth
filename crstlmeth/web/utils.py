# crstlmeth/web/utils.py
"""
crstlmeth.web.utils

streamlit utility helpers for temporary output folders, timestamping,
and discovery routines used across interactive pages
"""

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

_TMP = Path.cwd() / "tmp"


def timestamp() -> str:
    """
    generate a unique timestamp suitable for filenames
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_tmp() -> Path:
    """
    ensure that a temporary output folder exists and return its path
    """
    _TMP.mkdir(exist_ok=True)
    return _TMP


def list_builtin_kits() -> Dict[str, Path]:
    """
    return builtin kits shipped with the tool as a mapping
    {kit_name: <bed file path>}
    """
    pkg_root = Path(__file__).resolve().parents[1]
    kits_dir = pkg_root / "kits"

    kits: Dict[str, Path] = {}
    for bed in kits_dir.glob("*_meth.bed"):
        kits[bed.stem.replace("_meth", "")] = bed

    return kits


def scan_bedmethyl_dir(path: str | Path) -> dict[str, dict[str, Path]]:
    """
    recursively scan a folder for indexed bedmethyl files and
    group them as {sample_id: {"1": ..., "2": ..., "ungrouped": ...}}

    only files with valid suffixes and index files are returned
    """
    _BED_RE = re.compile(
        r"""
        ^(?P<sample>[^_.]+?)        # sample id up to first _ or .
        (?:_mods)?                  # optional _mods infix
        [_.]                        # separator _ or .
        (?P<hap>1|2|ungrouped)      # haplotype label
        (?:\.[^.]+)*                # any extra suffixes
        \.bedmethyl(?:\.gz)?$       # required extension
        """,
        re.VERBOSE,
    )

    path = Path(path)
    out: dict[str, dict[str, Path]] = defaultdict(dict)

    for f in path.rglob("*.bedmethyl.gz"):
        if not (f.with_suffix(f.suffix + ".tbi")).exists():
            continue
        m = _BED_RE.match(f.name)
        if not m:
            continue
        sample_id = m["sample"]
        hap = m["hap"]
        out[sample_id][hap] = f.resolve()

    return {sid: haps for sid, haps in out.items()}
