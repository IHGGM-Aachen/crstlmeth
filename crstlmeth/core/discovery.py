"""
crstlmeth/core/discovery.py

helpers to discover and classify crstlmeth files in a folder structure
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

__all__ = [
    "scan_bedmethyl",
    "scan_region_beds",
    "scan_cmeth",
    "resolve_bedmethyl_glob",
]

# regular expressions to classify filenames
_BEDM_RE = re.compile(
    r"""
    ^(?P<sample>.+?)               # sample id (lazy)
    [._-]                          # separator before hap
    (?P<hap>1|2|ungrouped)         # hap tag
    (?:[._-]\w+)*                  # optional extra tokens (e.g., wf_mods, LR)
    \.bedmethyl(?:\.gz)?$          # extension
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CMETH_RE = re.compile(r".+\.cmeth$", re.IGNORECASE)
_BED_RE = re.compile(r".+\.bed$", re.IGNORECASE)


def scan_bedmethyl(folder: Path) -> Dict[str, Dict[str, Path]]:
    """
    scan a folder for bgzipped and tabix-indexed bedMethyl files

    returns:
        a dictionary of the form {sample_id: {hap: Path}}, where hap is
        one of "1", "2", or "ungrouped"
    """
    out: Dict[str, Dict[str, Path]] = {}
    if not folder or not folder.exists():
        return out

    for f in folder.rglob("*.bedmethyl.gz"):
        m = _BEDM_RE.match(f.name)
        if not m:
            continue

        # Require a matching *.tbi
        if not (f.parent / (f.name + ".tbi")).exists():
            continue

        sample = m["sample"]
        hap = m["hap"]
        out.setdefault(sample, {})[hap] = f.resolve()

    return out


def scan_region_beds(folder: Path) -> List[Path]:
    """
    scan for plain *.bed files in a folder (non-recursive)

    returns:
        a sorted list of Path objects, empty if the folder is missing
    """
    return (
        sorted(p for p in folder.glob("*.bed"))
        if folder and folder.exists()
        else []
    )


def scan_cmeth(folder: Path) -> List[Path]:
    """
    scan for *.cmeth reference files in a folder (non-recursive)

    returns:
        a sorted list of Path objects, empty if the folder is missing
    """
    return (
        sorted(p for p in folder.glob("*.cmeth") if _CMETH_RE.match(p.name))
        if folder and folder.exists()
        else []
    )


def resolve_bedmethyl_glob(patterns: List[str]) -> List[Path]:
    """
    resolve shell-style globs and directory paths into bedmethyl files

    this is used by the cli to allow wildcard and recursive input.

    accepts:
        list of strings, which can be globs, directories or filenames

    returns:
        a sorted list of resolved bedmethyl paths matching _BEDM_RE
    """
    files: List[Path] = []
    for pat in patterns:
        p = Path(pat).expanduser()
        if "*" in pat or "?" in pat or "[" in pat:
            files.extend(sorted(p.parent.rglob(p.name)))
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.bedmethyl*")))
        else:
            files.append(p)
    return [f.resolve() for f in files if _BEDM_RE.match(f.name)]
