"""
crstlmeth/core/discovery.py

helpers to discover and classify crstlmeth files in a folder structure
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from crstlmeth.core.samples import (
    discover_bedmethyl_files,
    parse_bedmethyl_name,
)

__all__ = [
    "scan_bedmethyl",
    "scan_region_beds",
    "scan_cmeth",
    "resolve_bedmethyl_glob",
]

# regular expressions to classify filenames
_CMETH_RE = re.compile(r".+\.cmeth(?:\.gz)?$", re.IGNORECASE)
_BED_RE = re.compile(r".+\.bed$", re.IGNORECASE)


def scan_bedmethyl(
    folder: Path, *, require_index: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Scan a folder for bedMethyl files and classify sample roles.

    Returns {sample_id: {"1"|"2"|"ungrouped": Path}}.  Index files are
    detected for readiness checks, but are never returned as target inputs.
    Use require_index=False in the web uploader when incomplete uploads should
    still be displayed to the user.
    """
    if not folder or not folder.exists():
        return {}
    return discover_bedmethyl_files(folder, require_index=require_index)


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
        sorted(
            [p for p in folder.glob("*.cmeth") if _CMETH_RE.match(p.name)]
            + [p for p in folder.glob("*.cmeth.gz") if _CMETH_RE.match(p.name)]
        )
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
    return [
        f.resolve()
        for f in files
        if parse_bedmethyl_name(f) is not None
        and not parse_bedmethyl_name(f).is_index
    ]
