"""
Sample discovery and bedMethyl role classification.

The web frontend and CLI helper code both need to recognise the same file
naming conventions.  This module intentionally keeps the rules small and
explicit: a bedMethyl file belongs to one sample and one role, where the role
is haplotype 1, haplotype 2, or an ungrouped/pooled track.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

ROLE_ALIASES: dict[str, str] = {
    "1": "1",
    "h1": "1",
    "hap1": "1",
    "hap_1": "1",
    "2": "2",
    "h2": "2",
    "hap2": "2",
    "hap_2": "2",
    "ungrouped": "ungrouped",
    "unphased": "ungrouped",
    "pooled": "ungrouped",
    "pool": "ungrouped",
}

ROLE_ORDER = ("1", "2", "ungrouped")

# sample<sep>role[optional extra tokens].bedmethyl[.gz]
# examples:
#   M43599_1.bedmethyl.gz
#   M24520LR.2.bedmethyl.gz
#   M43599-ungrouped.bedmethyl.gz
#   sample_hap1.modkit.bedmethyl.gz
_BEDMETHYL_NAME_RE = re.compile(
    r"""
    ^(?P<sample>.+?)
    [._-]
    (?P<role>1|2|h1|h2|hap1|hap2|hap_1|hap_2|ungrouped|unphased|pooled|pool)
    (?:[._-][^.]+)*
    \.bedmethyl(?:\.gz)?$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class BedMethylName:
    """Parsed bedMethyl filename information."""

    sample_id: str
    role: str
    data_name: str
    is_index: bool = False


def strip_tabix_suffix(path: str | Path) -> tuple[str, bool]:
    """Return (filename_without_optional_tbi, had_tbi_suffix)."""
    name = Path(path).name
    if name.endswith(".tbi"):
        return name[:-4], True
    return name, False


def parse_bedmethyl_name(path: str | Path) -> BedMethylName | None:
    """
    Parse a bedMethyl or bedMethyl index filename.

    Returns None if the filename does not match a supported convention.
    """
    data_name, is_index = strip_tabix_suffix(path)
    m = _BEDMETHYL_NAME_RE.match(data_name)
    if not m:
        return None
    role = ROLE_ALIASES[m.group("role").lower()]
    sample = m.group("sample").strip()
    if not sample:
        return None
    return BedMethylName(
        sample_id=sample, role=role, data_name=data_name, is_index=is_index
    )


def _iter_candidate_paths(root_or_paths: Path | Iterable[Path]) -> list[Path]:
    if isinstance(root_or_paths, (str, Path)):
        root = Path(root_or_paths)
        if not root.exists():
            return []
        if root.is_dir():
            paths = list(root.rglob("*.bedmethyl"))
            paths += list(root.rglob("*.bedmethyl.gz"))
            paths += list(root.rglob("*.bedmethyl.tbi"))
            paths += list(root.rglob("*.bedmethyl.gz.tbi"))
            return sorted(set(paths))
        return [root]
    return [Path(p) for p in root_or_paths]


def discover_bedmethyl_files(
    root_or_paths: Path | Iterable[Path],
    *,
    require_index: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Discover bedMethyl files grouped as {sample_id: {role: data_path}}.

    Index files are never returned as data inputs.  When require_index=True,
    a data file is returned only if <data>.tbi exists beside it.
    """
    by_name: dict[str, Path] = {}
    index_names: set[str] = set()

    for path in _iter_candidate_paths(root_or_paths):
        parsed = parse_bedmethyl_name(path)
        if parsed is None:
            continue
        if parsed.is_index:
            index_names.add(parsed.data_name)
        else:
            by_name[parsed.data_name] = path.resolve()
            if Path(str(path) + ".tbi").exists():
                index_names.add(parsed.data_name)

    out: dict[str, dict[str, Path]] = {}
    for data_name, path in sorted(by_name.items()):
        parsed = parse_bedmethyl_name(data_name)
        if parsed is None:
            continue
        if require_index and data_name not in index_names:
            continue
        out.setdefault(parsed.sample_id, {})[parsed.role] = path
    return out


def sample_status_table(
    samples: Mapping[str, Mapping[str, Path]],
) -> pd.DataFrame:
    """
    Build a frontend-friendly status table for grouped sample files.

    The table keeps data paths and index status separate.  This makes uploaded
    .tbi files visible to the user without accidentally passing them to CLI
    plotting commands.
    """
    rows: list[dict[str, object]] = []
    for sample_id, parts in sorted(
        samples.items(), key=lambda kv: kv[0].lower()
    ):
        row: dict[str, object] = {"sample_id": sample_id}
        ready_any = False
        for role in ROLE_ORDER:
            p = parts.get(role)
            row[f"{role}_file"] = Path(p).name if p else ""
            has_tbi = bool(p and Path(str(p) + ".tbi").exists())
            row[f"{role}_tbi"] = has_tbi
            if p and has_tbi:
                ready_any = True
        row["ready_haps"] = bool(
            parts.get("1")
            and Path(str(parts["1"]) + ".tbi").exists()
            and parts.get("2")
            and Path(str(parts["2"]) + ".tbi").exists()
        )
        row["ready_ungrouped"] = bool(
            parts.get("ungrouped")
            and Path(str(parts["ungrouped"]) + ".tbi").exists()
        )
        row["ready_any"] = ready_any
        rows.append(row)
    return pd.DataFrame(rows)


def ready_sample_ids(
    samples: Mapping[str, Mapping[str, Path]],
    *,
    require_haps: bool = False,
    require_ungrouped: bool = False,
) -> list[str]:
    """Return sample IDs with the indexed files required by an analysis."""
    if not samples:
        return []
    table = sample_status_table(samples)
    if table.empty:
        return []
    mask = table["ready_any"].astype(bool)
    if require_haps:
        mask &= table["ready_haps"].astype(bool)
    if require_ungrouped:
        mask &= table["ready_ungrouped"].astype(bool)
    return table.loc[mask, "sample_id"].astype(str).tolist()


def summarize_parts(parts: Mapping[str, Path]) -> str:
    """Compact sample part summary for CLI/web messages."""
    labels = {"1": "hap1", "2": "hap2", "ungrouped": "ungrouped"}
    chunks: list[str] = []
    for role in ROLE_ORDER:
        p = parts.get(role)
        if p:
            mark = "OK" if Path(str(p) + ".tbi").exists() else "missing .tbi"
            chunks.append(f"{labels[role]}: {Path(p).name} ({mark})")
    return " | ".join(chunks) if chunks else "no bedMethyl parts"
