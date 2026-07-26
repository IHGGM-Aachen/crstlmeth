"""
crstlmeth/core/parsers.py

Low-level IO utilities for reading bgzipped + tabix-indexed modkit bedMethyl files.

This parser is intentionally tolerant of whitespace-delimited input and supports:
- standard modkit pileup bedMethyl output
- modkit bedRMod output (header lines skipped, denominator taken from score)

Key normalization choices:
- split on arbitrary whitespace, not just literal tabs
- derive `mod_code` from the BED name column (e.g. "m,CG,0" -> "m")
- use BED score (column 5) as Nvalid_cov, which is stable across bedMethyl/bedRMod
"""

from __future__ import annotations

import os
import re
from typing import Literal

import pandas as pd
import pysam

Mode = Literal["m", "h", "mh", "any", "custom"]


def _chrom_aliases(chrom: str) -> list[str]:
    """
    Return likely aliases for a chromosome name to handle 'chr' vs no-'chr' mismatches.
    """
    if chrom.startswith("chr"):
        return [chrom, chrom[3:]]
    return [chrom, f"chr{chrom}"]


def _split_bed_line(line: str) -> list[str]:
    """
    Split a BED-like line robustly on arbitrary whitespace.

    This tolerates files that were written with spaces or mixed whitespace instead
    of strict tab delimiters.
    """
    return re.split(r"[ \t]+", line.strip())


def _normalize_mod_code(name_field: str) -> str:
    """
    Normalize the mod code from bedMethyl column 4.

    Examples
    --------
    "m" -> "m"
    "h" -> "h"
    "m,CG,0" -> "m"
    "a,GATC,1" -> "a"
    """
    return name_field.split(",", 1)[0]


def _parse_modkit_bed_line(line: str) -> dict[str, object] | None:
    """
    Parse one modkit pileup / bedRMod row into a normalized record.

    Returns None for comments, track/browser lines, blank lines, or malformed rows.

    Normalized fields:
        chrom, start, end, name, mod_code, strand,
        Nvalid_cov, total_cov, percent_modified, Nmod,
        Ncanonical, Nother_mod, Ndelete, Nfail, Ndiff, Nnocall
    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return None
    if stripped.startswith("track") or stripped.startswith("browser"):
        return None

    fields = _split_bed_line(stripped)

    # modkit pileup/bedRMod schema is 18 columns
    if len(fields) < 18:
        return None

    try:
        name = fields[3]

        # BED score (column 5) is Nvalid_cov in standard bedMethyl and remains
        # Nvalid_cov in bedRMod. Column 10 differs between those formats.
        nvalid_cov = int(fields[4])

        return {
            "chrom": fields[0],
            "start": int(fields[1]),
            "end": int(fields[2]),
            "name": name,
            "mod_code": _normalize_mod_code(name),
            "strand": fields[5],
            "Nvalid_cov": nvalid_cov,
            "total_cov": int(
                fields[9]
            ),  # standard: Nvalid_cov, bedRMod: total coverage
            "percent_modified": float(fields[10]),
            "Nmod": int(fields[11]),
            "Ncanonical": int(fields[12]),
            "Nother_mod": int(fields[13]),
            "Ndelete": int(fields[14]),
            "Nfail": int(fields[15]),
            "Ndiff": int(fields[16]),
            "Nnocall": int(fields[17]),
        }
    except (ValueError, IndexError):
        return None


def query_bedmethyl(
    filepath: str,
    chrom: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """
    Query a genomic interval from a bgzipped + tabix-indexed modkit bedMethyl file.

    Returns a normalized DataFrame. Input lines may be tab-delimited or separated
    by arbitrary whitespace.

    Parameters
    ----------
    filepath
        Path to bgzipped bedMethyl/bedRMod file.
    chrom, start, end
        Interval [start, end), 0-based, end-exclusive.
    """
    if not filepath.endswith(".gz"):
        raise ValueError(f"expected .gz file, got: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"no such file: {filepath}")
    if not os.path.exists(filepath + ".tbi"):
        raise FileNotFoundError(f"missing index: {filepath}.tbi")

    rows: list[dict[str, object]] = []

    with pysam.TabixFile(filepath) as tabix:
        for chrom_try in _chrom_aliases(chrom):
            try:
                for line in tabix.fetch(chrom_try, start, end):
                    row = _parse_modkit_bed_line(line)
                    if row is not None:
                        rows.append(row)
            except (ValueError, KeyError, OSError):
                # Most commonly: chromosome not present in index for this alias
                continue

            if rows:
                break

    cols = [
        "chrom",
        "start",
        "end",
        "name",
        "mod_code",
        "strand",
        "Nvalid_cov",
        "total_cov",
        "percent_modified",
        "Nmod",
        "Ncanonical",
        "Nother_mod",
        "Ndelete",
        "Nfail",
        "Ndiff",
        "Nnocall",
    ]
    return pd.DataFrame(rows, columns=cols)


def _filter_codes(
    df: pd.DataFrame,
    *,
    mode: Mode,
    codes: list[str] | tuple[str, ...] | None,
) -> pd.DataFrame:
    """
    Filter rows by requested modification code mode.
    """
    if mode == "m":
        return df[df["mod_code"] == "m"]
    if mode == "h":
        return df[df["mod_code"] == "h"]
    if mode == "mh":
        return df[df["mod_code"].isin(["m", "h"])]
    if mode == "any":
        return df
    if mode == "custom":
        if not codes:
            raise ValueError(
                "mode='custom' requires a non-empty `codes=` list/tuple"
            )
        return df[df["mod_code"].isin(list(codes))]

    raise ValueError(
        f"unknown mode={mode!r}. Expected one of "
        "{'m', 'h', 'mh', 'any', 'custom'}."
    )


def collapse_bedmethyl_to_loci(
    df: pd.DataFrame,
    *,
    group_by_strand: bool = False,
) -> pd.DataFrame:
    """
    Collapse a normalized bedMethyl query result to one row per locus.

    Deduplication rules:
    1. Collapse repeated motif annotations of the same locus/code/strand by max.
    2. Collapse across mod codes so the denominator is counted once per locus,
       while the numerator is summed across selected codes.

    Returns columns:
        chrom, start, end, [strand], Nvalid_cov, Nmod
    """
    if df.empty:
        cols = ["chrom", "start", "end"]
        if group_by_strand:
            cols.append("strand")
        cols.extend(["Nvalid_cov", "Nmod"])
        return pd.DataFrame(columns=cols)

    dedup_keys = ["chrom", "start", "end", "strand", "mod_code"]
    per_code = df.groupby(dedup_keys, as_index=False).agg(
        Nvalid_cov=("Nvalid_cov", "max"),
        Nmod=("Nmod", "max"),
    )

    locus_keys = ["chrom", "start", "end"]
    if group_by_strand:
        locus_keys.append("strand")

    grp = per_code.groupby(locus_keys, as_index=False).agg(
        Nvalid_cov=("Nvalid_cov", "max"),
        Nmod=("Nmod", "sum"),
    )
    return grp


def get_locus_table(
    filepath: str,
    chrom: str,
    start: int,
    end: int,
    *,
    mode: Mode = "m",
    codes: list[str] | tuple[str, ...] | None = None,
    group_by_strand: bool = False,
) -> pd.DataFrame:
    """
    Return per-locus methylation counts for an interval from one bedMethyl file.

    Output columns:
        chrom, start, end, [strand], Nvalid_cov, Nmod
    """
    df = query_bedmethyl(filepath, chrom, start, end)
    if df.empty:
        cols = ["chrom", "start", "end"]
        if group_by_strand:
            cols.append("strand")
        cols.extend(["Nvalid_cov", "Nmod"])
        return pd.DataFrame(columns=cols)

    df = _filter_codes(df, mode=mode, codes=codes)
    if df.empty:
        cols = ["chrom", "start", "end"]
        if group_by_strand:
            cols.append("strand")
        cols.extend(["Nvalid_cov", "Nmod"])
        return pd.DataFrame(columns=cols)

    return collapse_bedmethyl_to_loci(df, group_by_strand=group_by_strand)


def get_region_stats(
    filepath: str,
    chrom: str,
    start: int,
    end: int,
    *,
    mode: Mode = "m",
    codes: list[str] | tuple[str, ...] | None = None,
    group_by_strand: bool = False,
) -> tuple[int, int]:
    """
    Compute (sum_mod, sum_valid) for a genomic interval from a modkit bedMethyl file.

    Strategy
    --------
    1. Query rows from the interval.
    2. Filter to the requested modification codes.
    3. Deduplicate repeated motif annotations at the same locus/code/strand.
    4. Collapse per locus so the denominator (Nvalid_cov) is counted once per locus,
       while the numerator (Nmod) is summed across selected codes.
    """
    grp = get_locus_table(
        filepath,
        chrom,
        start,
        end,
        mode=mode,
        codes=codes,
        group_by_strand=group_by_strand,
    )
    if grp.empty:
        return (0, 0)
    return int(grp["Nmod"].sum()), int(grp["Nvalid_cov"].sum())


def get_region_stats_many(
    filepaths: list[str] | tuple[str, ...],
    chrom: str,
    start: int,
    end: int,
    *,
    mode: Mode = "m",
    codes: list[str] | tuple[str, ...] | None = None,
    group_by_strand: bool = False,
) -> tuple[int, int]:
    """
    Sum total Nmod and Nvalid_cov across multiple bedMethyl files
    for a single region [start, end) on chrom.

    Returns
    -------
    (sum_mod, sum_valid)
    """
    sum_mod = 0
    sum_valid = 0

    for fp in filepaths:
        m, v = get_region_stats(
            fp,
            chrom,
            start,
            end,
            mode=mode,
            codes=codes,
            group_by_strand=group_by_strand,
        )
        sum_mod += m
        sum_valid += v

    return sum_mod, sum_valid


def get_region_stats_by_haplotype(
    files_by_hap: dict[str, str],
    chrom: str,
    start: int,
    end: int,
    *,
    mode: Mode = "m",
    codes: list[str] | tuple[str, ...] | None = None,
    group_by_strand: bool = False,
) -> dict[str, tuple[int, int]]:
    """
    Compute per-haplotype and pooled stats for a region.

    files_by_hap keys are expected to be any subset of {"1", "2", "ungrouped"}.

    Returns
    -------
    {
      "1": (mod, valid),         # if present, else (0,0)
      "2": (mod, valid),         # if present, else (0,0)
      "ungrouped": (mod, valid), # if present, else (0,0)
      "pooled": (mod, valid)     # sum of all present
    }
    """
    out: dict[str, tuple[int, int]] = {
        "1": (0, 0),
        "2": (0, 0),
        "ungrouped": (0, 0),
    }

    pooled_mod = 0
    pooled_valid = 0

    for hap in ("1", "2", "ungrouped"):
        fp = files_by_hap.get(hap)
        if fp is None:
            continue

        m, v = get_region_stats(
            fp,
            chrom,
            start,
            end,
            mode=mode,
            codes=codes,
            group_by_strand=group_by_strand,
        )
        out[hap] = (m, v)
        pooled_mod += m
        pooled_valid += v

    out["pooled"] = (pooled_mod, pooled_valid)
    return out
