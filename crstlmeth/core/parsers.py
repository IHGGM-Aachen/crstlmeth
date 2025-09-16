"""
crstlmeth/core/parsers.py

low-level io utilities for reading bgzipped + tabix-indexed bedmethyl files
"""

import os

import pandas as pd
import pysam


def _chrom_aliases(chrom: str) -> list[str]:
    """
    return likely aliases for a chromosome name to handle 'chr' vs no-'chr' mismatches
    """
    if chrom.startswith("chr"):
        base = chrom[3:]
        return [chrom, base]
    else:
        return [chrom, f"chr{chrom}"]


def query_bedmethyl(
    filepath: str, chrom: str, start: int, end: int
) -> pd.DataFrame:
    if not filepath.endswith(".gz"):
        raise ValueError(f"expected .gz file, got: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"no such file: {filepath}")
    if not os.path.exists(filepath + ".tbi"):
        raise FileNotFoundError(f"missing index: {filepath}.tbi")

    tabix = pysam.TabixFile(filepath)
    rows = []

    # try primary chrom, then alias
    for chrom_try in _chrom_aliases(chrom):
        try:
            for line in tabix.fetch(chrom_try, start, end):
                fields = line.rstrip("\n").split("\t")
                # guard against malformed lines
                if len(fields) <= 11:
                    continue
                rows.append(
                    {
                        "chrom": fields[0],
                        "start": int(fields[1]),
                        "end": int(fields[2]),
                        "mod_code": fields[3],
                        "strand": fields[5],
                        "Nvalid_cov": int(fields[9]),
                        "Nmod": int(fields[11]),
                    }
                )
        except (ValueError, IndexError, KeyError):
            # ValueError when chrom not in index; keep trying aliases
            continue
        if rows:
            break  # got data with this alias

    cols = ["chrom", "start", "end", "mod_code", "strand", "Nvalid_cov", "Nmod"]
    return pd.DataFrame(rows, columns=cols)


def get_region_stats(
    filepath: str, chrom: str, start: int, end: int
) -> tuple[int, int]:
    df = query_bedmethyl(filepath, chrom, start, end)
    if df.empty:
        return (0, 0)

    # If you want "any modification of this base": sum Nmod across codes,
    # but take ONE Nvalid_cov per locus (+strand)
    grp = df.groupby(["chrom", "start", "end", "strand"], as_index=False).agg(
        Nvalid_cov=("Nvalid_cov", "max"),  # one denominator per locus
        Nmod=("Nmod", "sum"),  # sum across mod codes
    )
    return int(grp["Nmod"].sum()), int(grp["Nvalid_cov"].sum())


def get_region_stats_many(
    filepaths: list[str] | tuple[str, ...],
    chrom: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """
    sum total nmod and nvalid_cov across multiple bedmethyl files
    for a single region [start, end) on chrom.

    returns:
        (sum_mod, sum_valid)
    """
    sum_mod = 0
    sum_valid = 0
    for fp in filepaths:
        m, v = get_region_stats(fp, chrom, start, end)
        sum_mod += m
        sum_valid += v
    return sum_mod, sum_valid


def get_region_stats_by_haplotype(
    files_by_hap: dict[str, str],
    chrom: str,
    start: int,
    end: int,
) -> dict[str, tuple[int, int]]:
    """
    compute per-haplotype and pooled stats for a region.

    files_by_hap keys are expected to be any subset of {"1","2","ungrouped"}.

    returns:
        {
          "1": (mod, valid)         # if present, else (0,0)
          "2": (mod, valid)         # if present, else (0,0)
          "ungrouped": (mod, valid) # if present, else (0,0)
          "pooled": (mod, valid)    # sum of all present
        }
    """
    out = {"1": (0, 0), "2": (0, 0), "ungrouped": (0, 0)}
    pooled_mod = 0
    pooled_valid = 0

    for hap in ("1", "2", "ungrouped"):
        if hap in files_by_hap:
            m, v = get_region_stats(files_by_hap[hap], chrom, start, end)
            out[hap] = (m, v)
            pooled_mod += m
            pooled_valid += v

    out["pooled"] = (pooled_mod, pooled_valid)
    return out
