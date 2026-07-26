"""
Shared CpG-profile helpers.

These functions are used by both the CLI and the Streamlit frontend so that
sample track construction, region matching, and CpG table export behave the
same way everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from crstlmeth.core.parsers import get_locus_table
from crstlmeth.core.samples import parse_bedmethyl_name

REFERENCE_HAP_KEYS = ("pooled", "allele_low", "allele_high", "unphased")
SAMPLE_TRACK_CHOICES = (
    "pooled",
    "hap1",
    "hap2",
    "both_haps",
    "allele_low",
    "allele_high",
    "both_alleles",
    "unphased",
    "all",
)


def group_paths_by_sample(paths: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Group bedMethyl paths as {sample_id: {1|2|ungrouped: Path}}."""
    out: dict[str, dict[str, Path]] = {}
    for p in paths:
        p = Path(p)
        parsed = parse_bedmethyl_name(p)
        if parsed is None or parsed.is_index:
            continue
        out.setdefault(parsed.sample_id, {})[parsed.role] = p
    return out


def ordered_cpg_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return CpG rows sorted by genomic coordinate."""
    out = df.copy()
    if "start" not in out.columns or "end" not in out.columns:
        raise ValueError("CpG rows require start/end columns")
    out["start"] = pd.to_numeric(out["start"], errors="raise").astype(int)
    out["end"] = pd.to_numeric(out["end"], errors="raise").astype(int)
    return out.sort_values(
        ["chrom", "start", "end"], kind="mergesort"
    ).reset_index(drop=True)


def available_regions(ref_df: pd.DataFrame) -> list[str]:
    """List region IDs available in a CMETH table."""
    if (
        "feature_type" not in ref_df.columns
        or "region_id" not in ref_df.columns
    ):
        return []
    regions = ref_df[ref_df["feature_type"].astype(str) == "region"][
        "region_id"
    ]
    return regions.dropna().astype(str).drop_duplicates().tolist()


def match_region(
    ref_df: pd.DataFrame, region_query: str
) -> tuple[str, str, int, int]:
    """Resolve a region id/display-name query to (region_id, chrom, start, end)."""
    if "feature_type" not in ref_df.columns:
        raise ValueError(
            "CMETH is missing 'feature_type'; no region rows available"
        )
    region_rows = ref_df[ref_df["feature_type"].astype(str) == "region"].copy()
    if region_rows.empty:
        raise ValueError("CMETH contains no region rows")

    q = str(region_query)
    masks = [region_rows["region_id"].astype(str) == q]
    for col in ("display_name", "parent_region"):
        if col in region_rows.columns:
            masks.append(region_rows[col].astype(str) == q)
    hits = region_rows[np.logical_or.reduce(masks)]
    if hits.empty:
        examples = available_regions(ref_df)[:10]
        raise ValueError(
            f"Region {q!r} not found in CMETH. Example region ids: {examples}"
        )
    first = hits.iloc[0]
    return (
        str(first["region_id"]),
        str(first["chrom"]),
        int(first["start"]),
        int(first["end"]),
    )


def get_reference_cpg_rows(
    ref_df: pd.DataFrame,
    *,
    region_id: str,
    reference_hap: str = "pooled",
) -> pd.DataFrame:
    """Select ordered CMETH CpG rows for one parent region and reference hap key."""
    reference_hap = str(reference_hap).lower()
    rows = ref_df[
        (ref_df["feature_type"].astype(str) == "cpg")
        & (ref_df["parent_region"].astype(str) == str(region_id))
        & (ref_df["hap_key"].astype(str) == reference_hap)
    ].copy()
    return ordered_cpg_rows(rows)


def _locus_map(
    tbl: pd.DataFrame,
) -> dict[tuple[str, int, int], tuple[int, int]]:
    if tbl.empty:
        return {}
    return {
        (str(r.chrom), int(r.start), int(r.end)): (
            int(r.Nmod),
            int(r.Nvalid_cov),
        )
        for r in tbl.itertuples(index=False)
    }


def align_sample_tracks(
    cpg_rows: pd.DataFrame,
    sample_parts: Mapping[str, Path],
    chrom: str,
    start: int,
    end: int,
    *,
    mod_mode: str = "m",
    mod_codes: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Align one target sample's bedMethyl files to CMETH CpG rows.

    Returned tracks include direct hap1/hap2 tracks when available and derived
    allele_low/allele_high tracks for order-independent imprinting views.
    """
    ordered = ordered_cpg_rows(cpg_rows)
    loci = [
        (str(r.chrom), int(r.start), int(r.end))
        for r in ordered.itertuples(index=False)
    ]

    maps: dict[str, dict[tuple[str, int, int], tuple[int, int]]] = {}
    for hap in ("1", "2", "ungrouped"):
        fp = sample_parts.get(hap)
        if fp is None:
            maps[hap] = {}
            continue
        tbl = get_locus_table(
            str(fp),
            chrom,
            int(start),
            int(end),
            mode=str(mod_mode),
            codes=list(mod_codes) if mod_codes else None,
        )
        maps[hap] = _locus_map(tbl)

    n = len(loci)
    tracks = {
        "pooled": np.full(n, np.nan, dtype=float),
        "hap1": np.full(n, np.nan, dtype=float),
        "hap2": np.full(n, np.nan, dtype=float),
        "allele_low": np.full(n, np.nan, dtype=float),
        "allele_high": np.full(n, np.nan, dtype=float),
        "unphased": np.full(n, np.nan, dtype=float),
    }

    for i, locus in enumerate(loci):
        m1, v1 = maps["1"].get(locus, (0, 0))
        m2, v2 = maps["2"].get(locus, (0, 0))
        mu, vu = maps["ungrouped"].get(locus, (0, 0))
        mt = m1 + m2 + mu
        vt = v1 + v2 + vu
        if vt > 0:
            tracks["pooled"][i] = mt / vt
        if v1 > 0:
            tracks["hap1"][i] = m1 / v1
        if v2 > 0:
            tracks["hap2"][i] = m2 / v2
        if vu > 0:
            tracks["unphased"][i] = mu / vu
        if v1 > 0 and v2 > 0:
            b1 = m1 / v1
            b2 = m2 / v2
            tracks["allele_low"][i] = min(b1, b2)
            tracks["allele_high"][i] = max(b1, b2)

    return {
        k: v for k, v in tracks.items() if np.isfinite(v).any() or k == "pooled"
    }


def select_sample_tracks(
    sample_tracks: Mapping[str, np.ndarray],
    sample_track: str,
    legacy_show_alleles: bool | None = None,
) -> dict[str, np.ndarray]:
    """Select target sample tracks for plotting."""
    sample_track = str(sample_track).lower()
    if legacy_show_alleles is True and sample_track == "pooled":
        sample_track = "both_alleles"
    elif legacy_show_alleles is False:
        sample_track = "pooled"

    if sample_track == "pooled":
        return {"pooled": np.asarray(sample_tracks["pooled"], dtype=float)}
    if sample_track in {
        "hap1",
        "hap2",
        "allele_low",
        "allele_high",
        "unphased",
    }:
        if sample_track not in sample_tracks:
            raise ValueError(
                f"sample track {sample_track!r} is not available for this input"
            )
        return {
            sample_track: np.asarray(sample_tracks[sample_track], dtype=float)
        }
    if sample_track == "both_haps":
        out = {
            k: np.asarray(sample_tracks[k], dtype=float)
            for k in ("hap1", "hap2")
            if k in sample_tracks
        }
        if not out:
            raise ValueError(
                "hap1/hap2 tracks are not available; provide both hap1 and hap2 files"
            )
        return out
    if sample_track == "both_alleles":
        out = {
            k: np.asarray(sample_tracks[k], dtype=float)
            for k in ("allele_low", "allele_high")
            if k in sample_tracks
        }
        if not out:
            raise ValueError(
                "allele_low/allele_high tracks are not available; provide both hap1 and hap2 files"
            )
        return out
    if sample_track == "all":
        return {k: np.asarray(v, dtype=float) for k, v in sample_tracks.items()}
    raise ValueError(f"unsupported sample track: {sample_track!r}")


def build_cpg_profile_table(
    cpg_rows: pd.DataFrame,
    sample_tracks: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a doctor-facing CpG table with coordinates, reference stats, and sample beta values."""
    ordered = ordered_cpg_rows(cpg_rows)
    table = pd.DataFrame(
        {
            "cpg_index": np.arange(1, len(ordered) + 1),
            "chrom": ordered["chrom"].astype(str),
            "start": ordered["start"].astype(int),
            "end": ordered["end"].astype(int),
            "cpg_id": ordered["region_id"].astype(str),
            "parent_region": ordered["parent_region"].astype(str),
            "display_name": ordered.get(
                "display_name", pd.Series(".", index=ordered.index)
            ).astype(str),
            "ref_median": pd.to_numeric(
                ordered.get("meth_median"), errors="coerce"
            ),
            "ref_q25": pd.to_numeric(ordered.get("meth_q25"), errors="coerce"),
            "ref_q75": pd.to_numeric(ordered.get("meth_q75"), errors="coerce"),
            "ref_q05": pd.to_numeric(ordered.get("meth_q05"), errors="coerce"),
            "ref_q95": pd.to_numeric(ordered.get("meth_q95"), errors="coerce"),
        }
    )
    for key, values in (sample_tracks or {}).items():
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] == len(table):
            table[f"sample_{key}_beta"] = arr
            table[f"delta_{key}_vs_ref_median"] = arr - table[
                "ref_median"
            ].to_numpy(dtype=float)
    return table
