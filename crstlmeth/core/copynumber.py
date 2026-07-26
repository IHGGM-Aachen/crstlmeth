"""
crstlmeth.core.copynumber

core routines for extracting and normalizing copy-number coverage
from bedMethyl files over predefined genomic intervals.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .logging import log_event
from .parsers import get_region_stats
from .samples import parse_bedmethyl_name


def _parse_bedmethyl_name(path: str | Path) -> tuple[str, str] | None:
    parsed = parse_bedmethyl_name(path)
    if parsed is None or parsed.is_index:
        return None
    return parsed.sample_id, parsed.role


class CopyNumber:
    """
    copy-number computations using .bedmethyl inputs
    """

    # ------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------
    @staticmethod
    def _sample_id_from_path(p: str) -> str:
        """Return sample id from dot/underscore/dash bedMethyl hap filenames."""
        parsed = _parse_bedmethyl_name(p)
        if parsed:
            return parsed[0]
        return Path(p).name.split("_")[0]

    @staticmethod
    def _group_paths_by_sample(paths: List[str]) -> Dict[str, List[str]]:
        """Group flat list of bedmethyl files by sample id."""
        out: Dict[str, List[str]] = {}
        for p in paths:
            out.setdefault(CopyNumber._sample_id_from_path(p), []).append(p)
        return out

    @staticmethod
    def _classify_haps(paths: List[str]) -> Dict[str, str]:
        """Classify bedmethyl parts by hap key {1,2,ungrouped}; supports dot/underscore/dash names."""
        h: Dict[str, str] = {}
        for p in paths:
            parsed = _parse_bedmethyl_name(p)
            if parsed:
                _sid, hap = parsed
                h[hap] = p
        return h

    # ------------------------------------------------------------
    # full CN path (coverage/depth -> matrix -> cohort-normalised log2)
    # ------------------------------------------------------------
    @staticmethod
    def bedmethyl_coverage(
        bedmethyl_files: List[str],
        intervals: List[Tuple[str, int, int]],
        region_names: List[str],
        logger: logging.Logger | None = None,
    ) -> pd.DataFrame:
        """
        compute pooled coverage (Nvalid counts) per region and sample

        For each sample, sums coverage from hap1, hap2 and ungrouped.

        returns
        -------
        pandas.DataFrame with columns: region_name, coverage, sample_id
        """
        t0 = time.perf_counter()

        by_sample = CopyNumber._group_paths_by_sample(bedmethyl_files)
        rows: List[pd.DataFrame] = []

        for sid, paths in sorted(by_sample.items()):
            h = CopyNumber._classify_haps(paths)
            covs: List[int] = []
            for chrom, start, end in intervals:
                v_sum = 0
                for key in ("1", "2", "ungrouped"):
                    p = h.get(key)
                    if p:
                        _, v = get_region_stats(p, chrom, start, end)
                        v_sum += v
                covs.append(int(v_sum))

            rows.append(
                pd.DataFrame(
                    {
                        "region_name": region_names,
                        "coverage": covs,
                        "sample_id": sid,
                    }
                )
            )

        df = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                {"region_name": [], "coverage": [], "sample_id": []}
            )
        )

        if logger:
            log_event(
                logger,
                event="copy_number",
                cmd="CopyNumber.bedmethyl_coverage",
                params={
                    "n_samples": len(by_sample),
                    "n_regions": len(region_names),
                },
                runtime_s=time.perf_counter() - t0,
            )
        return df

    @staticmethod
    def bedmethyl_depth_per_bp(
        bedmethyl_files: List[str],
        intervals: List[Tuple[str, int, int]],
        region_names: List[str],
        logger: logging.Logger | None = None,
    ) -> pd.DataFrame:
        """
        compute pooled depth_per_bp per region and sample (matches .cmeth full 'depth_per_bp')

        For each sample, sums Nvalid across hap1/2/ungrouped and divides by region length.

        returns
        -------
        pandas.DataFrame with columns: region_name, depth_per_bp, sample_id
        """
        t0 = time.perf_counter()

        by_sample = CopyNumber._group_paths_by_sample(bedmethyl_files)
        rows: List[pd.DataFrame] = []

        lengths = np.array(
            [max(1, e - s) for (_, s, e) in intervals], dtype=float
        )

        for sid, paths in sorted(by_sample.items()):
            h = CopyNumber._classify_haps(paths)
            depth = np.zeros(len(intervals), dtype=float)
            for j, (c, s, e) in enumerate(intervals):
                v_sum = 0
                for key in ("1", "2", "ungrouped"):
                    p = h.get(key)
                    if p:
                        _, v = get_region_stats(p, c, s, e)
                        v_sum += v
                depth[j] = v_sum / lengths[j]
            rows.append(
                pd.DataFrame(
                    {
                        "region_name": region_names,
                        "depth_per_bp": depth,
                        "sample_id": sid,
                    }
                )
            )

        df = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                {"region_name": [], "depth_per_bp": [], "sample_id": []}
            )
        )

        if logger:
            log_event(
                logger,
                event="copy_number",
                cmd="CopyNumber.bedmethyl_depth_per_bp",
                params={
                    "n_samples": len(by_sample),
                    "n_regions": len(region_names),
                },
                runtime_s=time.perf_counter() - t0,
            )
        return df

    @staticmethod
    def compute_region_stats(
        bedcov_df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """
        pivot long-format coverage to (n_samples, n_regions) matrix

        expects columns: region_name, coverage, sample_id
        """
        pivot = bedcov_df.pivot_table(
            index="sample_id", columns="region_name", values="coverage"
        ).sort_index()
        return pivot.values, list(pivot.columns)

    @staticmethod
    def normalise_to_cohort_mean(mat: np.ndarray) -> np.ndarray:
        """
        divide each region by the cohort mean for that region
        """
        mu = np.nanmean(mat, axis=0)
        mu[~np.isfinite(mu)] = 1.0
        mu[mu == 0] = 1.0
        return mat / mu

    @staticmethod
    def to_log2(arr: np.ndarray) -> np.ndarray:
        """
        safe elementwise log2 (<=0 mapped to NaN)
        """
        x = np.asarray(arr, dtype=float)
        out = np.full_like(x, np.nan, dtype=float)
        np.log2(x, out=out, where=x > 0)
        return out

    # ------------------------------------------------------------
    # aggregated CN path (target-only recipe -> log2)
    # ------------------------------------------------------------
    @staticmethod
    def _counts_for_sample(
        paths: List[str],
        intervals: List[Tuple[str, int, int]],
    ) -> np.ndarray:
        """
        pooled Nvalid counts for one sample across its bedmethyl parts

        returns
        -------
        numpy.ndarray shape (n_regions,)
        """
        counts = np.zeros(len(intervals), dtype=float)
        for j, (c, s, e) in enumerate(intervals):
            v_tot = 0
            for p in paths:
                _, v = get_region_stats(p, c, s, e)
                v_tot += v
            counts[j] = v_tot
        return counts

    @staticmethod
    def target_log2_for_aggregated(
        bedmethyl_files: List[str],
        intervals: List[Tuple[str, int, int]],
        *,
        cn_norm: str = "per-sample-median",
        trim_frac: float | None = None,
        logger: logging.Logger | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        compute target log2 ratios compatible with aggregated CN references

        IMPORTANT: uses pooled Nvalid counts per region (NOT per-bp), because
        aggregated references were built from counts then per-sample normalization.

        steps
        -----
        1) group by sample; pool parts; compute Nvalid counts per region
        2) normalise within each sample using cn_norm:
           - 'per-sample-median' (default)
           - 'per-sample-trimmed-mean' with central (1-2*trim_frac), trim_frac default 0.1
        3) log2 transform

        returns
        -------
        (matrix shape (n_samples, n_regions), ordered_sample_ids)
        """
        t0 = time.perf_counter()

        by_sample = CopyNumber._group_paths_by_sample(bedmethyl_files)
        if not by_sample:
            if logger:
                log_event(
                    logger,
                    event="copy_number",
                    cmd="CopyNumber.target_log2_for_aggregated",
                    params={
                        "n_samples": 0,
                        "n_regions": len(intervals),
                        "cn_norm": cn_norm,
                    },
                    runtime_s=time.perf_counter() - t0,
                )
            return np.zeros((0, len(intervals)), dtype=float), []

        mats: List[np.ndarray] = []
        sids: List[str] = []

        for sid, paths in sorted(by_sample.items()):
            counts = CopyNumber._counts_for_sample(paths, intervals)
            x = counts[counts > 0]

            if cn_norm == "per-sample-trimmed-mean":
                if x.size == 0:
                    baseline = 1.0
                else:
                    tf = 0.1 if trim_frac is None else float(trim_frac)
                    lo = np.quantile(x, tf)
                    hi = np.quantile(x, 1 - tf)
                    keep = (x >= lo) & (x <= hi)
                    baseline = x[keep].mean() if keep.any() else x.mean()
            else:
                baseline = np.nanmedian(x) if x.size else 1.0

            baseline = baseline if baseline > 0 else 1.0
            ratio = counts / baseline
            mats.append(ratio.reshape(1, -1))
            sids.append(sid)

        mat = np.vstack(mats)
        log2 = CopyNumber.to_log2(mat)

        if logger:
            log_event(
                logger,
                event="copy_number",
                cmd="CopyNumber.target_log2_for_aggregated",
                params={
                    "n_samples": len(sids),
                    "n_regions": len(intervals),
                    "cn_norm": cn_norm,
                    "trim_frac": (
                        None if trim_frac is None else float(trim_frac)
                    ),
                },
                runtime_s=time.perf_counter() - t0,
            )
        return log2, sids


# ---------------------------------------------------------------------------
# Legacy small-array helpers kept for backwards-compatible tests/users.
# ---------------------------------------------------------------------------
def get_read_count(df: pd.DataFrame) -> np.ndarray:
    """Pivot long bedcov-like data to samples x regions read-count matrix."""
    piv = df.pivot_table(
        index="sample_id",
        columns="region_name",
        values="read_count",
        aggfunc="first",
    ).sort_index()
    return piv.to_numpy(dtype=float)


def get_average_read_count(df: pd.DataFrame) -> np.ndarray:
    """Mean read count per region in first-seen region order."""
    order = list(dict.fromkeys(df["region_name"].astype(str).tolist()))
    return (
        df.groupby("region_name", sort=False)["read_count"]
        .mean()
        .reindex(order)
        .to_numpy(dtype=float)
    )


def get_read_count_deviation(df: pd.DataFrame) -> np.ndarray:
    """Population standard deviation of read counts per region in first-seen order."""
    order = list(dict.fromkeys(df["region_name"].astype(str).tolist()))
    return (
        df.groupby("region_name", sort=False)["read_count"]
        .std(ddof=0)
        .reindex(order)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def get_target_deviation(
    refs: pd.DataFrame,
    target: pd.DataFrame,
    *,
    z_score_threshold: float = 2.0,
) -> np.ndarray:
    """Flag target regions whose read count differs from the reference mean by z threshold."""
    order = list(dict.fromkeys(target["region_name"].astype(str).tolist()))
    ref_mean = (
        refs.groupby("region_name", sort=False)["read_count"]
        .mean()
        .reindex(order)
        .to_numpy(dtype=float)
    )
    ref_sd = (
        refs.groupby("region_name", sort=False)["read_count"]
        .std(ddof=0)
        .reindex(order)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    tgt = (
        target.groupby("region_name", sort=False)["read_count"]
        .mean()
        .reindex(order)
        .to_numpy(dtype=float)
    )
    z = np.divide(
        tgt - ref_mean,
        ref_sd,
        out=np.zeros_like(tgt, dtype=float),
        where=ref_sd > 0,
    )
    return z >= float(z_score_threshold)
