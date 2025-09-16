"""
crstlmeth/core/references.py

wrappers around cmeth.py for building, reading, writing .cmeth files
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from crstlmeth.core.cmeth import CMethFile, _md5, parse_header_meta
from crstlmeth.core.logging import log_event
from crstlmeth.core.parsers import get_region_stats
from crstlmeth.core.regions import load_intervals

__all__ = [
    "create_cmeth_aggregated",
    "create_cmeth_full",
    "write_cmeth_aggregated",
    "write_cmeth_full",
    "parse_cmeth_header",
    "read_cmeth",
]


# ────────────────────────────────────────────────────────────────────
# tiny helpers for grouping and naming
# ────────────────────────────────────────────────────────────────────
def _sample_id_from_path(p: Path | str) -> str:
    """
    derive sample id from filename (prefix before first underscore)
    """
    name = Path(p).name
    return name.split("_")[0] if "_" in name else name.split(".")[0]


def _group_paths_by_sample(paths: List[Path]) -> Dict[str, List[Path]]:
    """
    return {sample_id: [paths...]} from a flat list of files
    """
    out: Dict[str, List[Path]] = {}
    for p in paths:
        out.setdefault(_sample_id_from_path(p), []).append(p)
    return out


def _classify_haps(paths: List[Path]) -> Dict[str, Path]:
    """
    map a list of paths into hap buckets {"1","2","ungrouped"} if present
    returns only keys that exist
    """
    h: Dict[str, Path] = {}
    for p in paths:
        name = p.name
        if "_1." in name and "bedmethyl" in name:
            h["1"] = p
        elif "_2." in name and "bedmethyl" in name:
            h["2"] = p
        elif "_ungrouped." in name and "bedmethyl" in name:
            h["ungrouped"] = p
    return h


# ────────────────────────────────────────────────────────────────────
# writers (thin wrappers over CMethFile)
# ────────────────────────────────────────────────────────────────────
def write_cmeth_aggregated(
    *,
    meth_df: pd.DataFrame | None,
    cn_df: pd.DataFrame | None,
    out: Path,
    meta: Dict[str, str | int] | None = None,
    bed_path: Path | None = None,
    bins_meth: list[float] | None = None,
    bins_cn_log2: list[float] | None = None,
    cn_norm: str = "per-sample-median",
    k_min: int = 5,
    logger: logging.Logger | None = None,
) -> Path:
    """
    write a .cmeth file (mode=aggregated) with anonymised, plot-ready stats
    """
    t0 = time.perf_counter()
    meta = dict(meta or {})
    meta.setdefault("mode", "aggregated")
    meta.setdefault("cn_norm", cn_norm)
    meta.setdefault("k_min", int(k_min))
    if bins_meth is not None:
        meta["bins_meth"] = ",".join(str(x) for x in bins_meth)
    if bins_cn_log2 is not None:
        meta["bins_cn_log2"] = ",".join(str(x) for x in bins_cn_log2)
    if bed_path and bed_path.exists():
        meta["md5_bed"] = _md5(bed_path)

    cm = CMethFile.build_aggregated(meth_df=meth_df, cn_df=cn_df, meta=meta)
    try:
        outp = cm.write(out)
        if logger:
            log_event(
                logger,
                event="write-cmeth",
                cmd="write_cmeth_aggregated",
                params={
                    "out": str(outp),
                    "mode": "aggregated",
                    "n_rows": int(len(cm.df)),
                },
                message="ok",
                runtime_s=time.perf_counter() - t0,
            )
        return outp
    except Exception as e:
        if logger:
            log_event(
                logger,
                level=logging.ERROR,
                event="write-cmeth",
                cmd="write_cmeth_aggregated",
                params={"out": str(out), "mode": "aggregated"},
                message=str(e),
                runtime_s=time.perf_counter() - t0,
            )
        raise


def write_cmeth_full(
    rows: pd.DataFrame,
    out: Path,
    meta: Dict[str, str | int] | None = None,
    bed_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """
    write a .cmeth file (mode=full) with per-sample rows
    """
    t0 = time.perf_counter()
    meta = dict(meta or {})
    meta.setdefault("mode", "full")
    meta.setdefault("denom_dedup", "true")
    if bed_path and bed_path.exists():
        meta["md5_bed"] = _md5(bed_path)

    cm = CMethFile.build_full(rows, meta=meta)
    try:
        outp = cm.write(out)
        if logger:
            log_event(
                logger,
                event="write-cmeth",
                cmd="write_cmeth_full",
                params={
                    "out": str(outp),
                    "mode": "full",
                    "n_rows": int(len(cm.df)),
                },
                message="ok",
                runtime_s=time.perf_counter() - t0,
            )
        return outp
    except Exception as e:
        if logger:
            log_event(
                logger,
                level=logging.ERROR,
                event="write-cmeth",
                cmd="write_cmeth_full",
                params={"out": str(out), "mode": "full"},
                message=str(e),
                runtime_s=time.perf_counter() - t0,
            )
        raise


# ────────────────────────────────────────────────────────────────────
# readers (thin wrappers over CMethFile)
# ────────────────────────────────────────────────────────────────────
def parse_cmeth_header(
    path: Path, logger: logging.Logger | None = None
) -> dict[str, str]:
    """
    return metadata dict parsed from a .cmeth file header
    """
    t0 = time.perf_counter()
    try:
        meta = parse_header_meta(path)
        if logger:
            log_event(
                logger,
                event="parse-cmeth-header",
                cmd="parse_cmeth_header",
                params={"path": str(path)},
                message="ok",
                runtime_s=time.perf_counter() - t0,
            )
        return meta
    except Exception as e:
        if logger:
            log_event(
                logger,
                level=logging.ERROR,
                event="parse-cmeth-header",
                cmd="parse_cmeth_header",
                params={"path": str(path)},
                message=str(e),
                runtime_s=time.perf_counter() - t0,
            )
        raise


def read_cmeth(
    path: Path, logger: logging.Logger | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    load a .cmeth file and return (dataframe, metadata)
    """
    t0 = time.perf_counter()
    try:
        cm = CMethFile.read(path)
        if logger:
            log_event(
                logger,
                event="read-cmeth",
                cmd="read_cmeth",
                params={
                    "path": str(path),
                    "mode": cm.mode,
                    "n_rows": int(len(cm.df)),
                },
                message="ok",
                runtime_s=time.perf_counter() - t0,
            )
        return cm.df, cm.meta
    except Exception as e:
        if logger:
            log_event(
                logger,
                level=logging.ERROR,
                event="read-cmeth",
                cmd="read_cmeth",
                params={"path": str(path)},
                message=str(e),
                runtime_s=time.perf_counter() - t0,
            )
        raise


# ────────────────────────────────────────────────────────────────────
# builders (cohort math lives here; IO delegated to CMethFile)
# ────────────────────────────────────────────────────────────────────
def _nan_percentile(x: np.ndarray, q: float) -> float:
    """
    nan-robust percentile for a 1-d vector
    """
    if x.size == 0:
        return float("nan")
    return float(np.nanpercentile(x, q))


def create_cmeth_aggregated(
    *,
    kit: str | Path,
    bedmethyl_paths: List[Path],
    out: Path,
    logger: logging.Logger | None = None,
    cn_norm: str = "per-sample-median",
    k_min: int = 5,
    include_optional_stats: bool = True,
    bins_meth: list[float] | None = None,
    bins_cn_log2: list[float] | None = None,
) -> Path:
    """
    build an anonymised aggregated reference and write *.cmeth (mode=aggregated)

    produces two section blocks:
      - section="meth": per-region, per-hap_key (pooled, 1, 2) cohort quantiles
      - section="cn"  : per-region cohort log2-ratio quantiles
    """
    t0 = time.perf_counter()
    intervals, region_names = load_intervals(kit)
    if not intervals:
        raise ValueError(f"no intervals found for kit/BED: {kit!r}")

    groups = _group_paths_by_sample([Path(p) for p in bedmethyl_paths])
    sample_ids = sorted(groups)
    n_samples_total = len(sample_ids)
    if n_samples_total < max(2, k_min):
        raise ValueError(
            f"need at least {max(2, k_min)} samples for aggregated reference"
        )

    # preallocate per-region per-sample arrays for meth and depth
    n_regions = len(intervals)
    # keys we aggregate over
    hap_keys = ("pooled", "1", "2")

    # store per-hap_key matrices: shape (n_samples, n_regions), filled with NaN
    meth_by_hap = {
        k: np.full((n_samples_total, n_regions), np.nan, dtype=float)
        for k in hap_keys
    }
    depth_by_hap = {
        k: np.full((n_samples_total, n_regions), np.nan, dtype=float)
        for k in hap_keys
    }

    # also track frac_ungrouped at pooled level
    frac_ungrouped = np.full((n_samples_total, n_regions), np.nan, dtype=float)

    # coverage per sample per region for CN
    cov_mat = np.zeros((n_samples_total, n_regions), dtype=float)

    # fill per-sample metrics
    for i, sid in enumerate(sample_ids):
        h = _classify_haps(groups[sid])

        for j, (chrom, start, end) in enumerate(intervals):
            L = max(1, end - start)

            # hap parts
            m1 = v1 = m2 = v2 = mu = vu = 0
            if "1" in h:
                m1, v1 = get_region_stats(str(h["1"]), chrom, start, end)
            if "2" in h:
                m2, v2 = get_region_stats(str(h["2"]), chrom, start, end)
            if "ungrouped" in h:
                mu, vu = get_region_stats(
                    str(h["ungrouped"]), chrom, start, end
                )

            # pooled totals
            mt = m1 + m2 + mu
            vt = v1 + v2 + vu

            # methylation levels per hap_key (if covered)
            if vt > 0:
                meth_by_hap["pooled"][i, j] = mt / vt
                depth_by_hap["pooled"][i, j] = vt / L
                frac_ungrouped[i, j] = (vu / vt) if vt > 0 else np.nan

            if v1 > 0:
                meth_by_hap["1"][i, j] = m1 / v1
                depth_by_hap["1"][i, j] = v1 / L

            if v2 > 0:
                meth_by_hap["2"][i, j] = m2 / v2
                depth_by_hap["2"][i, j] = v2 / L

            # CN coverage uses pooled valid counts
            cov_mat[i, j] = vt

    # build methylation aggregated rows
    meth_rows: List[dict] = []
    for j, ((chrom, start, end), region) in enumerate(
        zip(intervals, region_names, strict=False)
    ):
        for k in hap_keys:
            vec = meth_by_hap[k][:, j]
            cov_vec = depth_by_hap[k][:, j]
            n_covered = int(np.isfinite(vec).sum())
            if n_covered == 0:
                continue  # no cohort signal for this hap_key at this interval

            row = {
                "section": "meth",
                "region": region,
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "hap_key": k,
                "n": int(n_samples_total),
                "n_covered": int(n_covered),
                "meth_median": _nan_percentile(vec, 50),
                "meth_q25": _nan_percentile(vec, 25),
                "meth_q75": _nan_percentile(vec, 75),
            }
            if include_optional_stats:
                row["meth_mean"] = float(np.nanmean(vec))
                row["meth_sd"] = float(np.nanstd(vec, ddof=0))
                row["meth_q05"] = _nan_percentile(vec, 5)
                row["meth_q95"] = _nan_percentile(vec, 95)
                # depth summaries
                row["depth_mean"] = float(np.nanmean(cov_vec))
                row["depth_q05"] = _nan_percentile(cov_vec, 5)
                row["depth_q25"] = _nan_percentile(cov_vec, 25)
                row["depth_median"] = _nan_percentile(cov_vec, 50)
                row["depth_q75"] = _nan_percentile(cov_vec, 75)
                row["depth_q95"] = _nan_percentile(cov_vec, 95)
                # ungrouped fraction only on pooled row to avoid duplication
                if k == "pooled":
                    fu = frac_ungrouped[:, j]
                    if np.isfinite(fu).any():
                        row["frac_ungrouped_mean"] = float(np.nanmean(fu))
                        row["frac_ungrouped_median"] = _nan_percentile(fu, 50)

            meth_rows.append(row)

    meth_df = pd.DataFrame.from_records(meth_rows) if meth_rows else None

    # build copy-number aggregated rows
    cn_rows: List[dict] = []
    if n_samples_total >= max(2, k_min):
        # per-sample normalisation
        if cn_norm.lower() == "per-sample-median":
            norms = np.nanmedian(np.where(cov_mat > 0, cov_mat, np.nan), axis=1)
        else:
            raise ValueError(f"unsupported cn_norm: {cn_norm!r}")

        # guard against zeros
        norms = np.where(norms <= 0, np.nan, norms)

        # ratios and log2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = cov_mat / norms[:, None]
            ratio = np.where(ratio > 0, ratio, np.nan)
            log2_ratio = np.log2(ratio)

        for j, ((chrom, start, end), region) in enumerate(
            zip(intervals, region_names, strict=False)
        ):
            vec = log2_ratio[:, j]
            n_used = int(np.isfinite(vec).sum())
            if n_used == 0:
                continue

            row = {
                "section": "cn",
                "region": region,
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "n": int(n_samples_total),
                "ratio_median_log2": _nan_percentile(vec, 50),
                "ratio_q25_log2": _nan_percentile(vec, 25),
                "ratio_q75_log2": _nan_percentile(vec, 75),
            }
            if include_optional_stats:
                row["ratio_mean_log2"] = float(np.nanmean(vec))
                row["ratio_sd_log2"] = float(np.nanstd(vec, ddof=0))
                row["ratio_q05_log2"] = _nan_percentile(vec, 5)
                row["ratio_q95_log2"] = _nan_percentile(vec, 95)
                if bins_cn_log2:
                    counts, _ = np.histogram(
                        vec[np.isfinite(vec)],
                        bins=np.array(bins_cn_log2, dtype=float),
                    )
                    row["ratio_hist_log2"] = ",".join(
                        str(int(c)) for c in counts
                    )

            cn_rows.append(row)

    cn_df = pd.DataFrame.from_records(cn_rows) if cn_rows else None

    # emit file
    outp = write_cmeth_aggregated(
        meth_df=meth_df,
        cn_df=cn_df,
        out=out,
        meta={"kit": str(kit)},
        bed_path=Path(kit) if Path(kit).suffix.lower() == ".bed" else None,
        bins_meth=bins_meth,
        bins_cn_log2=bins_cn_log2,
        cn_norm=cn_norm,
        k_min=k_min,
        logger=logger,
    )

    if logger:
        log_event(
            logger,
            event="reference-create",
            cmd="create_cmeth_aggregated",
            params={"mode": "aggregated", "kit": str(kit), "out": str(outp)},
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return outp


def create_cmeth_full(
    *,
    kit: str | Path,
    bedmethyl_paths: List[Path],
    out: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """
    build a per-sample reference and write *.cmeth* (mode=full)

    rows:
      sample_id, region, chrom, start, end, hap, n_valid, n_mod, meth, depth_per_bp

    for each sample and region, emit one row per present hap in {"1","2","ungrouped"}
    plus one pooled row that sums all present hap parts.
    """
    t0 = time.perf_counter()
    intervals, region_names = load_intervals(kit)
    if not intervals:
        raise ValueError(f"no intervals found for kit/BED: {kit!r}")

    groups = _group_paths_by_sample([Path(p) for p in bedmethyl_paths])
    sample_ids = sorted(groups)
    rows: List[dict] = []

    for sid in sample_ids:
        parts = _classify_haps(groups[sid])

        for (chrom, start, end), region in zip(
            intervals, region_names, strict=False
        ):
            L = max(1, end - start)

            # hap-specific
            recs: List[Tuple[str, int, int]] = []  # (hap_key, n_mod, n_valid)
            for hk in ("1", "2", "ungrouped"):
                if hk in parts:
                    m, v = get_region_stats(str(parts[hk]), chrom, start, end)
                    if v > 0:
                        rows.append(
                            {
                                "sample_id": sid,
                                "region": region,
                                "chrom": chrom,
                                "start": int(start),
                                "end": int(end),
                                "hap": hk,
                                "n_valid": int(v),
                                "n_mod": int(m),
                                "meth": float(m / v),
                                "depth_per_bp": float(v / L),
                            }
                        )
                    recs.append((hk, m, v))

            # pooled across present parts
            if recs:
                pm = sum(m for _, m, _ in recs)
                pv = sum(v for _, _, v in recs)
                if pv > 0:
                    rows.append(
                        {
                            "sample_id": sid,
                            "region": region,
                            "chrom": chrom,
                            "start": int(start),
                            "end": int(end),
                            "hap": "pooled",
                            "n_valid": int(pv),
                            "n_mod": int(pm),
                            "meth": float(pm / pv),
                            "depth_per_bp": float(pv / L),
                        }
                    )

    df_full = pd.DataFrame.from_records(
        rows,
        columns=[
            "sample_id",
            "region",
            "chrom",
            "start",
            "end",
            "hap",
            "n_valid",
            "n_mod",
            "meth",
            "depth_per_bp",
        ],
    )

    outp = write_cmeth_full(
        df_full,
        out=out,
        meta={"kit": str(kit)},
        bed_path=Path(kit) if Path(kit).suffix.lower() == ".bed" else None,
        logger=logger,
    )

    if logger:
        log_event(
            logger,
            event="reference-create",
            cmd="create_cmeth_full",
            params={
                "mode": "full",
                "kit": str(kit),
                "out": str(outp),
                "n_rows": int(len(df_full)),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return outp
