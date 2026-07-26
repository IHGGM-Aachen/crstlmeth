"""
crstlmeth/core/references.py

Build, read, query, and write CMETH references.

The reference is a rich aggregated cohort summary with an embedded target BED
block. It is intended to be written as `.cmeth.gz` and queried through tabix.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from crstlmeth.core.cmeth import (
    CMETH_COLUMNS,
    CMETH_VERSION,
    CN_LOG2_HIST_BINS,
    METH_HIST_BINS,
    MVAL_HIST_BINS,
    CMethFile,
    parse_header_meta,
    read_cmeth_region,
    read_target_bed,
)
from crstlmeth.core.logging import log_event
from crstlmeth.core.parsers import get_locus_table
from crstlmeth.core.regions import load_intervals

__all__ = [
    "create_cmeth_reference",
    "write_cmeth_reference",
    "parse_cmeth_header",
    "read_cmeth",
    "query_cmeth",
    "read_cmeth_target_bed",
]

_BEDM_RE = re.compile(
    r"""^(?P<sample>.+?)[._-](?P<hap>1|2|ungrouped)(?:[._-]\w+)*\.bedmethyl(?:\.gz)?$""",
    re.IGNORECASE,
)


def _group_paths_by_sample_and_hap(
    paths: Iterable[Path],
) -> Dict[str, Dict[str, Path]]:
    """Return {sample_id: {"1"|"2"|"ungrouped": Path}} from bedMethyl paths."""
    out: Dict[str, Dict[str, Path]] = {}
    for p in paths:
        p = Path(p)
        m = _BEDM_RE.match(p.name)
        if not m:
            continue
        out.setdefault(m["sample"], {})[m["hap"].lower()] = p
    return out


def _target_bed_lines(
    intervals: list[tuple[str, int, int]],
    region_names: list[str],
) -> list[str]:
    return [
        f"{chrom}\t{int(start)}\t{int(end)}\t{region}"
        for (chrom, start, end), region in zip(
            intervals, region_names, strict=False
        )
    ]


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def _q(x: np.ndarray, p: float) -> float:
    x = _finite(x)
    return float(np.nanpercentile(x, p)) if x.size else float("nan")


def _mad(x: np.ndarray) -> float:
    x = _finite(x)
    if not x.size:
        return float("nan")
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _hist_counts(x: np.ndarray, bins: Iterable[float]) -> str:
    x = _finite(x)
    bins_arr = np.asarray(tuple(bins), dtype=float)
    if not x.size:
        return "."
    counts, _ = np.histogram(x, bins=bins_arr)
    return ",".join(str(int(c)) for c in counts)


def _under_over(
    x: np.ndarray, bins: Iterable[float]
) -> tuple[int | float, int | float]:
    x = _finite(x)
    bins_arr = np.asarray(tuple(bins), dtype=float)
    if not x.size:
        return (float("nan"), float("nan"))
    return (int((x < bins_arr[0]).sum()), int((x > bins_arr[-1]).sum()))


def _summary(
    prefix: str, x: np.ndarray, *, hist_bins: Iterable[float] | None = None
) -> dict[str, object]:
    x = _finite(x)
    out: dict[str, object] = {
        f"{prefix}_mean": float(np.nanmean(x)) if x.size else float("nan"),
        f"{prefix}_sd": float(np.nanstd(x, ddof=0)) if x.size else float("nan"),
        f"{prefix}_median": float(np.nanmedian(x)) if x.size else float("nan"),
        f"{prefix}_mad": _mad(x),
        f"{prefix}_min": float(np.nanmin(x)) if x.size else float("nan"),
        f"{prefix}_q01": _q(x, 1),
        f"{prefix}_q05": _q(x, 5),
        f"{prefix}_q10": _q(x, 10),
        f"{prefix}_q25": _q(x, 25),
        f"{prefix}_q75": _q(x, 75),
        f"{prefix}_q90": _q(x, 90),
        f"{prefix}_q95": _q(x, 95),
        f"{prefix}_q99": _q(x, 99),
        f"{prefix}_max": float(np.nanmax(x)) if x.size else float("nan"),
    }
    if hist_bins is not None:
        out[f"{prefix}_hist10"] = _hist_counts(x, hist_bins)
        if prefix in {"mval", "cn_log2"}:
            under, over = _under_over(x, hist_bins)
            out[f"{prefix}_hist_underflow"] = under
            out[f"{prefix}_hist_overflow"] = over
    return out


def _compact_summary(prefix: str, x: np.ndarray) -> dict[str, object]:
    x = _finite(x)
    return {
        f"{prefix}_mean": float(np.nanmean(x)) if x.size else float("nan"),
        f"{prefix}_sd": float(np.nanstd(x, ddof=0)) if x.size else float("nan"),
        f"{prefix}_median": float(np.nanmedian(x)) if x.size else float("nan"),
        f"{prefix}_q25": _q(x, 25),
        f"{prefix}_q75": _q(x, 75),
    }


def _m_values(beta: np.ndarray, eps: float) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    out = np.full_like(b, np.nan, dtype=float)
    valid = np.isfinite(b)
    b = np.clip(b, 0.0, 1.0)
    out[valid] = np.log2((b[valid] + eps) / (1.0 - b[valid] + eps))
    return out


def _beta_fit(mean: float, sd: float) -> tuple[float, float, str]:
    if not (np.isfinite(mean) and np.isfinite(sd)):
        return (float("nan"), float("nan"), "no_data")
    if mean <= 0 or mean >= 1:
        return (float("nan"), float("nan"), "mean_out_of_bounds")
    var = sd**2
    max_var = mean * (1.0 - mean)
    if var <= 0:
        return (float("nan"), float("nan"), "zero_variance")
    if var >= max_var:
        return (float("nan"), float("nan"), "overdispersed")
    common = max_var / var - 1.0
    return (float(mean * common), float((1.0 - mean) * common), "ok")


def _status(n: int) -> str:
    return "ok" if int(n) > 0 else "no_data"


def _empty_row_base(
    *,
    chrom: str,
    start: int,
    end: int,
    region: str,
    hap_key: str,
    n_ref: int,
    feature_type: str = "region",
    parent_region: str = ".",
    display_name: str | None = None,
    strand: str = ".",
) -> dict[str, object]:
    row = {col: np.nan for col in CMETH_COLUMNS}
    row.update(
        {
            "chrom": str(chrom),
            "start": int(start),
            "end": int(end),
            "feature_type": feature_type,
            "region_id": str(region),
            "parent_region": str(parent_region),
            "display_name": str(display_name or region),
            "hap_key": hap_key,
            "strand": strand,
            "length_bp": int(max(1, int(end) - int(start))),
            "cpg_count": np.nan,
            "probe_count": np.nan,
            "gene": ".",
            "transcript": ".",
            "annotation": ".",
            "n_ref": int(n_ref),
            "n_meth": 0,
            "n_cn": 0,
            "n_depth": 0,
            "n_hap_resolved": 0,
            "n_unphased": 0,
            "meth_status": "no_data",
            "cn_status": "no_data",
            "phasing_status": "no_data",
            "row_status": "no_data",
        }
    )
    return row


def _add_methylation_stats(
    row: dict[str, object],
    *,
    beta: np.ndarray,
    nmod: np.ndarray,
    nvalid: np.ndarray,
    depth: np.ndarray,
    mvalue_eps: float,
) -> None:
    beta = np.asarray(beta, dtype=float)
    nmod = np.asarray(nmod, dtype=float)
    nvalid = np.asarray(nvalid, dtype=float)
    depth = np.asarray(depth, dtype=float)
    valid_beta = np.isfinite(beta)
    row["n_meth"] = int(valid_beta.sum())
    row["n_depth"] = int(np.isfinite(depth).sum())
    row["meth_status"] = _status(int(row["n_meth"]))
    row["meth_nmod_sum"] = (
        int(np.nansum(nmod[valid_beta])) if valid_beta.any() else np.nan
    )
    row["meth_nvalid_sum"] = (
        int(np.nansum(nvalid[valid_beta])) if valid_beta.any() else np.nan
    )
    row.update(_summary("meth", beta, hist_bins=METH_HIST_BINS))
    mval = _m_values(beta, mvalue_eps)
    row.update(_summary("mval", mval, hist_bins=MVAL_HIST_BINS))
    alpha, beta_param, fit_status = _beta_fit(
        float(row["meth_mean"]), float(row["meth_sd"])
    )
    row["beta_alpha"] = alpha
    row["beta_beta"] = beta_param
    row["beta_fit_status"] = fit_status
    row.update(_summary("depth", depth))
    row.update(_compact_summary("nvalid", nvalid))


def _add_phasing_stats(
    row: dict[str, object],
    *,
    frac_unphased: np.ndarray,
    hap_balance: np.ndarray,
    n_hap_resolved: int,
    n_unphased: int,
) -> None:
    row["n_hap_resolved"] = int(n_hap_resolved)
    row["n_unphased"] = int(n_unphased)
    row["phasing_status"] = _status(int(n_hap_resolved))
    row.update(_summary("frac_unphased", frac_unphased))
    row.update(_compact_summary("hap_balance", hap_balance))


def _add_allele_pair_stats(
    row: dict[str, object], *, allele_low: np.ndarray, allele_high: np.ndarray
) -> None:
    low = np.asarray(allele_low, dtype=float)
    high = np.asarray(allele_high, dtype=float)
    valid = np.isfinite(low) & np.isfinite(high)
    gap = np.full_like(low, np.nan, dtype=float)
    mean = np.full_like(low, np.nan, dtype=float)
    gap[valid] = high[valid] - low[valid]
    mean[valid] = (high[valid] + low[valid]) / 2.0
    row.update(_summary("allele_gap", gap))
    row.update(_compact_summary("allele_mean", mean))


def _add_cn_stats(
    row: dict[str, object], *, cn_log2: np.ndarray, cn_cov: np.ndarray
) -> None:
    row["n_cn"] = int(np.isfinite(cn_log2).sum())
    row["cn_status"] = _status(int(row["n_cn"]))
    row.update(_summary("cn_log2", cn_log2, hist_bins=CN_LOG2_HIST_BINS))
    row.update(_compact_summary("cn_cov", cn_cov))


def _ensure_output_suffix(path: Path) -> Path:
    path = Path(path)
    if path.suffix == ".cmeth":
        return path.with_suffix(".cmeth.gz")
    return path


def write_cmeth_reference(
    rows: pd.DataFrame,
    *,
    out: Path,
    meta: dict[str, object] | None = None,
    target_bed: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Write a CMETH reference."""
    t0 = time.perf_counter()
    out = _ensure_output_suffix(Path(out))
    cm = CMethFile.build_reference(
        rows, meta=dict(meta or {}), target_bed=list(target_bed or [])
    )
    outp = cm.write(out)
    if logger:
        log_event(
            logger,
            event="write-cmeth",
            cmd="write_cmeth_reference",
            params={
                "out": str(outp),
                "n_rows": int(len(cm.df)),
                "version": CMETH_VERSION,
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return outp


def parse_cmeth_header(
    path: Path, logger: logging.Logger | None = None
) -> dict[str, str]:
    t0 = time.perf_counter()
    meta = parse_header_meta(Path(path))
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


def read_cmeth(
    path: Path, logger: logging.Logger | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    t0 = time.perf_counter()
    cm = CMethFile.read(Path(path))
    if logger:
        log_event(
            logger,
            event="read-cmeth",
            cmd="read_cmeth",
            params={
                "path": str(path),
                "version": cm.version,
                "n_rows": int(len(cm.df)),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return cm.df, cm.meta


def read_cmeth_target_bed(path: Path) -> pd.DataFrame:
    return read_target_bed(Path(path))


def query_cmeth(
    path: Path,
    chrom: str,
    start: int,
    end: int,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    t0 = time.perf_counter()
    meta = parse_header_meta(Path(path))
    df = read_cmeth_region(Path(path), chrom, int(start), int(end))
    if logger:
        log_event(
            logger,
            event="query-cmeth",
            cmd="query_cmeth",
            params={
                "path": str(path),
                "chrom": chrom,
                "start": start,
                "end": end,
                "n_rows": int(len(df)),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return df, meta


def _build_locus_maps(
    table: pd.DataFrame,
) -> dict[tuple[str, int, int], tuple[int, int]]:
    if table.empty:
        return {}
    out: dict[tuple[str, int, int], tuple[int, int]] = {}
    for row in table.itertuples(index=False):
        out[(str(row.chrom), int(row.start), int(row.end))] = (
            int(row.Nmod),
            int(row.Nvalid_cov),
        )
    return out


def create_cmeth_reference(
    *,
    kit: str | Path,
    bedmethyl_paths: List[Path],
    out: Path,
    logger: logging.Logger | None = None,
    cn_norm: str = "per-sample-median",
    mvalue_eps: float = 0.001,
    description: str | None = None,
    include_cpgs: bool = True,
    mod_mode: str = "m",
    mod_codes: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """
    Build a rich aggregated CMETH reference from bgzipped, indexed bedMethyl files.

    Emits one row per region and reference hap key:
      pooled, allele_low, allele_high, unphased
    and, when include_cpgs=True, one additional row per observed CpG/locus and hap key.
    """
    t0 = time.perf_counter()
    intervals, region_names = load_intervals(kit)
    if not intervals:
        raise ValueError(f"no intervals found for kit/BED: {kit!r}")

    groups = _group_paths_by_sample_and_hap([Path(p) for p in bedmethyl_paths])
    sample_ids = sorted(groups)
    n_samples = len(sample_ids)
    if n_samples == 0:
        raise ValueError(
            "no bedMethyl files matched expected hap suffixes: _1, _2, _ungrouped"
        )

    n_regions = len(intervals)
    lengths = np.array([max(1, e - s) for _, s, e in intervals], dtype=float)

    beta_pooled = np.full((n_samples, n_regions), np.nan, dtype=float)
    beta_h1 = np.full_like(beta_pooled, np.nan)
    beta_h2 = np.full_like(beta_pooled, np.nan)
    beta_unphased = np.full_like(beta_pooled, np.nan)
    beta_low = np.full_like(beta_pooled, np.nan)
    beta_high = np.full_like(beta_pooled, np.nan)

    nmod_pooled = np.full_like(beta_pooled, np.nan)
    nvalid_pooled = np.full_like(beta_pooled, np.nan)
    nmod_h1 = np.full_like(beta_pooled, np.nan)
    nvalid_h1 = np.full_like(beta_pooled, np.nan)
    nmod_h2 = np.full_like(beta_pooled, np.nan)
    nvalid_h2 = np.full_like(beta_pooled, np.nan)
    nmod_unphased = np.full_like(beta_pooled, np.nan)
    nvalid_unphased = np.full_like(beta_pooled, np.nan)
    nmod_low = np.full_like(beta_pooled, np.nan)
    nvalid_low = np.full_like(beta_pooled, np.nan)
    nmod_high = np.full_like(beta_pooled, np.nan)
    nvalid_high = np.full_like(beta_pooled, np.nan)

    depth_pooled = np.full_like(beta_pooled, np.nan)
    depth_h1 = np.full_like(beta_pooled, np.nan)
    depth_h2 = np.full_like(beta_pooled, np.nan)
    depth_unphased = np.full_like(beta_pooled, np.nan)
    depth_low = np.full_like(beta_pooled, np.nan)
    depth_high = np.full_like(beta_pooled, np.nan)

    frac_unphased = np.full_like(beta_pooled, np.nan)
    hap_balance = np.full_like(beta_pooled, np.nan)
    cov_mat = np.zeros((n_samples, n_regions), dtype=float)

    cpg_union: list[set[tuple[str, int, int]]] = [
        set() for _ in range(n_regions)
    ]
    cpg_by_region_sample: list[
        list[dict[str, dict[tuple[str, int, int], tuple[int, int]]]]
    ] = [[dict() for _ in range(n_samples)] for _ in range(n_regions)]

    for i, sid in enumerate(sample_ids):
        parts = groups[sid]
        for j, (chrom, start, end) in enumerate(intervals):
            L = lengths[j]
            per_hap_tables: dict[str, pd.DataFrame] = {}
            for hap_key in ("1", "2", "ungrouped"):
                if hap_key in parts:
                    per_hap_tables[hap_key] = get_locus_table(
                        str(parts[hap_key]),
                        chrom,
                        start,
                        end,
                        mode=str(mod_mode).lower(),
                        codes=list(mod_codes) if mod_codes else None,
                    )
                else:
                    per_hap_tables[hap_key] = pd.DataFrame(
                        columns=["chrom", "start", "end", "Nvalid_cov", "Nmod"]
                    )

            m1 = (
                int(per_hap_tables["1"]["Nmod"].sum())
                if not per_hap_tables["1"].empty
                else 0
            )
            v1 = (
                int(per_hap_tables["1"]["Nvalid_cov"].sum())
                if not per_hap_tables["1"].empty
                else 0
            )
            m2 = (
                int(per_hap_tables["2"]["Nmod"].sum())
                if not per_hap_tables["2"].empty
                else 0
            )
            v2 = (
                int(per_hap_tables["2"]["Nvalid_cov"].sum())
                if not per_hap_tables["2"].empty
                else 0
            )
            mu = (
                int(per_hap_tables["ungrouped"]["Nmod"].sum())
                if not per_hap_tables["ungrouped"].empty
                else 0
            )
            vu = (
                int(per_hap_tables["ungrouped"]["Nvalid_cov"].sum())
                if not per_hap_tables["ungrouped"].empty
                else 0
            )

            mt = m1 + m2 + mu
            vt = v1 + v2 + vu
            cov_mat[i, j] = vt

            if vt > 0:
                beta_pooled[i, j] = mt / vt
                nmod_pooled[i, j] = mt
                nvalid_pooled[i, j] = vt
                depth_pooled[i, j] = vt / L
                frac_unphased[i, j] = vu / vt
            if v1 > 0:
                beta_h1[i, j] = m1 / v1
                nmod_h1[i, j] = m1
                nvalid_h1[i, j] = v1
                depth_h1[i, j] = v1 / L
            if v2 > 0:
                beta_h2[i, j] = m2 / v2
                nmod_h2[i, j] = m2
                nvalid_h2[i, j] = v2
                depth_h2[i, j] = v2 / L
            if vu > 0:
                beta_unphased[i, j] = mu / vu
                nmod_unphased[i, j] = mu
                nvalid_unphased[i, j] = vu
                depth_unphased[i, j] = vu / L
            if v1 > 0 and v2 > 0:
                if beta_h1[i, j] <= beta_h2[i, j]:
                    lo = (beta_h1[i, j], m1, v1, depth_h1[i, j])
                    hi = (beta_h2[i, j], m2, v2, depth_h2[i, j])
                else:
                    lo = (beta_h2[i, j], m2, v2, depth_h2[i, j])
                    hi = (beta_h1[i, j], m1, v1, depth_h1[i, j])
                (
                    beta_low[i, j],
                    nmod_low[i, j],
                    nvalid_low[i, j],
                    depth_low[i, j],
                ) = lo
                (
                    beta_high[i, j],
                    nmod_high[i, j],
                    nvalid_high[i, j],
                    depth_high[i, j],
                ) = hi
                hap_balance[i, j] = min(v1, v2) / max(v1, v2)

            if include_cpgs:
                maps = {
                    hap: _build_locus_maps(tbl)
                    for hap, tbl in per_hap_tables.items()
                }
                cpg_by_region_sample[j][i] = maps
                all_keys: set[tuple[str, int, int]] = set()
                for d in maps.values():
                    all_keys.update(d.keys())
                cpg_union[j].update(all_keys)

    if cn_norm.lower() == "per-sample-median":
        norms = np.nanmedian(np.where(cov_mat > 0, cov_mat, np.nan), axis=1)
    else:
        raise ValueError(f"unsupported cn_norm: {cn_norm!r}")
    norms = np.where(norms <= 0, np.nan, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        cn_ratio = cov_mat / norms[:, None]
        cn_ratio = np.where(cn_ratio > 0, cn_ratio, np.nan)
        cn_log2 = np.log2(cn_ratio)

    tracks = {
        "pooled": (beta_pooled, nmod_pooled, nvalid_pooled, depth_pooled),
        "allele_low": (beta_low, nmod_low, nvalid_low, depth_low),
        "allele_high": (beta_high, nmod_high, nvalid_high, depth_high),
        "unphased": (
            beta_unphased,
            nmod_unphased,
            nvalid_unphased,
            depth_unphased,
        ),
    }

    rows: list[dict[str, object]] = []
    for j, ((chrom, start, end), region) in enumerate(
        zip(intervals, region_names, strict=False)
    ):
        region_cpg_count = len(cpg_union[j]) if include_cpgs else np.nan
        n_hap_resolved = int(
            (np.isfinite(beta_low[:, j]) & np.isfinite(beta_high[:, j])).sum()
        )
        n_unphased = int(np.isfinite(beta_unphased[:, j]).sum())
        for hap_key, (
            beta_mat,
            nmod_mat,
            nvalid_mat,
            depth_mat,
        ) in tracks.items():
            row = _empty_row_base(
                chrom=chrom,
                start=start,
                end=end,
                region=region,
                hap_key=hap_key,
                n_ref=n_samples,
                feature_type="region",
            )
            row["cpg_count"] = region_cpg_count
            _add_methylation_stats(
                row,
                beta=beta_mat[:, j],
                nmod=nmod_mat[:, j],
                nvalid=nvalid_mat[:, j],
                depth=depth_mat[:, j],
                mvalue_eps=float(mvalue_eps),
            )
            if hap_key == "pooled":
                _add_phasing_stats(
                    row,
                    frac_unphased=frac_unphased[:, j],
                    hap_balance=hap_balance[:, j],
                    n_hap_resolved=n_hap_resolved,
                    n_unphased=n_unphased,
                )
                _add_allele_pair_stats(
                    row, allele_low=beta_low[:, j], allele_high=beta_high[:, j]
                )
                _add_cn_stats(row, cn_log2=cn_log2[:, j], cn_cov=cov_mat[:, j])
            statuses = [str(row.get("meth_status", "no_data"))]
            if hap_key == "pooled":
                statuses.extend(
                    [
                        str(row.get("cn_status", "no_data")),
                        str(row.get("phasing_status", "no_data")),
                    ]
                )
            row["row_status"] = "ok" if "ok" in statuses else "no_data"
            rows.append(row)

        if include_cpgs:
            ordered_cpgs = sorted(
                cpg_union[j], key=lambda x: (x[0], x[1], x[2])
            )
            for idx, locus in enumerate(ordered_cpgs, start=1):
                lchrom, lstart, lend = locus
                pooled_beta = np.full(n_samples, np.nan, dtype=float)
                pooled_nmod = np.full(n_samples, np.nan, dtype=float)
                pooled_nvalid = np.full(n_samples, np.nan, dtype=float)
                pooled_depth = np.full(n_samples, np.nan, dtype=float)
                low_beta = np.full(n_samples, np.nan, dtype=float)
                low_nmod = np.full(n_samples, np.nan, dtype=float)
                low_nvalid = np.full(n_samples, np.nan, dtype=float)
                low_depth = np.full(n_samples, np.nan, dtype=float)
                high_beta = np.full(n_samples, np.nan, dtype=float)
                high_nmod = np.full(n_samples, np.nan, dtype=float)
                high_nvalid = np.full(n_samples, np.nan, dtype=float)
                high_depth = np.full(n_samples, np.nan, dtype=float)
                unph_beta = np.full(n_samples, np.nan, dtype=float)
                unph_nmod = np.full(n_samples, np.nan, dtype=float)
                unph_nvalid = np.full(n_samples, np.nan, dtype=float)
                unph_depth = np.full(n_samples, np.nan, dtype=float)
                cpg_frac_unphased = np.full(n_samples, np.nan, dtype=float)
                cpg_hap_balance = np.full(n_samples, np.nan, dtype=float)

                for i in range(n_samples):
                    maps = cpg_by_region_sample[j][i]
                    m1, v1 = maps.get("1", {}).get(locus, (0, 0))
                    m2, v2 = maps.get("2", {}).get(locus, (0, 0))
                    mu, vu = maps.get("ungrouped", {}).get(locus, (0, 0))
                    mt = m1 + m2 + mu
                    vt = v1 + v2 + vu
                    Lc = max(1, lend - lstart)
                    if vt > 0:
                        pooled_beta[i] = mt / vt
                        pooled_nmod[i] = mt
                        pooled_nvalid[i] = vt
                        pooled_depth[i] = vt / Lc
                        cpg_frac_unphased[i] = vu / vt
                    if vu > 0:
                        unph_beta[i] = mu / vu
                        unph_nmod[i] = mu
                        unph_nvalid[i] = vu
                        unph_depth[i] = vu / Lc
                    if v1 > 0 and v2 > 0:
                        b1 = m1 / v1
                        b2 = m2 / v2
                        cpg_hap_balance[i] = min(v1, v2) / max(v1, v2)
                        if b1 <= b2:
                            (
                                low_beta[i],
                                low_nmod[i],
                                low_nvalid[i],
                                low_depth[i],
                            ) = (b1, m1, v1, v1 / Lc)
                            (
                                high_beta[i],
                                high_nmod[i],
                                high_nvalid[i],
                                high_depth[i],
                            ) = (b2, m2, v2, v2 / Lc)
                        else:
                            (
                                low_beta[i],
                                low_nmod[i],
                                low_nvalid[i],
                                low_depth[i],
                            ) = (b2, m2, v2, v2 / Lc)
                            (
                                high_beta[i],
                                high_nmod[i],
                                high_nvalid[i],
                                high_depth[i],
                            ) = (b1, m1, v1, v1 / Lc)

                cpg_region_id = f"{region}:CpG_{idx:03d}"
                cpg_display_name = f"{region} CpG {idx}"
                cpg_tracks = {
                    "pooled": (
                        pooled_beta,
                        pooled_nmod,
                        pooled_nvalid,
                        pooled_depth,
                    ),
                    "allele_low": (low_beta, low_nmod, low_nvalid, low_depth),
                    "allele_high": (
                        high_beta,
                        high_nmod,
                        high_nvalid,
                        high_depth,
                    ),
                    "unphased": (unph_beta, unph_nmod, unph_nvalid, unph_depth),
                }
                cpg_n_hap_resolved = int(
                    (np.isfinite(low_beta) & np.isfinite(high_beta)).sum()
                )
                cpg_n_unphased = int(np.isfinite(unph_beta).sum())
                for hap_key, (
                    beta_vec,
                    nmod_vec,
                    nvalid_vec,
                    depth_vec,
                ) in cpg_tracks.items():
                    row = _empty_row_base(
                        chrom=lchrom,
                        start=lstart,
                        end=lend,
                        region=cpg_region_id,
                        hap_key=hap_key,
                        n_ref=n_samples,
                        feature_type="cpg",
                        parent_region=region,
                        display_name=cpg_display_name,
                    )
                    row["cpg_count"] = 1
                    _add_methylation_stats(
                        row,
                        beta=beta_vec,
                        nmod=nmod_vec,
                        nvalid=nvalid_vec,
                        depth=depth_vec,
                        mvalue_eps=float(mvalue_eps),
                    )
                    if hap_key == "pooled":
                        _add_phasing_stats(
                            row,
                            frac_unphased=cpg_frac_unphased,
                            hap_balance=cpg_hap_balance,
                            n_hap_resolved=cpg_n_hap_resolved,
                            n_unphased=cpg_n_unphased,
                        )
                        _add_allele_pair_stats(
                            row, allele_low=low_beta, allele_high=high_beta
                        )
                    statuses = [str(row.get("meth_status", "no_data"))]
                    if hap_key == "pooled":
                        statuses.append(
                            str(row.get("phasing_status", "no_data"))
                        )
                    row["row_status"] = "ok" if "ok" in statuses else "no_data"
                    rows.append(row)

    df = pd.DataFrame.from_records(rows, columns=CMETH_COLUMNS)
    kit_path = Path(kit)
    target_name = kit_path.stem if kit_path.is_file() else str(kit)
    target_bed = _target_bed_lines(intervals, region_names)
    feature_types = ["region"] + (["cpg"] if include_cpgs else [])
    meta: dict[str, object] = {
        "kind": "reference",
        "coordinate": "bed0",
        "description": description or ".",
        "target_name": target_name,
        "target_count": len(intervals),
        "target_bed_columns": "chrom,start,end,name",
        "target_bed_count": len(target_bed),
        "source_sample_count": n_samples,
        "source_file_count": len(bedmethyl_paths),
        "cn_norm": cn_norm,
        "mvalue_eps": f"{float(mvalue_eps):g}",
        "meth_hist_bins": ",".join(f"{x:g}" for x in METH_HIST_BINS),
        "mval_hist_bins": ",".join(f"{x:g}" for x in MVAL_HIST_BINS),
        "cn_log2_hist_bins": ",".join(f"{x:g}" for x in CN_LOG2_HIST_BINS),
        "feature_types": ",".join(feature_types),
        "modkit_compatibility": "bedMethyl,bedRMod",
        "mod_mode": str(mod_mode).lower(),
        "mod_codes": ",".join(mod_codes or []),
    }

    outp = write_cmeth_reference(
        df, out=out, meta=meta, target_bed=target_bed, logger=logger
    )
    if logger:
        log_event(
            logger,
            event="reference-create",
            cmd="create_cmeth_reference",
            params={
                "kit": str(kit),
                "out": str(outp),
                "n_files": len(bedmethyl_paths),
                "n_samples": n_samples,
                "include_cpgs": bool(include_cpgs),
                "mod_mode": str(mod_mode).lower(),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    return outp
