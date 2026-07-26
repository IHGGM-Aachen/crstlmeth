"""
crstlmeth/cli/plot/meth_plot_cmd.py

redraw a methylation plot from:
  - one *.cmeth reference cohort (mode=full or mode=aggregated)
  - one or more target bedmethyl files (resolved automatically)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from crstlmeth.core.discovery import resolve_bedmethyl_glob
from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.methylation import Methylation
from crstlmeth.core.references import read_cmeth
from crstlmeth.core.regions import load_intervals
from crstlmeth.core.samples import parse_bedmethyl_name
from crstlmeth.viz.meth_plot import (
    plot_methylation_from_quantiles,
    plot_methylation_levels_from_arrays,
    plot_normalized_phased_methylation_from_arrays,
    plot_normalized_phased_methylation_from_quantiles,
)


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _parse_bedmethyl_name(path: str | Path) -> tuple[str, str] | None:
    parsed = parse_bedmethyl_name(path)
    if parsed is None or parsed.is_index:
        return None
    return parsed.sample_id, parsed.role


def _group_paths_by_sample(paths: List[str]) -> Dict[str, List[str]]:
    """
    Group bedMethyl data files by parsed sample ID.

    Index files are ignored. This keeps legacy methylation plotting code working
    while using the central filename parser from crstlmeth.core.samples.
    """
    out: Dict[str, List[str]] = {}
    for p in paths:
        parsed = _parse_bedmethyl_name(p)
        if parsed is None:
            continue
        sample_id, _role = parsed
        out.setdefault(sample_id, []).append(str(p))
    return out


def _classify_haps(paths: List[str]) -> Dict[str, str]:
    """
    Classify one sample's paths as 1, 2, and/or ungrouped.
    """
    out: Dict[str, str] = {}
    for p in paths:
        parsed = _parse_bedmethyl_name(p)
        if parsed is None:
            continue
        _sample_id, role = parsed
        out[role] = str(p)
    return out


def _reference_mode(meta: dict, ref_df: pd.DataFrame) -> str:
    raw = str(meta.get("mode", "")).lower()
    if raw in {"aggregated", "full"}:
        return raw
    if {"feature_type", "hap_key"}.issubset(ref_df.columns):
        return "aggregated"
    if {"sample_id", "region"}.issubset(ref_df.columns):
        return "full"
    return "aggregated"


def _region_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "feature_type" in out.columns:
        out = out[out["feature_type"].astype(str) == "region"].copy()
    if "region" not in out.columns and "region_id" in out.columns:
        out = out.rename(columns={"region_id": "region"})
    return out


def _pooled_levels_for_paths(
    paths: List[str], intervals: List[tuple[str, int, int]]
) -> np.ndarray:
    """
    pool arbitrary bedmethyl files by summing Nmod/Nvalid over paths per interval
    returns shape (n_intervals,)
    """
    from crstlmeth.core.parsers import get_region_stats  # lazy import

    n = len(intervals)
    out = np.zeros(n, dtype=float)
    for j, (c, s, e) in enumerate(intervals):
        tot_m = 0
        tot_v = 0
        for p in paths:
            m, v = get_region_stats(p, c, s, e)
            tot_m += m
            tot_v += v
        out[j] = (tot_m / tot_v) if tot_v > 0 else np.nan
    return out


def _run_normalized_haps_plot(
    *,
    cmeth_ref: Path,
    ref_df: pd.DataFrame,
    mode: str,
    regions: list[str],
    intervals: list[tuple[str, int, int]],
    by_sample: dict[str, list[str]],
    out_png: Path,
    min_hap_regions: int,
    logger,
) -> None:
    """
    Plot phased/unphased methylation normalized to pooled = 1.
    Requires exactly one target sample with _1 and _2 files.
    """
    if len(by_sample) != 1:
        raise click.UsageError(
            "--plot-type normalized-haps requires exactly one target sample"
        )

    sid, paths = next(iter(by_sample.items()))
    parts = _classify_haps(paths)
    if "1" not in parts or "2" not in parts:
        raise click.UsageError(
            "--plot-type normalized-haps needs both _1 and _2 files for the sample"
        )

    h1, h2, overall = Methylation.get_levels_by_haplotype(parts, intervals)

    finite_h1 = int(np.isfinite(h1).sum())
    finite_h2 = int(np.isfinite(h2).sum())
    if finite_h1 < min_hap_regions or finite_h2 < min_hap_regions:
        raise click.ClickException(
            "normalized-haps plot requires phased coverage in the requested regions.\n"
            f"Finite hap regions: hap1={finite_h1}, hap2={finite_h2} "
            f"(min required per hap: {min_hap_regions})."
        )

    pooled_t, h1_t, h2_t = Methylation.normalize_target_haps_to_overall(
        h1, h2, overall, min_overall=0.05, eps=0.02
    )

    qc = Methylation.assess_phasing_quality(parts, intervals, thresh=0.45)
    qc_mask = qc.get("flag_mask", None)

    if mode == "full":
        pooled_ref, h1_ref, h2_ref, pooled_valid = (
            Methylation.get_normalized_hap_matrices_from_full_reference(
                ref_df,
                regions,
                intervals=intervals,
                min_pooled=0.05,
                eps=0.02,
            )
        )

        keep = pooled_valid & (
            np.isfinite(pooled_t)
            | np.isfinite(h1_t)
            | np.isfinite(h2_t)
            | np.isfinite(h1_ref).any(axis=0)
            | np.isfinite(h2_ref).any(axis=0)
        )

        if not np.any(keep):
            raise click.ClickException(
                "No usable regions left after pooled normalization."
            )

        regions_used = [r for r, k in zip(regions, keep, strict=False) if k]
        qc_used = qc_mask[keep] if qc_mask is not None else None

        plot_normalized_phased_methylation_from_arrays(
            hap1_ref_norm=h1_ref[:, keep],
            hap2_ref_norm=h2_ref[:, keep],
            pooled_target_norm=pooled_t[keep],
            hap1_target_norm=h1_t[keep],
            hap2_target_norm=h2_t[keep],
            region_names=regions_used,
            save=str(out_png),
            title=f"methylation  -  phased vs pooled (normalized)  -  {sid}",
            qc_mask=qc_used,
        )

    elif mode == "aggregated":
        h1_q, h2_q, pooled_valid = (
            Methylation.get_normalized_hap_quantiles_from_aggregated_reference(
                ref_df,
                regions,
                intervals=intervals,
                min_pooled=0.05,
                eps=0.02,
            )
        )

        keep = pooled_valid & (
            np.isfinite(pooled_t)
            | np.isfinite(h1_t)
            | np.isfinite(h2_t)
            | np.isfinite(h1_q["q50"].to_numpy(dtype=float))
            | np.isfinite(h2_q["q50"].to_numpy(dtype=float))
        )

        if not np.any(keep):
            raise click.ClickException(
                "No usable regions left after pooled normalization."
            )

        regions_used = [r for r, k in zip(regions, keep, strict=False) if k]
        qc_used = qc_mask[keep] if qc_mask is not None else None

        plot_normalized_phased_methylation_from_quantiles(
            hap1_ref_q_norm=h1_q.iloc[keep].copy(),
            hap2_ref_q_norm=h2_q.iloc[keep].copy(),
            pooled_target_norm=pooled_t[keep],
            hap1_target_norm=h1_t[keep],
            hap2_target_norm=h2_t[keep],
            region_names=regions_used,
            save=str(out_png),
            title=f"methylation  -  phased vs pooled (normalized; approx.)  -  {sid}",
            qc_mask=qc_used,
        )

    else:
        raise click.ClickException(f"unsupported cmeth mode: {mode!r}")

    click.secho(f"figure written -> {out_png.resolve()}", fg="green")

    log_event(
        logger,
        event="plot_methylation",
        cmd="plot methylation",
        params=dict(
            cmeth=str(cmeth_ref),
            out=str(out_png),
            mode=mode,
            plot_type="normalized-haps",
            n_targets=1,
        ),
        message="ok",
    )


def _pick(
    df: pd.DataFrame, col: str, fallback: str | None = None
) -> np.ndarray | None:
    if col in df.columns:
        return df[col].to_numpy()
    if fallback and fallback in df.columns:
        return df[fallback].to_numpy()
    return None


def _dedup_aggregated_meth(df: pd.DataFrame) -> pd.DataFrame:
    """
    keep a single row per region in aggregated meth table using hap priority:
        pooled -> ungrouped -> 1 -> 2  (else first seen)
    returns a frame with unique 'region' rows
    """
    if "hap_key" in df.columns:
        prio = {"pooled": 0, "ungrouped": 1, "1": 2, "2": 3}
        df = df.assign(_prio=df["hap_key"].map(prio).fillna(99))
        df = (
            df.sort_values(["region", "_prio"])
            .drop_duplicates(subset="region", keep="first")
            .drop(columns="_prio")
        )
    else:
        df = df.drop_duplicates(subset="region", keep="first")
    return df


def _unique_order(seq: List[str]) -> List[str]:
    """preserve order while dropping duplicates"""
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _agg_quantiles(
    ref_df: pd.DataFrame,
    regions: List[str],
    intervals: List[tuple[str, int, int]],
    hap_key: str,
) -> pd.DataFrame:
    """
    Return q25/q50/q75(/sd) for aggregated references for a given hap_key.
    Uses name alignment with coordinate fallback.
    """
    q = Methylation.hap_quantiles_for_reference(
        ref_df=_region_rows(ref_df),
        mode="aggregated",
        regions=regions,
        hap_key=str(hap_key),
        intervals=intervals,
    )
    # q has columns: q25,q50,q75,sd (sd may be nan)
    return q


def _agg_pick_pooled_quantiles(
    ref_df: pd.DataFrame,
    regions: List[str],
    intervals: List[tuple[str, int, int]],
) -> tuple[pd.DataFrame, str]:
    """
    For aggregated pooled view: prefer hap_key pooled -> ungrouped -> 1 -> 2.
    Returns (quantiles_df, key_used). Raises if nothing usable.
    """
    if "hap_key" not in ref_df.columns:
        raise click.ClickException(
            "Aggregated reference is missing column 'hap_key' (needed to choose pooled/hap quantiles)."
        )

    for key in (
        "pooled",
        "unphased",
        "ungrouped",
        "allele_high",
        "allele_low",
        "1",
        "2",
    ):
        q = _agg_quantiles(ref_df, regions, intervals, key)
        ok = (
            np.isfinite(q["q25"].to_numpy())
            & np.isfinite(q["q50"].to_numpy())
            & np.isfinite(q["q75"].to_numpy())
        )
        if int(ok.sum()) > 0:
            return q, key

    raise click.ClickException(
        "Aggregated reference has no usable quantiles for pooled/unphased/allele rows "
        "after name/coordinate alignment."
    )


# -----------------------------------------------------------------------------
# click command
# -----------------------------------------------------------------------------
@click.command(
    name="methylation",
    help=(
        "draw methylation plot using a *.cmeth reference and one or more targets. "
        "supports mode=full (true cohort boxes) and mode=aggregated (quantile boxes)."
    ),
)
@click.option(
    "--cmeth",
    "cmeth_ref",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="reference cohort created with  crstlmeth reference create",
)
@click.option(
    "--plot-type",
    type=click.Choice(["default", "normalized-haps"], case_sensitive=False),
    default="default",
    show_default=True,
    help=(
        "default = existing interval methylation plot; "
        "normalized-haps = phased/unphased plot with pooled normalized to 1 "
        "(exact for full references, approximate for aggregated references)."
    ),
)
@click.option(
    "--kit",
    "--bed",
    "kit_or_bed",
    required=True,
    help="mlpa kit name or custom bed defining methylation regions",
)
@click.option(
    "--haplotypes/--no-haplotypes",
    default=False,
    show_default=True,
    help="when true and exactly one sample with _1/_2 is provided, overlay both haps (pooled reference view)",
)
@click.option(
    "--hap-ref-plot/--no-hap-ref-plot",
    default=False,
    show_default=True,
    help=(
        "plot against an allele-specific cohort view (requires exactly one sample with _1 and _2). "
        "In full references, hap 1 = higher-methylated allele, hap 2 = lower-methylated allele."
    ),
)
@click.option(
    "--ref-hap",
    type=click.Choice(
        ["1", "2", "allele_high", "allele_low"], case_sensitive=False
    ),
    default="1",
    show_default=True,
    help="reference allele to show with --hap-ref-plot: 1/allele_high=higher, 2/allele_low=lower.",
)
@click.option(
    "--min-hap-regions",
    type=int,
    default=10,
    show_default=True,
    help="Minimum number of regions with finite methylation in each hap (hap1 and hap2).",
)
@click.option(
    "--hap-debug/--no-hap-debug",
    default=False,
    show_default=True,
    help="print hap-level diagnostics in hap-ref-plot mode.",
)
@click.option(
    "--shade/--no-shade",
    default=True,
    show_default=True,
    help="shade intervals with BH-FDR<0.05 (aggregated: approx z from quantiles; full: z-test vs cohort)",
)
@click.argument(
    "target",
    nargs=-1,
    required=True,
    type=str,
)
@click.option(
    "--out",
    "out_png",
    type=click.Path(path_type=Path, dir_okay=False),
    default="methylation.png",
    show_default=True,
    help="destination png",
)
@click.pass_context
def methylation(
    ctx: click.Context,
    cmeth_ref: Path,
    plot_type: str,
    kit_or_bed: str,
    target: tuple[str, ...],
    haplotypes: bool,
    hap_ref_plot: bool,
    ref_hap: str,
    min_hap_regions: int,
    hap_debug: bool,
    shade: bool,
    out_png: Path,
) -> None:
    """
    produce out_png showing cohort distribution and target methylation levels
    """
    logger = get_logger_from_cli(ctx)

    # resolve and flatten target bedmethyl globs
    tgt_paths: List[Path] = []
    for t in target:
        tgt_paths.extend(resolve_bedmethyl_glob([str(t)]))
    if not tgt_paths:
        raise click.UsageError("no target bedmethyl files resolved")

    # region set
    intervals_raw, region_names_raw = load_intervals(kit_or_bed)

    # if region names are not unique, we must also deduplicate intervals
    # so that len(regions) == len(intervals) everywhere
    regions: List[str] = []
    intervals: List[tuple[str, int, int]] = []
    seen: set[str] = set()
    for iv, rn in zip(intervals_raw, region_names_raw, strict=False):
        if rn not in seen:
            seen.add(rn)
            regions.append(rn)
            intervals.append(iv)

    # load cmeth reference
    ref_df, meta = read_cmeth(cmeth_ref, logger=logger)
    mode = _reference_mode(meta, ref_df)
    ref_df = _region_rows(ref_df) if mode == "aggregated" else ref_df

    # build target arrays grouped by sample ID
    by_sample = _group_paths_by_sample([str(p) for p in tgt_paths])

    # plot type
    if str(plot_type).lower() == "normalized-haps":
        _run_normalized_haps_plot(
            cmeth_ref=cmeth_ref,
            ref_df=ref_df,
            mode=mode,
            regions=regions,
            intervals=intervals,
            by_sample=by_sample,
            out_png=out_png,
            min_hap_regions=min_hap_regions,
            logger=logger,
        )
        return

    # -------------------------------------------------------------------------
    # HAPLOTYPE-SPECIFIC REFERENCE PLOT (allele-based)
    # -------------------------------------------------------------------------
    if hap_ref_plot:
        if len(by_sample) != 1:
            raise click.UsageError(
                "--hap-ref-plot requires exactly one target sample"
            )
        sid, paths = next(iter(by_sample.items()))
        parts = _classify_haps(paths)
        if "1" not in parts or "2" not in parts:
            raise click.UsageError(
                "--hap-ref-plot needs both _1 and _2 files for the sample"
            )

        # target hap-specific levels, aligned to kit intervals
        h1, h2, overall = Methylation.get_levels_by_haplotype(parts, intervals)

        # guard: no finite coverage at all -> log + WARN, then fall back to pooled view
        finite_h1 = int(np.isfinite(h1).sum())
        finite_h2 = int(np.isfinite(h2).sum())

        if finite_h1 < min_hap_regions or finite_h2 < min_hap_regions:
            raise click.ClickException(
                "Haplotype reference plot requires phased coverage in the requested regions.\n"
                f"Finite hap regions: hap1={finite_h1}, hap2={finite_h2} (min required per hap: {min_hap_regions}).\n"
                "Likely causes: no phased reads for this kit/BED, wrong target files, or wrong region definition."
            )
        else:
            # we do have hap data -> allele-specific full-reference view
            # phasing QC mask (frac_ungrouped >= 45%)
            qc = Methylation.assess_phasing_quality(
                parts, intervals, thresh=0.45
            )
            qc_mask = qc.get("flag_mask", None)
            qc_note = "QC: frac ungrouped >= 45%"

            # allele orientation for target: per region high vs low allele
            tgt_stack = np.vstack([h1, h2])  # (2, n_regions)
            target_hi = np.nanmax(
                tgt_stack, axis=0
            )  # higher-methylated allele per region
            target_lo = np.nanmin(
                tgt_stack, axis=0
            )  # lower-methylated allele per region

            if mode == "aggregated":
                # aggregated allele-specific view: use hap_key-specific quantiles (1 or 2)
                q_key = (
                    "allele_high"
                    if str(ref_hap).lower() in {"1", "allele_high"}
                    else "allele_low"
                )
                q = _agg_quantiles(ref_df, regions, intervals, hap_key=q_key)

                ok = (
                    np.isfinite(q["q25"].to_numpy())
                    & np.isfinite(q["q50"].to_numpy())
                    & np.isfinite(q["q75"].to_numpy())
                )
                if int(ok.sum()) == 0:
                    raise click.ClickException(
                        f"Aggregated reference does not contain usable methylation quantiles for hap_key={ref_hap!r} "
                        "after name/coordinate alignment. Ensure the aggregated reference was built with hap_key=1/2 rows."
                    )

                # target allele view (per-region high/low from phased target)
                tgt_stack = np.vstack([h1, h2])  # (2, n_regions)
                target_hi = np.nanmax(tgt_stack, axis=0)
                target_lo = np.nanmin(tgt_stack, axis=0)

                if str(ref_hap).lower() in {"1", "allele_high"}:
                    tgt_vec = target_hi
                    allele_label = "higher-methylated allele"
                    tgt_label = f"{sid}_hi"
                else:
                    tgt_vec = target_lo
                    allele_label = "lower-methylated allele"
                    tgt_label = f"{sid}_lo"

                # restrict to regions with defined reference quantiles
                sel = np.where(ok)[0].tolist()
                regions_used = [regions[i] for i in sel]
                keep_cols = [
                    c
                    for c in (
                        "q05",
                        "q10",
                        "q25",
                        "q50",
                        "q75",
                        "q90",
                        "q95",
                        "sd",
                    )
                    if c in q.columns
                ]
                qdf = q.iloc[sel][keep_cols].copy()

                # apply QC mask on the same selection
                qc = Methylation.assess_phasing_quality(
                    parts, intervals, thresh=0.45
                )
                qc_mask = qc.get("flag_mask", None)
                qc_sel = (
                    qc_mask[sel]
                    if qc_mask is not None and len(qc_mask) == len(regions)
                    else None
                )

                plot_methylation_from_quantiles(
                    regions=regions_used,
                    quantiles=qdf,
                    targets=tgt_vec[sel].reshape(1, -1),
                    target_labels=[tgt_label],
                    save=str(out_png),
                    title=f"methylation  -  {allele_label}",
                    shade_outliers=shade,
                    qc_mask=qc_sel,
                    qc_note="QC: frac ungrouped >= 45%",
                )

                click.secho(
                    f"figure written -> {out_png.resolve()}", fg="green"
                )
                log_event(
                    logger,
                    event="plot_methylation",
                    cmd="plot methylation",
                    params=dict(
                        cmeth=str(cmeth_ref),
                        kit=str(kit_or_bed),
                        out=str(out_png),
                        n_targets=1,
                        mode=mode,
                        ref_hap=str(ref_hap),
                        hap_plot=True,
                        shade=bool(shade),
                        allele_view=True,
                    ),
                    message="ok",
                )
                return

            elif mode == "full":
                # build allele-specific reference matrices from hap=1/2 rows
                ref_haps = ref_df[
                    pd.Series(
                        (
                            ref_df["hap"].astype(str)
                            if "hap" in ref_df.columns
                            else [""] * len(ref_df)
                        ),
                        index=ref_df.index,
                    ).isin(["1", "2"])
                ].copy()
                if ref_haps.empty:
                    raise click.ClickException(
                        f"{cmeth_ref.name}: no hap=1/2 rows in full reference"
                    )

                # pivot hap1 and hap2 separately: rows = sample_id, cols = region
                ref_h1 = ref_haps[
                    ref_haps["hap"].astype(str) == "1"
                ].pivot_table(
                    index="sample_id",
                    columns="region",
                    values="meth",
                    aggfunc="first",
                )
                ref_h2 = ref_haps[
                    ref_haps["hap"].astype(str) == "2"
                ].pivot_table(
                    index="sample_id",
                    columns="region",
                    values="meth",
                    aggfunc="first",
                )

                # unify sample index and region columns
                all_samples = sorted(set(ref_h1.index) | set(ref_h2.index))
                col_candidates = set(ref_h1.columns) | set(ref_h2.columns)
                cols = [c for c in regions if c in col_candidates]
                if not cols:
                    raise click.ClickException(
                        "Haplotype plot aborted: no overlapping region names between "
                        "reference hap rows and kit/BED."
                    )

                ref_h1_mat = ref_h1.reindex(
                    index=all_samples, columns=cols, fill_value=np.nan
                ).to_numpy(dtype=float)
                ref_h2_mat = ref_h2.reindex(
                    index=all_samples, columns=cols, fill_value=np.nan
                ).to_numpy(dtype=float)

                # allele-based reference: higher vs lower methylated allele per sample+region
                ref_hi = np.where(
                    np.isnan(ref_h1_mat) & np.isnan(ref_h2_mat),
                    np.nan,
                    np.where(
                        np.isnan(ref_h1_mat),
                        ref_h2_mat,
                        np.where(
                            np.isnan(ref_h2_mat),
                            ref_h1_mat,
                            np.maximum(ref_h1_mat, ref_h2_mat),
                        ),
                    ),
                )
                ref_lo = np.where(
                    np.isnan(ref_h1_mat) & np.isnan(ref_h2_mat),
                    np.nan,
                    np.where(
                        np.isnan(ref_h1_mat),
                        ref_h2_mat,
                        np.where(
                            np.isnan(ref_h2_mat),
                            ref_h1_mat,
                            np.minimum(ref_h1_mat, ref_h2_mat),
                        ),
                    ),
                )

                # align target hi/lo to the same region subset
                idx_map = {r: i for i, r in enumerate(regions)}
                sel = [idx_map[c] for c in cols]
                tgt_hi_sel = target_hi[sel]
                tgt_lo_sel = target_lo[sel]

                if ref_hap == "1":
                    ref_mat = ref_hi
                    tgt_vec = tgt_hi_sel
                    allele_label = "higher-methylated allele"
                    tgt_label = f"{sid}_hi"
                else:
                    ref_mat = ref_lo
                    tgt_vec = tgt_lo_sel
                    allele_label = "lower-methylated allele"
                    tgt_label = f"{sid}_lo"

                if hap_debug:
                    n_samp, n_reg = ref_mat.shape
                    n_ref_finite = np.isfinite(ref_mat).sum()
                    n_tgt_finite = int(np.isfinite(tgt_vec).sum())
                    click.secho("[hap-ref-plot] allele-based view", fg="yellow")
                    click.echo(
                        f"  samples={n_samp} regions={n_reg} allele={allele_label}"
                    )
                    click.echo(
                        f"  ref finite={n_ref_finite}  target finite={n_tgt_finite}"
                    )

                # apply QC mask to selected columns
                qc_sel = (
                    qc_mask[sel]
                    if qc_mask is not None and len(qc_mask) == len(regions)
                    else None
                )

                plot_methylation_levels_from_arrays(
                    sample_lv=ref_mat,
                    target_lv=tgt_vec.reshape(1, -1),
                    region_names=cols,
                    save=str(out_png),
                    target_labels=[tgt_label],
                    shade_outliers=shade,
                    qc_mask=qc_sel,
                    qc_note=qc_note,
                    title=f"methylation  -  {allele_label}",
                )

                click.secho(
                    f"figure written -> {out_png.resolve()}", fg="green"
                )

                log_event(
                    logger,
                    event="plot_methylation",
                    cmd="plot methylation",
                    params=dict(
                        cmeth=str(cmeth_ref),
                        kit=str(kit_or_bed),
                        out=str(out_png),
                        n_targets=1,
                        mode=mode,
                        ref_hap=str(ref_hap),
                        hap_plot=True,
                        shade=bool(shade),
                        allele_view=True,
                    ),
                    message="ok",
                )
                return  # done

            else:
                raise click.ClickException(f"unsupported cmeth mode: {mode!r}")

    # -------------------------------------------------------------------------
    # POOLED REFERENCE VIEW (default)
    # -------------------------------------------------------------------------
    # phasing QC: if exactly one sample, compute mask from its parts (45%)
    qc_mask = None
    qc_note = None
    if len(by_sample) == 1:
        sid0, paths0 = next(iter(by_sample.items()))
        h0 = _classify_haps(paths0)
        if h0:
            qc = Methylation.assess_phasing_quality(h0, intervals, thresh=0.45)
            qc_mask = qc.get("flag_mask", None)
            qc_note = "QC: frac ungrouped >= 45%"

    if haplotypes:
        if len(by_sample) != 1:
            raise click.UsageError(
                "haplotype overlay requires exactly one target sample"
            )
        sid, paths = next(iter(by_sample.items()))
        h = _classify_haps(paths)
        if "1" not in h or "2" not in h:
            raise click.UsageError(
                "haplotype overlay needs both _1 and _2 files for the sample"
            )
        h1, h2, _overall = Methylation.get_levels_by_haplotype(h, intervals)
        tgt_mat = np.vstack([h1, h2])
        tgt_labels = [f"{sid}_1", f"{sid}_2"]
        plot_title = f"methylation - {sid} hap1/hap2"
    else:
        rows: List[np.ndarray] = []
        labels: List[str] = []
        for sid, paths in sorted(by_sample.items()):
            h = _classify_haps(paths)
            if h:
                # pooled view must use all present parts: hap1 + hap2 + ungrouped
                _h1, _h2, overall = Methylation.get_levels_by_haplotype(
                    h, intervals
                )
                rows.append(overall.reshape(1, -1))
            else:
                rows.append(
                    _pooled_levels_for_paths(paths, intervals).reshape(1, -1)
                )
            labels.append(sid)
        tgt_mat = (
            np.vstack(rows)
            if rows
            else np.zeros((0, len(intervals)), dtype=float)
        )
        tgt_labels = labels
        plot_title = "methylation per interval"

    # -------------------------------------------------------------------------
    # plot depending on reference mode (pooled view)
    # -------------------------------------------------------------------------
    if mode == "full":
        pooled = ref_df[
            pd.Series(
                (
                    ref_df["hap"].astype(str)
                    if "hap" in ref_df.columns
                    else ["pooled"] * len(ref_df)
                ),
                index=ref_df.index,
            )
            == "pooled"
        ]
        if pooled.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no pooled hap rows in full reference"
            )

        ref_piv = pooled.pivot_table(
            index="sample_id", columns="region", values="meth"
        )

        # align regions: use kit order but drop duplicates
        cols = [c for c in regions if c in ref_piv.columns]
        if not cols:
            raise click.ClickException(
                "No overlapping region names between reference and kit/BED."
            )
        ref_mat = ref_piv.reindex(columns=cols, fill_value=np.nan).to_numpy(
            dtype=float
        )

        # align targets to same selection
        idx_map = {r: i for i, r in enumerate(regions)}
        sel = [idx_map[c] for c in cols]
        tgt_mat = tgt_mat[:, sel]
        regions_used = cols

        plot_methylation_levels_from_arrays(
            sample_lv=ref_mat,
            target_lv=tgt_mat,
            region_names=regions_used,
            save=str(out_png),
            target_labels=tgt_labels,
            shade_outliers=shade,
            qc_mask=(qc_mask[sel] if qc_mask is not None else None),
            qc_note=qc_note,
            title=plot_title,
        )

    elif mode == "aggregated":
        df_meth = ref_df.copy()
        if "section" in df_meth.columns:
            df_meth = df_meth[df_meth["section"].astype(str) == "meth"].copy()
        if df_meth.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no 'meth' section in aggregated reference"
            )

        # Require hap_key to properly choose pooled/hap quantiles
        if "hap_key" not in df_meth.columns:
            raise click.ClickException(
                f"{cmeth_ref.name}: aggregated reference missing 'hap_key' column. "
                "Rebuild the aggregated reference with hap_key rows (pooled/ungrouped/1/2)."
            )

        # Pick pooled-view quantiles with priority: pooled -> ungrouped -> 1 -> 2
        q, key_used = _agg_pick_pooled_quantiles(
            ref_df=df_meth,
            regions=regions,
            intervals=intervals,
        )

        ok = (
            np.isfinite(q["q25"].to_numpy())
            & np.isfinite(q["q50"].to_numpy())
            & np.isfinite(q["q75"].to_numpy())
        )
        sel = np.where(ok)[0].tolist()
        if not sel:
            raise click.ClickException(
                f"{cmeth_ref.name}: quantiles exist but none are finite after alignment (hap_key={key_used})."
            )

        regions_used = [regions[i] for i in sel]
        keep_cols = [
            c
            for c in ("q05", "q10", "q25", "q50", "q75", "q90", "q95", "sd")
            if c in q.columns
        ]
        qdf = q.iloc[sel][keep_cols].copy()

        # align targets to the same selected regions
        tgt_mat = tgt_mat[:, sel]

        plot_methylation_from_quantiles(
            regions=regions_used,
            quantiles=qdf,
            targets=tgt_mat,
            target_labels=tgt_labels,
            save=str(out_png),
            title=plot_title,
            shade_outliers=shade,
            qc_mask=(qc_mask[sel] if qc_mask is not None else None),
            qc_note=qc_note,
        )
    else:
        raise click.ClickException(f"unsupported cmeth mode: {mode!r}")

    click.secho(f"figure written -> {out_png.resolve()}", fg="green")

    # log
    log_event(
        logger,
        event="plot_methylation",
        cmd="plot methylation",
        params=dict(
            cmeth=str(cmeth_ref),
            kit=str(kit_or_bed),
            out=str(out_png),
            n_targets=int(tgt_mat.shape[0]),
            mode=mode,
            hap_ref_plot=bool(hap_ref_plot),
            shade=bool(shade),
        ),
        message="ok",
    )
