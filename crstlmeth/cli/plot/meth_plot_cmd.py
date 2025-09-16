"""
crstlmeth/cli/plot/meth_plot_cmd.py

CLI interface to redraw a methylation plot from:
  - one *.cmeth reference cohort (mode=full or mode=aggregated)
  - one or more target bedmethyl files (resolved automatically)

Supports pooled plots, haplotype overlay, and hap-aware reference plots with
auto-matching of target hap1/hap2 to reference hap1/hap2.
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
from crstlmeth.viz.meth_plot import (
    plot_methylation_from_quantiles,
    plot_methylation_levels_from_arrays,
)


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _group_paths_by_sample(paths: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    from pathlib import Path as _P

    for p in paths:
        out.setdefault(_P(p).name.split("_")[0], []).append(p)
    return out


def _classify_haps(paths: List[str]) -> Dict[str, str]:
    """
    classify a list of paths into hap keys {"1","2","ungrouped"} if present
    returns only keys that were found
    """
    h: Dict[str, str] = {}
    from pathlib import Path as _P

    for p in paths:
        name = _P(p).name
        if "_1." in name and "bedmethyl" in name:
            h["1"] = p
        elif "_2." in name and "bedmethyl" in name:
            h["2"] = p
        elif "_ungrouped." in name and "bedmethyl" in name:
            h["ungrouped"] = p
    return h


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
    "--kit",
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
    help="plot against a haplotype-specific cohort (reference hap 1 or 2); requires exactly one target sample with _1 and _2",
)
@click.option(
    "--ref-hap",
    type=click.Choice(["1", "2"], case_sensitive=False),
    default="1",
    show_default=True,
    help="which haplotype of the reference to plot against (used with --hap-ref-plot)",
)
@click.option(
    "--auto-hap-match/--no-auto-hap-match",
    default=True,
    show_default=True,
    help="auto-map target hap1/hap2 to reference hap1/hap2 using robust |z|-median distance",
)
@click.option(
    "--hap-clear-ratio",
    type=float,
    default=1.05,
    show_default=True,
    help="Clarity threshold: max(score)/min(score) must be ≥ this to be clear.",
)
@click.option(
    "--hap-clear-delta",
    type=float,
    default=0.05,
    show_default=True,
    help="Clarity threshold: |Δscore| must be ≥ this to be clear.",
)
@click.option(
    "--hap-warn-ratio",
    type=float,
    default=1.10,
    show_default=True,
    help="Warning threshold for weak separation: max/min below this emits a warning.",
)
@click.option(
    "--hap-warn-delta",
    type=float,
    default=0.10,
    show_default=True,
    help="Warning threshold for weak separation: |Δscore| below this emits a warning.",
)
@click.option(
    "--min-hap-regions",
    type=int,
    default=10,
    show_default=True,
    help="Minimum regions used by hap matching.",
)
@click.option(
    "--hap-debug/--no-hap-debug",
    default=False,
    show_default=True,
    help="print hap-matching diagnostics (scores, mapping, n_used)",
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
    kit_or_bed: str,
    target: tuple[str, ...],
    haplotypes: bool,
    hap_ref_plot: bool,
    ref_hap: str,
    auto_hap_match: bool,
    hap_clear_ratio: float,
    hap_clear_delta: float,
    hap_warn_ratio: float,
    hap_warn_delta: float,
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
    intervals, region_names = load_intervals(kit_or_bed)
    region_names_unique = _unique_order(list(region_names))

    # load cmeth reference
    ref_df, meta = read_cmeth(cmeth_ref, logger=logger)
    mode = str(meta.get("mode", "aggregated")).lower()

    # build target arrays
    by_sample = _group_paths_by_sample([str(p) for p in tgt_paths])

    # -------------------------------------------------------------------------
    # HAPLOTYPE-SPECIFIC REFERENCE PLOT (with strict erroring on mismatch)
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

        # compute target hap arrays aligned to kit order
        h1, h2, overall = Methylation.get_levels_by_haplotype(parts, intervals)

        # guard: no finite values at all -> specific error
        if (np.isfinite(h1).sum() + np.isfinite(h2).sum()) == 0:
            raise click.ClickException(
                "Haplotype plot aborted: target hap1/hap2 have no finite methylation "
                "values across the requested intervals (no coverage?). "
                "Check the kit/BED vs your bedmethyl coordinates and phasing files."
            )

        # phasing QC mask (frac_ungrouped ≥ 45%)
        qc = Methylation.assess_phasing_quality(parts, intervals, thresh=0.45)
        qc_mask = qc.get("flag_mask", None)
        qc_note = "QC: frac ungrouped ≥ 45%"

        if auto_hap_match:
            res = Methylation.auto_map_target_haps(
                ref_df=ref_df,
                mode=mode,
                regions=region_names_unique,
                target_h1=h1,
                target_h2=h2,
                intervals=intervals,  # pass coords for robust alignment
                min_regions=min_hap_regions,
                ratio_ambiguous=hap_clear_ratio,
                delta_ambiguous=hap_clear_delta,
            )

            # strict error on no aligned regions / unusable stats
            if int(res.n_used_min) == 0:
                raise click.ClickException(
                    "Haplotype plot aborted: zero aligned/usable regions for hap matching.\n"
                    "Likely causes:\n"
                    "  • region-name/coordinate mismatch between reference and kit/BED\n"
                    "  • hap-specific quantiles/rows missing in the reference for this kit\n"
                    "  • the target hap files have no coverage in the chosen intervals\n"
                    "Please verify your kit choice and that the reference was built from the same kit."
                )

            # ambiguous -> treat as error (no silent fallback)
            if not res.clear:
                raise click.ClickException(
                    "Haplotype plot ambiguous: target hap(s) do not clearly map to reference haps.\n"
                    f"Scores: {res.scores}, n_used_min={res.n_used_min}. "
                    "Consider using pooled mode or checking coverage."
                )

            # weak separation warning (non-fatal)
            def _sep(a: float, b: float) -> tuple[float, float]:
                a = float(a)
                b = float(b)
                ratio = max(a, b) / (min(a, b) if min(a, b) > 0 else 1.0)
                delta = abs(a - b)
                return ratio, delta

            h1_ratio, h1_delta = _sep(
                res.scores.get("h1_vs_ref1", np.inf),
                res.scores.get("h1_vs_ref2", np.inf),
            )
            h2_ratio, h2_delta = _sep(
                res.scores.get("h2_vs_ref1", np.inf),
                res.scores.get("h2_vs_ref2", np.inf),
            )
            if hap_debug:
                click.secho("[hap-match] diagnostics:", fg="yellow")
                click.echo(
                    f"  mapping={res.mapping}  n_used_min={res.n_used_min}  mode={res.mode}"
                )
                click.echo(
                    f"  h1 ratio={h1_ratio:.3f} Δ={h1_delta:.3f} | h2 ratio={h2_ratio:.3f} Δ={h2_delta:.3f}"
                )

            # select the target hap that maps to the requested reference hap
            mapped_target_hk = (
                "1" if res.mapping.get("hap1") == ref_hap else "2"
            )
            target_for_plot = h1 if mapped_target_hk == "1" else h2
            title_suffix = (
                f"ref hap {ref_hap} (mapped: {sid}_hap{mapped_target_hk})"
            )

            # PLOT against hap-specific reference
            if mode == "aggregated":
                q = Methylation.hap_quantiles_for_reference(
                    ref_df,
                    mode,
                    region_names_unique,
                    ref_hap,
                    intervals=intervals,
                )
                need = {"q25", "q50", "q75"}
                if not need.issubset(set(q.columns)):
                    raise click.ClickException(
                        f"{cmeth_ref.name}: missing hap-specific quantiles in aggregated reference"
                    )

                tgt_mat = target_for_plot.reshape(1, -1)
                plot_methylation_from_quantiles(
                    regions=region_names_unique,
                    quantiles=q,
                    targets=tgt_mat,
                    target_labels=[f"{sid}_hap{mapped_target_hk}"],
                    save=str(out_png),
                    title=f"methylation – {title_suffix}",
                    shade_outliers=shade,
                    qc_mask=qc_mask,
                    qc_note=qc_note,
                )

            elif mode == "full":
                dfh = ref_df[
                    ref_df.get("hap", "").astype(str) == str(ref_hap)
                ].copy()
                if dfh.empty:
                    raise click.ClickException(
                        f"{cmeth_ref.name}: no rows for hap {ref_hap!r} in full reference"
                    )
                piv = dfh.pivot_table(
                    index="sample_id",
                    columns="region",
                    values="meth",
                    aggfunc="first",
                )
                cols = [c for c in region_names_unique if c in piv.columns]
                if not cols:
                    raise click.ClickException(
                        "Haplotype plot aborted: no overlapping region names between reference and kit/BED."
                    )

                ref_mat = piv.reindex(columns=cols, fill_value=np.nan).to_numpy(
                    dtype=float
                )
                # align target to same cols
                idx_map = {r: i for i, r in enumerate(region_names_unique)}
                sel = [idx_map[c] for c in cols]
                tgt_vec = target_for_plot[sel].reshape(1, -1)

                plot_methylation_levels_from_arrays(
                    sample_lv=ref_mat,
                    target_lv=tgt_vec,
                    region_names=cols,
                    save=str(out_png),
                    target_labels=[f"{sid}_hap{mapped_target_hk}"],
                    shade_outliers=shade,
                    qc_mask=(qc_mask[sel] if qc_mask is not None else None),
                    qc_note=qc_note,
                    title=f"methylation – {title_suffix}",
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
                    kit=str(kit_or_bed),
                    out=str(out_png),
                    n_targets=1,
                    mode=mode,
                    ref_hap=str(ref_hap),
                    mapped_target=mapped_target_hk,
                    hap_plot=True,
                    shade=bool(shade),
                ),
                message="ok",
            )
            return  # done

        else:
            # No manual mapping path implemented (we require auto)
            raise click.UsageError(
                "--hap-ref-plot without --auto-hap-match currently unsupported"
            )

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
            qc_note = "QC: frac ungrouped ≥ 45%"

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
    else:
        rows: List[np.ndarray] = []
        labels: List[str] = []
        for sid, paths in sorted(by_sample.items()):
            h = _classify_haps(paths)
            if h:
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

    # plot depending on reference mode
    if mode == "full":
        pooled = ref_df[ref_df.get("hap", "pooled").astype(str) == "pooled"]
        if pooled.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no pooled hap rows in full reference"
            )

        ref_piv = pooled.pivot_table(
            index="sample_id", columns="region", values="meth"
        )

        # align regions: use kit order but drop duplicates
        cols = [c for c in region_names_unique if c in ref_piv.columns]
        if not cols:
            raise click.ClickException(
                "No overlapping region names between reference and kit/BED."
            )
        ref_mat = ref_piv.reindex(columns=cols, fill_value=np.nan).to_numpy(
            dtype=float
        )

        # align targets to same selection
        idx_map = {r: i for i, r in enumerate(region_names_unique)}
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
            title="methylation per interval",
        )

    elif mode == "aggregated":
        # select meth section (if present) and deduplicate per region (pooled rows win)
        df = ref_df.copy()
        if "section" in df.columns:
            df = df[df["section"].astype(str) == "meth"].copy()
        if df.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no 'meth' section in aggregated reference"
            )

        df = _dedup_aggregated_meth(df)

        # index by region and drop any residual duplicates safely
        df = df.set_index("region")
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        # align to unique kit region order
        df = df.reindex(region_names_unique)
        df = df.reset_index().rename(columns={"index": "region"})
        regions_used = df["region"].tolist()

        # pull available quantiles
        q25 = _pick(df, "meth_q25")
        q50 = _pick(df, "meth_median", fallback="meth_q50")
        q75 = _pick(df, "meth_q75")
        q10 = _pick(df, "meth_q10", fallback="meth_q05")
        q90 = _pick(df, "meth_q90", fallback="meth_q95")

        if q25 is None or q50 is None or q75 is None:
            raise click.ClickException(
                f"{cmeth_ref.name}: missing methylation quantiles in aggregated reference"
            )

        q_cols: Dict[str, np.ndarray] = {"q25": q25, "q50": q50, "q75": q75}
        if q10 is not None:
            q_cols["q10"] = q10
        if q90 is not None:
            q_cols["q90"] = q90
        qdf = pd.DataFrame(q_cols, index=regions_used)

        # align targets to regions_used
        idx_map = {r: i for i, r in enumerate(region_names_unique)}
        sel = [idx_map[r] for r in regions_used if r in idx_map]
        tgt_mat = tgt_mat[:, sel]

        plot_methylation_from_quantiles(
            regions=regions_used,
            quantiles=qdf,
            targets=tgt_mat,
            target_labels=tgt_labels,
            save=str(out_png),
            title="methylation per interval",
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
