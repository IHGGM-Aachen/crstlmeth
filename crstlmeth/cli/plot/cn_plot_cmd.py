"""
crstlmeth/cli/plot/cn_plot_cmd.py

redraw a copy-number plot from:
  - one *.cmeth reference cohort (mode=full or mode=aggregated)
  - one or more target bedmethyl files (resolved automatically)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import click
import numpy as np

from crstlmeth.core.copynumber import CopyNumber
from crstlmeth.core.discovery import resolve_bedmethyl_glob
from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.methylation import Methylation
from crstlmeth.core.references import read_cmeth
from crstlmeth.core.regions import load_intervals
from crstlmeth.viz.cn_plot import (
    plot_cn_box_from_arrays,
    plot_cn_from_quantiles,
)


def _unique_order(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _first_occurrence_selector(
    all_regions: List[str], unique_regions: List[str]
) -> List[int]:
    """
    Return indices into all_regions picking the first occurrence for each name in unique_regions.
    """
    first_idx: Dict[str, int] = {}
    for i, r in enumerate(all_regions):
        if r not in first_idx:
            first_idx[r] = i
    sel: List[int] = []
    for r in unique_regions:
        if r in first_idx:
            sel.append(first_idx[r])
    return sel


@click.command(
    name="copynumber",
    help=(
        "draw copy-number plot using a *.cmeth reference and one or more targets. "
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
    "--bed",
    "kit_or_bed",
    required=True,
    help="mlpa kit name or custom bed defining cn regions",
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
    default="copy_number.png",
    show_default=True,
    help="destination png",
)
@click.pass_context
def copynumber(
    ctx: click.Context,
    cmeth_ref: Path,
    kit_or_bed: str,
    target: tuple[str, ...],
    out_png: Path,
) -> None:
    """
    Produce out_png showing cohort distribution and target log2 ratios.
    """
    logger = get_logger_from_cli(ctx)

    # resolve target bedmethyl files
    tgt_paths: List[Path] = []
    for t in target:
        tgt_paths.extend(resolve_bedmethyl_glob([str(t)]))
    if not tgt_paths:
        raise click.UsageError("no target bedmethyl files resolved")

    # regions
    intervals, region_names = load_intervals(kit_or_bed)
    region_names_unique = _unique_order(list(region_names))
    sel_unique = _first_occurrence_selector(
        list(region_names), region_names_unique
    )

    # load reference
    ref_df, meta = read_cmeth(cmeth_ref, logger=logger)
    mode = str(meta.get("mode", "full")).lower()
    if mode not in {"aggregated", "full"}:
        raise click.ClickException(
            f"{cmeth_ref.name}: unsupported reference mode {mode!r} "
            "(expected 'aggregated' or 'full')"
        )

    # optional phasing QC mask (from a single target with hap-part files)
    qc_mask = None
    qc_note = None

    def _group_paths_by_sample(paths: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        from pathlib import Path as _P

        for p in paths:
            out.setdefault(_P(p).name.split("_")[0], []).append(p)
        return out

    def _classify(paths: List[str]) -> Dict[str, str]:
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

    grouped = _group_paths_by_sample([str(p) for p in tgt_paths])
    if len(grouped) == 1:
        sid0, paths0 = next(iter(grouped.items()))
        parts0 = _classify(paths0)
        if parts0:
            qc = Methylation.assess_phasing_quality(
                parts0, intervals, thresh=0.45
            )
            qc_mask = qc.get("flag_mask", None)
            qc_note = "QC: frac ungrouped ≥ 45%"

    # ──────────────────────────────────────────────────────────────
    # Aggregated reference: both ref & targets use Nvalid counts
    # ──────────────────────────────────────────────────────────────
    if mode == "aggregated":
        df = ref_df.copy()
        if "section" not in df.columns:
            raise click.ClickException(
                f"{cmeth_ref.name}: aggregated CN reference requires a 'section' column "
                "with rows marked as 'cn'"
            )
        df = df[df["section"].astype(str) == "cn"]
        if df.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no 'cn' section found in aggregated reference"
            )

        if "region" not in df.columns:
            raise click.ClickException(
                f"{cmeth_ref.name}: aggregated CN expects a 'region' column"
            )

        # pick quantile columns and rename to q25,q50,q75,(q10,q90)
        colmap = {}
        for a, b in [
            ("ratio_q25_log2", "q25"),
            ("ratio_median_log2", "q50"),
            ("ratio_q75_log2", "q75"),
            ("ratio_q10_log2", "q10"),
            ("ratio_q90_log2", "q90"),
            ("ratio_q05_log2", "q10"),
            ("ratio_q95_log2", "q90"),
        ]:
            if a in df.columns and b not in colmap:
                colmap[a] = b
        missing_core = {
            "ratio_q25_log2",
            "ratio_median_log2",
            "ratio_q75_log2",
        } - set(df.columns)
        if missing_core:
            raise click.ClickException(
                f"{cmeth_ref.name}: missing CN quantiles in aggregated reference"
            )

        df = df.set_index("region")
        df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(region_names_unique)

        q = df[list(colmap.keys())].rename(columns=colmap)

        # cn_norm must be present in the header
        try:
            cn_norm = str(meta["cn_norm"]).lower()
        except KeyError as exc:
            raise click.ClickException(
                f"{cmeth_ref.name}: aggregated reference missing required 'cn_norm' header"
            ) from exc

        # compute target log2 under the recorded normalization recipe
        tgt_log2, labels = CopyNumber.target_log2_for_aggregated(
            [str(p) for p in tgt_paths],
            intervals,
            cn_norm=cn_norm,
            logger=logger,
        )

        # align targets to unique region selection
        tgt_log2 = tgt_log2[:, sel_unique]

        plot_cn_from_quantiles(
            regions=region_names_unique,
            quantiles=q,
            targets_log2=tgt_log2,
            target_labels=labels,
            save=str(out_png),
            title="copy number (log2 ratio)",
            qc_mask=(qc_mask[sel_unique] if qc_mask is not None else None),
            qc_note=qc_note,
        )

    # ──────────────────────────────────────────────────────────────
    # Full reference: both ref & targets use depth_per_bp
    # ──────────────────────────────────────────────────────────────
    elif mode == "full":
        pooled = ref_df[ref_df.get("hap", "pooled").astype(str) == "pooled"]
        if pooled.empty:
            raise click.ClickException(
                f"{cmeth_ref.name}: no pooled rows in full reference"
            )

        # reference depth_per_bp matrix: (n_ref_samples, n_regions)
        piv = pooled.pivot_table(
            index="sample_id", columns="region", values="depth_per_bp"
        )
        cols = [c for c in region_names_unique if c in piv.columns]
        if not cols:
            raise click.ClickException(
                "No overlapping region names between reference and kit/BED."
            )

        ref_depth = piv.reindex(columns=cols, fill_value=np.nan).to_numpy(
            dtype=float
        )

        # cohort-mean normalisation per region
        mu = np.nanmean(ref_depth, axis=0)
        mu[~np.isfinite(mu)] = 1.0
        mu[mu == 0] = 1.0

        ref_ratio = ref_depth / mu
        ref_log2 = np.log2(
            ref_ratio,
            where=np.isfinite(ref_ratio),
            out=np.full_like(ref_ratio, np.nan),
        )

        # IMPORTANT: target must use depth_per_bp as well (not raw counts)
        tgt_df = CopyNumber.bedmethyl_depth_per_bp(
            [str(p) for p in tgt_paths],
            intervals,
            region_names,
            logger=logger,
        )
        tgt_piv = tgt_df.pivot_table(
            index="sample_id", columns="region_name", values="depth_per_bp"
        )
        tgt_depth = tgt_piv.reindex(columns=cols, fill_value=np.nan).to_numpy(
            dtype=float
        )

        tgt_ratio = tgt_depth / mu
        tgt_log2 = np.log2(
            tgt_ratio,
            where=np.isfinite(tgt_ratio),
            out=np.full_like(tgt_ratio, np.nan),
        )

        # debug-ish summary in the TSV log so we can sanity-check scales
        if logger is not None:
            ref_flat = ref_log2[np.isfinite(ref_log2)]
            tgt_flat = tgt_log2[np.isfinite(tgt_log2)]
            ref_q = (
                (
                    float(np.nanpercentile(ref_flat, 5))
                    if ref_flat.size
                    else None
                ),
                (float(np.nanmedian(ref_flat)) if ref_flat.size else None),
                (
                    float(np.nanpercentile(ref_flat, 95))
                    if ref_flat.size
                    else None
                ),
            )
            tgt_q = (
                (
                    float(np.nanpercentile(tgt_flat, 5))
                    if tgt_flat.size
                    else None
                ),
                (float(np.nanmedian(tgt_flat)) if tgt_flat.size else None),
                (
                    float(np.nanpercentile(tgt_flat, 95))
                    if tgt_flat.size
                    else None
                ),
            )

            log_event(
                logger,
                event="cn-debug",
                cmd="copynumber-full",
                params=dict(
                    n_ref_samples=int(ref_log2.shape[0]),
                    n_targets=int(tgt_log2.shape[0]),
                    n_regions=int(ref_log2.shape[1]),
                    ref_log2_q5_med_q95=ref_q,
                    tgt_log2_q5_med_q95=tgt_q,
                ),
                message="ref/target log2 summary",
            )

        plot_cn_box_from_arrays(
            ref_log2=ref_log2,
            tgt_log2=tgt_log2,
            region_names=cols,
            save=str(out_png),
            ref_label=f"{cmeth_ref.name}",
            tgt_labels=list(tgt_piv.index),
            title="copy number (log2 ratio)",
            qc_mask=(
                qc_mask[_first_occurrence_selector(region_names, cols)]
                if qc_mask is not None
                else None
            ),
            qc_note=qc_note,
        )

    click.echo(f"figure written -> {out_png.resolve()}")

    log_event(
        logger,
        event="plot_copynumber",
        cmd="plot copynumber",
        params=dict(
            cmeth=str(cmeth_ref),
            kit=str(kit_or_bed),
            out=str(out_png),
            mode=mode,
            n_targets=len(tgt_paths),
            n_regions_all=int(len(region_names)),
            n_regions_unique=int(len(region_names_unique)),
        ),
        message="ok",
    )
