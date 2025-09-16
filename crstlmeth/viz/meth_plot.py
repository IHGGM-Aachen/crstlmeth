"""
crstlmeth/viz/meth_plot.py

methylation plotting helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from crstlmeth.core.stats import (
    approx_normal_params_from_quantiles,
    one_sample_z_test,
)
from crstlmeth.viz.style import PALETTE
from crstlmeth.viz.style import apply as apply_theme

__all__ = [
    "plot_methylation_from_quantiles",
    "plot_methylation_levels_from_arrays",
]


# ────────────────────────────────────────────────────────────────────
# internals
# ────────────────────────────────────────────────────────────────────
def _target_palette(
    labels: Sequence[str], base: str = "pastel"
) -> Mapping[str, tuple]:
    labs = list(labels)
    if not labs:
        return {}
    pal = sns.color_palette(base, n_colors=len(labs))
    return {lab: pal[i] for i, lab in enumerate(labs)}


def _legend_from_labels(
    ax: plt.Axes, labels: Sequence[str], color_map: Mapping[str, tuple]
) -> None:
    if not labels:
        return
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map.get(lab, PALETTE.tgt_meth),
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.2,
            label=lab,
        )
        for lab in labels
    ]
    # Add a single legend entry for "off-scale" cross (if needed, caller appends it)
    existing = ax.get_legend()
    if existing is None:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title=None,
        )
    else:
        for h in handles:
            existing._legend_box._children[0]._children.append(h)  # type: ignore[attr-defined]


def _legend_add_offscale(ax: plt.Axes) -> None:
    # Add an "off-scale" cross marker entry once
    handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markeredgecolor="black",
            markerfacecolor="none",
            markeredgewidth=1.2,
            markersize=7.5,
            label="off-scale",
        )
    ]
    leg = ax.get_legend()
    if leg is None:
        ax.legend(handles=handles, loc="upper right", frameon=False)
    else:
        leg._legend_box._children[0]._children.extend(handles)  # type: ignore[attr-defined]


def _legend_add_shading(ax: plt.Axes) -> None:
    # a tiny legend snippet for shading semantics (placed above the sample legend)
    items = [
        Line2D(
            [0],
            [0],
            color=PALETTE.shade_flag,
            lw=6,
            alpha=0.2,
            label="flag: BH-FDR<0.05",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE.shade_qc,
            lw=6,
            alpha=0.3,
            label="qc: ≥45% ungrouped",
        ),
    ]
    ax.legend(
        items,
        ["flag: BH-FDR<0.05", "qc: ≥45% ungrouped"],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.0, 1.02),
    )


def _shade_regions(
    ax: plt.Axes, mask: np.ndarray, color: str, alpha: float, z: int
) -> None:
    for i, bad in enumerate(mask):
        if bool(bad):
            ax.axvspan(i - 0.45, i + 0.45, color=color, alpha=alpha, zorder=z)


def _style_boxes_like_aggregated(ax: plt.Axes, face, edge) -> None:
    for artist in getattr(ax, "artists", []):
        artist.set_facecolor(face)
        artist.set_alpha(0.28)
        artist.set_edgecolor(edge)
        artist.set_linewidth(1.0)
    for line in getattr(ax, "lines", []):
        line.set_color(edge)
        line.set_linewidth(1.0)


def _clip_and_mark_offscale(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color,
    label: str,
    ymin: float,
    ymax: float,
    jitter: float = 0.0,
) -> tuple[bool, bool]:
    """
    Scatter visible points within [ymin,ymax]. For off-scale points, draw an 'x'
    at the boundary (ymin or ymax). Returns (had_low_off, had_high_off).
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    finite = np.isfinite(yy) & np.isfinite(xx)
    if not finite.any():
        return False, False

    xx = xx[finite]
    yy = yy[finite]

    had_low = (yy < ymin).any()
    had_high = (yy > ymax).any()

    # inside
    mask_in = (yy >= ymin) & (yy <= ymax)
    if mask_in.any():
        ax.scatter(
            xx[mask_in] + (jitter if jitter else 0.0),
            yy[mask_in],
            s=38,
            color=color,
            edgecolor="black",
            linewidths=0.8,
            zorder=10,
            clip_on=False,
            label=label if label else None,
        )

    # off-scale low
    mask_lo = yy < ymin
    if mask_lo.any():
        ax.scatter(
            xx[mask_lo] + (jitter if jitter else 0.0),
            np.full(mask_lo.sum(), ymin),
            s=52,
            marker="x",
            color="black",
            linewidths=1.2,
            zorder=11,
            clip_on=False,
        )

    # off-scale high
    mask_hi = yy > ymax
    if mask_hi.any():
        ax.scatter(
            xx[mask_hi] + (jitter if jitter else 0.0),
            np.full(mask_hi.sum(), ymax),
            s=52,
            marker="x",
            color="black",
            linewidths=1.2,
            zorder=11,
            clip_on=False,
        )

    return bool(had_low), bool(had_high)


# ────────────────────────────────────────────────────────────────────
# Aggregated reference
# ────────────────────────────────────────────────────────────────────
def plot_methylation_from_quantiles(
    regions: Sequence[str],
    quantiles: pd.DataFrame,  # columns: q25,q50,q75 (opt: q10,q90), index aligned to regions
    targets: np.ndarray,  # (n_targets, n_regions) – methylation levels in [0,1]
    target_labels: Sequence[str],
    save: str | Path,
    title: str = "methylation per interval",
    *,
    shade_outliers: bool = True,
    fdr_alpha: float = 0.05,
    qc_mask: np.ndarray | None = None,
    qc_note: str | None = None,
) -> Dict[str, Any]:
    """
    Draw per-interval 'box' glyphs from aggregated quantiles and overlay targets.
    Optional: approximate BH-FDR shading using sigma from quantiles.
    Optional: QC shading (e.g., frac ungrouped ≥ 45%).
    """
    apply_theme()
    sns.set_style("whitegrid")

    q = quantiles.reindex(regions)
    for c in ("q25", "q50", "q75"):
        if c not in q.columns:
            raise ValueError(f"quantiles missing column {c!r}")

    # drop regions with no q50 to avoid categorical conversion errors
    mask = q["q50"].notna()
    regions_used = [
        r for r, m in zip(regions, mask, strict=False) if m
    ] or list(regions)
    q = q.loc[regions_used]

    q25 = q["q25"].to_numpy(float)
    q50 = q["q50"].to_numpy(float)
    q75 = q["q75"].to_numpy(float)
    q10 = q["q10"].to_numpy(float) if "q10" in q.columns else None
    q90 = q["q90"].to_numpy(float) if "q90" in q.columns else None

    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    # align/pad targets to regions_used
    n_keep = len(regions_used)
    if targets.shape[1] != n_keep:
        if targets.shape[1] > n_keep:
            targets = targets[:, :n_keep]
        else:
            pad = np.full((targets.shape[0], n_keep - targets.shape[1]), np.nan)
            targets = np.hstack([targets, pad])

    x = np.arange(n_keep)
    fig, ax = plt.subplots(figsize=(max(10, n_keep * 0.62), 6.2))

    # QC shading (beneath everything)
    if qc_mask is not None and len(qc_mask) == len(regions):
        qc_mask_used = np.asarray(
            [qc_mask[list(regions).index(r)] for r in regions_used], dtype=bool
        )
        _shade_regions(
            ax, qc_mask_used, color=PALETTE.shade_qc, alpha=0.30, z=0
        )
        if qc_note:
            ax.text(
                0.99,
                1.015,
                qc_note,
                transform=ax.transAxes,
                fontsize=8.6,
                ha="right",
                va="bottom",
            )

    # draw IQR boxes + whiskers like aggregated plot
    box_w = 0.62
    for i in range(n_keep):
        if np.isfinite(q25[i]) and np.isfinite(q75[i]):
            rect = Rectangle(
                (x[i] - box_w / 2.0, q25[i]),
                width=box_w,
                height=max(0.0, q75[i] - q25[i]),
                facecolor=PALETTE.iqr_face_meth,
                edgecolor=PALETTE.iqr_edge_meth,
                linewidth=1.0,
                alpha=0.28,
                zorder=1,
            )
            ax.add_patch(rect)
        if np.isfinite(q50[i]):
            ax.plot(
                [x[i] - box_w / 2.0, x[i] + box_w / 2.0],
                [q50[i], q50[i]],
                color=PALETTE.iqr_edge_meth,
                linewidth=1.2,
                zorder=2,
            )
        if (
            q10 is not None
            and q90 is not None
            and np.isfinite(q10[i])
            and np.isfinite(q90[i])
        ):
            ax.plot(
                [x[i], x[i]],
                [q10[i], q25[i]],
                color=PALETTE.iqr_edge_meth,
                linewidth=1.0,
                zorder=1,
            )
            ax.plot(
                [x[i], x[i]],
                [q75[i], q90[i]],
                color=PALETTE.iqr_edge_meth,
                linewidth=1.0,
                zorder=1,
            )

    # overlay targets (pastel, edged, halo)
    labels = list(target_labels)
    present = [lab for lab in labels]  # keep requested order
    pal = _target_palette(present, base="pastel")
    n_tgt = targets.shape[0]
    offsets = np.linspace(-0.16, 0.16, num=max(1, n_tgt))
    y_min, y_max = -0.10, 1.10

    had_any_off = False
    for k, lab in enumerate(present):
        had_lo, had_hi = _clip_and_mark_offscale(
            ax,
            x + (offsets[k] if n_tgt > 1 else 0.0),
            targets[k],
            color=pal.get(lab, PALETTE.tgt_meth),
            label=lab,
            ymin=y_min,
            ymax=y_max,
            jitter=0.0,
        )
        had_any_off |= had_lo or had_hi

    if had_any_off:
        _legend_add_offscale(ax)

    # optional approx-FDR shading
    if shade_outliers:
        mu, sd = approx_normal_params_from_quantiles(q25, q50, q75, q10, q90)
        sd = np.where((~np.isfinite(sd)) | (sd < 1e-6), np.nan, sd)
        flags_any = np.zeros(n_keep, dtype=bool)

        for row in targets:
            valid = (
                np.isfinite(row) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
            )
            if not np.any(valid):
                continue
            z = np.zeros_like(mu)
            z[valid] = (row[valid] - mu[valid]) / sd[valid]
            # normal CDF using scipy-like erf (avoid numpy.math)
            from math import erf, sqrt

            p = np.ones_like(mu)
            p[valid] = 2.0 * (
                1.0
                - 0.5
                * (
                    1.0
                    + np.vectorize(lambda a: erf(a / sqrt(2.0)))(
                        np.abs(z[valid])
                    )
                )
            )

            pf = p[np.isfinite(p)]
            if pf.size:
                order = np.argsort(pf)
                ranks = np.arange(1, len(pf) + 1, dtype=float)
                qv = np.empty_like(pf)
                qv[order] = pf[order] * len(pf) / ranks
                qv[order] = np.minimum.accumulate(qv[order][::-1])[::-1]
                flags_local = np.zeros_like(p, dtype=bool)
                flags_local[np.isfinite(p)] = qv < fdr_alpha
                flags_any |= flags_local

        _shade_regions(ax, flags_any, color=PALETTE.shade_flag, alpha=0.20, z=2)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("methylation level")
    ax.set_xlabel("interval")
    ax.set_xticks(x)
    ax.set_xticklabels(regions_used, rotation=90, ha="right")
    ax.set_title(title, pad=16)

    _legend_from_labels(ax, present, pal)
    _legend_add_shading(ax)

    # halo around markers for visibility
    for c in ax.collections:
        c.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])

    # spacing
    fig.subplots_adjust(top=0.86, right=0.80, bottom=0.16, left=0.07)
    fig.tight_layout(rect=[0.05, 0.12, 0.80, 0.86])
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    return {
        "method": "approx-z" if shade_outliers else "none",
        "fdr_alpha": fdr_alpha,
    }


# ────────────────────────────────────────────────────────────────────
# Full reference
# ────────────────────────────────────────────────────────────────────
def plot_methylation_levels_from_arrays(
    sample_lv: np.ndarray,  # (n_refs, n_regions) cohort methylation (0..1)
    target_lv: np.ndarray,  # (n_targets, n_regions)
    region_names: Sequence[str],
    save: str | Path,
    target_labels: Sequence[str],
    title: str = "methylation per interval",
    *,
    shade_outliers: bool = True,
    fdr_alpha: float = 0.05,
    qc_mask: np.ndarray | None = None,
    qc_note: str | None = None,
) -> Dict[str, Any]:
    """
    Full reference variant: boxplots from per-sample levels + target dots.
    Optional: z-test shading vs cohort.
    Optional: QC shading (e.g., frac ungrouped ≥ 45%).
    """
    apply_theme()
    sns.set_style("whitegrid")

    sample_lv = np.asarray(sample_lv, dtype=float)
    target_lv = np.asarray(target_lv, dtype=float)
    if target_lv.ndim == 1:
        target_lv = target_lv.reshape(1, -1)

    n = len(region_names)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.62), 6.2))

    # QC shading
    if qc_mask is not None and len(qc_mask) == n:
        _shade_regions(
            ax,
            np.asarray(qc_mask, dtype=bool),
            color=PALETTE.shade_qc,
            alpha=0.30,
            z=0,
        )
        if qc_note:
            ax.text(
                0.99,
                1.015,
                qc_note,
                transform=ax.transAxes,
                fontsize=8.6,
                ha="right",
                va="bottom",
            )

    # significance shading (z-test vs cohort)
    if shade_outliers and sample_lv.size and target_lv.size:
        _, _, _, flags = one_sample_z_test(
            sample_lv, target_lv, axis=0, fdr_alpha=fdr_alpha
        )
        flags_any = flags.any(axis=0) if flags.ndim == 2 else flags.astype(bool)
        _shade_regions(ax, flags_any, color=PALETTE.shade_flag, alpha=0.20, z=1)

    # cohort boxes styled like aggregated IQR boxes
    df = pd.DataFrame(sample_lv, columns=region_names)
    sns.boxplot(data=df, color="white", linewidth=1.0, fliersize=0, ax=ax)
    _style_boxes_like_aggregated(
        ax, face=PALETTE.iqr_face_meth, edge=PALETTE.iqr_edge_meth
    )

    # overlay targets (ensure they’re always visible)
    labels = list(target_labels)
    present = labels[:]  # keep order
    pal = _target_palette(present, base="pastel")

    # numeric x positions for all regions
    x = np.arange(n)
    y_min, y_max = -0.10, 1.10
    n_tgt = target_lv.shape[0]
    offsets = np.linspace(-0.16, 0.16, num=max(1, n_tgt))

    had_any_off = False
    for k, lab in enumerate(present):
        had_lo, had_hi = _clip_and_mark_offscale(
            ax,
            x + (offsets[k] if n_tgt > 1 else 0.0),
            target_lv[k],
            color=pal.get(lab, PALETTE.tgt_meth),
            label=lab,
            ymin=y_min,
            ymax=y_max,
            jitter=0.0,
        )
        had_any_off |= had_lo or had_hi

    if had_any_off:
        _legend_add_offscale(ax)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("methylation level")
    ax.set_xlabel("interval")
    ax.set_title(title, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=90, ha="right")

    _legend_from_labels(ax, present, pal)
    _legend_add_shading(ax)

    # halo on collections
    for c in ax.collections:
        c.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])

    fig.subplots_adjust(top=0.86, right=0.80, bottom=0.16, left=0.07)
    fig.tight_layout(rect=[0.05, 0.12, 0.80, 0.86])
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    return {"method": "z" if shade_outliers else "none", "fdr_alpha": fdr_alpha}
