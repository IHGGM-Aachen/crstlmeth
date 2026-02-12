"""
crstlmeth/viz/cn_plot.py

copy-number plotting helpers
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

from crstlmeth.viz.style import PALETTE
from crstlmeth.viz.style import apply as apply_theme

__all__ = [
    "plot_cn_from_quantiles",
    "plot_cn_box_from_arrays",
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
            markerfacecolor=color_map.get(lab, PALETTE.tgt_cn),
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.2,
            label=lab,
        )
        for lab in labels
    ]
    leg = ax.get_legend()
    if leg is None:
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title=None,
        )
    else:
        # merge with existing legend
        existing_handles = leg.legend_handles
        existing_labels = [t.get_text() for t in leg.texts]
        merged_handles = existing_handles + handles
        merged_labels = existing_labels + [h.get_label() for h in handles]
        leg.remove()
        ax.legend(
            handles=merged_handles,
            labels=merged_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title=None,
        )


def _legend_add_offscale(ax: plt.Axes) -> None:
    off = Line2D(
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

    leg = ax.get_legend()
    if leg is None:
        ax.legend(
            handles=[off],
            labels=["off-scale"],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
        return

    existing_handles = leg.legend_handles
    existing_labels = [t.get_text() for t in leg.texts]

    merged_handles = existing_handles + [off]
    merged_labels = existing_labels + ["off-scale"]

    leg.remove()
    ax.legend(
        handles=merged_handles,
        labels=merged_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _legend_add_shading(ax: plt.Axes) -> None:
    """
    Add legend entries for shaded intervals (QC, flags).
    For CN we currently only use QC shading; the "flag" slot is kept
    for visual consistency with methylation plots and future use.
    """
    new_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE.shade_flag,
            lw=6,
            alpha=0.20,
            label="flag (red): BH-FDR < 0.05",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE.shade_qc,
            lw=6,
            alpha=0.30,
            label="QC (blue): ungrouped ≥ threshold",
        ),
    ]

    leg = ax.get_legend()
    if leg is None:
        ax.legend(
            handles=new_handles,
            labels=[h.get_label() for h in new_handles],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
        return

    existing_handles = leg.legend_handles
    existing_labels = [t.get_text() for t in leg.texts]

    merged_handles = existing_handles + new_handles
    merged_labels = existing_labels + [h.get_label() for h in new_handles]

    leg.remove()
    ax.legend(
        handles=merged_handles,
        labels=merged_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _shade_regions(
    ax: plt.Axes, mask: np.ndarray, color: str, alpha: float, z: int
) -> None:
    for i, bad in enumerate(mask):
        if bool(bad):
            ax.axvspan(i - 0.45, i + 0.45, color=color, alpha=alpha, zorder=z)


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
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    finite = np.isfinite(yy) & np.isfinite(xx)
    if not finite.any():
        return False, False

    xx = xx[finite]
    yy = yy[finite]

    had_low = (yy < ymin).any()
    had_high = (yy > ymax).any()

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


def _auto_cn_limits_from_quantiles(
    q: pd.DataFrame, targets_log2: np.ndarray
) -> tuple[float, float]:
    """
    Robust CN y-limits from reference quantiles *and* targets, symmetric around 0.
    """
    vals: list[np.ndarray] = []
    for col in ("q10", "q25", "q50", "q75", "q90"):
        if col in q.columns:
            vals.append(q[col].to_numpy(float))

    ref_arr = np.concatenate(vals) if vals else np.array([], dtype=float)
    ref_arr = ref_arr[np.isfinite(ref_arr)]

    tgt_arr = np.asarray(targets_log2, dtype=float).ravel()
    tgt_arr = tgt_arr[np.isfinite(tgt_arr)]

    if ref_arr.size == 0 and tgt_arr.size == 0:
        return (-2.0, 2.0)

    all_vals = (
        np.concatenate([ref_arr, tgt_arr])
        if ref_arr.size and tgt_arr.size
        else (ref_arr if ref_arr.size else tgt_arr)
    )

    lo = float(np.nanpercentile(all_vals, 2))
    hi = float(np.nanpercentile(all_vals, 98))

    # symmetric around 0, enforce minimum span, cap to +/-3, add small pad
    max_abs = max(abs(lo), abs(hi), 0.6)
    max_abs = min(max_abs + 0.15, 3.0)
    return (-max_abs, max_abs)


def _auto_cn_limits_from_ref(
    ref_log2: np.ndarray, tgt_log2: np.ndarray
) -> tuple[float, float]:
    """
    Robust CN y-limits from full reference values *and* targets, symmetric around 0.
    """
    ref_arr = np.asarray(ref_log2, dtype=float).ravel()
    ref_arr = ref_arr[np.isfinite(ref_arr)]

    tgt_arr = np.asarray(tgt_log2, dtype=float).ravel()
    tgt_arr = tgt_arr[np.isfinite(tgt_arr)]

    if ref_arr.size == 0 and tgt_arr.size == 0:
        return (-2.0, 2.0)

    all_vals = (
        np.concatenate([ref_arr, tgt_arr])
        if ref_arr.size and tgt_arr.size
        else (ref_arr if ref_arr.size else tgt_arr)
    )

    lo = float(np.nanpercentile(all_vals, 2))
    hi = float(np.nanpercentile(all_vals, 98))

    max_abs = max(abs(lo), abs(hi), 0.6)
    max_abs = min(max_abs + 0.15, 3.0)
    return (-max_abs, max_abs)


# ────────────────────────────────────────────────────────────────────
# Aggregated reference
# ────────────────────────────────────────────────────────────────────
def plot_cn_from_quantiles(
    regions: Sequence[str],
    quantiles: pd.DataFrame,  # columns: q25,q50,q75 (opt: q10,q90), index aligned to regions
    targets_log2: np.ndarray,  # (n_targets, n_regions)
    target_labels: Sequence[str],
    save: str | Path,
    title: str = "copy number (log2 ratio)",
    *,
    qc_mask: np.ndarray | None = None,
    qc_note: str | None = None,
) -> Dict[str, Any]:
    """
    Aggregated CN: draw 'box' glyphs from log2 quantiles and overlay target log2 dots.
    """
    apply_theme()
    sns.set_style("whitegrid")

    q = quantiles.reindex(regions)
    for c in ("q25", "q50", "q75"):
        if c not in q.columns:
            raise ValueError(f"quantiles missing column {c!r}")

    # keep rows with q50
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

    tgt = np.asarray(targets_log2, dtype=float)
    if tgt.ndim == 1:
        tgt = tgt.reshape(1, -1)
    n_keep = len(regions_used)
    if tgt.shape[1] != n_keep:
        if tgt.shape[1] > n_keep:
            tgt = tgt[:, :n_keep]
        else:
            pad = np.full((tgt.shape[0], n_keep - tgt.shape[1]), np.nan)
            tgt = np.hstack([tgt, pad])

    x = np.arange(n_keep)
    fig, ax = plt.subplots(figsize=(max(10, n_keep * 0.62), 6.2))

    # QC shading
    if qc_mask is not None and len(qc_mask) == len(regions):
        qc_mask_used = np.asarray(
            [qc_mask[list(regions).index(r)] for r in regions_used],
            dtype=bool,
        )
        _shade_regions(
            ax, qc_mask_used, color=PALETTE.shade_qc, alpha=0.30, z=0
        )
        if qc_note:
            # keep QC description in legend only
            pass

    # draw quantile "boxes"
    box_w = 0.62
    for i in range(n_keep):
        if np.isfinite(q25[i]) and np.isfinite(q75[i]):
            rect = Rectangle(
                (x[i] - box_w / 2.0, q25[i]),
                width=box_w,
                height=max(0.0, q75[i] - q25[i]),
                facecolor=PALETTE.iqr_face_cn,
                edgecolor=PALETTE.iqr_edge_cn,
                linewidth=1.0,
                alpha=0.28,
                zorder=1,
            )
            ax.add_patch(rect)
        if np.isfinite(q50[i]):
            ax.plot(
                [x[i] - box_w / 2.0, x[i] + box_w / 2.0],
                [q50[i], q50[i]],
                color=PALETTE.iqr_edge_cn,
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
                color=PALETTE.iqr_edge_cn,
                linewidth=1.0,
                zorder=1,
            )
            ax.plot(
                [x[i], x[i]],
                [q75[i], q90[i]],
                color=PALETTE.iqr_edge_cn,
                linewidth=1.0,
                zorder=1,
            )

    # overlay targets
    labels = list(target_labels)
    present = labels[:]
    pal = _target_palette(present, base="pastel")
    n_tgt = tgt.shape[0]
    offsets = np.linspace(-0.16, 0.16, num=max(1, n_tgt))

    y_lo, y_hi = _auto_cn_limits_from_quantiles(q, tgt)

    had_any_off = False
    for k, lab in enumerate(present):
        had_lo, had_hi = _clip_and_mark_offscale(
            ax,
            x + (offsets[k] if n_tgt > 1 else 0.0),
            tgt[k],
            color=pal.get(lab, PALETTE.tgt_cn),
            label=lab,
            ymin=y_lo,
            ymax=y_hi,
            jitter=0.0,
        )
        had_any_off |= had_lo or had_hi
    if had_any_off:
        _legend_add_offscale(ax)

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("log2 ratio")
    ax.set_xlabel("interval")
    ax.set_xticks(x)
    ax.set_xticklabels(regions_used, rotation=90, ha="right")
    ax.set_title(title, pad=16)

    _legend_from_labels(ax, present, pal)
    _legend_add_shading(ax)

    for c in ax.collections:
        c.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])

    fig.subplots_adjust(top=0.86, right=0.80, bottom=0.16, left=0.07)
    fig.tight_layout(rect=[0.05, 0.12, 0.80, 0.86])
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    return {"method": "quantiles"}


# ────────────────────────────────────────────────────────────────────
# Full reference
# ────────────────────────────────────────────────────────────────────
def plot_cn_box_from_arrays(
    ref_log2: np.ndarray,  # (n_refs, n_regions)
    tgt_log2: np.ndarray,  # (n_targets, n_regions)
    region_names: Sequence[str],
    save: str | Path,
    ref_label: str,
    tgt_labels: Sequence[str],
    title: str = "copy number (log2 ratio)",
    *,
    qc_mask: np.ndarray | None = None,
    qc_note: str | None = None,
) -> Dict[str, Any]:
    """
    Full CN: cohort boxplots of log2 and target log2 dots.
    """
    apply_theme()
    sns.set_style("whitegrid")

    ref_log2 = np.asarray(ref_log2, dtype=float)
    tgt_log2 = np.asarray(tgt_log2, dtype=float)
    if tgt_log2.ndim == 1:
        tgt_log2 = tgt_log2.reshape(1, -1)

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
            pass

    # cohort boxes
    df = pd.DataFrame(ref_log2, columns=region_names)
    sns.boxplot(data=df, color="white", linewidth=1.0, fliersize=0, ax=ax)
    # recolor like aggregated
    for artist in getattr(ax, "artists", []):
        artist.set_facecolor(PALETTE.iqr_face_cn)
        artist.set_alpha(0.28)
        artist.set_edgecolor(PALETTE.iqr_edge_cn)
        artist.set_linewidth(1.0)
    for line in getattr(ax, "lines", []):
        line.set_color(PALETTE.iqr_edge_cn)
        line.set_linewidth(1.0)

    # overlay targets
    labels = list(tgt_labels)
    present = labels[:]
    pal = _target_palette(present, base="pastel")

    x = np.arange(n)
    y_lo, y_hi = _auto_cn_limits_from_ref(ref_log2, tgt_log2)
    n_tgt = tgt_log2.shape[0]
    offsets = np.linspace(-0.16, 0.16, num=max(1, n_tgt))

    had_any_off = False
    for k, lab in enumerate(present):
        had_lo, had_hi = _clip_and_mark_offscale(
            ax,
            x + (offsets[k] if n_tgt > 1 else 0.0),
            tgt_log2[k],
            color=pal.get(lab, PALETTE.tgt_cn),
            label=lab,
            ymin=y_lo,
            ymax=y_hi,
            jitter=0.0,
        )
        had_any_off |= had_lo or had_hi
    if had_any_off:
        _legend_add_offscale(ax)

    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel("log2 ratio")
    ax.set_xlabel("interval")
    ax.set_title(title, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, rotation=90, ha="right")

    _legend_from_labels(ax, present, pal)
    _legend_add_shading(ax)

    for c in ax.collections:
        c.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])

    fig.subplots_adjust(top=0.86, right=0.80, bottom=0.16, left=0.07)
    fig.tight_layout(rect=[0.05, 0.12, 0.80, 0.86])
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    return {"method": "boxes"}
