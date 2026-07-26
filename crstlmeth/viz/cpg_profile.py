"""
CpG-resolution methylation profile plots.

Matplotlib is used for stable CLI PNG output. Plotly is used by the web
frontend and optional HTML export to provide hoverable CpG coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crstlmeth.core.cpg_profile import build_cpg_profile_table, ordered_cpg_rows

LabelMode = str


def _tick_positions(
    n: int, x: np.ndarray, label_cpgs: LabelMode
) -> tuple[np.ndarray, list[str]]:
    if label_cpgs == "all" and n <= 60:
        idx = np.arange(n)
    elif label_cpgs in {"ticks", "all"}:
        step = max(1, int(np.ceil(n / 20)))
        idx = np.arange(0, n, step)
    else:
        idx = np.array([], dtype=int)
    return x[idx], [f"CpG {i + 1}" for i in idx]


def plot_cpg_profile(
    *,
    reference_rows: pd.DataFrame,
    sample_tracks: Mapping[str, np.ndarray],
    title: str,
    save: str | Path,
    x_mode: str = "index",
    highlight_indices: Sequence[int] | None = None,
    highlight_positions: Sequence[int] | None = None,
    sample_labels: Mapping[str, str] | None = None,
    show_outer_band: bool = False,
    sample_line: bool = False,
    label_cpgs: LabelMode = "none",
    genomic_track: bool = False,
) -> None:
    """
    Plot a CpG-resolution methylation profile to PNG/SVG/PDF.

    Defaults are intentionally clinical/presentation-friendly: reference median
    plus IQR band, sample points, and no dense CpG labels unless requested.
    """
    df = ordered_cpg_rows(reference_rows)
    if df.empty:
        raise ValueError("No CpG rows available for plotting")

    n = len(df)
    starts = df["start"].to_numpy(dtype=int)
    if str(x_mode).lower() == "genomic":
        x = starts.astype(float)
        xlabel = "genomic position (BED start)"
    else:
        x = np.arange(1, n + 1, dtype=float)
        xlabel = "CpG index within region"

    q05 = pd.to_numeric(
        df.get("meth_q05", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=float)
    q25 = pd.to_numeric(
        df.get("meth_q25", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=float)
    q50 = pd.to_numeric(
        df.get("meth_median", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    q75 = pd.to_numeric(
        df.get("meth_q75", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=float)
    q95 = pd.to_numeric(
        df.get("meth_q95", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=float)

    fig_width = max(10.0, min(18.0, 8.0 + n / 35.0))
    if genomic_track:
        fig, (ax_top, ax) = plt.subplots(
            2,
            1,
            figsize=(fig_width, 6.2),
            gridspec_kw={"height_ratios": [0.7, 5.0]},
            sharex=True,
        )
        ax_top.hlines(
            0.5, float(x.min()), float(x.max()), linewidth=4, alpha=0.35
        )
        ax_top.vlines(x, 0.25, 0.75, linewidth=0.7, alpha=0.55)
        ax_top.set_yticks([])
        ax_top.set_ylabel("loci", rotation=0, labelpad=24)
        ax_top.spines[["left", "right", "top"]].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    if show_outer_band and np.isfinite(q05).any() and np.isfinite(q95).any():
        ax.fill_between(x, q05, q95, alpha=0.12, label="reference q05-q95")
    if np.isfinite(q25).any() and np.isfinite(q75).any():
        ax.fill_between(x, q25, q75, alpha=0.28, label="reference q25-q75")
    ax.plot(x, q50, linewidth=2.0, label="reference median")

    labels = dict(sample_labels or {})
    track_order = [
        k
        for k in (
            "pooled",
            "hap1",
            "hap2",
            "allele_low",
            "allele_high",
            "unphased",
        )
        if k in sample_tracks
    ]
    track_order += [k for k in sample_tracks if k not in track_order]
    for key in track_order:
        y = np.asarray(sample_tracks[key], dtype=float)
        if y.shape[0] != n:
            raise ValueError(
                f"sample track {key!r} has length {y.shape[0]}, expected {n}"
            )
        label = labels.get(key, key)
        if sample_line:
            ax.plot(
                x, y, marker="o", linewidth=1.2, markersize=4.0, label=label
            )
        else:
            ax.scatter(x, y, s=24, label=label, zorder=5)

    highlight_x: list[float] = []
    highlight_labels: list[str] = []
    if highlight_indices:
        for idx in highlight_indices:
            idx_i = int(idx)
            if 1 <= idx_i <= n:
                highlight_x.append(float(x[idx_i - 1]))
                highlight_labels.append(f"CpG {idx_i}")

    if highlight_positions and str(x_mode).lower() == "genomic":
        start_to_idx = {int(pos): i for i, pos in enumerate(starts)}
        for pos in highlight_positions:
            pos_i = int(pos)
            if pos_i in start_to_idx:
                i = start_to_idx[pos_i]
                highlight_x.append(float(x[i]))
                highlight_labels.append(f"CpG {i + 1}")
            else:
                highlight_x.append(float(pos_i))
                highlight_labels.append(str(pos_i))

    for xv in highlight_x:
        ax.axvline(xv, linestyle="--", linewidth=1.0, alpha=0.55)

    if label_cpgs == "highlighted":
        for xv, lab in zip(highlight_x, highlight_labels, strict=False):
            ax.annotate(
                lab,
                xy=(xv, 1.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

    ymin, ymax = ax.get_ylim()
    rug_y0 = max(0.0, ymin)
    rug_y1 = rug_y0 + 0.018 * max(0.2, ymax - ymin)
    ax.vlines(x, rug_y0, rug_y1, linewidth=0.55, alpha=0.50)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("methylation beta (Nmod / Nvalid)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", alpha=0.25)

    if str(x_mode).lower() == "genomic":
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    else:
        ticks, tick_labels = _tick_positions(n, x, label_cpgs)
        if len(ticks):
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        else:
            step = max(1, int(np.ceil(n / 20)))
            ax.set_xticks(x[::step])

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=220)
    plt.close(fig)


def make_cpg_profile_plotly(
    *,
    reference_rows: pd.DataFrame,
    sample_tracks: Mapping[str, np.ndarray],
    title: str,
    x_mode: str = "index",
    sample_labels: Mapping[str, str] | None = None,
    show_outer_band: bool = False,
    sample_line: bool = False,
    genomic_track: bool = True,
):
    """Return a Plotly figure with CpG coordinate hover data."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = ordered_cpg_rows(reference_rows)
    if df.empty:
        raise ValueError("No CpG rows available for plotting")

    table = build_cpg_profile_table(df, sample_tracks)
    n = len(table)
    if str(x_mode).lower() == "genomic":
        x = table["start"].astype(float).to_numpy()
        xlabel = "genomic position (BED start)"
    else:
        x = table["cpg_index"].astype(float).to_numpy()
        xlabel = "CpG index within region"

    custom = np.stack(
        [
            table["cpg_index"].to_numpy(),
            table["chrom"].astype(str).to_numpy(),
            table["start"].to_numpy(),
            table["end"].to_numpy(),
            table["cpg_id"].astype(str).to_numpy(),
        ],
        axis=1,
    )
    hover_common = (
        "CpG %{customdata[0]}<br>"
        "%{customdata[1]}:%{customdata[2]}-%{customdata[3]}<br>"
        "%{customdata[4]}<br>"
        "beta=%{y:.3f}<extra>%{fullData.name}</extra>"
    )

    rows = 2 if genomic_track else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.18, 0.82] if genomic_track else [1.0],
    )
    plot_row = 2 if genomic_track else 1

    if genomic_track:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.zeros(n),
                mode="markers",
                marker={"size": 8, "symbol": "line-ns-open"},
                name="CpG loci",
                customdata=custom,
                hovertemplate="CpG %{customdata[0]}<br>%{customdata[1]}:%{customdata[2]}-%{customdata[3]}<extra>genomic locus</extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(visible=False, row=1, col=1)

    q05 = table["ref_q05"].to_numpy(dtype=float)
    q25 = table["ref_q25"].to_numpy(dtype=float)
    q50 = table["ref_median"].to_numpy(dtype=float)
    q75 = table["ref_q75"].to_numpy(dtype=float)
    q95 = table["ref_q95"].to_numpy(dtype=float)

    if show_outer_band and np.isfinite(q05).any() and np.isfinite(q95).any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q95,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q05,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(100,100,100,0.13)",
                name="reference q05-q95",
                hoverinfo="skip",
            ),
            row=plot_row,
            col=1,
        )

    if np.isfinite(q25).any() and np.isfinite(q75).any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q75,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=plot_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q25,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.20)",
                name="reference q25-q75",
                hoverinfo="skip",
            ),
            row=plot_row,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=q50,
            mode="lines+markers",
            name="reference median",
            customdata=custom,
            hovertemplate=hover_common,
        ),
        row=plot_row,
        col=1,
    )

    labels = dict(sample_labels or {})
    track_order = [
        k
        for k in (
            "pooled",
            "hap1",
            "hap2",
            "allele_low",
            "allele_high",
            "unphased",
        )
        if k in sample_tracks
    ]
    track_order += [k for k in sample_tracks if k not in track_order]
    for key in track_order:
        y = np.asarray(sample_tracks[key], dtype=float)
        mode = "lines+markers" if sample_line else "markers"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                marker={"size": 8},
                name=labels.get(key, key),
                customdata=custom,
                hovertemplate=hover_common,
            ),
            row=plot_row,
            col=1,
        )

    fig.update_layout(
        title=title,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 60, "r": 20, "t": 80, "b": 60},
        height=620 if genomic_track else 540,
    )
    fig.update_xaxes(title_text=xlabel, row=plot_row, col=1)
    fig.update_yaxes(
        title_text="methylation beta (Nmod / Nvalid)",
        range=[-0.02, 1.02],
        row=plot_row,
        col=1,
    )
    return fig
