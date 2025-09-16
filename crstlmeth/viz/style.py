"""
crstlmeth.viz.style

shared plot styling (pastel theme, palette, rc tweaks)
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib as mpl
import seaborn as sns

# ────────────────────────────────────────────────────────────────────
# palette (soft pastels + functional colors)
# ────────────────────────────────────────────────────────────────────
PALETTE = SimpleNamespace(
    # methylation
    iqr_face_meth="#A3D5FF",  # pastel blue
    iqr_edge_meth="#5A8BB0",
    # copy number
    iqr_face_cn="#FFD8A8",  # pastel orange
    iqr_edge_cn="#C26E00",
    # targets
    tgt_meth="#A3F7B5",  # default green fallback (actual colors come from pastel palette)
    tgt_cn="#A3F7B5",
    # shading
    shade_flag="#F94144",  # red-ish for flagged
    shade_qc="#B197FC",  # lilac for QC
    # text / grid
    grid="#E9ECEF",
    text="#222222",
)


# ────────────────────────────────────────────────────────────────────
# theme / rcParams
# ────────────────────────────────────────────────────────────────────
def apply() -> None:
    """
    apply a soft pastel theme with readable defaults
    """
    sns.set_theme(context="notebook", style="whitegrid")
    sns.set_palette("pastel")

    mpl.rcParams.update(
        {
            # text
            "text.color": PALETTE.text,
            "axes.labelcolor": PALETTE.text,
            "axes.titlesize": 12,
            "axes.titleweight": 600,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # grid / spines
            "axes.grid": True,
            "grid.color": PALETTE.grid,
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#AAAAAA",
            "axes.linewidth": 0.8,
            # legend
            "legend.frameon": False,
            # figure
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
