import numpy as np
import pandas as pd

from crstlmeth.viz.cpg_profile import make_cpg_profile_plotly, plot_cpg_profile


def _rows():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [10, 20, 30],
            "end": [11, 21, 31],
            "feature_type": ["cpg"] * 3,
            "region_id": ["R:CpG_001", "R:CpG_002", "R:CpG_003"],
            "parent_region": ["R"] * 3,
            "display_name": ["R CpG 1", "R CpG 2", "R CpG 3"],
            "hap_key": ["pooled"] * 3,
            "meth_median": [0.5, 0.6, 0.7],
            "meth_q25": [0.4, 0.5, 0.6],
            "meth_q75": [0.6, 0.7, 0.8],
            "meth_q05": [0.2, 0.3, 0.4],
            "meth_q95": [0.8, 0.9, 0.95],
        }
    )


def test_plot_cpg_profile_png(tmp_path):
    out = tmp_path / "cpg.png"
    plot_cpg_profile(
        reference_rows=_rows(),
        sample_tracks={"pooled": np.array([0.45, 0.65, 0.9])},
        title="test",
        save=out,
        genomic_track=True,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_make_cpg_profile_plotly_hoverdata():
    fig = make_cpg_profile_plotly(
        reference_rows=_rows(),
        sample_tracks={
            "hap1": np.array([0.1, 0.2, 0.3]),
            "hap2": np.array([0.8, 0.7, 0.6]),
        },
        title="test",
        genomic_track=True,
    )
    assert len(fig.data) >= 4
    assert "CpG" in fig.data[-1].hovertemplate
