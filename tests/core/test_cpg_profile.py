from pathlib import Path

import numpy as np
import pandas as pd

from crstlmeth.core import cpg_profile


def _rows():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 20],
            "end": [11, 21],
            "feature_type": ["cpg", "cpg"],
            "region_id": ["R:CpG_001", "R:CpG_002"],
            "parent_region": ["R", "R"],
            "display_name": ["R CpG 1", "R CpG 2"],
            "hap_key": ["pooled", "pooled"],
            "meth_median": [0.5, 0.6],
            "meth_q25": [0.4, 0.5],
            "meth_q75": [0.6, 0.7],
            "meth_q05": [0.2, 0.3],
            "meth_q95": [0.8, 0.9],
        }
    )


def test_build_cpg_profile_table_adds_sample_delta():
    rows = _rows()
    table = cpg_profile.build_cpg_profile_table(
        rows, {"pooled": np.array([0.25, 0.75])}
    )
    assert table["cpg_index"].tolist() == [1, 2]
    assert table["chrom"].tolist() == ["chr1", "chr1"]
    assert np.allclose(table["sample_pooled_beta"], [0.25, 0.75])
    assert np.allclose(table["delta_pooled_vs_ref_median"], [-0.25, 0.15])


def test_select_sample_tracks_both_haps():
    tracks = {
        "pooled": np.array([0.5]),
        "hap1": np.array([0.1]),
        "hap2": np.array([0.9]),
        "allele_low": np.array([0.1]),
        "allele_high": np.array([0.9]),
    }
    out = cpg_profile.select_sample_tracks(tracks, "both_haps")
    assert sorted(out) == ["hap1", "hap2"]


def test_align_sample_tracks_exposes_hap_and_allele_tracks(
    monkeypatch, tmp_path
):
    rows = _rows()

    def fake_get_locus_table(
        filepath, chrom, start, end, mode="m", codes=None, group_by_strand=False
    ):
        name = Path(filepath).name
        if ".1." in name:
            return pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1"],
                    "start": [10, 20],
                    "end": [11, 21],
                    "Nvalid_cov": [10, 10],
                    "Nmod": [1, 9],
                }
            )
        if ".2." in name:
            return pd.DataFrame(
                {
                    "chrom": ["chr1", "chr1"],
                    "start": [10, 20],
                    "end": [11, 21],
                    "Nvalid_cov": [10, 10],
                    "Nmod": [8, 2],
                }
            )
        return pd.DataFrame(
            columns=["chrom", "start", "end", "Nvalid_cov", "Nmod"]
        )

    monkeypatch.setattr(cpg_profile, "get_locus_table", fake_get_locus_table)
    tracks = cpg_profile.align_sample_tracks(
        rows,
        {
            "1": tmp_path / "S.1.bedmethyl.gz",
            "2": tmp_path / "S.2.bedmethyl.gz",
        },
        "chr1",
        0,
        100,
    )
    assert np.allclose(tracks["hap1"], [0.1, 0.9])
    assert np.allclose(tracks["hap2"], [0.8, 0.2])
    assert np.allclose(tracks["allele_low"], [0.1, 0.2])
    assert np.allclose(tracks["allele_high"], [0.8, 0.9])
    assert np.allclose(tracks["pooled"], [0.45, 0.55])
