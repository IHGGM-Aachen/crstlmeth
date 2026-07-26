import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from toy_crstlmeth import prepare_toy_dataset

from crstlmeth.core.references import create_cmeth_reference, read_cmeth


def test_create_cmeth_reference_contains_region_and_cpg_rows(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    out = tmp_path / "toy_reference.cmeth.gz"

    create_cmeth_reference(
        kit=toy.kit,
        bedmethyl_paths=toy.control_paths(),
        out=out,
        description="toy reference pytest",
        include_cpgs=True,
    )

    assert out.exists()
    assert Path(str(out) + ".tbi").exists()

    df, meta = read_cmeth(out)
    assert str(meta.get("source_sample_count")) == "4"
    assert set(df["feature_type"].astype(str)) == {"region", "cpg"}
    assert {"pooled", "allele_low", "allele_high", "unphased"}.issubset(
        set(df["hap_key"].astype(str))
    )

    region_rows = df[df["feature_type"].astype(str) == "region"]
    cpg_rows = df[df["feature_type"].astype(str) == "cpg"]
    assert region_rows["region_id"].nunique() == 4
    assert cpg_rows["parent_region"].nunique() == 4
    assert cpg_rows["start"].astype(int).nunique() == 24


def test_region_summary_is_not_just_one_cpg_row(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    out = tmp_path / "toy_reference.cmeth.gz"
    create_cmeth_reference(
        kit=toy.kit,
        bedmethyl_paths=toy.control_paths(),
        out=out,
        include_cpgs=True,
    )
    df, _meta = read_cmeth(out)

    region = df[
        (df["feature_type"].astype(str) == "region")
        & (df["region_id"].astype(str) == "TOY:ICR-balanced")
        & (df["hap_key"].astype(str) == "pooled")
    ].iloc[0]
    cpgs = df[
        (df["feature_type"].astype(str) == "cpg")
        & (df["parent_region"].astype(str) == "TOY:ICR-balanced")
        & (df["hap_key"].astype(str) == "pooled")
    ]

    assert len(cpgs) == 6
    assert float(region["meth_median"]) != float(cpgs.iloc[0]["meth_median"])
