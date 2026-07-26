import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from toy_crstlmeth import prepare_toy_dataset

from crstlmeth.core.parsers import get_locus_table, query_bedmethyl


def test_query_bedmethyl_reads_bgzipped_tabix_toy_file(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    path = toy.samples["CTRL_A01"]["ungrouped"]

    df = query_bedmethyl(str(path), "chr1", 1000, 1120)

    assert len(df) == 6
    assert set(df["mod_code"]) == {"m"}
    assert df["Nvalid_cov"].min() > 0
    assert df["Nmod"].between(0, df["Nvalid_cov"]).all()


def test_get_locus_table_mh_mode_collapses_denominator_once(tmp_path):
    toy = prepare_toy_dataset(tmp_path)

    loci = get_locus_table(str(toy.modmix), "chr3", 490, 520, mode="mh")

    assert len(loci) == 2
    first = loci.sort_values("start").iloc[0]
    assert int(first["Nvalid_cov"]) == 20
    # m row contributes 6/20 and h row contributes 2/20 -> numerator 8, denominator 20.
    assert int(first["Nmod"]) == 8
