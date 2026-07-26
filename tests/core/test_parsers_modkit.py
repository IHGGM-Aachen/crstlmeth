import pandas as pd

from crstlmeth.core.parsers import (
    _parse_modkit_bed_line,
    collapse_bedmethyl_to_loci,
)


def test_parse_modkit_bed_line_extracts_mod_code_and_counts():
    line = "chr1\t10\t11\tm,CG,0\t12\t+\t10\t11\t0,0,0\t12\t75.0\t9\t3\t0\t0\t0\t0\t0"
    row = _parse_modkit_bed_line(line)
    assert row["chrom"] == "chr1"
    assert row["mod_code"] == "m"
    assert row["Nvalid_cov"] == 12
    assert row["Nmod"] == 9


def test_collapse_loci_avoids_double_counting_denominator():
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 10],
            "end": [11, 11],
            "strand": ["+", "+"],
            "mod_code": ["m", "h"],
            "Nvalid_cov": [20, 20],
            "Nmod": [6, 2],
        }
    )
    collapsed = collapse_bedmethyl_to_loci(df)
    assert collapsed.loc[0, "Nvalid_cov"] == 20
    assert collapsed.loc[0, "Nmod"] == 8
