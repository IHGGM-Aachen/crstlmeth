"""
tests/core/test_regions.py

Unit tests for crstlmeth.core.regions module.
"""

from pathlib import Path

import pandas as pd

from crstlmeth.core.regions import (
    DEFAULT_KITS_DIR,
    load_intervals,
    split_haplotypes,
)

# -------------------------------------------------------------------
# Test load_intervals with built-in kit
# -------------------------------------------------------------------


def test_load_intervals_builtin_kit():
    kit_name = "ME030"
    bed_path = DEFAULT_KITS_DIR / f"{kit_name}_meth.bed"
    assert bed_path.exists(), f"Built-in kit file missing: {bed_path}"

    # Read the raw BED file for expected values
    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "name"],
    )
    expected_intervals = list(
        zip(
            df.chrom.astype(str),
            df.start.astype(int),
            df.end.astype(int),
            strict=False,
        )
    )
    expected_names = df.name.astype(str).tolist()

    intervals, names = load_intervals(kit_name)
    assert intervals == expected_intervals
    assert names == expected_names


# -------------------------------------------------------------------
# Test load_intervals with custom BED path
# -------------------------------------------------------------------


def test_load_intervals_custom_bed(tmp_path):
    # Write a small custom BED
    custom = tmp_path / "regions.bed"
    lines = [
        "chrA\t10\t20\trA",
        "chrB\t30\t40\trB",
        "chrC\t50\t60\trC",
    ]
    custom.write_text("\n".join(lines))

    intervals, names = load_intervals(str(custom))
    assert intervals == [("chrA", 10, 20), ("chrB", 30, 40), ("chrC", 50, 60)]
    assert names == ["rA", "rB", "rC"]


# -------------------------------------------------------------------
# Test split_haplotypes
# -------------------------------------------------------------------


def test_split_haplotypes():
    files = [
        "/foo/sample_1.x.bedmethyl.gz",
        "/foo/sample_2.x.bedmethyl.gz",
        "/foo/other.bedmethyl.gz",
        "/foo/sample_1.y.bedmethyl.gz",
    ]
    hap1, hap2 = split_haplotypes(files)
    # Haplotype 1 should include all with '_1.'
    assert all("_1." in Path(f).name for f in hap1)
    # Haplotype 2 should include all with '_2.'
    assert all("_2." in Path(f).name for f in hap2)
    # Files without either tag should be ignored
    combined = set(hap1 + hap2)
    assert "/foo/other.bedmethyl.gz" not in combined
