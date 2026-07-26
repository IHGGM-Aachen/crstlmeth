"""
tests/core/test_regions.py

Unit tests for crstlmeth.core.regions module.
"""

from pathlib import Path

from crstlmeth.core.regions import (
    load_intervals,
    split_haplotypes,
)

# -------------------------------------------------------------------
# Test load_intervals with built-in kit
# -------------------------------------------------------------------


def test_load_intervals_toy_kit_file():
    kit = Path("tests/data/toy_crstlmeth/toy_regions.bed")
    assert kit.exists(), f"Toy kit file missing: {kit}"

    intervals, names = load_intervals(str(kit))

    assert len(intervals) >= 1
    assert len(intervals) == len(names)
    assert "TOY:ICR-balanced" in names


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
