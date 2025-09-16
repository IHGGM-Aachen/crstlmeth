"""
tests/core/test_methylation.py

Unit tests for crstlmeth.core.methylation module.
"""

import numpy as np

from crstlmeth.core.methylation import (
    filter_empty_regions,
    get_confidence_intervals,
    get_methylation_levels,
    get_target_deviations,
    handle_ungrouped_samples,
)

# -------------------------------------------------------------------
# Dummy BedMethyl stub
# -------------------------------------------------------------------


class DummyBedMethyl:
    """
    Stub for BedMethyl.get_region_stats().
    Initialized with a list of (nmod, nvalid) tuples.
    """

    def __init__(self, stats):
        self._stats = list(stats)

    def get_region_stats(self, chrom, start, end):
        return self._stats.pop(0)


# -------------------------------------------------------------------
# Tests for get_methylation_levels & get_confidence_intervals
# -------------------------------------------------------------------


def test_get_methylation_levels_and_cis_zero_z():
    # Two samples × two intervals
    # Sample1: (1,2)->0.5; (3,4)->0.75
    # Sample2: (0,1)->0.0; (2,2)->1.0
    samples = [
        DummyBedMethyl([(1, 2), (3, 4)]),
        DummyBedMethyl([(0, 1), (2, 2)]),
    ]
    intervals = [("chr", 0, 1), ("chr", 1, 2)]

    levels = get_methylation_levels(samples, intervals)
    expected = np.array([[0.5, 0.75], [0.0, 1.0]])
    assert np.allclose(levels, expected)

    # CI with z_score=0 should collapse to levels
    samples_ci = [
        DummyBedMethyl([(1, 2), (3, 4)]),
        DummyBedMethyl([(0, 1), (2, 2)]),
    ]
    cis = get_confidence_intervals(samples_ci, intervals, z_score=0.0)
    assert cis.shape == (2, 2, 2)
    assert np.allclose(cis[:, :, 0], expected)
    assert np.allclose(cis[:, :, 1], expected)


# -------------------------------------------------------------------
# Tests for get_target_deviations
# -------------------------------------------------------------------


def test_get_target_deviations():
    # sample_levels shape (2,2): [[0,2],[2,4]] ⇒ mean=[1,3], std=[1,1]
    sample_levels = np.array([[0, 2], [2, 4]])
    target_levels = np.array([[3, 3]])
    # z = [(3-1)/1=2, (3-3)/1=0], threshold=1.5 ⇒ [True, False]
    devs = get_target_deviations(sample_levels, target_levels, z_thresh=1.5)
    assert devs.shape == (1, 2)
    assert bool(devs[0, 0]) is True
    assert bool(devs[0, 1]) is False


# -------------------------------------------------------------------
# Tests for filter_empty_regions
# -------------------------------------------------------------------


def test_filter_empty_regions():
    # levels_list = [ [0,0,1], [0,0,0.5] ], intervals [r1,r2,r3]
    levels_sample = np.array([[0.0, 0.0, 1.0]])
    levels_target = np.array([[0.0, 0.0, 0.5]])
    levels_list = [levels_sample.copy(), levels_target.copy()]
    intervals = [("c", 0, 1), ("c", 1, 2), ("c", 2, 3)]
    names = ["r1", "r2", "r3"]

    filter_empty_regions(levels_list, intervals, names)

    # r1,r2 dropped; only r3 remains
    assert intervals == [("c", 2, 3)]
    assert names == ["r3"]
    for arr in levels_list:
        assert arr.shape == (1, 1)


# -------------------------------------------------------------------
# Tests for handle_ungrouped_samples
# -------------------------------------------------------------------


def test_handle_ungrouped_samples():
    samples = [
        "/path/A/file1.gz",
        "/path/A/file2.gz",
        "/path/B/file3.gz",
    ]
    groups = handle_ungrouped_samples(samples)
    # Normalize order
    normalized = [sorted(g) for g in groups]
    assert sorted(normalized) == [
        sorted(["/path/A/file1.gz", "/path/A/file2.gz"]),
        ["/path/B/file3.gz"],
    ]
