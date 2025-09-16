# tests/core/test_coverage.py

"""
tests/core/test_coverage.py

Unit tests for crstlmeth.core.coverage module.
"""

import numpy as np
import pandas as pd

from crstlmeth.core.copynumber import (
    get_average_read_count,
    get_read_count,
    get_read_count_deviation,
    get_target_deviation,
)


# -------------------------------------------------------------------
# Helper to build a sample bedcov DataFrame
# -------------------------------------------------------------------
def make_bedcov_df():
    """
    Create a DataFrame mimicking samtools_bedcov output for two samples ('s1','s2')
    across three regions ('r1','r2','r3').
    """
    # Two reads per region per sample
    data = {
        "region_name": [
            "r1",
            "r1",  # sample s1 & s2 for region r1
            "r2",
            "r2",  # sample s1 & s2 for region r2
            "r3",
            "r3",  # sample s1 & s2 for region r3
        ],
        "read_count": [
            10,
            20,  # r1 counts
            30,
            40,  # r2 counts
            0,
            0,  # r3 counts (zero-depth region)
        ],
        "read_depth_region": [
            100,
            200,  # actual depth can differ
            300,
            400,
            0,
            0,
        ],
        "sample_id": ["s1", "s2", "s1", "s2", "s1", "s2"],
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------------
# Tests for get_read_count
# -------------------------------------------------------------------
def test_get_read_count_shape_and_values():
    df = make_bedcov_df()
    rc = get_read_count(df)
    # Should be a 2D array: one row per sample (s1, s2), one column per region (r1, r2, r3)
    assert isinstance(rc, np.ndarray)
    assert rc.shape == (2, 3)
    # First row = s1 counts = [10, 30, 0]
    # Second row = s2 counts = [20, 40, 0]
    assert np.array_equal(rc[0], [10, 30, 0])
    assert np.array_equal(rc[1], [20, 40, 0])


# -------------------------------------------------------------------
# Tests for get_average_read_count
# -------------------------------------------------------------------
def test_get_average_read_count():
    df = make_bedcov_df()
    avg = get_average_read_count(df)
    # Averages per region: r1=(10+20)/2=15, r2=35, r3=0
    assert isinstance(avg, np.ndarray)
    assert avg.shape == (3,)
    assert np.allclose(avg, [15.0, 35.0, 0.0])


# -------------------------------------------------------------------
# Tests for get_read_count_deviation
# -------------------------------------------------------------------
def test_get_read_count_deviation():
    df = make_bedcov_df()
    std = get_read_count_deviation(df)
    # Standard deviation per region: r1 std([10,20])=5, r2 std([30,40])=5, r3 std([0,0])=0
    assert isinstance(std, np.ndarray)
    assert std.shape == (3,)
    assert np.allclose(std, [5.0, 5.0, 0.0])


# -------------------------------------------------------------------
# Tests for get_target_deviation
# -------------------------------------------------------------------
def test_get_target_deviation_flags():
    df = make_bedcov_df()
    # Use r1 and r2 only (non-zero std) for flagging
    refs = df[df.region_name != "r3"]
    # Create target same shape as a single sample: reuse refs but treat all as one target
    # Here target counts = [10,30], mean_ref = [15,35], std_ref=[5,5]
    target = pd.DataFrame(
        {
            "region_name": ["r1", "r2"],
            "read_count": [
                20,
                30,
            ],  # deviations: z1=(20-15)/5=1, z2=(30-35)/5=-1
            "read_depth_region": [200, 300],
            "sample_id": ["t1", "t1"],
        }
    )
    flags = get_target_deviation(refs, target, z_score_threshold=0.8)
    # z_scores = [1.0, -1.0] → threshold=0.8 → [True, False]
    assert isinstance(flags, np.ndarray)
    assert flags.shape == (2,)
    assert flags[0]  # first flag must be True
    assert not flags[1]  # second flag must be False
