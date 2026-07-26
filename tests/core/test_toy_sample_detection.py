import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from toy_crstlmeth import prepare_toy_dataset

from crstlmeth.core.cpg_profile import group_paths_by_sample


def test_group_paths_by_sample_accepts_underscore_dot_and_dash(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    paths = []
    for sample_id in toy.controls + toy.cases:
        paths.extend(toy.sample_paths(sample_id))
        paths.extend(Path(str(p) + ".tbi") for p in toy.sample_paths(sample_id))

    grouped = group_paths_by_sample(paths)

    assert "CTRL_A01" in grouped
    assert "CTRL.B02" in grouped
    assert "CTRL-C03" in grouped
    assert "CASE_X01" in grouped
    assert set(grouped["CASE_X01"]) == {"1", "2", "ungrouped"}


def test_group_paths_by_sample_ignores_tbi_files(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    target = toy.sample_paths("CASE_X01")
    grouped = group_paths_by_sample(
        target + [Path(str(p) + ".tbi") for p in target]
    )
    assert list(grouped) == ["CASE_X01"]
    assert all(
        not str(p).endswith(".tbi") for p in grouped["CASE_X01"].values()
    )
