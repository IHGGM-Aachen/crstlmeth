from pathlib import Path

from crstlmeth.core.discovery import scan_bedmethyl
from crstlmeth.core.samples import (
    parse_bedmethyl_name,
    ready_sample_ids,
    sample_status_table,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_parse_bedmethyl_name_accepts_common_separators_and_aliases():
    cases = {
        "M43599_1.bedmethyl.gz": ("M43599", "1", False),
        "M43599_2.bedmethyl.gz": ("M43599", "2", False),
        "M43599_ungrouped.bedmethyl.gz": ("M43599", "ungrouped", False),
        "M24520LR.1.bedmethyl.gz": ("M24520LR", "1", False),
        "M24520LR.2.bedmethyl.gz": ("M24520LR", "2", False),
        "M24520LR.ungrouped.bedmethyl.gz": ("M24520LR", "ungrouped", False),
        "S-1.bedmethyl.gz": ("S", "1", False),
        "S-hap2.bedmethyl.gz": ("S", "2", False),
        "S-unphased.bedmethyl.gz": ("S", "ungrouped", False),
        "S_1.bedmethyl.gz.tbi": ("S", "1", True),
    }
    for name, expected in cases.items():
        parsed = parse_bedmethyl_name(name)
        assert parsed is not None
        assert (parsed.sample_id, parsed.role, parsed.is_index) == expected


def test_scan_bedmethyl_keeps_incomplete_visible_but_filters_when_required(
    tmp_path,
):
    for name in [
        "S1_1.bedmethyl.gz",
        "S1_2.bedmethyl.gz",
        "S1_ungrouped.bedmethyl.gz",
        "S2.1.bedmethyl.gz",
    ]:
        _touch(tmp_path / name)
    for name in [
        "S1_1.bedmethyl.gz.tbi",
        "S1_2.bedmethyl.gz.tbi",
        "S1_ungrouped.bedmethyl.gz.tbi",
    ]:
        _touch(tmp_path / name)

    visible = scan_bedmethyl(tmp_path, require_index=False)
    assert sorted(visible) == ["S1", "S2"]
    assert sorted(visible["S1"]) == ["1", "2", "ungrouped"]
    assert sorted(visible["S2"]) == ["1"]

    ready = scan_bedmethyl(tmp_path, require_index=True)
    assert sorted(ready) == ["S1"]
    assert sorted(ready["S1"]) == ["1", "2", "ungrouped"]


def test_sample_status_table_and_ready_ids(tmp_path):
    for name in [
        "S_1.bedmethyl.gz",
        "S_1.bedmethyl.gz.tbi",
        "S_2.bedmethyl.gz",
        "S_2.bedmethyl.gz.tbi",
        "P_ungrouped.bedmethyl.gz",
        "P_ungrouped.bedmethyl.gz.tbi",
    ]:
        _touch(tmp_path / name)
    samples = scan_bedmethyl(tmp_path, require_index=False)
    table = sample_status_table(samples).set_index("sample_id")
    assert bool(table.loc["S", "ready_haps"])
    assert bool(table.loc["P", "ready_ungrouped"])
    assert ready_sample_ids(samples, require_haps=True) == ["S"]
    assert ready_sample_ids(samples, require_ungrouped=True) == ["P"]
