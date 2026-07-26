import pandas as pd
import pytest

from crstlmeth.core.cmeth import CMETH_COLUMNS, CMethFile


def _row(
    feature_type="region", region_id="R", parent_region=".", hap_key="pooled"
):
    row = {c: "." for c in CMETH_COLUMNS}
    row.update(
        {
            "chrom": "chr1",
            "start": 10,
            "end": 20,
            "feature_type": feature_type,
            "region_id": region_id,
            "parent_region": parent_region,
            "display_name": region_id,
            "hap_key": hap_key,
            "n_ref": 2,
            "meth_status": "ok",
            "row_status": "ok",
        }
    )
    return row


def test_cmeth_validation_accepts_region_and_cpg_parent():
    rows = pd.DataFrame(
        [
            _row("region", "R", ".", "pooled"),
            _row("cpg", "R:CpG_001", "R", "pooled") | {"start": 12, "end": 13},
        ],
        columns=CMETH_COLUMNS,
    )
    cm = CMethFile.build_reference(
        rows,
        meta={"source_sample_count": 2, "source_file_count": 6},
        target_bed=["chr1\t10\t20\tR"],
    )
    cm.validate()


def test_cmeth_validation_rejects_cpg_without_parent():
    rows = pd.DataFrame(
        [
            _row("region", "R", ".", "pooled"),
            _row("cpg", "R:CpG_001", ".", "pooled") | {"start": 12, "end": 13},
        ],
        columns=CMETH_COLUMNS,
    )
    with pytest.raises(ValueError, match="parent_region"):
        CMethFile.build_reference(
            rows,
            meta={"source_sample_count": 2, "source_file_count": 6},
            target_bed=["chr1\t10\t20\tR"],
        )
