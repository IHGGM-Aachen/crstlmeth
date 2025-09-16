"""
crstlmeth/core/regions.py

load region intervals from mlpa kits or custom bed files
"""

import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

default_kits_dir = Path(__file__).parent.parent / "kits"


def load_intervals(
    kit_or_path: str, kits_dir: Union[str, Path, None] = None
) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """
    return (intervals, region_names) from either a built-in kit or a user bed

    if a kit name is passed, the corresponding bed is searched under kits_dir
    """
    p = Path(kit_or_path)
    if p.is_file():
        bed_path = p
    else:
        base = Path(
            kits_dir
            or os.environ.get("CRSTL_METH_KITS_DIR")
            or default_kits_dir
        )
        bed_path = base / f"{kit_or_path}_meth.bed"
        if not bed_path.exists():
            raise FileNotFoundError(f"mlpa kit bed not found at {bed_path!r}")

    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "name"],
    )

    intervals = list(
        zip(
            df.chrom.astype(str),
            df.start.astype(int),
            df.end.astype(int),
            strict=False,
        )
    )
    region_names = df.name.astype(str).tolist()
    return intervals, region_names


def split_haplotypes(files: List[str]) -> Tuple[List[str], List[str]]:
    """
    split input file paths into haplotype 1 and 2 groups by filename pattern
    """
    hap1, hap2 = [], []
    for f in files:
        fn = Path(f).name
        if "_1." in fn:
            hap1.append(f)
        elif "_2." in fn:
            hap2.append(f)
    return hap1, hap2
