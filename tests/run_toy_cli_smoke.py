#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

from helpers.toy_crstlmeth import prepare_toy_dataset


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        toy = prepare_toy_dataset(root)
        out = root / "out"
        out.mkdir()
        ref = out / "toy_reference.cmeth.gz"
        run(
            [
                "crstlmeth",
                "reference",
                "create",
                "--kit",
                str(toy.kit),
                "--include-cpgs",
                "-o",
                str(ref),
                *map(str, toy.control_paths()),
            ]
        )
        run(["crstlmeth", "reference", "validate", "--strict", str(ref)])
        run(
            [
                "crstlmeth",
                "plot",
                "cpg-profile",
                "--cmeth",
                str(ref),
                "--region",
                "TOY:LOM-case",
                "--sample-track",
                "both_haps",
                "--export-cpg-table",
                str(out / "cpg.tsv"),
                "--out-html",
                str(out / "cpg.html"),
                "--out",
                str(out / "cpg.png"),
                *map(str, toy.case_paths()),
            ]
        )
        run(
            [
                "crstlmeth",
                "plot",
                "methylation",
                "--cmeth",
                str(ref),
                "--kit",
                str(toy.kit),
                "--out",
                str(out / "meth.png"),
                str(toy.samples["CASE_X01"]["ungrouped"]),
            ]
        )
        run(
            [
                "crstlmeth",
                "plot",
                "copynumber",
                "--cmeth",
                str(ref),
                "--kit",
                str(toy.kit),
                "--out",
                str(out / "cn.png"),
                *map(str, toy.case_paths()),
            ]
        )
        print(f"\nSmoke test outputs in temporary directory: {out}")
        for p in sorted(out.iterdir()):
            print(p)


if __name__ == "__main__":
    main()
