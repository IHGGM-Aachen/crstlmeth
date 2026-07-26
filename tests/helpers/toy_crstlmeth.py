from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

TOY_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "toy_crstlmeth"
BEDM_RE = re.compile(
    r"^(?P<sample>.+?)[._-](?P<hap>1|2|ungrouped)(?:[._-]\w+)*\.bedmethyl(?:\.gz)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToyDataset:
    root: Path
    kit: Path
    samples: dict[str, dict[str, Path]]
    modmix: Path

    @property
    def controls(self) -> list[str]:
        return sorted([s for s in self.samples if s.startswith("CTRL")])

    @property
    def cases(self) -> list[str]:
        return sorted([s for s in self.samples if s.startswith("CASE")])

    def sample_paths(
        self, sample_id: str, roles: Iterable[str] = ("1", "2", "ungrouped")
    ) -> list[Path]:
        parts = self.samples[sample_id]
        return [parts[r] for r in roles if r in parts]

    def control_paths(
        self, roles: Iterable[str] = ("1", "2", "ungrouped")
    ) -> list[Path]:
        out: list[Path] = []
        for sample_id in self.controls:
            out.extend(self.sample_paths(sample_id, roles=roles))
        return out

    def case_paths(
        self, roles: Iterable[str] = ("1", "2", "ungrouped")
    ) -> list[Path]:
        out: list[Path] = []
        for sample_id in self.cases:
            out.extend(self.sample_paths(sample_id, roles=roles))
        return out


def _bgzip_and_index(src: Path) -> Path:
    pysam = pytest.importorskip("pysam")
    gz = Path(str(src) + ".gz")
    with src.open("rb") as inp, pysam.BGZFile(str(gz), "wb") as out:
        shutil.copyfileobj(inp, out)
    pysam.tabix_index(str(gz), preset="bed", force=True)
    return gz


def _discover_samples(root: Path) -> dict[str, dict[str, Path]]:
    samples: dict[str, dict[str, Path]] = {}
    for p in sorted((root / "samples").glob("*/*.bedmethyl.gz")):
        m = BEDM_RE.match(p.name)
        if not m:
            continue
        samples.setdefault(m.group("sample"), {})[m.group("hap").lower()] = p
    return samples


def prepare_toy_dataset(tmp_path: Path) -> ToyDataset:
    """Copy bundled plain toy data to tmp_path and create bgzip/tabix files."""
    work = tmp_path / "toy_crstlmeth"
    shutil.copytree(TOY_SOURCE_ROOT, work)

    for src in sorted(work.glob("samples/*/*.bedmethyl")):
        _bgzip_and_index(src)
    modmix_gz = _bgzip_and_index(work / "modkit" / "modmix_mh.bedmethyl")

    return ToyDataset(
        root=work,
        kit=work / "toy_regions.bed",
        samples=_discover_samples(work),
        modmix=modmix_gz,
    )
