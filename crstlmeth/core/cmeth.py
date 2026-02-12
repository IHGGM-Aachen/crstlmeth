"""
crstlmeth/core/cmeth.py

versioned, sectioned *.cmeth* handling (spec, registry, read/write/validate)

design:
  - a *.cmeth* file has a header (## key: value) and a single TSV body
  - "mode" selects a set of allowed "sections" (logical blocks of rows)
  - each section has required/optional columns
  - modes:
      mode=aggregated -> sections: "meth" (methylation), "cn" (copy-number)
      mode=full       -> single unsectioned per-sample table
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping

import pandas as pd

# ────────────────────────────────────────────────────────────────────
# basic header helpers
# ────────────────────────────────────────────────────────────────────
_FMT_VERSION = "v0.1"
_HEADER_TAG = f"## crstlmeth-cmeth {_FMT_VERSION}"


def _today() -> str:
    return dt.date.today().isoformat()


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _read_header_lines(path: Path) -> tuple[list[str], str]:
    """
    Extract header lines and body text from a .cmeth file.

    Returns
    -------
    list_of_header_lines, tsv_body_string
    """
    lines: List[str] = []
    with path.open() as fh:
        first = fh.readline().rstrip("\n")
        if not first.startswith(_HEADER_TAG):
            raise ValueError(
                f"{path}: unsupported reference format – "
                f"first line was {first!r}, expected prefix {_HEADER_TAG!r}"
            )
        lines.append(first)

        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line.startswith("##"):
                fh.seek(pos)
                break
            lines.append(line.rstrip("\n"))

        body = fh.read()
    return lines, body


def parse_header_meta(path: Path) -> dict[str, str]:
    hdr, _ = _read_header_lines(path)
    meta: dict[str, str] = {}
    for line in hdr[1:]:
        if line.startswith("##") and ":" in line:
            key, _, val = line[2:].partition(":")
            meta[key.strip()] = val.strip()
    # defaults
    meta.setdefault("version", _FMT_VERSION)
    meta.setdefault("mode", "aggregated")
    return meta


# ────────────────────────────────────────────────────────────────────
# spec registry
# ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SectionSpec:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModeSpec:
    name: str
    sections: Mapping[str, SectionSpec] | None = None  # aggregated has sections
    required_header: tuple[str, ...] = ()  # keys that must exist


@dataclass(frozen=True)
class VersionSpec:
    version: str
    modes: Mapping[str, ModeSpec]


def _v01_spec() -> VersionSpec:
    # aggregated v0.1
    meth = SectionSpec(
        name="meth",
        required=(
            "region",
            "chrom",
            "start",
            "end",
            "hap_key",
            "n",
            "n_covered",
            "meth_median",
            "meth_q25",
            "meth_q75",
        ),
        optional=(
            "meth_mean",
            "meth_sd",
            "meth_q05",
            "meth_q95",
            "meth_hist",
            "depth_mean",
            "depth_q05",
            "depth_q25",
            "depth_median",
            "depth_q75",
            "depth_q95",
            "frac_ungrouped_mean",
            "frac_ungrouped_median",
        ),
    )
    cn = SectionSpec(
        name="cn",
        required=(
            "region",
            "chrom",
            "start",
            "end",
            "n",
            "ratio_median_log2",
            "ratio_q25_log2",
            "ratio_q75_log2",
        ),
        optional=(
            "ratio_mean_log2",
            "ratio_sd_log2",
            "ratio_q05_log2",
            "ratio_q95_log2",
            "ratio_hist_log2",
            "metric",  # compatibility from earlier drafts
        ),
    )
    aggregated = ModeSpec(
        name="aggregated",
        sections={"meth": meth, "cn": cn},
        required_header=("k_min", "cn_norm"),
    )

    # full v0.1
    full = ModeSpec(
        name="full",
        sections=None,  # single table, no section column
        required_header=("denom_dedup",),
    )

    return VersionSpec(
        version=_FMT_VERSION, modes={"aggregated": aggregated, "full": full}
    )


_REGISTRY: dict[str, VersionSpec] = {_FMT_VERSION: _v01_spec()}


def get_version_spec(version: str | None) -> VersionSpec:
    return _REGISTRY.get(version or _FMT_VERSION, _REGISTRY[_FMT_VERSION])


# ────────────────────────────────────────────────────────────────────
# core object
# ────────────────────────────────────────────────────────────────────
@dataclass
class CMethFile:
    meta: dict[str, str]
    df: pd.DataFrame

    @property
    def mode(self) -> str:
        return str(self.meta.get("mode", "aggregated")).lower()

    @property
    def version(self) -> str:
        return str(self.meta.get("version", _FMT_VERSION))

    # ---- section utilities ---------------------------------------------------
    def has_section_column(self) -> bool:
        """
        Return True if the dataframe has a 'section' column.
        """
        return "section" in self.df.columns

    def infer_section(self) -> pd.Series:
        """
        Return the section labels as a string Series.

        In aggregated mode the 'section' column is required.
        """
        if not self.has_section_column():
            raise ValueError(
                "aggregated .cmeth files require a 'section' column "
                "with values such as 'meth' or 'cn'."
            )
        return self.df["section"].astype(str)

    def split_sections(self) -> dict[str, pd.DataFrame]:
        """
        Return {section: dataframe} for aggregated mode; for full mode, return
        a single entry {'full': df}.
        """
        if self.mode != "aggregated":
            return {"full": self.df.copy()}

        sec = self.infer_section()
        out: dict[str, pd.DataFrame] = {}
        for sname, block in self.df.groupby(sec):
            out[str(sname)] = block.copy()
        return out

    # ---- validation ----------------------------------------------------------
    def validate(self) -> None:
        """
        Validate header keys and table schema against the versioned specification.
        """
        spec = get_version_spec(self.version)
        mode_spec = spec.modes.get(self.mode)
        if mode_spec is None:
            raise ValueError(f"unsupported mode: {self.mode!r}")

        # header keys
        for k in mode_spec.required_header:
            if k not in self.meta:
                raise ValueError(f"missing header key: {k!r}")

        # full mode: per-sample schema
        if mode_spec.sections is None:
            req = {
                "sample_id",
                "region",
                "chrom",
                "start",
                "end",
                "hap",
                "n_valid",
                "n_mod",
                "meth",
                "depth_per_bp",
            }
            missing = req - set(self.df.columns)
            if missing:
                raise ValueError(f"missing full columns: {sorted(missing)}")
            return

        # aggregated mode: explicit 'section' column and section-wise checks
        if "section" not in self.df.columns:
            raise ValueError(
                "aggregated .cmeth files require a 'section' column "
                f"with values in {sorted(mode_spec.sections.keys())!r}"
            )

        valid_sections = set(mode_spec.sections.keys())
        sec_values = set(self.df["section"].astype(str).unique())
        unknown = sec_values - valid_sections
        if unknown:
            raise ValueError(
                "aggregated .cmeth: unknown section values "
                f"{sorted(unknown)}; expected subset of "
                f"{sorted(valid_sections)}"
            )

        sec = self.split_sections()
        for sname, block in sec.items():
            s = mode_spec.sections.get(sname)
            if not s:
                continue
            missing = set(s.required) - set(block.columns)
            if missing:
                raise ValueError(
                    f"section {sname!r}: missing columns {sorted(missing)}"
                )

    # ---- write ---------------------------------------------------------------
    def write(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as fh:
            fh.write(f"{_HEADER_TAG}\n")
            for k, v in self.meta.items():
                fh.write(f"## {k}: {v}\n")
            if len(self.df):
                self.df.to_csv(fh, sep="\t", index=False)
        return path

    # ---- constructors --------------------------------------------------------
    @classmethod
    def read(cls, path: Path) -> "CMethFile":
        meta = parse_header_meta(path)
        _, body = _read_header_lines(path)
        df = (
            pd.read_csv(io.StringIO(body), sep="\t")
            if body.strip()
            else pd.DataFrame()
        )
        obj = cls(meta=meta, df=df)
        obj.validate()
        return obj

    @classmethod
    def build_full(
        cls, rows: pd.DataFrame, *, meta: dict[str, str]
    ) -> "CMethFile":
        m = dict(meta)
        m.setdefault("version", _FMT_VERSION)
        m.setdefault("mode", "full")
        m.setdefault("date", _today())
        obj = cls(meta=m, df=rows.copy())
        obj.validate()
        return obj

    @classmethod
    def build_aggregated(
        cls,
        meth_df: pd.DataFrame | None,
        cn_df: pd.DataFrame | None,
        *,
        meta: dict[str, str],
    ) -> "CMethFile":
        if (meth_df is None) and (cn_df is None):
            raise ValueError(
                "build_aggregated: need at least one of meth_df or cn_df"
            )
        m = dict(meta)
        m.setdefault("version", _FMT_VERSION)
        m.setdefault("mode", "aggregated")
        m.setdefault("date", _today())

        parts: list[pd.DataFrame] = []
        if meth_df is not None and len(meth_df):
            dfm = meth_df.copy()
            if "section" not in dfm.columns:
                dfm.insert(0, "section", "meth")
            parts.append(dfm)
        if cn_df is not None and len(cn_df):
            dfc = cn_df.copy()
            if "section" not in dfc.columns:
                dfc.insert(0, "section", "cn")
            parts.append(dfc)

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        obj = cls(meta=m, df=df)
        obj.validate()
        return obj
