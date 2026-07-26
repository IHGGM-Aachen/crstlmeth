"""
crstlmeth/core/cmeth.py

CMETH reference handling.

A CMETH reference is a BED-like, bgzip/tabix-indexable cohort summary.
It stores rich aggregated reference statistics and an embedded target BED block,
but it never stores per-sample rows or source sample filenames.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import pysam

CMETH_VERSION = "0.1.2"
_MAGIC = "## cmeth"

METH_HIST_BINS = tuple(i / 10 for i in range(11))
MVAL_HIST_BINS = tuple(range(-10, 12, 2))
CN_LOG2_HIST_BINS = (
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
)

CMETH_COLUMNS: tuple[str, ...] = (
    "chrom",
    "start",
    "end",
    "feature_type",
    "region_id",
    "parent_region",
    "display_name",
    "hap_key",
    "strand",
    "length_bp",
    "cpg_count",
    "probe_count",
    "gene",
    "transcript",
    "annotation",
    "n_ref",
    "n_meth",
    "n_cn",
    "n_depth",
    "n_hap_resolved",
    "n_unphased",
    "meth_nmod_sum",
    "meth_nvalid_sum",
    "meth_mean",
    "meth_sd",
    "meth_median",
    "meth_mad",
    "meth_min",
    "meth_q01",
    "meth_q05",
    "meth_q10",
    "meth_q25",
    "meth_q75",
    "meth_q90",
    "meth_q95",
    "meth_q99",
    "meth_max",
    "meth_hist10",
    "mval_mean",
    "mval_sd",
    "mval_median",
    "mval_mad",
    "mval_min",
    "mval_q01",
    "mval_q05",
    "mval_q10",
    "mval_q25",
    "mval_q75",
    "mval_q90",
    "mval_q95",
    "mval_q99",
    "mval_max",
    "mval_hist10",
    "mval_hist_underflow",
    "mval_hist_overflow",
    "beta_alpha",
    "beta_beta",
    "beta_fit_status",
    "depth_mean",
    "depth_sd",
    "depth_median",
    "depth_mad",
    "depth_min",
    "depth_q05",
    "depth_q25",
    "depth_q75",
    "depth_q95",
    "depth_max",
    "nvalid_mean",
    "nvalid_sd",
    "nvalid_median",
    "nvalid_q25",
    "nvalid_q75",
    "frac_unphased_mean",
    "frac_unphased_sd",
    "frac_unphased_median",
    "frac_unphased_q25",
    "frac_unphased_q75",
    "frac_unphased_q95",
    "hap_balance_mean",
    "hap_balance_sd",
    "hap_balance_median",
    "hap_balance_q25",
    "hap_balance_q75",
    "allele_gap_mean",
    "allele_gap_sd",
    "allele_gap_median",
    "allele_gap_q05",
    "allele_gap_q25",
    "allele_gap_q75",
    "allele_gap_q95",
    "allele_mean_mean",
    "allele_mean_sd",
    "allele_mean_median",
    "allele_mean_q25",
    "allele_mean_q75",
    "cn_log2_mean",
    "cn_log2_sd",
    "cn_log2_median",
    "cn_log2_mad",
    "cn_log2_min",
    "cn_log2_q01",
    "cn_log2_q05",
    "cn_log2_q10",
    "cn_log2_q25",
    "cn_log2_q75",
    "cn_log2_q90",
    "cn_log2_q95",
    "cn_log2_q99",
    "cn_log2_max",
    "cn_log2_hist10",
    "cn_log2_hist_underflow",
    "cn_log2_hist_overflow",
    "cn_cov_mean",
    "cn_cov_sd",
    "cn_cov_median",
    "cn_cov_q25",
    "cn_cov_q75",
    "meth_status",
    "cn_status",
    "phasing_status",
    "row_status",
)

REQUIRED_HEADER_KEYS: tuple[str, ...] = (
    "kind",
    "coordinate",
    "created",
    "description",
    "target_name",
    "target_count",
    "target_bed_columns",
    "target_bed_count",
    "source_sample_count",
    "source_file_count",
    "cn_norm",
    "mvalue_eps",
    "meth_hist_bins",
    "mval_hist_bins",
    "cn_log2_hist_bins",
)

HEADER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("format", ("kind", "coordinate", "created")),
    ("description", ("description",)),
    (
        "target",
        (
            "target_name",
            "target_count",
            "target_bed_columns",
            "target_bed_count",
        ),
    ),
    ("cohort", ("source_sample_count", "source_file_count")),
    ("calculation", ("cn_norm", "mvalue_eps")),
    ("histograms", ("meth_hist_bins", "mval_hist_bins", "cn_log2_hist_bins")),
)

VALID_HAP_KEYS = {"pooled", "allele_low", "allele_high", "unphased"}
VALID_FEATURE_TYPES = {"region", "cpg", "probe", "bin", "custom"}


def _today() -> str:
    return dt.date.today().isoformat()


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def _format_bins(bins: Iterable[float]) -> str:
    return ",".join(f"{float(x):g}" for x in bins)


def _clean_meta_value(value: object) -> str:
    text = str(value)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()) or "."


def _parse_first_line(first: str, path: Path) -> str:
    parts = first.strip().split()
    if len(parts) != 3 or parts[0] != "##" or parts[1].lower() != "cmeth":
        raise ValueError(
            f"{path}: unsupported CMETH first line {first!r}; expected '## cmeth {CMETH_VERSION}'"
        )
    version = parts[2]
    if version != CMETH_VERSION:
        raise ValueError(
            f"{path}: unsupported CMETH version {version!r}; expected {CMETH_VERSION!r}"
        )
    return version


def _read_header_and_body(path: Path) -> tuple[list[str], str]:
    path = Path(path)
    header: list[str] = []
    body_lines: list[str] = []
    with _open_text(path) as fh:
        first = fh.readline().rstrip("\n")
        if not first:
            raise ValueError(f"{path}: empty CMETH file")
        _parse_first_line(first, path)
        header.append(first)
        for line in fh:
            if line.startswith("##"):
                header.append(line.rstrip("\n"))
            else:
                body_lines.append(line)
                break
        body_lines.extend(fh.readlines())
    return header, "".join(body_lines)


def parse_header(path: Path) -> tuple[dict[str, str], list[str]]:
    header, _ = _read_header_and_body(Path(path))
    meta: dict[str, str] = {"version": CMETH_VERSION}
    target_bed: list[str] = []
    in_target_bed = False
    for line in header[1:]:
        if line.startswith("## ["):
            continue
        if line == "## target_bed_begin":
            in_target_bed = True
            continue
        if line == "## target_bed_end":
            in_target_bed = False
            continue
        if in_target_bed:
            if line.startswith("## target_bed:"):
                target_bed.append(line.partition(":")[2].lstrip())
            continue
        if line.startswith("##") and ":" in line:
            key, _, val = line[2:].partition(":")
            meta[key.strip()] = val.strip()
    return meta, target_bed


def parse_header_meta(path: Path) -> dict[str, str]:
    meta, target_bed = parse_header(Path(path))
    meta.setdefault("target_bed_count", str(len(target_bed)))
    for key in REQUIRED_HEADER_KEYS:
        if key not in meta:
            raise ValueError(
                f"{path}: missing required CMETH header key {key!r}"
            )
    if meta.get("kind") != "reference":
        raise ValueError(f"{path}: CMETH kind must be 'reference'")
    if meta.get("coordinate") != "bed0":
        raise ValueError(f"{path}: CMETH coordinate must be 'bed0'")
    return meta


def read_target_bed(path: Path) -> pd.DataFrame:
    _meta, target_bed = parse_header(Path(path))
    if not target_bed:
        return pd.DataFrame(columns=["chrom", "start", "end", "name"])
    text = "\n".join(target_bed) + "\n"
    return pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name"],
        usecols=[0, 1, 2, 3],
    )


def _body_to_dataframe(body: str, path: Path) -> pd.DataFrame:
    if not body.strip():
        return pd.DataFrame(columns=CMETH_COLUMNS)
    lines = body.splitlines()
    if not lines:
        return pd.DataFrame(columns=CMETH_COLUMNS)
    header = lines[0].rstrip("\n")
    if header.startswith("#"):
        header = header[1:]
    if not header.startswith("chrom\t") and header != "chrom":
        raise ValueError(
            f"{path}: CMETH table header must start with '#chrom\tstart\tend'"
        )
    tsv = "\n".join([header, *lines[1:]]) + "\n"
    return pd.read_csv(
        io.StringIO(tsv), sep="\t", na_values=["."], keep_default_na=True
    )


@dataclass
class CMethFile:
    """In-memory CMETH reference."""

    meta: dict[str, str]
    df: pd.DataFrame
    target_bed: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        return CMETH_VERSION

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind", ""))

    def validate(self) -> None:
        if self.kind != "reference":
            raise ValueError("CMETH only supports kind='reference'")
        if self.meta.get("coordinate") != "bed0":
            raise ValueError("CMETH requires coordinate='bed0'")
        if len(self.target_bed) == 0:
            raise ValueError("CMETH requires an embedded target_bed block")
        self.meta["target_bed_count"] = str(len(self.target_bed))
        for key in REQUIRED_HEADER_KEYS:
            if key not in self.meta:
                raise ValueError(f"missing required CMETH header key: {key!r}")
        missing = [col for col in CMETH_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"missing CMETH columns: {missing}")
        if list(self.df.columns[:3]) != ["chrom", "start", "end"]:
            raise ValueError("CMETH columns must start with chrom,start,end")
        if len(self.df) == 0:
            return
        starts = pd.to_numeric(self.df["start"], errors="coerce")
        ends = pd.to_numeric(self.df["end"], errors="coerce")
        if starts.isna().any() or ends.isna().any():
            raise ValueError("CMETH start/end columns must be numeric")
        if (starts < 0).any() or (ends <= starts).any():
            raise ValueError("CMETH coordinates must satisfy 0 <= start < end")
        bad_haps = set(self.df["hap_key"].dropna().astype(str)) - VALID_HAP_KEYS
        if bad_haps:
            raise ValueError(
                f"unknown CMETH hap_key values: {sorted(bad_haps)}"
            )
        bad_types = (
            set(self.df["feature_type"].dropna().astype(str))
            - VALID_FEATURE_TYPES
        )
        if bad_types:
            raise ValueError(
                f"unknown CMETH feature_type values: {sorted(bad_types)}"
            )
        key_cols = [
            "chrom",
            "start",
            "end",
            "feature_type",
            "region_id",
            "hap_key",
        ]
        if all(c in self.df.columns for c in key_cols):
            dup = self.df.duplicated(subset=key_cols, keep=False)
            if dup.any():
                raise ValueError(
                    f"CMETH contains duplicated table rows: {int(dup.sum())} duplicated entries"
                )
        if "cpg" in set(self.df["feature_type"].dropna().astype(str)):
            cpg = self.df[self.df["feature_type"].astype(str) == "cpg"]
            parent = cpg["parent_region"]
            blank = parent.isna() | parent.astype(str).isin(
                ["", ".", "nan", "None"]
            )
            if blank.any():
                raise ValueError("CMETH CpG rows require parent_region")
            region_ids = set(
                self.df[self.df["feature_type"].astype(str) == "region"][
                    "region_id"
                ]
                .dropna()
                .astype(str)
            )
            missing = set(parent.dropna().astype(str)) - region_ids
            if missing:
                raise ValueError(
                    f"CMETH CpG parent_region values missing from region_id rows: {sorted(missing)[:5]}"
                )

    def normalized_df(self) -> pd.DataFrame:
        df = self.df.copy()
        for col in CMETH_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df.loc[:, CMETH_COLUMNS]
        if len(df):
            df["chrom"] = df["chrom"].astype(str)
            df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
            df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)
            df = df.sort_values(
                [
                    "chrom",
                    "start",
                    "end",
                    "feature_type",
                    "region_id",
                    "hap_key",
                ],
                kind="mergesort",
            ).reset_index(drop=True)
        return df

    def write(self, path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self.normalized_df()
        self.validate()
        if path.suffix == ".gz":
            with tempfile.NamedTemporaryFile(
                "w", delete=False, suffix=".cmeth", dir=path.parent
            ) as tmp:
                tmp_path = Path(tmp.name)
                self._write_plain_handle(tmp)
            try:
                pysam.tabix_compress(str(tmp_path), str(path), force=True)
                pysam.tabix_index(
                    str(path),
                    seq_col=0,
                    start_col=1,
                    end_col=2,
                    meta_char="#",
                    zerobased=True,
                    force=True,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            return path
        with path.open("w", newline="") as fh:
            self._write_plain_handle(fh)
        return path

    def _write_plain_handle(self, fh) -> None:
        meta = {
            str(k): _clean_meta_value(v)
            for k, v in self.meta.items()
            if str(k) != "version"
        }
        meta["target_bed_count"] = str(len(self.target_bed))
        fh.write(f"{_MAGIC} {CMETH_VERSION}\n")
        written: set[str] = set()
        for group, keys in HEADER_GROUPS:
            fh.write(f"## [{group}]\n")
            for key in keys:
                if key in meta:
                    fh.write(f"## {key}: {meta[key]}\n")
                    written.add(key)
            if group == "target":
                fh.write("## target_bed_begin\n")
                for line in self.target_bed:
                    fh.write(f"## target_bed: {line.rstrip()}\n")
                fh.write("## target_bed_end\n")
        extra = [k for k in meta if k not in written]
        if extra:
            fh.write("## [extra]\n")
            for key in extra:
                fh.write(f"## {key}: {meta[key]}\n")
        fh.write("#" + "\t".join(CMETH_COLUMNS) + "\n")
        if len(self.df):
            self.df.to_csv(fh, sep="\t", index=False, header=False, na_rep=".")

    @classmethod
    def build_reference(
        cls,
        rows: pd.DataFrame,
        *,
        meta: dict[str, object],
        target_bed: list[str],
    ) -> "CMethFile":
        m = {str(k): _clean_meta_value(v) for k, v in dict(meta).items()}
        m.setdefault("kind", "reference")
        m.setdefault("coordinate", "bed0")
        m.setdefault("created", _today())
        m.setdefault("description", ".")
        m.setdefault("target_name", "unknown")
        m.setdefault("target_count", str(len(target_bed)))
        m.setdefault("target_bed_columns", "chrom,start,end,name")
        m.setdefault("target_bed_count", str(len(target_bed)))
        if "source_sample_count" not in m:
            if "n_ref" in rows.columns and len(rows):
                m["source_sample_count"] = str(
                    pd.to_numeric(rows["n_ref"], errors="coerce").max()
                )
            else:
                m["source_sample_count"] = "unknown"
        m.setdefault("source_file_count", "unknown")
        m.setdefault("cn_norm", "per-sample-median")
        m.setdefault("mvalue_eps", "0.001")
        m.setdefault("meth_hist_bins", _format_bins(METH_HIST_BINS))
        m.setdefault("mval_hist_bins", _format_bins(MVAL_HIST_BINS))
        m.setdefault("cn_log2_hist_bins", _format_bins(CN_LOG2_HIST_BINS))
        obj = cls(meta=m, df=rows.copy(), target_bed=list(target_bed))
        obj.df = obj.normalized_df()
        obj.validate()
        return obj

    @classmethod
    def read(cls, path: Path) -> "CMethFile":
        path = Path(path)
        meta, target_bed = parse_header(path)
        meta.setdefault("target_bed_count", str(len(target_bed)))
        _, body = _read_header_and_body(path)
        df = _body_to_dataframe(body, path)
        obj = cls(meta=meta, df=df, target_bed=target_bed)
        obj.validate()
        return obj


def read_cmeth_region(
    path: Path, chrom: str, start: int, end: int
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix != ".gz":
        raise ValueError("interval queries require a bgzipped .cmeth.gz file")
    if not Path(str(path) + ".tbi").exists():
        raise FileNotFoundError(f"missing tabix index: {path}.tbi")
    rows: list[str] = []
    aliases = (
        [chrom, chrom[3:]]
        if chrom.startswith("chr")
        else [chrom, f"chr{chrom}"]
    )
    with pysam.TabixFile(str(path)) as tbx:
        for c in aliases:
            try:
                rows = list(tbx.fetch(c, int(start), int(end)))
            except (ValueError, KeyError, OSError):
                continue
            if rows:
                break
    if not rows:
        return pd.DataFrame(columns=CMETH_COLUMNS)
    tsv = "\t".join(CMETH_COLUMNS) + "\n" + "\n".join(rows) + "\n"
    df = pd.read_csv(
        io.StringIO(tsv), sep="\t", na_values=["."], keep_default_na=True
    )
    return df.loc[:, list(CMETH_COLUMNS)]
