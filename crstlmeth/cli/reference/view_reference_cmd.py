# crstlmeth/cli/reference/view_reference_cmd.py

"""
CLI to inspect a CMETH reference file.

The CMETH schema is intentionally wide, so the default view is compact:
metadata summary, row counts, status counts, and a useful subset of columns.
Use --all-columns when the full raw table preview is needed.
"""

from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import click
import pandas as pd

from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.references import read_cmeth

_META_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (
        "format",
        (
            "version",
            "kind",
            "coordinate",
            "created",
        ),
    ),
    (
        "description",
        ("description",),
    ),
    (
        "target",
        (
            "target_name",
            "target_count",
            "target_bed_columns",
            "target_bed_count",
        ),
    ),
    (
        "cohort",
        (
            "source_sample_count",
            "source_file_count",
        ),
    ),
    (
        "calculation",
        (
            "cn_norm",
            "mvalue_eps",
        ),
    ),
    (
        "histograms",
        (
            "meth_hist_bins",
            "mval_hist_bins",
            "cn_log2_hist_bins",
        ),
    ),
]


_COMPACT_COLUMNS: tuple[str, ...] = (
    "chrom",
    "start",
    "end",
    "feature_type",
    "region_id",
    "hap_key",
    "n_ref",
    "n_meth",
    "n_cn",
    "meth_mean",
    "meth_sd",
    "meth_median",
    "meth_q25",
    "meth_q75",
    "meth_q05",
    "meth_q95",
    "depth_median",
    "frac_unphased_median",
    "hap_balance_median",
    "allele_gap_median",
    "cn_log2_mean",
    "cn_log2_sd",
    "cn_log2_median",
    "cn_log2_q25",
    "cn_log2_q75",
    "meth_status",
    "cn_status",
    "phasing_status",
    "row_status",
)


def _existing_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _echo_meta(meta: dict[str, str]) -> None:
    seen: set[str] = set()

    click.echo("--- metadata ---")
    for group_name, keys in _META_GROUPS:
        present = [(k, meta.get(k)) for k in keys if k in meta]
        if not present:
            continue

        click.echo(f"\n[{group_name}]")
        for key, value in present:
            seen.add(key)
            click.echo(f"{key:>22} : {value}")

    extras = [
        (k, v)
        for k, v in meta.items()
        if k not in seen and not str(k).startswith("target_bed")
    ]
    if extras:
        click.echo("\n[extra]")
        for key, value in extras:
            click.echo(f"{key:>22} : {value}")


def _echo_counts(df: pd.DataFrame) -> None:
    click.echo("\n--- table ---")
    click.echo(f"{'rows':>22} : {len(df)}")
    click.echo(f"{'columns':>22} : {len(df.columns)}")

    for col in (
        "feature_type",
        "hap_key",
        "row_status",
        "meth_status",
        "cn_status",
    ):
        if col not in df.columns:
            continue
        counts = df[col].fillna(".").astype(str).value_counts(dropna=False)
        compact = ", ".join(f"{idx}={val}" for idx, val in counts.items())
        click.echo(f"{col:>22} : {compact}")


def _format_preview(df: pd.DataFrame, rows: int, *, all_columns: bool) -> str:
    if all_columns:
        preview = df.head(rows).copy()
    else:
        cols = _existing_columns(df, _COMPACT_COLUMNS)
        preview = df.loc[:, cols].head(rows).copy()

    # Keep terminal output readable.
    preview = preview.fillna(".")

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        42,
        "display.float_format",
        lambda x: f"{x:.4g}",
    ):
        return preview.to_string(index=False)


@click.command(
    name="view",
    help=dedent("""
        Inspect a CMETH reference file.

        By default this prints a compact, readable preview. Use --all-columns
        to show the full wide schema.
        """),
)
@click.argument(
    "cmeth_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "-n",
    "--rows",
    default=12,
    show_default=True,
    type=int,
    help="Number of body rows to preview.",
)
@click.option(
    "--all-columns",
    is_flag=True,
    help="Show all columns in the preview instead of a compact subset.",
)
@click.pass_context
def view(
    ctx: click.Context,
    cmeth_file: Path,
    rows: int,
    all_columns: bool,
) -> None:
    """
    CLI command to inspect metadata and a compact preview of a CMETH file.
    """
    logger = get_logger_from_cli(ctx)
    t0 = time.perf_counter()

    try:
        df, meta = read_cmeth(cmeth_file, logger=logger)

        _echo_meta(meta)
        _echo_counts(df)

        click.echo("\n--- preview ---")
        click.echo(
            _format_preview(df, max(1, int(rows)), all_columns=all_columns)
        )

        if not all_columns:
            click.echo(
                "\nTip: use --all-columns to inspect the complete rich schema."
            )

        log_event(
            logger,
            event="reference-view",
            cmd="reference.view",
            params={
                "file": str(cmeth_file),
                "version": meta.get("version", "?"),
                "n_rows": int(len(df)),
                "n_columns": int(len(df.columns)),
                "all_columns": bool(all_columns),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        log_event(
            logger,
            level=40,  # logging.ERROR
            event="reference-view",
            cmd="reference.view",
            params={"file": str(cmeth_file)},
            message=str(exc),
            runtime_s=time.perf_counter() - t0,
        )
        raise
