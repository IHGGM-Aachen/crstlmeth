"""
Validate CMETH reference files.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import pandas as pd

from crstlmeth.core.cmeth import CMethFile
from crstlmeth.core.logging import get_logger_from_cli, log_event


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).isin(["", ".", "nan", "None"])


def _extra_validation(cm: CMethFile) -> list[str]:
    """Return non-fatal warnings after CMethFile.validate() has passed."""
    warnings: list[str] = []
    df = cm.df
    if df.empty:
        warnings.append("table is empty")
        return warnings

    if "region" not in set(df["feature_type"].dropna().astype(str)):
        warnings.append("no feature_type=region rows found")

    key_cols = ["chrom", "start", "end", "feature_type", "region_id", "hap_key"]
    dups = df.duplicated(subset=key_cols, keep=False)
    if dups.any():
        warnings.append(
            f"{int(dups.sum())} duplicated CMETH rows by {','.join(key_cols)}"
        )

    if "cpg" in set(df["feature_type"].dropna().astype(str)):
        cpg = df[df["feature_type"].astype(str) == "cpg"]
        blank_parent = _is_blank(cpg["parent_region"])
        if blank_parent.any():
            warnings.append(
                f"{int(blank_parent.sum())} CpG rows have empty parent_region"
            )
        region_ids = set(
            df[df["feature_type"].astype(str) == "region"]["region_id"]
            .dropna()
            .astype(str)
        )
        parent_ids = set(cpg["parent_region"].dropna().astype(str)) - {
            "",
            ".",
            "nan",
            "None",
        }
        missing = sorted(parent_ids - region_ids)
        if missing:
            preview = ", ".join(missing[:5])
            more = " ..." if len(missing) > 5 else ""
            warnings.append(
                f"CpG parent_region values not present as region_id: {preview}{more}"
            )

    declared = cm.meta.get("target_bed_count")
    if declared is not None:
        try:
            if int(declared) != len(cm.target_bed):
                warnings.append(
                    f"target_bed_count={declared}, but embedded target BED has {len(cm.target_bed)} rows"
                )
        except ValueError:
            warnings.append(f"target_bed_count is not an integer: {declared!r}")

    return warnings


@click.command(
    name="validate",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Validate a CMETH reference schema/header/table.",
)
@click.argument(
    "cmeth", type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.option(
    "--strict", is_flag=True, help="Treat validation warnings as errors."
)
@click.pass_context
def validate(ctx: click.Context, cmeth: Path, strict: bool) -> None:
    logger = get_logger_from_cli(ctx)
    t0 = time.perf_counter()
    try:
        cm = CMethFile.read(cmeth)
        warnings = _extra_validation(cm)
        click.echo(f"OK: {cmeth}")
        click.echo(f"version: {cm.version}")
        click.echo(f"rows: {len(cm.df)}")
        if "feature_type" in cm.df.columns and len(cm.df):
            click.echo("feature_type counts:")
            for key, val in (
                cm.df["feature_type"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            ):
                click.echo(f"  {key}: {int(val)}")
        if "hap_key" in cm.df.columns and len(cm.df):
            click.echo("hap_key counts:")
            for key, val in (
                cm.df["hap_key"].astype(str).value_counts().sort_index().items()
            ):
                click.echo(f"  {key}: {int(val)}")
        if warnings:
            click.echo("warnings:")
            for msg in warnings:
                click.echo(f"  - {msg}")
            if strict:
                raise click.ClickException(
                    "validation warnings present under --strict"
                )
        log_event(
            logger,
            event="reference-validate",
            cmd="reference.validate",
            params={
                "cmeth": str(cmeth),
                "strict": strict,
                "warnings": len(warnings),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    except click.ClickException:
        raise
    except Exception as exc:
        log_event(
            logger,
            level=40,
            event="reference-validate",
            cmd="reference.validate",
            params={"cmeth": str(cmeth), "strict": strict},
            message=str(exc),
            runtime_s=time.perf_counter() - t0,
        )
        raise click.ClickException(str(exc)) from exc
