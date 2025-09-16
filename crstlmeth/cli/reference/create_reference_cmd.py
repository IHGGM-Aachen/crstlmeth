"""
crstlmeth/cli/reference/create_reference_cmd.py

CLI to create a .cmeth reference file from a set of bedMethyl files
and an MLPA kit (built-in) or custom BED.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.references import (
    create_cmeth_aggregated,
    create_cmeth_full,
)


@click.command(
    name="create",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Build a .cmeth reference from bedMethyl files and a kit/BED.",
)
@click.option(
    "--kit",
    required=True,
    help="MLPA kit name (built-in) or path to custom BED file.",
)
@click.option(
    "-o",
    "--out",
    "out_file",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    required=True,
    help="Output .cmeth filename.",
)
@click.option(
    "--mode",
    type=click.Choice(
        ["aggregated", "truncated", "full"], case_sensitive=False
    ),
    default="aggregated",
    show_default=True,
    help="Reference format: 'aggregated' (quantiles; shareable) or 'full' (per-sample; plot-ready).",
)
@click.argument(
    "bedmethyl_paths",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_context
def create(
    ctx: click.Context,
    kit: str | Path,
    out_file: Path,
    bedmethyl_paths: tuple[Path, ...],
    mode: str,
) -> None:
    """
    Create a cohort reference (.cmeth) by delegating to core helpers.
    """
    logger = get_logger_from_cli(ctx)
    t0 = time.perf_counter()

    out_path = Path(out_file).resolve()
    mode_norm = mode.lower()
    if mode_norm == "truncated":
        mode_norm = "aggregated"  # accept legacy spelling

    try:
        if mode_norm == "aggregated":
            create_cmeth_aggregated(
                kit=kit,
                bedmethyl_paths=list(bedmethyl_paths),
                out=out_path,
                logger=logger,
            )
        else:
            create_cmeth_full(
                kit=kit,
                bedmethyl_paths=list(bedmethyl_paths),
                out=out_path,
                logger=logger,
            )
        click.echo(f"wrote reference (mode={mode_norm}) to {out_path}")
        log_event(
            logger,
            event="reference-create",
            cmd="reference.create",
            params={
                "mode": mode_norm,
                "kit": str(kit),
                "out": str(out_path),
                "n_files": len(bedmethyl_paths),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        log_event(
            logger,
            level=40,  # logging.ERROR
            event="reference-create",
            cmd="reference.create",
            params={
                "mode": mode_norm,
                "kit": str(kit),
                "out": str(out_path),
                "n_files": len(bedmethyl_paths),
            },
            message=str(exc),
            runtime_s=time.perf_counter() - t0,
        )
        raise
