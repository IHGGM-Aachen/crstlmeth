"""
crstlmeth/cli/reference/view_reference_cmd.py

CLI to display metadata and a short preview of a *.cmeth* reference file
"""

from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent

import click

from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.references import read_cmeth


@click.command(
    name="view",
    help=dedent(
        """
        show header and preview of a *.cmeth* file

        future versions may add summary statistics or plots
        """
    ),
)
@click.argument(
    "cmeth_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.pass_context
def view(ctx: click.Context, cmeth_file: Path) -> None:
    """
    cli command to inspect the metadata and preview of a *.cmeth* file
    """
    logger = get_logger_from_cli(ctx)
    t0 = time.perf_counter()

    try:
        df, header = read_cmeth(cmeth_file, logger=logger)

        click.echo("--- header ---")
        for k, v in header.items():
            click.echo(f"{k:>12} : {v}")
        click.echo("\n--- preview ---")
        click.echo(df.head(10).to_string(index=False))

        log_event(
            logger,
            event="reference-view",
            cmd="reference.view",
            params={
                "file": str(cmeth_file),
                "mode": header.get("mode", "?"),
                "version": header.get("version", "?"),
                "n_rows": int(len(df)),
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
