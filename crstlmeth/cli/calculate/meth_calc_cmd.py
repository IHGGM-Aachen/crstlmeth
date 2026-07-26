"""
crstlmeth/cli/calculate/meth_calc_cmd.py

CLI for a single-interval methylation calculator using a bedmethyl file:
returns Nmod, Nvalid_cov, and percentage methylation.
"""

from __future__ import annotations

from pathlib import Path

import click

from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.parsers import query_bedmethyl


@click.command(
    name="methylation",
    help="return Nmod / Nvalid_cov and percent for one interval",
)
@click.argument(
    "bedmethyl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("chrom")
@click.argument("start", type=int)
@click.argument("end", type=int)
def methylation(bedmethyl: Path, chrom: str, start: int, end: int) -> None:
    """
    cli entry point for computing methylation percentage in a given interval.
    """
    if end <= start:
        raise click.BadParameter("end must be greater than start")

    log = get_logger_from_cli(click.get_current_context())
    log_event(
        log,
        event="methylation_start",
        details={
            "file": str(bedmethyl),
            "chrom": chrom,
            "start": start,
            "end": end,
        },
    )

    try:
        df = query_bedmethyl(str(bedmethyl), chrom, start, end)
        n_mod = int(df["Nmod"].sum())
        n_valid = int(df["Nvalid_cov"].sum())
        pct = 100 * n_mod / n_valid if n_valid else 0.0

        click.echo(
            f"nmod={n_mod:,}  nvalid={n_valid:,}  " f"methylation={pct:0.2f}%"
        )
        log_event(
            log,
            event="methylation_done",
            details={"nmod": n_mod, "nvalid": n_valid, "pct": f"{pct:.2f}"},
        )
    except Exception as exc:  # noqa: BLE001
        log_event(log, event="methylation_error", details={"error": str(exc)})
        raise click.ClickException(str(exc)) from exc
