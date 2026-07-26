"""
crstlmeth/cli/calculate/devi_calc_cmd.py

CLI to compute per-region z-score deviations for a target sample
against a cohort of reference samples using methylation levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import click
import pandas as pd

from crstlmeth.core.discovery import scan_bedmethyl
from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.methylation import Methylation
from crstlmeth.core.regions import load_intervals


def _resolve_by_id(data_dir: Path, ids: List[str]) -> List[Path]:
    """
    resolve sample IDs to bedmethyl files using scan_bedmethyl().
    raises click.BadParameter if any ids are missing.
    """
    found = scan_bedmethyl(data_dir)
    missing = [sid for sid in ids if sid not in found]
    if missing:
        raise click.BadParameter(
            f"sample(s) not found in {data_dir}: {', '.join(missing)}"
        )
    out: list[Path] = []
    for sid in ids:
        out.extend(found[sid].values())
    return out


@click.command(
    name="deviation",
    help="compute per-region z-score deviation of target vs reference cohort",
)
@click.option(
    "--kit",
    required=True,
    help="mlpa kit name or custom bed file with *_meth.bed layout",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="folder with <sample>_(1|2|ungrouped).bedmethyl.gz; if set, "
    "--reference/--target are interpreted as sample ids",
)
@click.option(
    "--reference",
    "ref_args",
    multiple=True,
    required=True,
    help="reference bedmethyl files or sample ids",
)
@click.option(
    "--target",
    "tgt_arg",
    required=True,
    help="target bedmethyl file or sample id",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default="deviation_table.tsv",
    show_default=True,
    help="output tsv with z-scores and flags",
)
@click.option(
    "--yes/--ask",
    default=False,
    help="skip confirmation when using --data-dir discovery mode",
)
def deviation(
    kit: str,
    data_dir: Path | None,
    ref_args: tuple[str, ...],
    tgt_arg: str,
    out: Path,
    yes: bool,
) -> None:
    """
    entry point for calculating per-region z-scores from methylation levels.
    """
    if data_dir:
        ref_paths = _resolve_by_id(data_dir, list(ref_args))
        tgt_paths = _resolve_by_id(data_dir, [tgt_arg])
        if not yes:
            click.echo("target :", err=True)
            click.echo(f"    {tgt_paths[0]}", err=True)
            click.echo("references :")
            for p in ref_paths:
                click.echo(f"    {p}", err=True)
            if not click.confirm("continue?", default=True):
                click.echo("aborted.")
                return
    else:
        ref_paths = [Path(p) for p in ref_args]
        tgt_paths = [Path(tgt_arg)]

    log = get_logger_from_cli(click.get_current_context())
    log_event(
        log,
        event="deviation_start",
        details={
            "kit": kit,
            "references": ";".join(str(p) for p in ref_paths),
            "target": str(tgt_paths[0]),
        },
    )

    try:
        intervals, regions = load_intervals(kit)
        ref_lv = Methylation.get_levels([str(p) for p in ref_paths], intervals)
        tgt_lv = Methylation.get_levels([str(tgt_paths[0])], intervals)

        mean_ref = ref_lv.mean(axis=0)
        std_ref = ref_lv.std(axis=0)
        z_scores = (tgt_lv[0] - mean_ref) / std_ref

        table = pd.DataFrame(
            dict(
                region=regions,
                mean_ref=mean_ref,
                target=tgt_lv[0],
                z_score=z_scores,
            )
        )
        table.to_csv(out, sep="\t", index=False)
        click.echo(f"deviation table written to {out.resolve()}")

        log_event(log, event="deviation_done", details={"out": str(out)})
    except Exception as exc:  # noqa: BLE001
        log_event(log, event="deviation_error", details={"error": str(exc)})
        raise click.ClickException(str(exc)) from exc
