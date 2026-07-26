"""
crstlmeth/cli/reference/create_reference_cmd.py

CLI to create a CMETH reference from bedMethyl files and a region BED/kit.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.references import create_cmeth_reference


@click.command(
    name="create",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Build a CMETH cohort reference from bedMethyl files and a kit/BED.",
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
    help="Output filename. A bare .cmeth suffix is written as .cmeth.gz.",
)
@click.option(
    "--description",
    default="",
    show_default=False,
    help="Optional description written into the CMETH header.",
)
@click.option(
    "--cn-norm",
    default="per-sample-median",
    show_default=True,
    help="Copy-number normalization strategy.",
)
@click.option(
    "--mvalue-eps",
    type=float,
    default=0.001,
    show_default=True,
    help="Epsilon used for M-value summaries: log2((beta+eps)/(1-beta+eps)).",
)
@click.option(
    "--include-cpgs/--no-include-cpgs",
    default=True,
    show_default=True,
    help="Include CpG/locus-level CMETH rows in addition to region rows.",
)
@click.option(
    "--mod-mode",
    type=click.Choice(["m", "h", "mh", "any", "custom"], case_sensitive=False),
    default="m",
    show_default=True,
    help="Modification code mode for modkit bedMethyl/bedRMod input.",
)
@click.option(
    "--mod-code",
    multiple=True,
    help="Modification code for --mod-mode custom. Can be repeated.",
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
    description: str,
    cn_norm: str,
    mvalue_eps: float,
    include_cpgs: bool,
    mod_mode: str,
    mod_code: tuple[str, ...],
    bedmethyl_paths: tuple[Path, ...],
) -> None:
    """Create a rich aggregated CMETH reference."""
    logger = get_logger_from_cli(ctx)
    t0 = time.perf_counter()
    out_path = Path(out_file).resolve()

    try:
        outp = create_cmeth_reference(
            kit=kit,
            bedmethyl_paths=list(bedmethyl_paths),
            out=out_path,
            logger=logger,
            cn_norm=cn_norm,
            mvalue_eps=float(mvalue_eps),
            description=description.strip() or None,
            include_cpgs=bool(include_cpgs),
            mod_mode=str(mod_mode).lower(),
            mod_codes=list(mod_code) if mod_code else None,
        )
        click.echo(f"wrote CMETH reference to {outp}")
        log_event(
            logger,
            event="reference-create",
            cmd="reference.create",
            params={
                "kit": str(kit),
                "out": str(outp),
                "n_files": len(bedmethyl_paths),
                "cn_norm": cn_norm,
                "mvalue_eps": mvalue_eps,
                "include_cpgs": bool(include_cpgs),
                "mod_mode": str(mod_mode).lower(),
                "mod_codes": ",".join(mod_code),
            },
            message="ok",
            runtime_s=time.perf_counter() - t0,
        )
    except Exception as exc:
        log_event(
            logger,
            level=40,
            event="reference-create",
            cmd="reference.create",
            params={
                "kit": str(kit),
                "out": str(out_path),
                "n_files": len(bedmethyl_paths),
            },
            message=str(exc),
            runtime_s=time.perf_counter() - t0,
        )
        raise
