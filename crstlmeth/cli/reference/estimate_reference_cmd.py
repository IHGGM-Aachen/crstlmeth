"""
Estimate CMETH row count and approximate size before building large references.
"""

from __future__ import annotations

from pathlib import Path

import click

from crstlmeth.core.cmeth import CMETH_COLUMNS
from crstlmeth.core.regions import load_intervals


@click.command(
    name="estimate",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Estimate row count and rough storage needs for a target-BED CMETH reference.",
)
@click.option("--kit", required=True, help="MLPA kit name or path to BED file.")
@click.option(
    "--include-cpgs/--no-include-cpgs", default=True, show_default=True
)
@click.option(
    "--observed-cpgs",
    type=int,
    default=None,
    help="Known total number of observed CpG/locus positions. Overrides --mean-cpgs-per-region.",
)
@click.option(
    "--mean-cpgs-per-region",
    type=float,
    default=50.0,
    show_default=True,
    help="Fallback estimate when total observed CpGs is unknown.",
)
@click.option(
    "--hap-keys",
    type=int,
    default=4,
    show_default=True,
    help="Number of hap/reference tracks stored per feature. CMETH default is 4.",
)
def estimate(
    kit: str | Path,
    include_cpgs: bool,
    observed_cpgs: int | None,
    mean_cpgs_per_region: float,
    hap_keys: int,
) -> None:
    intervals, _names = load_intervals(kit)
    n_regions = len(intervals)
    if n_regions == 0:
        raise click.ClickException(f"No intervals found for {kit!r}")

    region_rows = n_regions * int(hap_keys)
    if include_cpgs:
        n_cpgs = (
            int(observed_cpgs)
            if observed_cpgs is not None
            else int(round(n_regions * float(mean_cpgs_per_region)))
        )
        cpg_rows = n_cpgs * int(hap_keys)
    else:
        n_cpgs = 0
        cpg_rows = 0
    total_rows = region_rows + cpg_rows

    # Broad engineering estimate, not a promise: wide TSV with many numeric stats.
    approx_uncompressed_mb = (
        total_rows * len(CMETH_COLUMNS) * 12 / (1024 * 1024)
    )
    approx_gzip_mb_low = approx_uncompressed_mb * 0.15
    approx_gzip_mb_high = approx_uncompressed_mb * 0.35

    click.echo("CMETH reference size estimate")
    click.echo(f"  target intervals: {n_regions:,}")
    click.echo(f"  estimated CpG/locus positions: {n_cpgs:,}")
    click.echo(f"  hap/reference tracks per feature: {hap_keys}")
    click.echo(f"  region rows: {region_rows:,}")
    click.echo(f"  CpG rows: {cpg_rows:,}")
    click.echo(f"  total rows: {total_rows:,}")
    click.echo(f"  columns: {len(CMETH_COLUMNS):,}")
    click.echo(f"  rough uncompressed TSV: {approx_uncompressed_mb:,.1f} MB")
    click.echo(
        f"  rough .gz range: {approx_gzip_mb_low:,.1f} - {approx_gzip_mb_high:,.1f} MB"
    )
    click.echo(
        "  note: real size depends on number formatting, sparsity, and compression."
    )
