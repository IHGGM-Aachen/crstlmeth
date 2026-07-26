"""
Draw a CpG-resolution methylation profile from CMETH CpG rows and one target sample.
"""

from __future__ import annotations

from pathlib import Path

import click

from crstlmeth.core.cpg_profile import (
    SAMPLE_TRACK_CHOICES,
    align_sample_tracks,
    build_cpg_profile_table,
    get_reference_cpg_rows,
    group_paths_by_sample,
    match_region,
    select_sample_tracks,
)
from crstlmeth.core.logging import get_logger_from_cli, log_event
from crstlmeth.core.references import read_cmeth
from crstlmeth.viz.cpg_profile import make_cpg_profile_plotly, plot_cpg_profile


@click.command(
    name="cpg-profile",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Draw a CpG-resolution methylation profile from CMETH CpG rows and one target sample.",
)
@click.option(
    "--cmeth",
    "cmeth_ref",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="CMETH reference with embedded CpG rows.",
)
@click.option(
    "--region",
    "region_query",
    required=True,
    help="Region id/display name to plot, for example 'SNURF:TSS-DMR'.",
)
@click.option(
    "--reference-hap",
    type=click.Choice(
        ["pooled", "allele_low", "allele_high", "unphased"],
        case_sensitive=False,
    ),
    default="pooled",
    show_default=True,
    help="Reference hap/track to draw from CMETH.",
)
@click.option(
    "--sample-track",
    type=click.Choice(list(SAMPLE_TRACK_CHOICES), case_sensitive=False),
    default="pooled",
    show_default=True,
    help="Target sample track(s). Use both_haps for direct hap1+hap2; both_alleles for low/high order-independent view.",
)
@click.option(
    "--show-alleles/--pooled-only",
    default=None,
    show_default=False,
    help="Deprecated compatibility option. Prefer --sample-track.",
)
@click.option(
    "--x-mode",
    type=click.Choice(["index", "genomic"], case_sensitive=False),
    default="index",
    show_default=True,
    help="X-axis mode.",
)
@click.option(
    "--mod-mode",
    type=click.Choice(["m", "h", "mh", "any", "custom"], case_sensitive=False),
    default="m",
    show_default=True,
    help="Modification code mode passed to the modkit bedMethyl parser.",
)
@click.option(
    "--mod-code",
    multiple=True,
    help="Modification code for --mod-mode custom. Can be repeated, e.g. --mod-code m --mod-code h.",
)
@click.option(
    "--show-outer-band/--hide-outer-band",
    default=False,
    show_default=True,
    help="Show the wider reference q05-q95 band. The q25-q75 band is always shown when available.",
)
@click.option(
    "--sample-line/--sample-points-only",
    default=False,
    show_default=True,
    help="Connect target sample CpG points with a line. Points-only is cleaner for clinical views.",
)
@click.option(
    "--genomic-track/--no-genomic-track",
    default=True,
    show_default=True,
    help="Add a compact CpG locus ruler above the PNG plot.",
)
@click.option(
    "--label-cpgs",
    type=click.Choice(
        ["none", "ticks", "highlighted", "all"], case_sensitive=False
    ),
    default="none",
    show_default=True,
    help="CpG label mode. 'highlighted' labels only CpGs selected with --highlight-index/position.",
)
@click.option(
    "--highlight-index",
    type=int,
    multiple=True,
    help="1-based CpG index to highlight. Can be repeated.",
)
@click.option(
    "--highlight-position",
    type=int,
    multiple=True,
    help="Genomic start coordinate to highlight. Can be repeated.",
)
@click.option(
    "--export-cpg-table",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional TSV with CpG coordinates, reference statistics, and sample beta values.",
)
@click.option(
    "--out-html",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional interactive Plotly HTML output with CpG coordinate hover data.",
)
@click.option(
    "--out",
    "out_png",
    type=click.Path(path_type=Path, dir_okay=False),
    default="cpg_profile.png",
    show_default=True,
    help="Destination static image file.",
)
@click.argument(
    "target",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.pass_context
def cpg_profile_cmd(
    ctx: click.Context,
    cmeth_ref: Path,
    region_query: str,
    reference_hap: str,
    sample_track: str,
    show_alleles: bool | None,
    x_mode: str,
    mod_mode: str,
    mod_code: tuple[str, ...],
    show_outer_band: bool,
    sample_line: bool,
    genomic_track: bool,
    label_cpgs: str,
    highlight_index: tuple[int, ...],
    highlight_position: tuple[int, ...],
    export_cpg_table: Path | None,
    out_html: Path | None,
    out_png: Path,
    target: tuple[Path, ...],
) -> None:
    logger = get_logger_from_cli(ctx)
    ref_df, meta = read_cmeth(cmeth_ref, logger=logger)

    if "feature_type" not in ref_df.columns:
        raise click.ClickException(
            "CMETH is missing 'feature_type'; no CpG rows available"
        )

    try:
        region_id, chrom, start, end = match_region(ref_df, region_query)
        cpg_rows = get_reference_cpg_rows(
            ref_df,
            region_id=region_id,
            reference_hap=str(reference_hap).lower(),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if cpg_rows.empty:
        raise click.ClickException(
            f"No CpG rows found for region={region_id!r} hap_key={reference_hap!r}. Rebuild the CMETH with --include-cpgs."
        )

    grouped = group_paths_by_sample(list(target))
    if len(grouped) != 1:
        raise click.ClickException(
            "cpg-profile requires target files from exactly one sample"
        )
    sample_id, sample_parts = next(iter(grouped.items()))

    try:
        all_sample_tracks = align_sample_tracks(
            cpg_rows,
            sample_parts,
            chrom,
            start,
            end,
            mod_mode=str(mod_mode).lower(),
            mod_codes=list(mod_code) if mod_code else None,
        )
        selected_tracks = select_sample_tracks(
            all_sample_tracks,
            str(sample_track).lower(),
            show_alleles,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    sample_labels = {
        "pooled": f"{sample_id} pooled",
        "hap1": f"{sample_id} hap1",
        "hap2": f"{sample_id} hap2",
        "allele_low": f"{sample_id} allele low",
        "allele_high": f"{sample_id} allele high",
        "unphased": f"{sample_id} unphased",
    }

    plot_cpg_profile(
        reference_rows=cpg_rows,
        sample_tracks=selected_tracks,
        title=f"CpG methylation profile  -  {region_id}",
        save=str(out_png),
        x_mode=str(x_mode).lower(),
        highlight_indices=list(highlight_index),
        highlight_positions=list(highlight_position),
        sample_labels=sample_labels,
        show_outer_band=bool(show_outer_band),
        sample_line=bool(sample_line),
        label_cpgs=str(label_cpgs).lower(),
        genomic_track=bool(genomic_track),
    )

    if export_cpg_table is not None:
        table = build_cpg_profile_table(cpg_rows, selected_tracks)
        Path(export_cpg_table).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(export_cpg_table, sep="\t", index=False)
        click.echo(f"wrote CpG table to {Path(export_cpg_table).resolve()}")

    if out_html is not None:
        fig = make_cpg_profile_plotly(
            reference_rows=cpg_rows,
            sample_tracks=selected_tracks,
            title=f"CpG methylation profile  -  {region_id}",
            x_mode=str(x_mode).lower(),
            sample_labels=sample_labels,
            show_outer_band=bool(show_outer_band),
            sample_line=bool(sample_line),
            genomic_track=bool(genomic_track),
        )
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        click.echo(
            f"wrote interactive CpG profile to {Path(out_html).resolve()}"
        )

    click.echo(f"wrote CpG profile to {Path(out_png).resolve()}")
    log_event(
        logger,
        event="plot_cpg_profile",
        cmd="plot cpg-profile",
        params={
            "cmeth": str(cmeth_ref),
            "region": region_id,
            "reference_hap": reference_hap,
            "sample": sample_id,
            "sample_track": sample_track,
            "out": str(out_png),
            "out_html": str(out_html) if out_html else "",
            "x_mode": x_mode,
            "mod_mode": mod_mode,
            "mod_codes": ",".join(mod_code),
            "target_name": meta.get("target_name", "."),
            "export_cpg_table": (
                str(export_cpg_table) if export_cpg_table else ""
            ),
        },
        message="ok",
    )
