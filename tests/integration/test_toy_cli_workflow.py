import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from toy_crstlmeth import prepare_toy_dataset

from crstlmeth.cli import cli


def test_full_cli_toy_workflow_creates_reference_and_all_plots(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    ref = outdir / "toy_reference.cmeth.gz"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "reference",
            "create",
            "--kit",
            str(toy.kit),
            "--include-cpgs",
            "--description",
            "toy integration reference",
            "-o",
            str(ref),
            *map(str, toy.control_paths()),
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["reference", "validate", "--strict", str(ref)])
    assert result.exit_code == 0, result.output

    cpg_png = outdir / "case_cpg_profile.png"
    cpg_html = outdir / "case_cpg_profile.html"
    cpg_tsv = outdir / "case_cpg_profile.tsv"
    result = runner.invoke(
        cli,
        [
            "plot",
            "cpg-profile",
            "--cmeth",
            str(ref),
            "--region",
            "TOY:LOM-case",
            "--sample-track",
            "both_haps",
            "--x-mode",
            "index",
            "--export-cpg-table",
            str(cpg_tsv),
            "--out-html",
            str(cpg_html),
            "--out",
            str(cpg_png),
            *map(str, toy.case_paths()),
        ],
    )
    assert result.exit_code == 0, result.output
    assert cpg_png.exists() and cpg_png.stat().st_size > 0
    assert cpg_html.exists() and cpg_html.stat().st_size > 0
    assert cpg_tsv.exists() and "sample_hap1_beta" in cpg_tsv.read_text()

    meth_png = outdir / "case_methylation.png"
    result = runner.invoke(
        cli,
        [
            "plot",
            "methylation",
            "--cmeth",
            str(ref),
            "--kit",
            str(toy.kit),
            "--out",
            str(meth_png),
            str(toy.samples["CASE_X01"]["ungrouped"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert meth_png.exists() and meth_png.stat().st_size > 0

    cn_png = outdir / "case_copy_number.png"
    result = runner.invoke(
        cli,
        [
            "plot",
            "copynumber",
            "--cmeth",
            str(ref),
            "--kit",
            str(toy.kit),
            "--out",
            str(cn_png),
            *map(str, toy.case_paths()),
        ],
    )
    assert result.exit_code == 0, result.output
    assert cn_png.exists() and cn_png.stat().st_size > 0
