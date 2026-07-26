from click.testing import CliRunner

from crstlmeth.cli import cli


def test_reference_estimate_toy_kit_runs():
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "reference",
            "estimate",
            "--kit",
            "tests/data/toy_crstlmeth/toy_regions.bed",
            "--observed-cpgs",
            "20",
            "--hap-keys",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output

    out = result.output.lower()
    assert "cmeth reference size estimate" in out
    assert "target intervals" in out
    assert "estimated cpg" in out
    assert "total rows" in out
