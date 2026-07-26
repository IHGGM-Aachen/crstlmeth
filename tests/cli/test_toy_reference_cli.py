import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from toy_crstlmeth import prepare_toy_dataset

from crstlmeth.cli.reference import reference


def test_reference_create_validate_view_cli_on_toy_data(tmp_path):
    toy = prepare_toy_dataset(tmp_path)
    out = tmp_path / "toy_reference.cmeth.gz"
    runner = CliRunner()

    create_result = runner.invoke(
        reference,
        [
            "create",
            "--kit",
            str(toy.kit),
            "--include-cpgs",
            "--description",
            "toy CLI reference",
            "-o",
            str(out),
            *map(str, toy.control_paths()),
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    assert out.exists()

    validate_result = runner.invoke(
        reference, ["validate", "--strict", str(out)]
    )
    assert validate_result.exit_code == 0, validate_result.output
    assert "feature_type counts" in validate_result.output

    view_result = runner.invoke(reference, ["view", str(out)])
    assert view_result.exit_code == 0, view_result.output
    assert "TOY:ICR-balanced" in view_result.output
