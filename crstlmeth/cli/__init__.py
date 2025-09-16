"""
crstlmeth/cli/__init__.py

root dispatcher and cli entry point
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import click

from crstlmeth.core.logging import get_logger_from_cli, log_event

from .calculate import calculate
from .plot import plot
from .reference import reference

# ascii banner shown for `crstlmeth` without args
_BANNER = dedent(
    r"""
     ██████ ██████  ███████ ████████ ██      ███    ███ ████████ ██   ██
    ██      ██   ██ ██         ██    ██      ████  ████    ██    ██   ██
    ██      ██████  ███████    ██    ██      ██ ████ ██    ██    ███████
    ██      ██   ██      ██    ██    ██      ██  ██  ██    ██    ██   ██
     ██████ ██   ██ ███████    ██    ███████ ██      ██    ██    ██   ██
"""
)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    help=dedent(
        """
        crstlmeth - clinical and research tool for analysis of methylation data

        every sub-command appends a tsv line to a shared log (see --log-file).
        """
    ),
    epilog="",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path, dir_okay=True, writable=True),
    default="crstlmeth.log.tsv",
    show_default=True,
    help="global tsv log file or directory; all sub-commands append here",
)
@click.pass_context
def cli(ctx: click.Context, log_file: Path) -> None:
    """
    root cli dispatcher

    stores the log file path in `ctx.obj` so nested commands can access it
    via `click.get_current_context().obj["log_file"]`
    """
    ctx.ensure_object(dict)
    ctx.obj["log_file"] = log_file

    if ctx.invoked_subcommand is None:
        click.echo(_BANNER)
        click.echo(cli.get_help(ctx))


# register commands
cli.add_command(reference)
cli.add_command(calculate)
cli.add_command(plot)


@cli.command("web", help="launch the streamlit gui")
@click.option(
    "--port",
    type=int,
    default=8501,
    show_default=True,
    help="port for the streamlit server",
)
@click.pass_context
def web_cmd(ctx: click.Context, port: int) -> None:
    """
    launch streamlit gui

    the selected log file path is exported as CRSTLMETH_LOGFILE so
    the web ui can append to the same log
    """
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError as exc:
        click.echo(
            "streamlit is not installed - run  pip install streamlit  "
            "or use the provided docker image",
            err=True,
        )
        raise SystemExit(1) from exc

    app_py = Path(__file__).parent.parent / "web" / "app.py"
    log_path = Path(ctx.obj["log_file"]).resolve()

    click.echo(f"[crstlmeth] starting streamlit at :{port}  (log → {log_path})")
    log = get_logger_from_cli(ctx)
    log_event(
        log,
        cmd="web",
        event="web_start",
        params={"port": port},
        message=str(app_py),
    )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
    ]
    env = dict(**os.environ, CRSTLMETH_LOGFILE=str(log_path))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:  # pragma: no cover
    """
    allow `python -m crstlmeth.cli` for direct execution
    """
    cli(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
