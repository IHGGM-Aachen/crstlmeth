"""
crstlmeth/cli/plot/__init__.py

cli group for re-drawing figures from previously generated inputs
"""

from __future__ import annotations

import click

from .cn_plot_cmd import copynumber
from .cpg_profile_cmd import cpg_profile_cmd
from .meth_plot_cmd import methylation


@click.group(help="redraw figures from existing inputs (bedmethyl, cmeth ...)")
def plot() -> None:
    """
    entry point for figure re-draw commands
    """
    pass


plot.add_command(copynumber)
plot.add_command(methylation)
plot.add_command(cpg_profile_cmd)
