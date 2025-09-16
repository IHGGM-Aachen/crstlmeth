"""
crstlmeth/cli/calculate/__init__.py

grouped entry point for ad-hoc calculators:
  - methylation: compute % methylation for one interval
  - deviation:   z-score scan across multiple regions
"""

from __future__ import annotations

import click

from .devi_calc_cmd import deviation
from .meth_calc_cmd import methylation


@click.group(
    name="calculate",
    help="convenience calculators for ad-hoc inspection",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def calculate() -> None:
    """group dispatcher for simple methylation/deviation tools"""
    pass


calculate.add_command(methylation)
calculate.add_command(deviation)
