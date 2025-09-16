"""
crstlmeth/cli/reference/__init__.py

command group: create or inspect *.cmeth* reference files
"""

from __future__ import annotations

import click

from .create_reference_cmd import create
from .view_reference_cmd import view


@click.group(help="create or inspect *.cmeth* reference files")
def reference() -> None:
    pass


reference.add_command(create)
reference.add_command(view)
