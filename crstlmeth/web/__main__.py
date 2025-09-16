"""
crstlmeth/web/__main__.py

entry point for launching the streamlit gui.

usage:
    python -m crstlmeth.web
    or: crstlmeth web

this starts the multi-page streamlit app bundled with crstlmeth.
pages are discovered automatically from the ./pages/ directory.
"""

import subprocess
import sys
from pathlib import Path

# path to app.py (entrypoint)
here = Path(__file__).resolve().parent
app = here / "app.py"

# build the streamlit run command
cmd = [sys.executable, "-m", "streamlit", "run", str(app)]

# forward extra cli flags passed after “web …”
if len(sys.argv) > 1:
    cmd.extend(sys.argv[1:])

print("[crstlmeth] starting streamlit...")
sys.exit(subprocess.call(cmd))
