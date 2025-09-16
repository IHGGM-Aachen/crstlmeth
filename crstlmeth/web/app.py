"""
crstlmeth/web/app.py

registers the streamlit pages and hands control to the navigator.
only scripts inside ./pages are registered.
"""

from pathlib import Path

import streamlit as st

# page source directory
page_dir = Path(__file__).parent / "pages"

# page labels and titles
icon_map = {
    "00_home.py": ("home", ":material/home:"),
    "01_analyze.py": ("analyze", ":material/analytics:"),
    "02_references.py": ("reference", ":material/database:"),
    "03_calculate.py": ("calculate", ":material/calculate:"),
    "04_kits.py": ("kits", ":material/arrow_range:"),
    "90_log.py": ("log", ":material/bug_report:"),
    "99_faq.py": ("faq", ":material/help:"),
}

# build navigation list
pages = [
    st.Page(str(page_dir / file), title=title, icon=icon)
    for file, (title, icon) in icon_map.items()
]

# launch multipage navigation
st.navigation(pages).run()
