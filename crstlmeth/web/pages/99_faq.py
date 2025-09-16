"""
crstlmeth.web.pages.99_faq
frequently asked questions about inputs and concepts
"""

from __future__ import annotations

import streamlit as st

from crstlmeth.web.sidebar import render_sidebar

# page setup
st.set_page_config(page_title="crstlmeth - faq", page_icon=":material/help:")
st.title("faq")
render_sidebar()

# questions & answers
st.markdown(
    """
**Q: What inputs do I need?**
You'll need a set of bgzipped & indexed *bedMethyl* files - both for the reference cohort and the target.
You'll also need a BED file defining the MLPA regions (or use one of the built-in kits).

**Q: Where does the copy-number come from?**
It's estimated from the `Nvalid_cov` field in bedMethyl, averaged over each MLPA region.

**Q: Can I create my own reference cohort?**
Yes. Go to the **references** page and select samples from your dataset to build a `.cmeth` file.

**Q: What are these `.cmeth` files?**
They store precomputed region-level methylation & coverage statistics across a reference cohort.
They're used for plotting and comparison against new targets.

**Q: How do I upload custom regions?**
Provide a BED file with at least 4 columns (chrom, start, end, name) in the sidebar under “optional folders”.

**Q: Where are the logs saved?**
All actions are logged to a TSV file whose path is defined by the environment variable `CRSTLMETH_LOGFILE`.

**Q: My BED files aren't recognized. Why?**
Make sure they are bgzipped (`.gz`) and indexed with Tabix (`.tbi`).
File names must follow the convention: `sample_(1|2|ungrouped).bedmethyl.gz`.
"""
)
