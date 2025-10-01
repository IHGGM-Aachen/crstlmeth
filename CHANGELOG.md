# Changelog

## [0.1.1] - 2025-10-01

### Added

* **Unified selectors**: single dropdowns for **references** and **regions** in analyze page that show `bundled · …` and `external · …` entries side-by-side.
* **Per-session output**: figures now write to a temp folder `…/crstlmeth_out/<session>/` (under `data_dir` if set, else system tmp); each plot gets a **Download** button.
* **Haplotype diagnostics**: on hap-plot failure, show a short coverage report (finite values per hap, regions with no coverage).
* **CLI visibility**: expanders for **argv** and **stdout/stderr** on pooled, hap1, hap2, and CN runs.

### Changed

* **Home**: removed “output directory”; simplified **scan folders**. Help text polish (`*.bed` wording).
* **Analyze**:

  * Defaults to data discovered on **home** page; uploads still supported (merged with discovered files).
  * Haplotype mode uses `--auto-hap-match` and validates presence of `_1`/`_2`.
* **Sidebar**: shows counts, paths, session id; no longer mutates session state and doesn't show outdir.
* **Streamlit config**: increased upload limit (≈1 GiB) to accommodate large `.bedmethyl.gz`.

### Fixed

* **Reference parsing crash** when external ref folder unset — now transparently falls back to bundled refs.
* **Sample selector reset** during selection — stabilized multiselect and session usage.
* **Haplotype discovery** edge cases (looser filename handling).


---

## [0.1.0] - 2025-09-16

* Initial release of **crstlmeth**
* CLI (`crstlmeth`) with modular subcommands
* Streamlit web UI with multi-page layout
* Methylation & copy-number analysis from bgzipped + tabix-indexed `.bedmethyl.gz`
* Cohort reference builder (`.cmeth`) for deviation plots
* Built-in MLPA kits (ME030, ME032, ME034, MLPA_all)
