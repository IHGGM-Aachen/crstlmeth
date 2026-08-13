# Changelog

## [0.1.3] - 2026-08-13

### Fixed

* **Docker web startup** no longer fails when `CRSTLMETH_LOGFILE` is already present in the container environment.
* **CLI web command** now safely overrides the log-file environment variable using a copied environment mapping.


## [0.1.2] - 2026-07-26

### Added

* **Extended CMETH reference format** with rich cohort-level region summaries.
* **CpG-level reference rows** generated from observed bedMethyl loci inside target regions.
* **CpG profile plotting** with pooled and haplotype-separated sample tracks.
* **Interactive Plotly CpG profile output** with genomic coordinates, hover labels, and exportable CpG tables.
* **Reference validation command** for checking CMETH structure and required fields.
* **Reference size estimate command** for estimating CMETH row counts and approximate output size.
* **Toy test dataset** covering sample detection, parsing, reference creation, CLI workflows, and plotting.
* **GitHub Actions workflows** for checks, package build, Docker build, linting, and release artifacts.

### Changed

* **Reference strategy** now uses one privacy-preserving aggregated CMETH format instead of separate aggregated/full modes.
* **Region summaries** are calculated as true per-sample DMR-level aggregates before cohort summarization.
* **Haplotype reference naming** now uses `pooled`, `allele_low`, `allele_high`, and `unphased`.
* **Sample-side haplotypes** remain `pooled`, `hap1`, `hap2`, and `unphased`.
* **Analyze web page** now separates CpG profile, methylation, and copy-number workflows.
* **Web sample collection** now recognizes underscore, dot, and dash role suffixes.
* **Uploaded `.tbi` files** are detected for index status but are no longer passed as analysis inputs.
* **Docker image** now installs the web dependencies and runs the Streamlit app on `0.0.0.0:8501`.

### Fixed

* **Methylation plotting with CpG-rich CMETH files** now filters to region rows where appropriate.
* **Copy-number plotting with CMETH 1.2.3** now handles `hap_key` and rich reference rows correctly.
* **Copy-number quantile handling** no longer creates duplicate renamed columns.
* **Missing pandas import** in copy-number plotting command.
* **Sample grouping helpers** restored for methylation plotting.
* **Streamlit deprecation warnings** by replacing `use_container_width` with `width`.
* **Tabix upload timestamp warning** by preserving matching `.bedmethyl.gz` and `.tbi` pair timestamps.
* **CI test failures** by committing the toy fixtures under `tests/data`.


## [0.1.1] - 2025-10-01

### Added

* **Unified selectors**: single dropdowns for **references** and **regions** in analyze page that show `bundled  -  ...` and `external  -  ...` entries side-by-side.
* **Per-session output**: figures now write to a temp folder `.../crstlmeth_out/<session>/` (under `data_dir` if set, else system tmp); each plot gets a **Download** button.
* **Haplotype diagnostics**: on hap-plot failure, show a short coverage report (finite values per hap, regions with no coverage).
* **CLI visibility**: expanders for **argv** and **stdout/stderr** on pooled, hap1, hap2, and CN runs.

### Changed

* **Home**: removed "output directory"; simplified **scan folders**. Help text polish (`*.bed` wording).
* **Analyze**:

  * Defaults to data discovered on **home** page; uploads still supported (merged with discovered files).
  * Haplotype mode uses `--auto-hap-match` and validates presence of `_1`/`_2`.
* **Sidebar**: shows counts, paths, session id; no longer mutates session state and doesn't show outdir.
* **Streamlit config**: increased upload limit (approx. 1 GiB) to accommodate large `.bedmethyl.gz`.

### Fixed

* **Reference parsing crash** when external ref folder unset - now transparently falls back to bundled refs.
* **Sample selector reset** during selection - stabilized multiselect and session usage.
* **Haplotype discovery** edge cases (looser filename handling).


## [0.1.0] - 2025-09-16

* Initial release of **crstlmeth**
* CLI (`crstlmeth`) with modular subcommands
* Streamlit web UI with multi-page layout
* Methylation & copy-number analysis from bgzipped + tabix-indexed `.bedmethyl.gz`
* Cohort reference builder (`.cmeth`) for deviation plots
* Built-in MLPA kits (ME030, ME032, ME034, MLPA_all)
