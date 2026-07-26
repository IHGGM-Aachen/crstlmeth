<p align="center">
  <img src="crstlmeth/web/assets/logo.svg" alt="crstlmeth logo" height="200">
</p>

<h1 align="center">crstlmeth</h1>

<p align="center">
  <a href="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/checks.yaml">
    <img alt="Checks" src="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/checks.yaml/badge.svg">
  </a>
  <a href="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/docker.yaml">
    <img alt="Docker" src="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/docker.yaml/badge.svg">
  </a>
  <a href="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/lint.yaml">
    <img alt="Lint" src="https://github.com/IHGGM-Aachen/crstlmeth/actions/workflows/lint.yaml/badge.svg">
  </a>
  <a href="https://pypi.org/project/crstlmeth/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/crstlmeth.svg?logo=pypi&label=PyPI">
  </a>
  <a href="https://pypi.org/project/crstlmeth/">
    <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/crstlmeth.svg">
  </a>
  <a href="https://github.com/IHGGM-Aachen/crstlmeth/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/IHGGM-Aachen/crstlmeth.svg">
  </a>
  <a href="https://github.com/IHGGM-Aachen/crstlmeth/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/IHGGM-Aachen/crstlmeth?style=social">
  </a>
</p>

<p align="center">
  <b>Clinical and ReSearch Tool for anaLysis of METHylation data</b>
</p>

`crstlmeth` is a modular toolkit for analyzing, visualizing, and inspecting
tabix-indexed `bedMethyl` data.

It supports haplotype-resolved methylation analysis, copy-number visualization,
cohort reference creation, CpG-level methylation profiles, and an interactive
Streamlit web interface.



## Features

- Command-line interface and Streamlit web UI
- Methylation and copy-number analysis from bgzipped `.bedmethyl.gz` files
- Haplotype-aware sample handling for hap1, hap2, and ungrouped tracks
- CMETH cohort references with region-level and CpG-level summaries
- Interactive CpG profile plots with genomic coordinates and CpG tables
- Support for built-in kits and custom BED region files
- Reproducible CLI workflows and structured logs



## Installation

Requirements:

- Python >= 3.12
- `tabix`
- bgzipped and tabix-indexed `.bedmethyl.gz` input files

### From PyPI

```bash
pip install crstlmeth
````

### From source

```bash
git clone https://github.com/IHGGM-Aachen/crstlmeth.git
cd crstlmeth
pip install -e .
```

### Development install

```bash
git clone https://github.com/IHGGM-Aachen/crstlmeth.git
cd crstlmeth
pip install -e ".[all]"
```



## Usage

### Launch the web interface

```bash
crstlmeth web
```

This starts the multi-page Streamlit app on port `8501`.

### Show available commands

```bash
crstlmeth --help
crstlmeth reference --help
crstlmeth plot --help
```

### Create a CMETH reference

```bash
crstlmeth reference create \
  --kit regions.bed \
  --include-cpgs \
  --description "Example cohort reference" \
  -o reference.cmeth.gz \
  CTRL001_1.bedmethyl.gz \
  CTRL001_2.bedmethyl.gz \
  CTRL001_ungrouped.bedmethyl.gz \
  CTRL002_1.bedmethyl.gz \
  CTRL002_2.bedmethyl.gz \
  CTRL002_ungrouped.bedmethyl.gz
```

### Validate a CMETH reference

```bash
crstlmeth reference validate reference.cmeth.gz
```

### Plot a CpG methylation profile

```bash
crstlmeth plot cpg-profile \
  --cmeth reference.cmeth.gz \
  --region "SNURF:TSS-DMR" \
  --sample-track both_haps \
  --x-mode index \
  --export-cpg-table snurf_cpg_profile.tsv \
  --out-html snurf_cpg_profile.html \
  --out snurf_cpg_profile.png \
  SAMPLE_1.bedmethyl.gz \
  SAMPLE_2.bedmethyl.gz \
  SAMPLE_ungrouped.bedmethyl.gz
```

### Plot methylation

```bash
crstlmeth plot methylation \
  --cmeth reference.cmeth.gz \
  --kit regions.bed \
  --out methylation.png \
  SAMPLE_ungrouped.bedmethyl.gz
```

### Plot copy number

```bash
crstlmeth plot copynumber \
  --cmeth reference.cmeth.gz \
  --kit regions.bed \
  --out copy_number.png \
  SAMPLE_1.bedmethyl.gz \
  SAMPLE_2.bedmethyl.gz \
  SAMPLE_ungrouped.bedmethyl.gz
```



## Input expectations

Input files must be bgzipped and tabix-indexed:

```text
sample.bedmethyl.gz
sample.bedmethyl.gz.tbi
```

Supported sample role suffixes are:

```text
SAMPLE_1.bedmethyl.gz
SAMPLE_2.bedmethyl.gz
SAMPLE_ungrouped.bedmethyl.gz

SAMPLE.1.bedmethyl.gz
SAMPLE.2.bedmethyl.gz
SAMPLE.ungrouped.bedmethyl.gz

SAMPLE-1.bedmethyl.gz
SAMPLE-2.bedmethyl.gz
SAMPLE-ungrouped.bedmethyl.gz
```

The web sample collector detects hap1, hap2, ungrouped, and matching `.tbi`
files before running plots.



## Region input

Regions can be provided as:

* built-in kit names, for example `ME030`, `ME032`, `ME034`, or `MLPA_all`
* custom BED files

Custom BED files should contain at least four columns:

```text
chrom    start    end    name
```

Coordinates are expected to be BED-style zero-based half-open intervals.



## CMETH references

CMETH references store cohort-level summary statistics.

A CMETH file can contain two feature levels:

```text
feature_type = region
feature_type = cpg
```

Region rows store DMR-level aggregate summaries.

CpG rows store observed CpG or methylation-locus summaries inside each parent
region. These rows are used for CpG-resolution profile plots.

Small clinical or demo CMETH references may be packaged under:

```text
crstlmeth/refs/
```

Large genome-wide CpG references should be generated or downloaded as external
artifacts instead of being committed into the Python package.



## Development checks

Run the local test suite:

```bash
python -m py_compile $(find crstlmeth tests -name "*.py")
pytest -q
```

Build the package:

```bash
python -m build
twine check dist/*
```

Build the Docker image:

```bash
docker build -t crstlmeth:dev .
docker run --rm crstlmeth:dev crstlmeth --help
```



## License

MIT. See `LICENSE`.
