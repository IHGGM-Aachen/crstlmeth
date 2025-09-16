<p align="center">
  <img src="crstlmeth/web/assets/logo.svg" alt="crstlmeth logo" height="200">
</p>

<h1 align="center">crstlmeth</h1>

<p align="center">
  <a href="https://pypi.org/project/crstlmeth/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/crstlmeth.svg?logo=pypi&label=PyPI">
  </a>
  <a href="https://pypi.org/project/crstlmeth/">
    <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/crstlmeth.svg">
  </a>
  <a href="https://github.com/ihggm-aachen/crstlmeth/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/ihggm-aachen/crstlmeth.svg">
  </a>
  <img alt="Code style: Black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  <img alt="Linter: Ruff" src="https://img.shields.io/badge/linter-ruff-0a7aca.svg">
</p>

`crstlmeth` [*Clinical and ReSearch Tool for anaLysis of METHylation data*]

modular toolkit for analyzing and visualizing **bedmethyl** data  
supports haplotype-resolved MLPA analysis, cohort references, plotting and inspection.

---

features
--------

- cli and streamlit web-ui
- methylation + copy-number analysis from bgzipped **.bedmethyl.gz**
- **.cmeth** reference builder for cohort-based deviation plots
- built-in mlpa kits + packaged aggregated/full references (and custom BED support)
- clean logs and reproducible cli

install
-------

requires:

- python ≥3.12
- `tabix`-indexed **.bedmethyl.gz** files

**from pypi**

```bash
pip install crstlmeth
````

**from source (dev)**

```bash
git clone https://github.com/ihggm-aachen/crstlmeth
cd crstlmeth
pip install -e .
```

## usage

### launch the gui

```bash
crstlmeth web
```

launches the multi-page streamlit app on port `8501`.

### run via cli

```bash
crstlmeth analyze \
  --kit ME030 \
  --target data/sample_1.bedmethyl.gz \
  --ref data/refA_1.bedmethyl.gz data/refA_2.bedmethyl.gz
```

see all subcommands:

```bash
crstlmeth --help
```

## input expectations

**.bedmethyl.gz** inputs must:

* be **bgzipped** and **tabix-indexed** (`.tbi` alongside)
* follow one of:

  * `SAMPLE_1.bedmethyl.gz`, `SAMPLE_2.bedmethyl.gz`
  * `SAMPLE_ungrouped.bedmethyl.gz` (optional pooled)

regions:

* use built-in MLPA kits (e.g. `ME030`, `ME032`, `ME034`, `MLPA_all`)
* or provide a custom BED with at least 4 columns: `chrom  start  end  name`

## license

MIT – see `LICENSE`
