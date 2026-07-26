# Toy crstlmeth dataset

Small deterministic bedMethyl source data for unit and CLI integration tests.
The files in `samples/` are intentionally plain `.bedmethyl` text files. The
pytest fixture in `tests/helpers/toy_crstlmeth.py` creates temporary bgzipped
and tabix-indexed `.bedmethyl.gz` files during each test run using `pysam`.

Regions:

- `TOY:ICR-balanced`: normal imprinting-like hap1-high/hap2-low pattern.
- `TOY:LOM-case`: case sample shows loss of methylation on both hap tracks.
- `TOY:CN-gain`: case sample has increased coverage.
- `TOY:CN-loss`: case sample has decreased coverage.

Filename styles intentionally mix `_`, `.`, and `-` separators to test sample
collector/name parsing.
