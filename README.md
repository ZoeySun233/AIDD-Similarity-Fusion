# BACE-1 molecular model fusion: a reproducible negative result

This repository is a curated public-facing extract of an exploratory AI drug-discovery project. It compares a Morgan-fingerprint random forest (RF), a frozen ChemBERTa encoder with an MLP head, calibrated fixed fusion (C1), and a similarity-aware dynamic gate (D2) for BACE-1 activity classification.

**Main finding:** D2 did not show a reliable advantage over C1. The practical comparator remains the simpler calibrated fixed fusion. This is not a claim that all dynamic fusion methods fail, nor evidence of prospective screening utility.

## Results at a glance

Average precision (AP; `average_precision_score`) is the primary ranking metric. Lower Brier score indicates better probability accuracy.

| Evaluation | RF AP | ChemBERTa AP | C1 AP | D2 AP | D2 − C1 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repeated nested scaffold CV (2 × 5 outer folds) | 0.8465 | 0.8031 | 0.8474 | 0.8473 | −0.000085 |
| Primary external assay, 158 molecules | 0.8394 | 0.7890 | 0.8413 | 0.8404 | −0.000912 |
| Separate older-source assay, 191 molecules | 0.7390 | 0.7915 | 0.7629 | 0.7633 | +0.000354 |

The prespecified primary external comparison used a paired scaffold-cluster bootstrap: 95% CI for D2 − C1 AP **[−0.005997, +0.000180]**. The required +0.01 AP improvement was not met. The main external cohort contained 110 actives among 158 molecules (69.6% prevalence), so its AP must be read against an already high prevalence baseline. The second assay was a separate sensitivity cohort, not pooled with the primary cohort.

The external numbers evaluate a **newly retrained and frozen full-BACE deployment model**. They are not a replay of the historical cross-validation fold checkpoints. See [Results and interpretation](docs/RESULTS.md) for the exact scope and limitations.

## What is included

- `src/`: data audit, scaffold splitting, model training, calibration, fusion, evaluation, external-cohort curation, and exploratory analyses.
- `configs/` and `configs_v2/`: recorded experiment settings.
- `tests/`: source tests; data-dependent integration tests require the datasets and generated outputs described below.
- `protocol_v2.md`, `docs/external_eligibility_protocol.md`, and `docs/external_wp3_amended_protocol.md`: protocols and freeze boundaries.
- `results_summary/`: compact aggregate tables and one calibration figure. Per-molecule predictions, raw assay extracts, and fitted weights are deliberately excluded.

The original, unfiltered research repository was snapshotted at Git commit `a88de3eb85ce91cdb36139e703bcd86a91048b4d` (`archive-2026-10-22`). This public extract changes three plotting scripts only to make a private figure-layout QA helper optional. These edits do not change model fitting or scoring, but this export is not byte-identical to that historical commit.

## Reproduction starting point

1. Create the environment with `conda env create -f environment.yml` and activate `aidd-similarity-fusion`.
2. Obtain the original BACE CSV from the [DeepChem dataset URL](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv) and save it as `data/raw/bace.csv`.
3. Run `python src/preprocessing_v2.py`, then `python src/splits_v2.py`. Additional stages and required inputs are described in [Reproduction notes](docs/REPRODUCIBILITY.md).

The source datasets and ChemBERTa weights are downloaded from their original providers; they are not bundled here. Reproducing the complete external evaluation also requires rebuilding the ChEMBL snapshot and the frozen deployment bundle. The supplied summary tables are published results, **not** a substitute for those omitted artifacts.

## Research boundaries

- The first Week 1–5 test was already viewed and is treated as historical, not a fresh test for version-2 method selection.
- The two scaffold-CV repeats reuse the same 1,513 original molecules; ten fold-repeat comparisons are not ten independent datasets.
- The external primary cohort comes from one patent/assay source. It is a focused transfer test, not a general deployment benchmark.
- Original DeepChem labels include some non-IC50 endpoints; the external IC50 threshold is justified for the original IC50-labelled subset, not every training row.
- Phase-7 expert variants and the accept/defer analysis are exploratory after earlier outcomes were viewed. They do not rescue or revise the primary D2 conclusion.

See [Data and provenance](docs/DATA.md) before reusing any labels. No software license has been selected for this extract; do not assume permission to redistribute its code or third-party data until a license and source terms have been reviewed.
