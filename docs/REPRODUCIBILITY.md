# Reproduction notes

This is a **curated source-and-summary extract**, not a turnkey rerun of the historical one-time external test. The original raw/processed molecular datasets, all per-molecule prediction files, and fitted deployment checkpoints are intentionally absent. The aggregate tables allow inspection of published numbers, but they cannot reconstruct the paired scaffold bootstrap or the original artifact hashes on their own.

## Environment and original data

Create the Conda environment with `conda env create -f environment.yml`, then activate `aidd-similarity-fusion`. The upstream BACE file can be retrieved from the URL in `docs/DATA.md` and placed unchanged at `data/raw/bace.csv`.

The primary version-2 pipeline is ordered as follows from the repository root:

```bash
python src/preprocessing_v2.py
python src/splits_v2.py
python src/train_base_models_v2.py
python src/calibrate_models_v2.py
python src/fusion_v2.py
python src/phase6_independent_inner.py
python src/evaluate_phase6.py
```

These stages create many generated outputs under `data/processed_v2/`, `data/splits_v2/`, and `results_v2/`. The ChemBERTa stage downloads a separate pinned encoder checkpoint on first use and requires materially more time and memory than RF. Integration tests under `tests/` require their corresponding generated outputs; running the entire suite immediately after cloning will fail because the datasets and artifacts are deliberately excluded.

The original protocols are retained in `protocol_v2.md` and `docs/`. The source code references the original directory layout; do not substitute external labels into training or use the already viewed historical test to choose new settings.

## External evaluation boundary

`src/curate_external_wp2.py` contains the ChEMBL extraction and eligibility logic. `src/train_external_wp3_deployment.py` creates a **new** model from original BACE training data, while `src/external_wp3_preflight.py` and `src/evaluate_external_wp3_deployment.py` implement the freeze and scoring checks. The historical one-time external result should not be rerun and represented as a second independent evaluation. A new substantive model or threshold change requires a further independent external cohort.

## Public-export differences

Three plotting scripts (`calibrate_models_v2.py`, `plot_phase6_reliability.py`, `analyze_wp4_screening.py`) treat the original private `audit_panel_alignment` figure-layout helper as optional. This only skips a layout QA sidecar when the helper is unavailable; it is **not** part of model fitting or metric calculation. Original historical hashes therefore do not apply to this public source copy.

The export excludes notebook outputs, downloaded data, original supporting archives, raw ChEMBL pages, detailed prediction files, and trained weights. If exact historical reconstruction is required, use the full private archive at commit `a88de3eb85ce91cdb36139e703bcd86a91048b4d` rather than treating this public extract as the immutable run snapshot.
