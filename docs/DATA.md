# Data and provenance

## Original BACE training data

The original CSV is provided by [DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv) and is **not redistributed** here. Place the downloaded file at `data/raw/bace.csv`. The main source fields are `mol` (SMILES), `Class` (binary label), and `pIC50` (potency field). Preserve the raw file unchanged and generate standardized copies separately.

The original BACE study's publisher-hosted [supporting dataset](https://acs.figshare.com/articles/dataset/Computational_Modeling_of_Secretase_1_BACE-1_Inhibitors_Using_Ligand_Based_Approaches/4003788) was checked during endpoint audit. In its 1,478 IC50-labelled records, the source `Class` agreed with `pIC50 >= 7`. The local DeepChem CSV had 17 disagreements with that simple threshold: eight mapped to Ki, eight to Kd, and one did not match the source sheet. Do **not** relabel those rows from model predictions or assume the threshold reconstructs every original label.

## External cohorts

The external audit used a ChEMBL_37 snapshot from the [ChEMBL API](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services), human BACE-1 target `CHEMBL4822`, biochemical IC50 assays, exact `=` relations, and nM units. Activity was defined as IC50 ≤100 nM. Standardized structures overlapping the original training set were excluded. Replicates straddling the threshold were excluded rather than repaired.

| Role | ChEMBL assay | Eligible structures | Active / inactive | Scaffold count | Source scope |
| --- | --- | ---: | ---: | ---: | --- |
| Primary | `CHEMBL5736880` | 158 | 110 / 48 | 62 | One patent/assay source |
| Sensitivity only | `CHEMBL5734173` | 191 | 135 / 56 | 101 | Separate older-source patent/assay |

The two external cohorts were never pooled. Zero observed structure/scaffold overlap does not prove universal provenance independence; the original supporting sheet lacks exact ChEMBL assay and patent identifiers for a full direct-ID audit. The external binary endpoint is comparable to the original **IC50-labelled subset**, not necessarily to every mixed-endpoint DeepChem row.

Raw ChEMBL responses, the curated molecule-level cohort, and the curation manifest are excluded from this compact public extract. The curation implementation and frozen eligibility rules are included in `src/curate_external_wp2.py`, `configs_v2/external_wp2_chembl.yaml`, and `docs/external_eligibility_protocol.md`. Check each upstream provider's current terms before redistributing its data or derived structures.

## Third-party model

The encoder is [`DeepChem/ChemBERTa-77M-MLM`](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM), pinned in the deployment configuration to revision `ed8a5374f2024ec8da53760af91a33fb8f6a15ff`. The checkpoint is not bundled. Verify its current access and license terms independently.
