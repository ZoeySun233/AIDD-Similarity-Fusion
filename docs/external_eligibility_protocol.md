# Work Package 2: External BACE-1 eligibility protocol

**Status:** Curation protocol; no external model performance has been examined.
**Written:** 2026-10-20, after a preliminary, label-aware candidate search but before the reproducible extract and overlap audit below. Candidate counts from that search are exploratory, not a blinded eligibility decision.
**Candidate source:** ChEMBL activity API, human BACE-1 target `CHEMBL4822`, assays `CHEMBL5736880` and `CHEMBL5734173`. Keep the assays as separate cohorts.

## Intended claim and decision boundary

The intended claim is transfer of the frozen MoleculeNet/DeepChem BACE `Class` predictor to independently sourced human BACE-1 biochemical inhibition data. A new IC50 cutoff must **not** be inferred merely by optimizing agreement with the existing labels or external model predictions. If the original `Class` construction cannot be recovered or its endpoint shown comparable, the binary external confirmation is **no-go**. An IC50-defined classification or pIC50 regression analysis would be a separate, explicitly nonconfirmatory study.

## Source and record eligibility

1. Fetch the complete raw activity responses for each named assay, plus target, assay, document, source and API-status metadata. Save query URLs, UTC retrieval time, source release (if supplied), response hashes and software versions. Preserve raw responses without overwriting earlier snapshots.
2. Require `target_chembl_id == CHEMBL4822`, a single-protein human BACE-1 target, `assay_type == B`, and an assay description showing direct biochemical enzyme inhibition. The two named assays use FRET readouts. Do not pool BACE2, cellular Aβ readouts, binding constants (`Ki`/`Kd`), or phenotypic assays with these biochemical IC50 records.
3. Require `standard_type == IC50`, `standard_units == nM`, `standard_relation == '='`, finite positive `standard_value`, a nonempty molecule structure, `potential_duplicate == 0`, and no adverse `data_validity_comment`. Preserve rejected rows and the exact reason for each rejection. Do not treat censored values as exact measurements.
4. Retain ChEMBL's numeric `activity_id`, `molecule_chembl_id`, `parent_molecule_chembl_id` when available, `assay_chembl_id`, `document_chembl_id`, DOI, PMID, patent identifier, source ID, document year, target ID, confidence score, raw SMILES, IC50 value and any ChEMBL quality flags. Missing parent IDs trigger review; they are not silently inferred from the molecule ID.

## Structure handling and provenance

Apply the unchanged `configs_v2/phase1_data_standardization.yaml` rules: RDKit parse, largest organic fragment, functional-group normalization, reionization, uncharging, canonical tautomer, isomeric canonical SMILES and InChIKey. Preserve raw and intermediate representations. Group replicate records by standardized stereospecific structure **within assay**; retain the number, range and median of log10(IC50 nM). If stereochemistry is unspecified, do not assign it; compare both stereospecific and stereo-stripped keys. Review tautomer-convergent groups and conflicts rather than choosing a favorable measurement.

Compare each candidate to *all* original BACE rows using raw canonical SMILES, standardized isomeric SMILES, InChIKey, stereo-stripped standardized SMILES, and scaffold. Exact or standardized overlaps are excluded from external evaluation. Stereo-only, tautomer-only and close analog relationships are reported separately; scaffold novelty is descriptive, not a substitute for compound-level independence. Compare assay and document IDs plus DOI, PMID, patent identifiers and, where possible, patent family or cited source literature with the original BACE supporting information. A later ChEMBL ingestion or document year alone does not establish independent experimental provenance.

## Outcome and replicate rule: original-label verification

The original `Class` is not reconstructed by `pIC50 >= 7` across the entire DeepChem CSV: Phase 1 found 17 disagreements. First inspect the original Subramanian et al. paper and supporting data, DeepChem/MoleculeNet provenance, and the 17 discrepant rows for an explicit class rule and possible source-specific annotations. If a defensible original rule is recovered, document it with a source citation and apply it unchanged to external IC50 only if the assay endpoint is comparable. For replicate values, aggregate the log IC50 within each assay by the median only when all records yield the same class under the verified rule; otherwise mark the structure unresolved and exclude it from the binary cohort. Do not use model outputs to settle conflicts.

**Evidence addendum, 2026-10-20, before external model scoring:** The publisher's [original BACE supporting dataset](https://acs.figshare.com/articles/dataset/Computational_Modeling_of_Secretase_1_BACE-1_Inhibitors_Using_Ligand_Based_Approaches/4003788) was retrieved and verified against its published MD5. In sheet `S1-bace1547_liter`, all 1,478 IC50 records with a source class follow `Class = 1` exactly when `pIC50 >= 7`; the highest source IC50 inactive is 6.9829664, and the lowest active is 7. The 35 Ki, 8 Kd, 1 cellular and 25 endpoint-unspecified source records have no source class. In the local DeepChem CSV, 17 threshold disagreements are all mapped to Ki/Kd or one row without a CID match to the source sheet. The external IC50 operational rule is therefore frozen as `IC50 <= 100 nM` active, `IC50 > 100 nM` inactive, within the named biochemical assays. It is comparable to the original **IC50-labelled subset**, not a perfect reconstruction of the mixed-endpoint DeepChem CSV. The frozen trained models themselves still include those non-IC50 records, a limitation to report rather than a license to retrain.

## Go/no-go and freeze

Go only if (i) the binary endpoint is traceable and comparable, (ii) compound and source independence can be defended, (iii) each proposed assay cohort has at least 30 resolved active and 30 resolved inactive structures, and (iv) exclusions and remaining uncertainty are fully recorded. This minimum is a feasibility floor, not evidence of precise external estimates. Otherwise report no-go and do not run frozen models as confirmatory external validation. If go, freeze the exact row IDs, structure keys, label rule, hashes, model/code revision and evaluation script before scoring; evaluate each assay once and separately.

## References

- [DeepChem BACE loader](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/bace_datasets.py)
- [Original BACE paper](https://doi.org/10.1021/acs.jcim.6b00290) and [publisher-hosted supporting dataset](https://acs.figshare.com/articles/dataset/Computational_Modeling_of_Secretase_1_BACE-1_Inhibitors_Using_Ligand_Based_Approaches/4003788)
- [ChEMBL activity-data guidance](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)
- [ChEMBL assay CHEMBL5736880](https://www.ebi.ac.uk/chembl/api/data/assay/CHEMBL5736880.json)
- [ChEMBL assay CHEMBL5734173](https://www.ebi.ac.uk/chembl/api/data/assay/CHEMBL5734173.json)
