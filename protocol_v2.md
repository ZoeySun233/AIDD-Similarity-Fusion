# Protocol v2: Reliability-Aware Molecular Model Fusion

**Status:** Frozen before v2 model development
**Protocol date:** 2026-10-19
**Dataset:** BACE-1 binary activity classification
**Historical experiment:** `v1_original_experiment/`

## 1. Purpose and research question

Version 1 showed that the maximum-Tanimoto dynamic fusion rule did not
reliably outperform a validation-optimized fixed-weight ensemble on the
historical scaffold test set. Version 2 asks:

> Can a calibrated, regularized, reliability-aware fusion rule improve
> activity ranking over calibrated fixed fusion under repeated scaffold-level
> distribution shifts?

The primary comparison is the proposed reliability-aware dynamic fusion model
against calibrated fixed-weight fusion. All other comparisons are supportive
or exploratory.

## 2. Historical-test boundary

The original 227-molecule test set was evaluated during Week 5 and is now a
`historical_test`. It may be retained for exact reproduction and retrospective
error analysis, but it must not be used to:

- choose v2 preprocessing or molecular standardization rules;
- select representations, features, hyperparameters, calibration methods, or
  fusion equations;
- select classification thresholds or stopping epochs;
- decide which v2 result to present as the final result.

No v2 model-performance claim will be based on a new evaluation of this
historical test set.

## 3. Data protocol

The raw BACE CSV is immutable. Version 2 preprocessing will create separate
outputs under `data/processed_v2/` and will record every exclusion or structure
change. The audit will cover molecular validity, multi-component structures,
largest-fragment selection, charge normalization, tautomer handling,
stereochemistry-sensitive and stereochemistry-insensitive duplicates,
conflicting labels, and consistency between `Class` and `pIC50`.

Labels will not be silently corrected. Ambiguous records will be retained,
excluded, or analyzed separately only according to a documented rule fixed
before model evaluation.

## 4. Primary evaluation design

The primary internal evaluation will use repeated nested scaffold cross-validation:

- outer evaluation: 2 repeats × 5 folds;
- inner model selection: 3 scaffold-disjoint folds;
- training seeds per outer fold: 11, 22, and 33;
- the same outer and inner assignments for every compared model;
- no scaffold overlap between development and outer-test data.

Hyperparameters, early stopping, calibration, fusion parameters, feature
selection, and decision thresholds will be determined inside the outer
development set. The outer-test fold will be used only after those choices are
frozen. Seeds will first be summarized within each outer fold; outer folds,
not seeds, are the primary statistical units.

Canonical DeepChem scaffold splitting and structure-cluster splitting may be
reported as prespecified sensitivity analyses. A random split, if included,
is an optimistic sanity check and cannot replace the primary evaluation.

## 5. Base models and cross-fitting

The initial v2 base models are:

1. Morgan fingerprint plus Random Forest.
2. Frozen ChemBERTa representation plus a lightweight classifier.

Within each outer-development set, scaffold-aware inner cross-fitting will
generate out-of-fold probabilities for every development molecule. Only these
out-of-fold probabilities may be used to train calibrators, stacking models,
or dynamic fusion gates. After selection, base models will be refitted on the
complete outer-development set before outer-test prediction.

Any stronger representation, partial fine-tuning, SMILES augmentation, or
graph model will be a separately identified experiment using the same folds.

## 6. Calibration protocol

Base-model probabilities will be calibrated before fusion. Platt scaling is
the primary method; temperature scaling is permitted for the neural model.
Isotonic regression is a sensitivity analysis because of the limited sample
size. Calibrators must be fitted using inner out-of-fold predictions only.

Calibration will be evaluated using Brier score, log loss, calibration slope,
calibration intercept, and a reliability diagram. Fixed and dynamic fusion
must receive the same calibrated component probabilities.

## 7. Prespecified model comparisons

The minimum experiment set is:

| ID | Model | Role |
|---|---|---|
| A | Calibrated Morgan RF | Classical baseline |
| B | Calibrated frozen ChemBERTa classifier | Representation baseline |
| C0 | Equal-weight fusion (`alpha_RF = 0.5`) | Parameter-free fusion control |
| C1 | Calibrated fixed-weight fusion | Primary control |
| D0 | Original maximum-Tanimoto sigmoid gate | Version 1 reference |
| D1 | Regularized gate shrunk toward C1 | Dynamic-gating ablation |
| D2 | D1 plus calibrated relative-confidence features | Proposed v2 model |
| E | Similarity-permuted gate | Negative control |
| F | Low-dimensional calibrated logistic stacking | Non-gating ensemble control |

The proposed shrinkage gate is nested within the fixed ensemble:

`alpha(x) = (1 - lambda) * alpha_fixed + lambda * sigmoid(g(x))`.

`lambda = 0` exactly recovers fixed fusion. Gate parameters will be estimated
with Brier score or log loss plus regularization, not by exhaustive direct
optimization of PR-AUC. A one-standard-error rule will prefer the simpler gate
when inner-CV performance is indistinguishable.

Candidate reliability features will be introduced through prespecified
ablations: maximum train-set Tanimoto similarity, relative calibrated entropy,
top-k similarity summaries, neighbor-label consistency, and model uncertainty.
All neighborhood features must use only the applicable training fold.

## 8. Outcomes and decision rule

The sole primary endpoint is PR-AUC. Key secondary endpoints are Brier score
and ROC-AUC. Additional reporting includes log loss, F1, balanced accuracy,
MCC, Precision at the top 5% and 10%, and recall at a fixed screening budget.
Threshold-dependent metrics use thresholds selected only from inner
out-of-fold predictions.

The primary reported effect is the paired outer-fold difference:

`PR-AUC(D2) - PR-AUC(C1)`.

Results will include the fold-level differences, mean, standard deviation,
95% confidence interval, and number of outer folds won. Confidence intervals
will respect scaffold clustering. Per-seed variation will not be presented as
independent evidence of chemical-space generalization.

The suggested criterion for a practically meaningful positive result is:

- mean PR-AUC improvement of at least 0.01 over C1;
- a paired 95% confidence-interval lower bound above zero;
- improvement in at least 70% of outer folds;
- no material Brier-score degradation;
- no systematic degradation in the low-similarity region;
- improvement not attributable to one dominant scaffold;
- performance exceeding the similarity-permuted negative control.

Failure to satisfy these criteria will be reported as evidence that the tested
dynamic gate does not reliably improve over fixed fusion.

## 9. Locked external confirmation

After all v2 decisions are frozen, an independently curated and deduplicated
BACE-1 dataset may be used once for external confirmation. Target, species,
assay type, units, activity transformation, structure standardization,
duplicate handling, and overlap removal must be fixed before accessing its
labels for evaluation. No external-set tuning or recalibration is permitted.

## 10. Reproducibility and output separation

- Version 1 remains under `v1_original_experiment/`.
- Version 2 configurations go to `configs_v2/`.
- Version 2 results go to `results_v2/`.
- Version 2 reports go to `report_v2/`.
- Version 2 processed data go to `data/processed_v2/`.
- Every run will save its resolved configuration, split identifiers, seeds,
  input hashes, dependency versions, predictions, and timing conditions.
- A configuration file must be consumed by the running code; documentation-only
  configuration files are not sufficient.

Changes to this protocol must be dated and recorded before the affected outer
test folds or external-test labels are examined.
