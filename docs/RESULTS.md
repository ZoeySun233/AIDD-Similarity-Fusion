# Results and interpretation

## Primary internal comparison

The version-2 protocol used 2 repeats × 5 outer scaffold folds, 3 inner scaffold folds, and training seeds 11, 22, and 33. Seeds were averaged within each outer context; the same original molecules recur across repeats. The primary endpoint is average precision (AP), not trapezoidal PR-curve area.

| Method | Mean AP ↑ | Mean Brier ↓ | Mean ROC-AUC ↑ |
| --- | ---: | ---: | ---: |
| Morgan RF | 0.846521 | 0.139073 | 0.881338 |
| Frozen ChemBERTa + MLP | 0.803114 | 0.151083 | 0.858897 |
| C1: calibrated fixed fusion | 0.847416 | 0.135738 | 0.885472 |
| D2: similarity + entropy gate | 0.847331 | 0.135732 | 0.885467 |

For D2 minus C1, mean paired AP difference was **−0.000085** across ten outer fold-repeat comparisons, with scaffold-cluster bootstrap 95% CI **[−0.000386, +0.000071]** and wins in 4/10 comparisons. This did not meet the prespecified success criteria. The ten comparisons do not represent ten independent chemical datasets.

## One-time external evaluation

The historical fold checkpoints were not available. Before external scoring, a new full-original-BACE deployment model was trained on 1,513 original molecules, calibrated, frozen, and checked against a manifest. This evaluation must be called an **external evaluation of the newly frozen model**, not recovery of the old checkpoints. Neither external cohort was used to refit or choose a component.

| Assay | n / actives | Model | AP ↑ | Brier ↓ | ROC-AUC ↑ |
| --- | ---: | --- | ---: | ---: | ---: |
| Primary `CHEMBL5736880` | 158 / 110 | RF | 0.8394 | 0.2173 | 0.6735 |
| Primary `CHEMBL5736880` | 158 / 110 | ChemBERTa | 0.7890 | 0.2084 | 0.6288 |
| Primary `CHEMBL5736880` | 158 / 110 | C1 | 0.8413 | 0.2051 | 0.6994 |
| Primary `CHEMBL5736880` | 158 / 110 | D2 | 0.8404 | 0.2051 | 0.6991 |
| Older-source `CHEMBL5734173` | 191 / 135 | RF | 0.7390 | 0.3353 | 0.5720 |
| Older-source `CHEMBL5734173` | 191 / 135 | ChemBERTa | 0.7915 | 0.3549 | 0.6469 |
| Older-source `CHEMBL5734173` | 191 / 135 | C1 | 0.7629 | 0.3343 | 0.6130 |
| Older-source `CHEMBL5734173` | 191 / 135 | D2 | 0.7633 | 0.3343 | 0.6135 |

On the same 158 primary compounds, `AP(D2) − AP(C1) = −0.0009125`. The paired scaffold-cluster bootstrap used 62 scaffolds and 10,000 valid resamples; its 95% CI was **[−0.005997, +0.000180]**. Brier difference was −0.0000253, practically negligible. The prespecified decision rule required at least +0.01 AP, a CI lower bound above zero, and no meaningful Brier deterioration. It failed. This supports retaining C1; it does not prove D2 is universally inferior.

Primary active prevalence was 69.6% and older-source prevalence was 70.7%. AP and early-screening precision should be interpreted relative to those high baselines. The older-source C1 Brier of 0.3343 was worse than an approximately 0.207 constant-prevalence reference, suggesting poor probability transfer. The two assays are not pooled.

## Mechanistic and exploratory observations

- RF and ChemBERTa residuals were strongly correlated in the exploratory failure analysis (mean within-fold Pearson correlation 0.888). Both experts were wrong in 404 of 831 prediction events where at least one was wrong. Shared errors limit the opportunity for a switching gate; this is descriptive, not a causal proof.
- D2's mean shrinkage coefficient was 0.006607 in nested CV. Its predictions were therefore very close to C1, rather than materially changing expert weight by similarity.
- Nine Phase-7 representation/head variants failed the predefined exploratory admission rule for a complementary expert. This does not invalidate the broader ChemBERTa or graph-model families.
- Post-external accept/defer rules selected easier subsets in some settings but did not produce a reliable transferable screening benefit. At 50% coverage on the primary assay, the C1-boundary rule deferred 46 of 110 known actives while leaving top-eight hits at 8/8. This was post hoc and is not a validated deployment policy.

The useful scientific result is a carefully bounded negative finding: increasing fusion complexity did not yield a confirmed AP gain under scaffold CV and one narrow external transfer test. Assay-source shift, endpoint provenance, and shared expert errors deserve more attention than further tuning on these already viewed outcomes.

Compact source tables are in `../results_summary/`. Full per-molecule data, model artifacts, bootstrap draws, and curation manifests are intentionally not part of this public extract.
