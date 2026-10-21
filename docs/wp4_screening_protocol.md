# WP4 exploratory reliability-oriented screening protocol

**Frozen before WP4 calculations:** 2026-10-20. This is a post-external-evaluation exploratory analysis of the already viewed WP3 paired predictions. It is not a second independent test, does not refit C1/D2, and cannot justify changing the deployed predictor or screening policy without a new prospective cohort.

## Question and analysis set

Can simple, label-free deferral rules select subsets on which the frozen C1 predictor has lower observed error while keeping useful early-screening precision? Analyze the 158 primary-assay compounds and the 191 older-source sensitivity-assay compounds **separately**. Use every eligible molecule from `paired_predictions.csv`; do not remove difficult cases or redefine the binary endpoint. The identical C1 probabilities and labels from WP3 are reused without model inference on the external cohort.

## Locked rules and budgets

For each assay, retain `ceil(c*N)` molecules for coverage `c` in `{1.0, 0.9, 0.8, 0.7, 0.6, 0.5}` and defer the rest. The reference at 100% coverage is identical for all rules. Exact rank ties are broken by ascending `assay_chembl_id|standardized_inchikey`.

1. **C1 boundary distance:** keep larger `abs(P_C1 - 0.5)`; defer predictions nearest 0.5. This is a heuristic uncertainty proxy, not a learned uncertainty estimate.
2. **Expert agreement:** keep smaller `abs(P_RF - P_ChemBERTa)`; defer the largest disagreement.
3. **Training similarity:** keep larger maximum Morgan Tanimoto similarity to original BACE training chemistry; defer the lowest-similarity compounds.

Do not tune a cutoff, transform, rule orientation, or coverage level using observed WP3 labels. All rules rank with the **same frozen C1 activity probability** after retention; only the accept/defer decision changes. `0.5` is used solely for a descriptive hard-classification error rate.

## Locked outcomes

- Coverage versus mean C1 Brier loss among retained molecules is the primary descriptive risk curve. Also report the retained 0.5-threshold error rate, n, active count, prevalence and deferred count so changing class mix is visible.
- Screening budget is **fixed at `ceil(0.05*N_original_assay)`**, equal to 8 primary and 10 secondary molecules at every coverage; it is not 5% of the shrinking retained set. Precision@that budget is the fraction active among the highest-C1-probability retained molecules. The denominator `k` and the number of hits are always displayed.
- Report the count of deferred actives/inactives and the C1 Brier loss in the deferred set. A lower retained Brier with many deferred actives is not automatically useful in discovery.
- No p-values or confirmatory intervals are assigned to coverage curves. Outcome comparisons are post hoc descriptions of the viewed external cohorts.
- Measure one cold end-to-end predictor call on the first 128 original BACE structures (not the external cohort) to estimate inference cost. State machine/runtime scope, batch size, whether model loading and encoder execution are included, and the bookkeeping time for the deferral rankings. This timing cannot be interpreted as a hardware-independent throughput guarantee.

## Figure contract

The Results-level question is whether any fixed rejection rule visibly improves error at a fixed screening budget. Use two separate **single-panel quantitative** figures for the primary cohort: coverage versus retained Brier loss and coverage versus fixed-budget Precision@Top 5%. The no-deferral value is the shared 100% reference. Use Python/Matplotlib, white background, editable PDF/SVG text, ≥5 pt final glyphs, restrained colors with rule identity consistent between figures, and explicit denominators in the report. Secondary-source curves are retained in source data and discussed separately, not pooled into the primary figure. Source CSV, code, SVG/PDF, and rendered QA belong together. Figures show descriptive points only, no unearned uncertainty ribbons.

## Interpretation boundary

If a rule appears useful in this viewed cohort, treat it as a candidate to **pre-register and test elsewhere**. A confidence proxy can fail under dataset shift; uncertainty is not a substitute for potency or activity prediction, and screening utility cannot be claimed without prospective experimental follow-up.
