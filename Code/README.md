# Q1 tasks 1 and 2 implementation

This directory implements the first two sub-tasks of Question 1:

1. multi-signal quality evaluation on A1/A2/A3;
2. conflict diagnosis and a constrained conflict-aware candidate score.

The implementation follows the proposal and incorporates the review decision that a non-zero conflict penalty is not identifiable without independent labels. Therefore the default reported `Q_primary` is the transparent weighted average (`Q_base`). The finite-compensation score (`Q_rule`) and `Q_eta_*` scenarios are still computed and exported. After an independent blind review or downstream validation, `primary_score` can be changed to `Q_rule` in `configs/q1_quality.json`.

## Run

From the repository root:

```powershell
python Code/run_q1.py `
  --data-root real_attachments `
  --config Code/configs/q1_quality.json `
  --out Code/outputs/q1
```

For a quick smoke run, add `--bootstrap 10 --permutations 20`. The default run processes all A1/A2/A3 records and uses only A1 to fit empirical reference scales, impute domain medians, freeze thresholds, and estimate the optional CRITIC comparison weights.

## Project structure

- `configs/q1_quality.json`: versioned scoring assumptions and review status.
- `src/llm_resource/q1/schema.py`: the 22 fields, list decoding, and field semantics.
- `data.py`: streaming xz-JSONL reader, provenance and overlap audit.
- `preprocess.py`: A1-fitted, frozen utility transformations.
- `scoring.py`: primary score, finite-compensation candidate, conflict diagnostics, and sensitivity scenarios.
- `diagnostics.py`: conditional pair analysis, permutation baseline, and descriptive factor analysis.
- `aggregate.py`: domain summaries, frozen-rule bootstrap intervals, and A1 versus extension comparisons.
- `cli.py`: end-to-end runner and CSV/JSON outputs.

## Interpretation rules

- `Q_primary` is the default domain quality score used in summaries.
- `Q_base` is the weighted average of all 22 positive-direction indicators.
- `Q_rule` subtracts a bounded pairwise disagreement penalty. It is a candidate preference function, not a learned ground-truth quality label.
- `has_high_low_difference`, `conflict_intensity`, and `within_domain_R` are diagnostics. A high diagnostic value is not automatically a defect.
- A2/A3 extension rows overlapping A1 are retained for file-level reporting but are not treated as independent records in the de-duplicated union.
- The expansion sets have no `content` field. Their scores can validate signal-level stability, but not the semantic cause of a conflict.

## Main outputs

`source_audit.csv`, `parse_issues.csv`, `sample_scores.csv`, `domain_summary.csv`, `score_sensitivity_by_domain.csv`, `extension_comparison.csv`, `conflict_pairs_*.csv`, `conflict_permutation_A1.csv`, `factor_*.csv`, `utility_dictionary.csv`, `fitted_scoring_model.json`, and `run_metadata.json`.

The full scoring table is intentionally called `sample_scores.csv` because the input is a sample/extension corpus rather than a census of the underlying 580B-token SlimPajama training set.
