# Wilcoxon Documentation

## Purpose

This folder contains the Wilcoxon statistical comparison for `SDCSA` against the same comparison algorithms used in the final graph variant `graphs_variant_sdcsa_v2`.

The goal was to answer:

- Is `SDCSA` statistically different from the compared algorithms?
- When a difference is significant, which algorithm performs better?

## Why This Test Was Done

The benchmark section already had repeated-run results for each algorithm, benchmark, and dimension.
Because each run was generated with matching run indices and matching seeds, the results can be treated as paired observations.

That makes the **Wilcoxon signed-rank test** appropriate for this comparison.

We used it to compare `SDCSA` against each included algorithm on:

- every 2D benchmark separately
- every 50D benchmark separately
- every 100D benchmark separately

We did **not** merge all dimensions into one raw test because `dim 2`, `dim 50`, and `dim 100` are different problem settings.

## Why Signed-Rank, Not Rank-Sum

- **Wilcoxon signed-rank test** is for paired or matched samples.
- **Wilcoxon rank-sum test** is for independent samples.

In this experiment, `run 1` of `SDCSA` is matched with `run 1` of the comparison algorithm, `run 2` with `run 2`, and so on.
Because the runs are pairable by run index and seed, signed-rank is the correct Wilcoxon variant here.

## Algorithm Set Used

The compared algorithms were taken from the final graph variant rules in:

- `test4_v2_results/graphs_variant_sdcsa_v2/manifest_variant_rules.txt`

Those rules were:

- exclude `IICO`
- exclude `MSHCSA`
- rename `HybridBase` to `FCSA-IICO`
- keep the final graph ordering

The Wilcoxon analysis used:

- reference algorithm: `SDCSA`
- compared algorithms: `FCSA`, `FCSA-IICO`, `CSA`, `ADECSA`, `DUSCSA`

Important note:

- In the graph variant, the display label `SDCSA` originally came from baseline `TSD`.
- In this Wilcoxon analysis, the reference algorithm is the **actual** `SDCSA` result set from `test6_dmmo_adjsuted/.../results_sdcsa_full_run01`.

## Data Sources Used

Reference side:

- actual `SDCSA` results from `test6_dmmo_adjsuted/sdcsa-substrate-drift-clonal-selection-algorithm/results_sdcsa_full_run01`

Comparison side:

- baseline comparison results from `test4_v2_results`

Source priority:

1. raw `run_*_per_gen.csv` folders
2. postprocessed finals CSV only as fallback

This priority was chosen because some aggregated `*_runs.csv` files in `test4_v2_results` were stale or truncated even when the raw run folders already had the full 100 runs.

## What Was Compared

For each benchmark and dimension:

- extract the final `best_fitness` from each per-run CSV
- pair runs by run number
- compute Wilcoxon signed-rank p-value
- record whether `p < 0.05`
- record which algorithm is better when the difference is significant

Significance threshold:

- `alpha = 0.05`

Test settings:

- two-sided alternative
- `zero_method = "pratt"`

## Output Files

Detailed results:

- `wilcoxon_sdsca_vs_graph_variant_detailed.csv`
- `wilcoxon_sdsca_vs_graph_variant_detailed.md`

Comparator summary:

- `wilcoxon_sdsca_vs_graph_variant_summary.csv`
- `wilcoxon_sdsca_vs_graph_variant_summary.md`

Paper-style p-value tables:

- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim2.csv`
- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim2.md`
- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim50.csv`
- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim50.md`
- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim100.csv`
- `wilcoxon_sdsca_vs_graph_variant_pvalues_dim100.md`

Run manifest:

- `manifest_wilcoxon_rules.txt`

## How To Read The Results

### Detailed table

Key columns:

- `p_value`: the Wilcoxon p-value
- `significant`: `True` if `p_value < 0.05`
- `significant_better_algorithm`: which algorithm is better when the result is significant
- `n_pairs`: number of usable paired values actually tested
- `n_excluded_nonfinite_pairs`: number of matched pairs removed because one or both values were non-finite

Interpretation:

- `p < 0.05` means the two algorithms are statistically different
- `p >= 0.05` means there is no statistically significant difference
- significance alone does **not** say `SDCSA` is better; use `significant_better_algorithm`

### Paper-style p-value tables

Each cell is the p-value for:

- `SDCSA` vs one comparator
- on one benchmark
- at one dimension

Interpretation:

- if the value is below `0.05`, the difference is significant
- if the value is `1` or above `0.05`, it is not significant
- `*` means the comparison used fewer than 100 usable paired values

## Important Caveat

After switching to the raw run folders, the old false partial cases disappeared.

The only remaining starred case is:

- `ADECSA` vs `SDCSA` on `Holdertable`, `dim 2`

Reason:

- the case still has 100 runs
- but many `ADECSA` final values are non-finite (`-inf`)
- only 9 usable finite pairs remained for the Wilcoxon test

So that star is a real data-quality warning, not a stale-file issue.

## Reproducibility

The script used to generate these files is:

- `test4_codes/generate_test4_v2_wilcoxon_variant.py`

Command used:

```powershell
python "test4_codes\generate_test4_v2_wilcoxon_variant.py"
```

## Practical Conclusion

These Wilcoxon outputs are ready to use for reporting.

The main interpretation is:

- most `SDCSA` vs comparator cases are statistically significant at `alpha = 0.05`
- the result should be reported as a **Wilcoxon signed-rank test**
- the only remaining caution is the `ADECSA` `Holdertable` `dim 2` case due to non-finite values
