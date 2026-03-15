"""
Generate SDCSA-vs-baseline Wilcoxon signed-rank results using the same
algorithm subset/naming policy as the final graph variant.

Variant provenance:
  - Graph rules come from `test4_codes/generate_test4_v2_graph_variant.py`
    and/or `test4_v2_results/graphs_variant_sdcsa_v2/manifest_variant_rules.txt`
  - Final graph names:
      TSD -> SDCSA
      HybridBase -> FCSA-IICO
      IICO, MSHCSA excluded

This script differs in one important way:
  - It uses the actual static SDCSA finals summary as the reference algorithm
    instead of reusing baseline TSD rows.

Outputs:
  - detailed CSV + Markdown: one row per (dimension, benchmark, comparator)
  - summary CSV + Markdown: one row per comparator
  - manifest TXT: records naming rules and source files
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "This script requires scipy. Install scipy in the same environment "
        "used for the rest of the thesis scripts."
    ) from exc


REFERENCE_DISPLAY = "SDCSA"
REFERENCE_RAW = "SDCSA"

DEFAULT_EXCLUDE_ALGORITHMS = {"IICO", "MSHCSA"}
DEFAULT_RENAME_ALGORITHM = {
    "TSD": "SDCSA",
    "HybridBase": "FCSA-IICO",
}
DEFAULT_ALGO_ORDER = ["SDCSA", "FCSA", "FCSA-IICO", "CSA", "ADECSA", "DUSCSA"]

BENCH_FILE_TO_DISPLAY = {
    "Schaffer_2": "Schaffer 2",
    "Schaffer_4": "Schaffer 4",
    "Ackley": "Ackley",
    "Griewank": "Griewank",
    "Rastrigin": "Rastrigin",
    "Shubert": "Shubert",
    "Eggholder": "Eggholder",
    "Holdertable": "Holdertable",
    "Levy": "Levy",
    "Schwefel": "Schwefel",
}

BENCH_FILE_TO_INTERNAL = {
    "Schaffer_2": "SchafferN2",
    "Schaffer_4": "SchafferN4",
    "Ackley": "Ackley",
    "Griewank": "Griewank",
    "Rastrigin": "Rastrigin",
    "Shubert": "Shubert",
    "Eggholder": "Eggholder",
    "Holdertable": "HolderTable",
    "Levy": "Levy",
    "Schwefel": "Schwefel",
}

RUN_DIR_RE = re.compile(r"^runs_(?P<algorithm>[^_]+)_(?P<benchmark>.+)_dim(?P<dim>\d+)$")
RUN_PER_GEN_RE = re.compile(r"^run_(?P<run>\d+)_per_gen\.csv$")


def load_variant_rules(manifest_path: Path) -> tuple[set[str], dict[str, str], list[str]]:
    exclude = set(DEFAULT_EXCLUDE_ALGORITHMS)
    rename = dict(DEFAULT_RENAME_ALGORITHM)
    order = list(DEFAULT_ALGO_ORDER)

    if not manifest_path.exists():
        return exclude, rename, order

    raw: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = value.strip()

    try:
        if "exclude_algorithms" in raw:
            exclude = set(ast.literal_eval(raw["exclude_algorithms"]))
        if "rename_algorithm" in raw:
            rename = dict(ast.literal_eval(raw["rename_algorithm"]))
        if "algorithm_order" in raw:
            order = list(ast.literal_eval(raw["algorithm_order"]))
    except Exception:
        # Keep safe fallbacks if manifest parsing fails.
        pass

    return exclude, rename, order


def normalize_algorithm(raw_name: str, exclude: set[str], rename: dict[str, str]) -> str | None:
    if raw_name in exclude:
        return None
    display_name = rename.get(raw_name, raw_name)
    if display_name in exclude:
        return None
    return display_name


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_final_best_from_per_gen_csv(path: Path) -> float | None:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        last_value: float | None = None
        for row in reader:
            value_text = row.get("best_fitness", "")
            if value_text in (None, ""):
                continue
            try:
                last_value = float(value_text)
            except Exception:
                continue
    return last_value


def build_reference_cases_from_run_dirs(results_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}

    for run_dir in sorted(results_root.glob("dim_*/runs_SDCSA_*_dim*")):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue

        bench_file = str(match.group("benchmark")).strip()
        dim = int(match.group("dim"))
        benchmark = BENCH_FILE_TO_INTERNAL.get(bench_file, bench_file)
        benchmark_display = BENCH_FILE_TO_DISPLAY.get(bench_file, bench_file.replace("_", " "))
        key = (benchmark, dim)
        case = cases.setdefault(
            key,
            {
                "benchmark": benchmark,
                "benchmark_display": benchmark_display,
                "dimension": dim,
                "runs": {},
                "source_runs_csv": str(run_dir),
            },
        )

        for per_gen_csv in sorted(run_dir.glob("run_*_per_gen.csv")):
            run_match = RUN_PER_GEN_RE.match(per_gen_csv.name)
            if not run_match:
                continue
            run = int(run_match.group("run"))
            final_best = read_final_best_from_per_gen_csv(per_gen_csv)
            if final_best is None:
                continue
            case["runs"][run] = final_best
    return cases


def build_comparator_cases_from_run_dirs(
    results_root: Path,
    exclude: set[str],
    rename: dict[str, str],
    allowed_display_names: set[str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    cases: dict[tuple[str, str, int], dict[str, Any]] = {}

    for run_dir in sorted(results_root.glob("dim_*/runs_*_dim*")):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue

        raw_algorithm = str(match.group("algorithm")).strip()
        display_algorithm = normalize_algorithm(raw_algorithm, exclude, rename)
        if display_algorithm is None or display_algorithm == REFERENCE_DISPLAY:
            continue
        if display_algorithm not in allowed_display_names:
            continue

        bench_file = str(match.group("benchmark")).strip()
        dim = int(match.group("dim"))
        benchmark = BENCH_FILE_TO_INTERNAL.get(bench_file, bench_file)
        benchmark_display = BENCH_FILE_TO_DISPLAY.get(bench_file, bench_file.replace("_", " "))
        key = (display_algorithm, benchmark, dim)
        case = cases.setdefault(
            key,
            {
                "compared_algorithm": display_algorithm,
                "compared_raw_algorithm": raw_algorithm,
                "benchmark": benchmark,
                "benchmark_display": benchmark_display,
                "dimension": dim,
                "runs": {},
                "source_runs_csv": str(run_dir),
            },
        )

        for per_gen_csv in sorted(run_dir.glob("run_*_per_gen.csv")):
            run_match = RUN_PER_GEN_RE.match(per_gen_csv.name)
            if not run_match:
                continue
            run = int(run_match.group("run"))
            final_best = read_final_best_from_per_gen_csv(per_gen_csv)
            if final_best is None:
                continue
            case["runs"][run] = final_best
    return cases


def build_reference_cases(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, Any]]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("algorithm", "")).strip() != REFERENCE_RAW:
            continue

        benchmark = str(row["benchmark"]).strip()
        benchmark_display = str(row.get("benchmark_display", benchmark)).strip()
        dim = int(row["dimension"])
        run = int(row["run"])
        final_best = float(row["final_best"])
        key = (benchmark, dim)

        case = cases.setdefault(
            key,
            {
                "benchmark": benchmark,
                "benchmark_display": benchmark_display,
                "dimension": dim,
                "runs": {},
                "source_runs_csv": str(row.get("source_runs_csv", "")).strip(),
            },
        )
        case["runs"][run] = final_best
        if not case["source_runs_csv"] and row.get("source_runs_csv"):
            case["source_runs_csv"] = str(row["source_runs_csv"]).strip()
    return cases


def build_comparator_cases(
    rows: list[dict[str, str]],
    exclude: set[str],
    rename: dict[str, str],
    allowed_display_names: set[str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    cases: dict[tuple[str, str, int], dict[str, Any]] = {}

    for row in rows:
        raw_algorithm = str(row.get("algorithm", "")).strip()
        display_algorithm = normalize_algorithm(raw_algorithm, exclude, rename)
        if display_algorithm is None:
            continue
        if display_algorithm == REFERENCE_DISPLAY:
            # Final graph label "SDCSA" came from baseline TSD; this analysis
            # uses actual SDCSA results as the reference instead.
            continue
        if display_algorithm not in allowed_display_names:
            continue

        benchmark = str(row["benchmark"]).strip()
        benchmark_display = str(row.get("benchmark_display", benchmark)).strip()
        dim = int(row["dimension"])
        run = int(row["run"])
        final_best = float(row["final_best"])
        key = (display_algorithm, benchmark, dim)

        case = cases.setdefault(
            key,
            {
                "compared_algorithm": display_algorithm,
                "compared_raw_algorithm": raw_algorithm,
                "benchmark": benchmark,
                "benchmark_display": benchmark_display,
                "dimension": dim,
                "runs": {},
                "source_runs_csv": str(row.get("source_runs_csv", "")).strip(),
            },
        )
        case["runs"][run] = final_best
        if not case["source_runs_csv"] and row.get("source_runs_csv"):
            case["source_runs_csv"] = str(row["source_runs_csv"]).strip()
    return cases


def _fmt_md(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown_table(md_path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = [_fmt_md(row.get(col, "")) for col in columns]
            fh.write("| " + " | ".join(vals) + " |\n")


def algo_sort_key(name: str, algorithm_order: list[str]) -> tuple[int, str]:
    try:
        idx = algorithm_order.index(name)
    except ValueError:
        idx = len(algorithm_order) + 1
    return (idx, name)


def format_p_value_for_matrix(value: Any) -> str:
    if value is None or value == "":
        return "NA"

    p_value = float(value)
    if abs(p_value - 1.0) < 1e-15:
        return "1"
    if p_value >= 0.001:
        return f"{p_value:.3f}".rstrip("0").rstrip(".")

    text = f"{p_value:.2E}"
    return re.sub(r"E([+-])0+(\d+)$", r"E\1\2", text)


def format_p_value_cell(row: dict[str, Any]) -> str:
    if str(row.get("test_status", "")) in {"no_finite_pairs", "too_few_pairs"}:
        return "NA"

    cell = format_p_value_for_matrix(row.get("p_value", ""))
    if str(row.get("pair_coverage", "")) != "complete":
        cell += "*"
    return cell


def compute_case_row(
    reference_case: dict[str, Any],
    comparator_case: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    reference_runs: dict[int, float] = reference_case["runs"]
    comparator_runs: dict[int, float] = comparator_case["runs"]
    common_run_ids = sorted(set(reference_runs) & set(comparator_runs))

    if not common_run_ids:
        raise ValueError("No common run ids were found for this comparison.")

    ref_all = np.asarray([reference_runs[run_id] for run_id in common_run_ids], dtype=float)
    cmp_all = np.asarray([comparator_runs[run_id] for run_id in common_run_ids], dtype=float)
    finite_mask = np.isfinite(ref_all) & np.isfinite(cmp_all)
    ref_vals = ref_all[finite_mask]
    cmp_vals = cmp_all[finite_mask]
    deltas = ref_vals - cmp_vals
    excluded_nonfinite_pairs = int(len(common_run_ids) - len(ref_vals))

    wins_reference = int(np.sum(deltas < 0.0))
    wins_compared = int(np.sum(deltas > 0.0))
    ties = int(np.sum(deltas == 0.0))
    n_nonzero_diffs = int(np.sum(deltas != 0.0))

    pair_coverage = "complete"
    if (
        len(ref_vals) != len(reference_runs)
        or len(ref_vals) != len(comparator_runs)
    ):
        pair_coverage = "partial"

    if len(ref_vals) > 0:
        median_delta = float(np.median(deltas))
        mean_delta = float(np.mean(deltas))
        reference_mean_final: float | str = float(np.mean(ref_vals))
        compared_mean_final: float | str = float(np.mean(cmp_vals))
        reference_median_final: float | str = float(np.median(ref_vals))
        compared_median_final: float | str = float(np.median(cmp_vals))
    else:
        median_delta = ""
        mean_delta = ""
        reference_mean_final = ""
        compared_mean_final = ""
        reference_median_final = ""
        compared_median_final = ""

    if isinstance(median_delta, float) and median_delta < 0.0:
        better_by_median = REFERENCE_DISPLAY
    elif isinstance(median_delta, float) and median_delta > 0.0:
        better_by_median = comparator_case["compared_algorithm"]
    else:
        better_by_median = "Tie"

    statistic: float | None
    p_value: float | None
    significant = False
    significant_better = "Tie"
    test_status = "ok"

    if len(ref_vals) == 0:
        statistic = None
        p_value = None
        test_status = "no_finite_pairs"
    elif len(ref_vals) < 2:
        statistic = None
        p_value = None
        test_status = "too_few_pairs"
    elif n_nonzero_diffs == 0:
        statistic = 0.0
        p_value = 1.0
        test_status = "all_pairs_equal"
    else:
        result = wilcoxon(ref_vals, cmp_vals, zero_method="pratt", alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        significant = bool(p_value < alpha)
        if significant:
            significant_better = better_by_median

    return {
        "reference_algorithm": REFERENCE_DISPLAY,
        "reference_raw_algorithm": REFERENCE_RAW,
        "compared_algorithm": comparator_case["compared_algorithm"],
        "compared_raw_algorithm": comparator_case["compared_raw_algorithm"],
        "benchmark": reference_case["benchmark"],
        "benchmark_display": reference_case["benchmark_display"],
        "dimension": int(reference_case["dimension"]),
        "n_reference_runs": int(len(reference_runs)),
        "n_compared_runs": int(len(comparator_runs)),
        "n_common_runs": int(len(common_run_ids)),
        "n_pairs": int(len(ref_vals)),
        "n_reference_only_runs": int(len(set(reference_runs) - set(comparator_runs))),
        "n_compared_only_runs": int(len(set(comparator_runs) - set(reference_runs))),
        "n_excluded_nonfinite_pairs": excluded_nonfinite_pairs,
        "pair_coverage": pair_coverage,
        "n_nonzero_diffs": n_nonzero_diffs,
        "reference_mean_final": reference_mean_final,
        "compared_mean_final": compared_mean_final,
        "reference_median_final": reference_median_final,
        "compared_median_final": compared_median_final,
        "mean_delta_reference_minus_compared": mean_delta,
        "median_delta_reference_minus_compared": median_delta,
        "wins_reference": wins_reference,
        "wins_compared": wins_compared,
        "ties": ties,
        "wilcoxon_statistic": statistic if statistic is not None else "",
        "p_value": p_value if p_value is not None else "",
        "alpha": float(alpha),
        "significant": significant,
        "better_algorithm_by_median_delta": better_by_median,
        "significant_better_algorithm": significant_better,
        "test_status": test_status,
        "reference_source_runs_csv": reference_case["source_runs_csv"],
        "compared_source_runs_csv": comparator_case["source_runs_csv"],
    }


def build_detailed_rows(
    reference_cases: dict[tuple[str, int], dict[str, Any]],
    comparator_cases: dict[tuple[str, str, int], dict[str, Any]],
    alpha: float,
    algorithm_order: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for (compared_algorithm, benchmark, dim), comparator_case in comparator_cases.items():
        reference_key = (benchmark, dim)
        reference_case = reference_cases.get(reference_key)
        if reference_case is None:
            continue
        rows.append(compute_case_row(reference_case, comparator_case, alpha))

    rows.sort(
        key=lambda row: (
            int(row["dimension"]),
            str(row["benchmark_display"]),
            algo_sort_key(str(row["compared_algorithm"]), algorithm_order),
        )
    )
    return rows


def build_summary_rows(detailed_rows: list[dict[str, Any]], algorithm_order: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_names: dict[str, str] = {}

    for row in detailed_rows:
        display_name = str(row["compared_algorithm"])
        grouped[display_name].append(row)
        raw_names.setdefault(display_name, str(row["compared_raw_algorithm"]))

    summary_rows: list[dict[str, Any]] = []
    for display_name, rows in grouped.items():
        total_cases = len(rows)
        complete_cases = sum(1 for row in rows if row["pair_coverage"] == "complete")
        partial_cases = sum(1 for row in rows if row["pair_coverage"] == "partial")
        all_pairs_equal_cases = sum(1 for row in rows if row["test_status"] == "all_pairs_equal")
        no_finite_pairs_cases = sum(1 for row in rows if row["test_status"] == "no_finite_pairs")
        too_few_pairs_cases = sum(1 for row in rows if row["test_status"] == "too_few_pairs")
        significant_reference_better = sum(
            1
            for row in rows
            if row["significant"] and row["significant_better_algorithm"] == REFERENCE_DISPLAY
        )
        significant_compared_better = sum(
            1
            for row in rows
            if row["significant"] and row["significant_better_algorithm"] == display_name
        )
        not_significant = sum(1 for row in rows if not row["significant"])
        n_pairs = [int(row["n_pairs"]) for row in rows]

        summary_rows.append(
            {
                "reference_algorithm": REFERENCE_DISPLAY,
                "compared_algorithm": display_name,
                "compared_raw_algorithm": raw_names.get(display_name, ""),
                "total_cases": total_cases,
                "complete_cases": complete_cases,
                "partial_cases": partial_cases,
                "all_pairs_equal_cases": all_pairs_equal_cases,
                "no_finite_pairs_cases": no_finite_pairs_cases,
                "too_few_pairs_cases": too_few_pairs_cases,
                "significant_reference_better": significant_reference_better,
                "significant_compared_better": significant_compared_better,
                "not_significant": not_significant,
                "min_pairs": min(n_pairs) if n_pairs else "",
                "max_pairs": max(n_pairs) if n_pairs else "",
                "mean_pairs": float(np.mean(n_pairs)) if n_pairs else "",
            }
        )

    summary_rows.sort(key=lambda row: algo_sort_key(str(row["compared_algorithm"]), algorithm_order))
    return summary_rows


def build_pvalue_matrix_rows(
    detailed_rows: list[dict[str, Any]],
    dimension: int,
    compared_algorithms: list[str],
) -> list[dict[str, Any]]:
    rows_for_dim = [row for row in detailed_rows if int(row["dimension"]) == dimension]
    benchmarks = sorted({str(row["benchmark_display"]) for row in rows_for_dim})
    index: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["benchmark_display"]), str(row["compared_algorithm"])): row for row in rows_for_dim
    }

    matrix_rows: list[dict[str, Any]] = []
    for benchmark_display in benchmarks:
        matrix_row: dict[str, Any] = {"benchmark_display": benchmark_display}
        for compared_algorithm in compared_algorithms:
            source_row = index.get((benchmark_display, compared_algorithm))
            matrix_row[compared_algorithm] = format_p_value_cell(source_row) if source_row else "NA"
        matrix_rows.append(matrix_row)
    return matrix_rows


def write_pvalue_matrix_markdown(
    md_path: Path,
    caption: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"{caption}\n\n")
        fh.write("| " + " | ".join(columns) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = [str(row.get(col, "NA")) for col in columns]
            fh.write("| " + " | ".join(vals) + " |\n")
        fh.write(
            "\nReference algorithm: `SDCSA`. Values are p-values from the Wilcoxon signed-rank test. "
            "`*` means the comparison used fewer than 100 usable paired values.\n"
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    baseline_finals: Path,
    reference_finals: Path,
    graph_manifest: Path,
    output_dir: Path,
    exclude: set[str],
    rename: dict[str, str],
    algorithm_order: list[str],
    compared_algorithms: list[str],
    compared_raw_algorithms: list[str],
    alpha: float,
) -> None:
    lines = [
        f"baseline_finals={baseline_finals.resolve()}",
        f"reference_finals={reference_finals.resolve()}",
        f"graph_variant_manifest={graph_manifest.resolve()}",
        f"output_dir={output_dir.resolve()}",
        f"exclude_algorithms={sorted(exclude)}",
        f"rename_algorithm={rename}",
        f"algorithm_order={algorithm_order}",
        f"reference_algorithm_display={REFERENCE_DISPLAY}",
        f"reference_algorithm_raw={REFERENCE_RAW}",
        f"compared_algorithms_display={compared_algorithms}",
        f"compared_algorithms_raw={compared_raw_algorithms}",
        "graph_variant_sdsca_note=Final graph label 'SDCSA' came from baseline TSD, "
        "but this Wilcoxon analysis uses actual SDCSA finals as the reference algorithm.",
        "source_priority=raw run_*_per_gen.csv folders first; postprocessed finals CSV only as fallback",
        "test=Wilcoxon signed-rank",
        "alternative=two-sided",
        "zero_method=pratt",
        f"alpha={alpha}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Wilcoxon signed-rank results for SDCSA vs graph-variant baseline algorithms."
    )
    parser.add_argument(
        "--baseline-finals",
        type=str,
        default="test4_v2_results/summary_test4_v2_finals_postprocessed.csv",
        help="Baseline per-run finals CSV from test4_v2.",
    )
    parser.add_argument(
        "--reference-finals",
        type=str,
        default="test6_dmmo_adjsuted/sdcsa-substrate-drift-clonal-selection-algorithm/results_sdcsa_full_run01/summary_SDCSA_finals_postprocessed.csv",
        help="Actual SDCSA per-run finals CSV.",
    )
    parser.add_argument(
        "--graph-manifest",
        type=str,
        default="test4_v2_results/graphs_variant_sdcsa_v2/manifest_variant_rules.txt",
        help="Variant manifest that defines the final graph algorithm rules.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test4_v2_results/wilcoxon_variant_sdcsa_v2",
        help="Output directory for Wilcoxon CSV/Markdown summaries.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove previously generated CSV/Markdown/TXT files in the output directory.",
    )
    args = parser.parse_args()

    baseline_finals = Path(args.baseline_finals).resolve()
    reference_finals = Path(args.reference_finals).resolve()
    graph_manifest = Path(args.graph_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not baseline_finals.exists():
        raise FileNotFoundError(f"Baseline finals CSV not found: {baseline_finals}")
    if not reference_finals.exists():
        raise FileNotFoundError(f"Reference finals CSV not found: {reference_finals}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        for pattern in ("*.csv", "*.md", "*.txt"):
            for file_path in output_dir.glob(pattern):
                if file_path.name.lower() == "readme.md":
                    continue
                file_path.unlink(missing_ok=True)

    exclude, rename, algorithm_order = load_variant_rules(graph_manifest)
    compared_algorithms = [name for name in algorithm_order if name != REFERENCE_DISPLAY]
    allowed_display_names = set(compared_algorithms)

    baseline_rows = read_csv_rows(baseline_finals)
    reference_rows = read_csv_rows(reference_finals)
    baseline_results_root = baseline_finals.parent
    reference_results_root = reference_finals.parent

    reference_cases = build_reference_cases_from_run_dirs(reference_results_root)
    if not reference_cases:
        reference_cases = build_reference_cases(reference_rows)
    else:
        fallback_reference_cases = build_reference_cases(reference_rows)
        for key, case in fallback_reference_cases.items():
            reference_cases.setdefault(key, case)

    comparator_cases = build_comparator_cases_from_run_dirs(
        baseline_results_root,
        exclude=exclude,
        rename=rename,
        allowed_display_names=allowed_display_names,
    )
    if not comparator_cases:
        comparator_cases = build_comparator_cases(
            baseline_rows,
            exclude=exclude,
            rename=rename,
            allowed_display_names=allowed_display_names,
        )
    else:
        fallback_comparator_cases = build_comparator_cases(
            baseline_rows,
            exclude=exclude,
            rename=rename,
            allowed_display_names=allowed_display_names,
        )
        for key, case in fallback_comparator_cases.items():
            comparator_cases.setdefault(key, case)

    if not reference_cases:
        raise RuntimeError("No SDCSA reference cases were loaded from the reference finals CSV.")
    if not comparator_cases:
        raise RuntimeError("No comparator cases matched the graph-variant rules.")

    detailed_rows = build_detailed_rows(
        reference_cases=reference_cases,
        comparator_cases=comparator_cases,
        alpha=float(args.alpha),
        algorithm_order=algorithm_order,
    )
    summary_rows = build_summary_rows(detailed_rows, algorithm_order=algorithm_order)

    detailed_csv = output_dir / "wilcoxon_sdsca_vs_graph_variant_detailed.csv"
    detailed_md = output_dir / "wilcoxon_sdsca_vs_graph_variant_detailed.md"
    summary_csv = output_dir / "wilcoxon_sdsca_vs_graph_variant_summary.csv"
    summary_md = output_dir / "wilcoxon_sdsca_vs_graph_variant_summary.md"
    manifest_txt = output_dir / "manifest_wilcoxon_rules.txt"

    detailed_fields = [
        "reference_algorithm",
        "reference_raw_algorithm",
        "compared_algorithm",
        "compared_raw_algorithm",
        "benchmark",
        "benchmark_display",
        "dimension",
        "n_reference_runs",
        "n_compared_runs",
        "n_common_runs",
        "n_pairs",
        "n_reference_only_runs",
        "n_compared_only_runs",
        "n_excluded_nonfinite_pairs",
        "pair_coverage",
        "n_nonzero_diffs",
        "reference_mean_final",
        "compared_mean_final",
        "reference_median_final",
        "compared_median_final",
        "mean_delta_reference_minus_compared",
        "median_delta_reference_minus_compared",
        "wins_reference",
        "wins_compared",
        "ties",
        "wilcoxon_statistic",
        "p_value",
        "alpha",
        "significant",
        "better_algorithm_by_median_delta",
        "significant_better_algorithm",
        "test_status",
        "reference_source_runs_csv",
        "compared_source_runs_csv",
    ]
    summary_fields = [
        "reference_algorithm",
        "compared_algorithm",
        "compared_raw_algorithm",
        "total_cases",
        "complete_cases",
        "partial_cases",
        "all_pairs_equal_cases",
        "no_finite_pairs_cases",
        "too_few_pairs_cases",
        "significant_reference_better",
        "significant_compared_better",
        "not_significant",
        "min_pairs",
        "max_pairs",
        "mean_pairs",
    ]

    write_csv(detailed_csv, detailed_rows, detailed_fields)
    write_markdown_table(detailed_md, detailed_rows, detailed_fields[:-2])
    write_csv(summary_csv, summary_rows, summary_fields)
    write_markdown_table(summary_md, summary_rows, summary_fields)

    matrix_columns = ["benchmark_display", *compared_algorithms]
    dimensions_present = sorted({int(row["dimension"]) for row in detailed_rows})
    for dimension in dimensions_present:
        matrix_rows = build_pvalue_matrix_rows(
            detailed_rows=detailed_rows,
            dimension=dimension,
            compared_algorithms=compared_algorithms,
        )
        matrix_csv = output_dir / f"wilcoxon_sdsca_vs_graph_variant_pvalues_dim{dimension}.csv"
        matrix_md = output_dir / f"wilcoxon_sdsca_vs_graph_variant_pvalues_dim{dimension}.md"
        write_csv(matrix_csv, matrix_rows, matrix_columns)
        write_pvalue_matrix_markdown(
            matrix_md,
            caption=f"Wilcoxon signed-rank p-values for Dimension {dimension} (reference: SDCSA)",
            rows=matrix_rows,
            columns=matrix_columns,
        )

    compared_raw_algorithms = []
    seen_raw: set[str] = set()
    for row in detailed_rows:
        raw_name = str(row["compared_raw_algorithm"])
        if raw_name not in seen_raw:
            compared_raw_algorithms.append(raw_name)
            seen_raw.add(raw_name)

    write_manifest(
        manifest_txt,
        baseline_finals=baseline_finals,
        reference_finals=reference_finals,
        graph_manifest=graph_manifest,
        output_dir=output_dir,
        exclude=exclude,
        rename=rename,
        algorithm_order=algorithm_order,
        compared_algorithms=compared_algorithms,
        compared_raw_algorithms=compared_raw_algorithms,
        alpha=float(args.alpha),
    )

    print(f"Saved detailed Wilcoxon CSV: {detailed_csv}")
    print(f"Saved detailed Wilcoxon Markdown: {detailed_md}")
    print(f"Saved Wilcoxon summary CSV: {summary_csv}")
    print(f"Saved Wilcoxon summary Markdown: {summary_md}")
    print(f"Saved manifest: {manifest_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
