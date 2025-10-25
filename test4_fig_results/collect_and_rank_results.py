#!/usr/bin/env python3
"""
collect_and_rank_results.py

python ./collect_and_rank_results.py "./dim_2" -o "./dim2_rank.txt"

Usage:
    python collect_and_rank_results.py <directory> [--out OUTPUT_FILE]

This script finds all JSON summary files in the given directory (non-recursive),
extracts mean_final, max_final and min_final values (preserving their non-scientific
decimal string representations when possible), writes a summary text file in the
directory, and prints/ranks entries by closeness of mean_final to known global
optima for the benchmark function.

The script is defensive: it first tries to capture the numeric values as strings
from the raw JSON file (so formatting is preserved). If not found it falls back
to loading JSON and formatting numbers with Decimal to avoid scientific notation.

Example:
    python collect_and_rank_results.py "e:/.../test4_fig_results/dim_2"

Output: writes `collected_results.txt` inside the given directory by default.
"""

from __future__ import annotations
import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import List, Dict, Optional, Tuple


NUM_RE = re.compile(r'([-+]?(?:\d+\.\d+|\d+|\d*\.\d+)(?:[eE][-+]?\d+)?)')
FIELD_RE = lambda name: re.compile(r'"' + re.escape(name) + r'"\s*:\s*' + NUM_RE.pattern)


# Known optimal values for benchmarks (from user-provided table)
OPTIMA: Dict[str, Decimal] = {
    'schaffer n.2': Decimal('0'),
    'schaffer n.4': Decimal('0.292579'),
    'ackley': Decimal('0'),
    'griewank': Decimal('0'),
    'rastrigin': Decimal('0'),
    'shubert': Decimal('-186.7309'),
    'eggholder': Decimal('-959.6407'),
    'holdertable': Decimal('-19.2085'),
    'schwefel': Decimal('0'),
    'levy': Decimal('0'),
}


def normalize_name(s: str) -> str:
    """Normalize benchmark/keys to lowercase alphanumeric string.

    Examples:
      'Schaffer n.2' -> 'schaffern2'
      'SchafferN2'   -> 'schaffern2'
    """
    if not s:
        return ''
    return ''.join(ch for ch in s.lower() if ch.isalnum())


# Precompute a normalized-optima map for robust matching
OPTIMA_NORMALIZED: Dict[str, Decimal] = {normalize_name(k): v for k, v in OPTIMA.items()}


def extract_numeric_strings_from_text(text: str, keys: List[str]) -> Dict[str, Optional[str]]:
    out = {}
    for k in keys:
        m = FIELD_RE(k).search(text)
        if m:
            out[k] = m.group(1)
        else:
            out[k] = None
    return out


def format_decimal_no_sci(value) -> str:
    """Return a decimal string without scientific notation.

    Accepts Decimal, int, float or numeric string.
    """
    try:
        d = Decimal(str(value)) if not isinstance(value, Decimal) else value
        # Use 'f' format to avoid exponent form
        return format(d, 'f')
    except (InvalidOperation, ValueError):
        return str(value)


def process_file(path: str) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    # try to extract as strings from raw text first (preserve formatting)
    keys = ['mean_final', 'max_final', 'min_final']
    numeric_strings = extract_numeric_strings_from_text(text, keys)

    # load json for other fields (benchmark, algorithm)
    try:
        data = json.loads(text)
    except Exception:
        data = {}

    algorithm = data.get('algorithm') or data.get('algo') or ''
    benchmark = data.get('benchmark') or data.get('benchmark_display') or data.get('function') or ''
    dim = data.get('dim') or data.get('dimension')

    # if any numeric strings missing, fallback to parsed JSON numbers with Decimal formatting
    for k in keys:
        if numeric_strings.get(k) is None:
            val = data.get(k)
            if val is None:
                numeric_strings[k] = None
            else:
                numeric_strings[k] = format_decimal_no_sci(val)

    # store also a Decimal for mean_final (if possible) for ranking
    mean_decimal: Optional[Decimal] = None
    if numeric_strings.get('mean_final') is not None:
        try:
            mean_decimal = Decimal(numeric_strings['mean_final'])
        except InvalidOperation:
            try:
                mean_decimal = Decimal(str(data.get('mean_final')))
            except Exception:
                mean_decimal = None

    return {
        'path': path,
        'algorithm': algorithm,
        'benchmark': benchmark,
        'dim': dim,
        'mean_str': numeric_strings.get('mean_final'),
        'max_str': numeric_strings.get('max_final'),
        'min_str': numeric_strings.get('min_final'),
        'mean_decimal': mean_decimal,
    }


def rank_for_benchmark(entries: List[Dict]) -> List[Tuple[Dict, Optional[Decimal]]]:
    """Given a list of entries for the same benchmark, compute abs(mean - opt)
    where opt is known. Returns list of (entry, diff) sorted by diff ascending.
    If optimum is unknown or mean missing, diff will be None and those entries
    will be placed after the ranked ones.
    """
    if not entries:
        return []

    bench_name = (entries[0].get('benchmark') or '').strip()
    norm = normalize_name(bench_name)
    opt = OPTIMA_NORMALIZED.get(norm)
    # fallback: if not exact normalized match, try substring matches in normalized space
    if opt is None:
        for k_norm, v in OPTIMA_NORMALIZED.items():
            if k_norm and (norm.startswith(k_norm) or k_norm.startswith(norm)):
                opt = v
                break

    ranked: List[Tuple[Dict, Optional[Decimal]]] = []
    for e in entries:
        if opt is not None and e.get('mean_decimal') is not None:
            diff = abs(e['mean_decimal'] - opt)
        else:
            diff = None
        ranked.append((e, diff))

    ranked.sort(key=lambda pair: (pair[1] is None, pair[1] if pair[1] is not None else Decimal('1E50')))
    return ranked


def write_simple_output(out_path: str, entries: List[Dict]):
    """Write a simple, readable text file grouped by benchmark. For each
    benchmark print algorithm mean/max/min lines and rank algorithms against the
    known optimum for that benchmark (if available).
    """
    # group entries by benchmark (use display name)
    groups: Dict[str, List[Dict]] = {}
    dims_by_bench: Dict[str, set] = {}
    for e in entries:
        bench = (e.get('benchmark') or 'unknown').strip()
        groups.setdefault(bench, []).append(e)
        dims_by_bench.setdefault(bench, set())
        if e.get('dim') is not None:
            dims_by_bench[bench].add(str(e.get('dim')))

    lines: List[str] = []
    for bench, items in sorted(groups.items()):
        dims = ','.join(sorted(dims_by_bench.get(bench, set()))) or 'unknown'
        lines.append(f"=== Benchmark: {bench} | Dimension: {dims} ===")
        algo_names = [ (it.get('algorithm') or 'unknown') for it in items ]
        lines.append(f"  Algorithms: {algo_names}")
        # print each algorithm's stats
        for it in items:
            algo = it.get('algorithm') or 'unknown'
            mean_s = it.get('mean_str') or 'N/A'
            max_s = it.get('max_str') or 'N/A'
            min_s = it.get('min_str') or 'N/A'
            lines.append(f"  {algo}: mean={mean_s}, max={max_s}, min={min_s}")

        # ranking for this benchmark
        ranked = rank_for_benchmark(items)
        # determine opt if available
        norm = normalize_name(bench)
        opt = OPTIMA_NORMALIZED.get(norm)
        if opt is None:
            for k_norm, v in OPTIMA_NORMALIZED.items():
                if k_norm and (norm.startswith(k_norm) or k_norm.startswith(norm)):
                    opt = v
                    break

        if opt is not None:
            lines.append(f"  Ranking to optimal value (opt={format_decimal_no_sci(opt)}):")
            r = 1
            for it, diff in ranked:
                if diff is None:
                    # can't rank this entry
                    lines.append(f"   {r}. {it.get('algorithm') or 'unknown'}: diff=N/A, mean={it.get('mean_str') or 'N/A'}")
                else:
                    lines.append(f"   {r}. {it.get('algorithm') or 'unknown'}: diff={format_decimal_no_sci(diff)}, mean={it.get('mean_str') or 'N/A'}")
                r += 1
        else:
            lines.append('  No known optimum for this benchmark — skipping ranking.')

        lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Wrote simple summary to {out_path} ({len(entries)} files, {len(groups)} benchmarks)")


def main():
    parser = argparse.ArgumentParser(description='Collect mean/max/min from JSON files and rank by closeness to optima')
    parser.add_argument('directory', help='Directory containing JSON summary files (non-recursive)')
    parser.add_argument('--out', '-o', help='Output text file path (defaults to <directory>/collected_results.txt)')
    args = parser.parse_args()

    d = args.directory
    if not os.path.isdir(d):
        print(f"Error: directory not found: {d}")
        return

    json_files = [os.path.join(d, fn) for fn in os.listdir(d) if fn.lower().endswith('.json')]
    json_files.sort()

    if not json_files:
        print(f"No JSON files found in {d}")
        return

    entries = []
    for p in json_files:
        r = process_file(p)
        if r:
            entries.append(r)

    # write grouped, simple output and per-benchmark rankings
    out_path = args.out or os.path.join(d, 'collected_results.txt')
    write_simple_output(out_path, entries)


if __name__ == '__main__':
    main()
