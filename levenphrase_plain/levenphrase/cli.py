"""Command-line preview for Metric LevenPhrase."""

from __future__ import annotations

import argparse

from .preview import preview_file, print_preview_table, write_results_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric LevenPhrase batch preview")
    parser.add_argument("path", help="Text file to load")
    parser.add_argument("--format", default="auto", choices=["auto", "tsv", "csv", "pipe", "pairs", "triples", "single"])
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--mode", default="wyrazow", choices=["liter", "znakow", "glosek", "wyrazow"])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gamma-index", type=int, default=2, help="Which gamma score to show in compact table; default 2 means gamma=0")
    parser.add_argument("--results-file", default=None, help="Write MATLAB-friendly results file, e.g. results1.txt")
    parser.add_argument("--results-kind", type=int, default=1, help="Numeric kind/id column written to results file; historically 1=letters, 2=glosek")
    parser.add_argument("--append-results", action="store_true", help="Append to --results-file instead of overwriting")
    parser.add_argument("--no-results-header", action="store_true", help="Do not write commented header lines to results file")
    args = parser.parse_args()

    rows = preview_file(
        args.path,
        format=args.format,
        encoding=args.encoding,
        language=args.language,
        mode=args.mode,
        epsilon=args.epsilon,
    )
    print_preview_table(rows, gamma_index=args.gamma_index)
    if args.results_file:
        write_results_file(
            rows,
            args.results_file,
            gamma_values=(-1.0, -0.5, 0.0, 0.5, 1.0),
            kind=args.results_kind,
            append=args.append_results,
            include_header=not args.no_results_header,
        )
        print(f"\nSaved MATLAB-friendly results to: {args.results_file}")


if __name__ == "__main__":
    main()
