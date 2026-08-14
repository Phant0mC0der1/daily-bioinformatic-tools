#!/usr/bin/env python3
"""
Conserved Residue Finder
=========================
Compares multiple aligned FASTA sequences (protein or nucleotide) and
identifies/highlights conserved residues at each alignment position.

Usage:
    python conserved_residue_finder.py input.fasta [--threshold 0.8] [--out report.txt]

Input:
    A FASTA file containing 2+ sequences that are already aligned
    (i.e. all sequences must be the same length, gaps represented by "-").

Output:
    - A console/text report showing, for each alignment column:
        * the consensus residue
        * conservation score (fraction of sequences matching consensus)
        * whether the column is "conserved" (>= threshold) or "strictly conserved" (100%)
    - A simple annotated alignment view marking conserved columns with "*"
      (strict conservation) or ":" (>= threshold, not strict), similar to
      ClustalW-style conservation lines.
    - Optionally saves the report to a file with --out.
"""

import argparse
import sys
from collections import Counter


def read_fasta(path):
    sequences = {}
    name = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def validate_alignment(sequences):
    lengths = {len(seq) for seq in sequences.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequences are not the same length ({sorted(lengths)}). "
            "Input must be a pre-aligned FASTA (e.g. from MUSCLE/Clustal/MAFFT)."
        )


def compute_conservation(sequences, threshold):
    names = list(sequences.keys())
    alignment_length = len(next(iter(sequences.values())))
    n_seqs = len(names)

    columns = []
    for pos in range(alignment_length):
        residues = [sequences[name][pos] for name in names]
        non_gap = [r for r in residues if r != "-"]
        if not non_gap:
            columns.append({
                "position": pos + 1,
                "consensus": "-",
                "score": 0.0,
                "strict": False,
                "conserved": False,
                "counts": Counter(residues),
            })
            continue

        counts = Counter(residues)
        consensus, top_count = counts.most_common(1)[0]
        score = top_count / n_seqs
        columns.append({
            "position": pos + 1,
            "consensus": consensus,
            "score": score,
            "strict": score == 1.0 and consensus != "-",
            "conserved": score >= threshold,
            "counts": counts,
        })
    return columns


def build_conservation_line(columns):
    symbols = []
    for col in columns:
        if col["strict"]:
            symbols.append("*")
        elif col["conserved"]:
            symbols.append(":")
        else:
            symbols.append(" ")
    return "".join(symbols)


def format_report(sequences, columns, threshold):
    lines = []
    n_seqs = len(sequences)
    alignment_length = len(columns)
    strict_count = sum(1 for c in columns if c["strict"])
    conserved_count = sum(1 for c in columns if c["conserved"])

    lines.append("Conserved Residue Finder Report")
    lines.append("=" * 40)
    lines.append(f"Sequences analysed : {n_seqs}")
    lines.append(f"Alignment length   : {alignment_length}")
    lines.append(f"Conservation threshold : {threshold:.0%}")
    lines.append(f"Strictly conserved columns (100%) : {strict_count}")
    lines.append(f"Conserved columns (>= threshold)  : {conserved_count}")
    lines.append("")

    lines.append("Alignment view (blocks of 60, with conservation line):")
    lines.append("  * = strictly conserved (100% identical, no gaps)")
    lines.append("  : = conserved (>= threshold, not strict)")
    lines.append("")

    names = list(sequences.keys())
    name_width = max(len(n) for n in names) + 2
    conservation_line = build_conservation_line(columns)
    aligned_length = len(next(iter(sequences.values())))

    block_size = 60
    for start in range(0, aligned_length, block_size):
        end = min(start + block_size, aligned_length)
        for name in names:
            seq_block = sequences[name][start:end]
            lines.append(f"{name:<{name_width}}{seq_block}")
        cons_block = conservation_line[start:end]
        lines.append(f"{'':<{name_width}}{cons_block}")
        lines.append("")

    lines.append("Per-position detail for conserved columns:")
    lines.append(f"{'Pos':>6}  {'Consensus':^9}  {'Score':>7}  Strict  Residue counts")
    for col in columns:
        if col["conserved"]:
            counts_str = ", ".join(f"{res}:{n}" for res, n in col["counts"].most_common())
            lines.append(
                f"{col['position']:>6}  {col['consensus']:^9}  {col['score']:>6.1%}  "
                f"{'yes' if col['strict'] else 'no':<6}  {counts_str}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Find and highlight conserved residues across aligned FASTA sequences.")
    parser.add_argument("fasta", help="Path to a pre-aligned FASTA file (2+ sequences, equal length).")
    parser.add_argument("--threshold", type=float, default=0.8,
                         help="Minimum fraction of sequences sharing a residue to call the column conserved (default: 0.8).")
    parser.add_argument("--out", help="Optional path to save the text report. If omitted, prints to stdout.")
    args = parser.parse_args()

    try:
        sequences = read_fasta(args.fasta)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.fasta}")

    if len(sequences) < 2:
        sys.exit("Error: need at least 2 sequences in the FASTA file to compare.")

    try:
        validate_alignment(sequences)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    if not (0.0 < args.threshold <= 1.0):
        sys.exit("Error: --threshold must be between 0 and 1.")

    columns = compute_conservation(sequences, args.threshold)
    report = format_report(sequences, columns, args.threshold)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
