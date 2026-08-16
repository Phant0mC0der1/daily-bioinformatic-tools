#!/usr/bin/env python3
"""
GC Content Analyzer
--------------------
Computes sliding-window GC% across one or more DNA sequences in a FASTA file
and plots the result.

Usage:
    python gc_content_analyzer.py input.fasta [-w WINDOW] [-s STEP] [-o OUTPUT_PREFIX]

Outputs:
    <prefix>_gc_content.csv   - per-window GC% for every sequence
    <prefix>_gc_content.png   - line plot of GC% along each sequence
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_fasta(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.upper())
    if header is not None:
        yield header, "".join(seq_chunks)


def sliding_gc_content(sequence, window, step):
    """Return (positions, gc_percentages) for a sliding window over sequence."""
    positions = []
    gc_percentages = []
    seq_len = len(sequence)
    if seq_len < window:
        return positions, gc_percentages
    for start in range(0, seq_len - window + 1, step):
        chunk = sequence[start:start + window]
        gc_count = chunk.count("G") + chunk.count("C")
        gc_percent = 100.0 * gc_count / window
        positions.append(start + 1)  # 1-based start position
        gc_percentages.append(gc_percent)
    return positions, gc_percentages


def main():
    parser = argparse.ArgumentParser(description="Sliding-window GC content analyzer")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("-w", "--window", type=int, default=100, help="Window size (default: 100)")
    parser.add_argument("-s", "--step", type=int, default=10, help="Step size (default: 10)")
    parser.add_argument("-o", "--output-prefix", default="gc_content", help="Output file prefix")
    args = parser.parse_args()

    if args.window <= 0 or args.step <= 0:
        sys.exit("Window and step sizes must be positive integers.")

    records = list(parse_fasta(args.fasta))
    if not records:
        sys.exit(f"No sequences found in {args.fasta}")

    csv_path = Path(f"{args.output_prefix}_gc_content.csv")
    plot_path = Path(f"{args.output_prefix}_gc_content.png")

    fig, ax = plt.subplots(figsize=(10, 5))

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sequence_id", "window_start", "gc_percent"])

        for header, sequence in records:
            seq_id = header.split()[0] if header else "sequence"
            positions, gc_percentages = sliding_gc_content(sequence, args.window, args.step)

            if not positions:
                print(f"Skipping '{seq_id}': sequence shorter than window size ({args.window}).")
                continue

            for pos, gc in zip(positions, gc_percentages):
                writer.writerow([seq_id, pos, f"{gc:.2f}"])

            ax.plot(positions, gc_percentages, label=seq_id, linewidth=1)

    overall_gc_values = []
    for _, sequence in records:
        gc_count = sequence.count("G") + sequence.count("C")
        if sequence:
            overall_gc_values.append(100.0 * gc_count / len(sequence))
    if overall_gc_values:
        mean_gc = sum(overall_gc_values) / len(overall_gc_values)
        ax.axhline(mean_gc, color="gray", linestyle="--", linewidth=1,
                    label=f"Mean overall GC% ({mean_gc:.1f}%)")

    ax.set_xlabel("Window start position (bp)")
    ax.set_ylabel("GC content (%)")
    ax.set_title(f"Sliding-Window GC Content (window={args.window}, step={args.step})")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
