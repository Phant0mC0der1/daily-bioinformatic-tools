#!/usr/bin/env python3
"""
Mutation Spectrum Analyzer

Counts mutations by functional class (Missense, Nonsense, Silent,
Frameshift, Splice) from a MAF/CSV mutation annotation file and
visualises their proportions as a bar chart and a pie chart.

Usage:
    python mutation_spectrum_analyzer.py --input mutations.maf.csv
    python mutation_spectrum_analyzer.py --input mutations.csv --column Consequence
    python mutation_spectrum_analyzer.py --demo

Input file requirements:
    A tab- or comma-delimited table (MAF-like) containing a column with
    per-variant functional annotations, e.g. "Variant_Classification"
    (standard MAF) or a VEP/SnpEff-style "Consequence" column.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORY_ORDER = ["Missense", "Nonsense", "Silent", "Frameshift", "Splice", "Other"]

CLASSIFICATION_MAP = {
    "missense_mutation": "Missense",
    "missense_variant": "Missense",
    "nonsense_mutation": "Nonsense",
    "stop_gained": "Nonsense",
    "nonstop_mutation": "Nonsense",
    "stop_lost": "Nonsense",
    "silent": "Silent",
    "synonymous_variant": "Silent",
    "synonymous_snv": "Silent",
    "frame_shift_ins": "Frameshift",
    "frame_shift_del": "Frameshift",
    "frameshift_variant": "Frameshift",
    "frameshift_insertion": "Frameshift",
    "frameshift_deletion": "Frameshift",
    "splice_site": "Splice",
    "splice_region": "Splice",
    "splice_region_variant": "Splice",
    "splice_donor_variant": "Splice",
    "splice_acceptor_variant": "Splice",
}


def classify(raw_value):
    key = raw_value.strip().lower()
    return CLASSIFICATION_MAP.get(key, "Other")


def sniff_delimiter(path):
    with open(path, newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def find_column(fieldnames, requested):
    if requested:
        for name in fieldnames:
            if name.lower() == requested.lower():
                return name
        raise ValueError(f"Column '{requested}' not found. Available columns: {fieldnames}")
    candidates = ["variant_classification", "consequence", "effect", "mutation_type"]
    for candidate in candidates:
        for name in fieldnames:
            if name.lower() == candidate:
                return name
    raise ValueError(
        f"Could not auto-detect a classification column. Available columns: {fieldnames}. "
        "Pass --column to specify one explicitly."
    )


def load_counts(path, column):
    delimiter = sniff_delimiter(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row.")
        col = find_column(reader.fieldnames, column)
        counts = Counter()
        for row in reader:
            raw = row.get(col, "")
            if raw is None or raw.strip() == "":
                continue
            counts[classify(raw)] += 1
    return counts


def make_demo_counts():
    return Counter({
        "Missense": 142,
        "Nonsense": 21,
        "Silent": 63,
        "Frameshift": 18,
        "Splice": 9,
        "Other": 7,
    })


def plot_spectrum(counts, out_prefix):
    ordered = [c for c in CATEGORY_ORDER if counts.get(c, 0) > 0]
    values = [counts[c] for c in ordered]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(ordered, values, color="#4C72B0")
    ax.set_ylabel("Mutation count")
    ax.set_title("Mutation Spectrum")
    for bar, value in zip(bars, values):
        pct = 100 * value / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    bar_path = f"{out_prefix}_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=ordered, autopct="%1.1f%%", startangle=90)
    ax.set_title("Mutation Spectrum Proportions")
    fig.tight_layout()
    pie_path = f"{out_prefix}_pie.png"
    fig.savefig(pie_path, dpi=150)
    plt.close(fig)

    return bar_path, pie_path


def write_summary(counts, out_prefix):
    total = sum(counts.values())
    summary_path = f"{out_prefix}_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Category", "Count", "Proportion"])
        for category in CATEGORY_ORDER:
            count = counts.get(category, 0)
            if count == 0:
                continue
            writer.writerow([category, count, f"{count / total:.4f}"])
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Count and visualise mutation spectrum by functional class.")
    parser.add_argument("--input", type=Path, help="Path to a MAF/CSV/TSV file with a mutation classification column.")
    parser.add_argument("--column", help="Name of the classification column (auto-detected if omitted).")
    parser.add_argument("--out-prefix", default="mutation_spectrum", help="Prefix for output files.")
    parser.add_argument("--demo", action="store_true", help="Run with built-in synthetic demo data instead of an input file.")
    args = parser.parse_args()

    if args.demo:
        counts = make_demo_counts()
    elif args.input:
        counts = load_counts(args.input, args.column)
    else:
        parser.error("Provide --input <file> or use --demo.")

    if sum(counts.values()) == 0:
        sys.exit("No classifiable mutations found in the input.")

    summary_path = write_summary(counts, args.out_prefix)
    bar_path, pie_path = plot_spectrum(counts, args.out_prefix)

    total = sum(counts.values())
    print(f"Total mutations classified: {total}")
    for category in CATEGORY_ORDER:
        count = counts.get(category, 0)
        if count:
            print(f"  {category:10s}: {count:6d} ({100 * count / total:5.1f}%)")
    print(f"Summary written to {summary_path}")
    print(f"Bar chart written to {bar_path}")
    print(f"Pie chart written to {pie_path}")


if __name__ == "__main__":
    main()
