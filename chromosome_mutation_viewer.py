#!/usr/bin/env python3
"""
Chromosome Mutation Viewer

Shows mutation density across chromosomes from a VCF file or a plain
chromosome/position table. Mutations are counted into fixed-size genomic
bins per chromosome and rendered as a stacked density track (one row per
chromosome), similar to a simple ideogram view.

Usage:
    python chromosome_mutation_viewer.py --input variants.vcf
    python chromosome_mutation_viewer.py --input mutations.csv --chrom-column Chr --pos-column Position
    python chromosome_mutation_viewer.py --input mutations.vcf --bin-size 500000
    python chromosome_mutation_viewer.py --demo

Input file requirements:
    Either a standard VCF (lines starting with "#CHROM" define the header;
    the first two columns are CHROM and POS), or a comma/tab-delimited
    table with a chromosome column and a position column (names are
    auto-detected, or supply --chrom-column / --pos-column explicitly).

Note:
    Chromosome extents are inferred from the highest observed mutation
    position per chromosome (no reference genome is required), so bins
    near the true chromosome end may be under-represented if mutations
    are sparse there.
"""

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHROM_ORDER_SPECIAL = {"X": 100, "Y": 101, "MT": 102, "M": 102}


def chrom_sort_key(chrom):
    name = chrom.lower().replace("chr", "").upper()
    if name in CHROM_ORDER_SPECIAL:
        return (CHROM_ORDER_SPECIAL[name], name)
    try:
        return (int(name), name)
    except ValueError:
        return (999, name)


def normalise_chrom(chrom):
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def is_vcf(path):
    if path.suffix.lower() in (".vcf",):
        return True
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            return line.startswith("#CHROM")
    return False


def load_mutations_vcf(path):
    mutations = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            chrom, pos = fields[0], fields[1]
            try:
                mutations.append((normalise_chrom(chrom), int(pos)))
            except ValueError:
                continue
    return mutations


def sniff_delimiter(path):
    with open(path, newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def find_column(fieldnames, requested, candidates):
    if requested:
        for name in fieldnames:
            if name.lower() == requested.lower():
                return name
        raise ValueError(f"Column '{requested}' not found. Available columns: {fieldnames}")
    for candidate in candidates:
        for name in fieldnames:
            if name.lower() == candidate:
                return name
    raise ValueError(
        f"Could not auto-detect a required column from {candidates}. "
        f"Available columns: {fieldnames}. Pass --chrom-column/--pos-column explicitly."
    )


def load_mutations_table(path, chrom_column, pos_column):
    delimiter = sniff_delimiter(path)
    mutations = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row.")
        chrom_col = find_column(reader.fieldnames, chrom_column, ["chrom", "chr", "chromosome"])
        pos_col = find_column(reader.fieldnames, pos_column, ["pos", "position", "start"])
        for row in reader:
            chrom, pos = row.get(chrom_col), row.get(pos_col)
            if not chrom or not pos:
                continue
            try:
                mutations.append((normalise_chrom(str(chrom).strip()), int(re.sub(r"[^\d]", "", str(pos)))))
            except ValueError:
                continue
    return mutations


def load_mutations(path, chrom_column, pos_column):
    if is_vcf(path):
        return load_mutations_vcf(path)
    return load_mutations_table(path, chrom_column, pos_column)


def make_demo_mutations():
    random.seed(42)
    mutations = []
    chrom_lengths = {"chr1": 249_000_000, "chr2": 243_000_000, "chr7": 159_000_000,
                      "chr17": 83_000_000, "chrX": 156_000_000}
    hotspots = {"chr1": [30_000_000], "chr7": [55_000_000, 140_000_000], "chr17": [7_500_000]}
    for chrom, length in chrom_lengths.items():
        n_background = random.randint(150, 300)
        for _ in range(n_background):
            mutations.append((chrom, random.randint(1, length)))
        for centre in hotspots.get(chrom, []):
            for _ in range(random.randint(80, 150)):
                pos = int(random.gauss(centre, 2_000_000))
                if 1 <= pos <= length:
                    mutations.append((chrom, pos))
    return mutations


def bin_mutations(mutations, bin_size):
    counts = defaultdict(lambda: defaultdict(int))
    for chrom, pos in mutations:
        bin_index = pos // bin_size
        counts[chrom][bin_index] += 1
    return counts


def write_summary(counts, bin_size, out_prefix):
    summary_path = f"{out_prefix}_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chrom", "bin_start", "bin_end", "mutation_count"])
        for chrom in sorted(counts, key=chrom_sort_key):
            for bin_index in sorted(counts[chrom]):
                start = bin_index * bin_size
                writer.writerow([chrom, start, start + bin_size, counts[chrom][bin_index]])
    return summary_path


def plot_density(counts, bin_size, out_prefix):
    chroms = sorted(counts, key=chrom_sort_key)
    fig, axes = plt.subplots(len(chroms), 1, figsize=(10, 1.1 * len(chroms)), sharex=False)
    if len(chroms) == 1:
        axes = [axes]

    max_count = max(count for chrom_counts in counts.values() for count in chrom_counts.values())

    for ax, chrom in zip(axes, chroms):
        chrom_counts = counts[chrom]
        max_bin = max(chrom_counts)
        xs = list(range(max_bin + 1))
        ys = [chrom_counts.get(b, 0) for b in xs]
        positions_mb = [x * bin_size / 1_000_000 for x in xs]
        ax.bar(positions_mb, ys, width=bin_size / 1_000_000, color="#C44E52", align="edge")
        ax.set_ylim(0, max_count * 1.1)
        ax.set_ylabel(chrom.replace("chr", ""), rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Position (Mb)")
    fig.suptitle(f"Mutation Density Across Chromosomes (bin size = {bin_size:,} bp)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = f"{out_prefix}_density.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Show mutation density across chromosomes.")
    parser.add_argument("--input", type=Path, help="Path to a VCF file or a chromosome/position table (CSV/TSV).")
    parser.add_argument("--chrom-column", help="Name of the chromosome column (auto-detected if omitted).")
    parser.add_argument("--pos-column", help="Name of the position column (auto-detected if omitted).")
    parser.add_argument("--bin-size", type=int, default=1_000_000, help="Genomic bin size in bp (default: 1,000,000).")
    parser.add_argument("--out-prefix", default="chromosome_mutation_viewer", help="Prefix for output files.")
    parser.add_argument("--demo", action="store_true", help="Run with built-in synthetic demo data instead of an input file.")
    args = parser.parse_args()

    if args.demo:
        mutations = make_demo_mutations()
    elif args.input:
        mutations = load_mutations(args.input, args.chrom_column, args.pos_column)
    else:
        parser.error("Provide --input <file> or use --demo.")

    if not mutations:
        sys.exit("No mutations with a valid chromosome and position were found in the input.")

    counts = bin_mutations(mutations, args.bin_size)
    summary_path = write_summary(counts, args.bin_size, args.out_prefix)
    plot_path = plot_density(counts, args.bin_size, args.out_prefix)

    total = len(mutations)
    print(f"Total mutations: {total}")
    for chrom in sorted(counts, key=chrom_sort_key):
        chrom_total = sum(counts[chrom].values())
        print(f"  {chrom:8s}: {chrom_total:6d} mutations across {len(counts[chrom])} bins")
    print(f"Summary written to {summary_path}")
    print(f"Density plot written to {plot_path}")


if __name__ == "__main__":
    main()
