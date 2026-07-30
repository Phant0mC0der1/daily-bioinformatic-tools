#!/usr/bin/env python3
"""
Cancer Mutation Dashboard

Summarises a cohort-level cancer mutation table (MAF/CSV) into a
one-stop dashboard: most-mutated genes, per-sample mutation
frequencies, and a gene x sample mutation heatmap.

Usage:
    python cancer_mutation_dashboard.py --input mutations.maf.csv
    python cancer_mutation_dashboard.py --input mutations.csv \
        --gene-column Hugo_Symbol --sample-column Tumor_Sample_Barcode
    python cancer_mutation_dashboard.py --demo

Input file requirements:
    A tab- or comma-delimited table with at least a gene column
    (e.g. "Hugo_Symbol", standard MAF) and a sample column
    (e.g. "Tumor_Sample_Barcode"). A variant classification column
    (e.g. "Variant_Classification") is used if present to annotate
    the top-genes report, but is not required.
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GENE_CANDIDATES = ["hugo_symbol", "gene", "gene_name", "gene_symbol"]
SAMPLE_CANDIDATES = ["tumor_sample_barcode", "sample", "sample_id", "sample_name"]
CLASS_CANDIDATES = ["variant_classification", "consequence", "effect", "mutation_type"]


def sniff_delimiter(path):
    with open(path, newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return ","


def find_column(fieldnames, requested, candidates, label):
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
        f"Could not auto-detect a {label} column. Available columns: {fieldnames}. "
        f"Pass --{label.replace(' ', '-')}-column to specify one explicitly."
    )


def load_mutations(path, gene_column, sample_column, class_column):
    delimiter = sniff_delimiter(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row.")
        gene_col = find_column(reader.fieldnames, gene_column, GENE_CANDIDATES, "gene")
        sample_col = find_column(reader.fieldnames, sample_column, SAMPLE_CANDIDATES, "sample")
        try:
            class_col = find_column(reader.fieldnames, class_column, CLASS_CANDIDATES, "class")
        except ValueError:
            class_col = None

        records = []
        for row in reader:
            gene = (row.get(gene_col) or "").strip()
            sample = (row.get(sample_col) or "").strip()
            if not gene or not sample:
                continue
            variant_class = (row.get(class_col) or "").strip() if class_col else ""
            records.append((gene, sample, variant_class))
    return records


def make_demo_records():
    genes = ["TP53", "KRAS", "PIK3CA", "PTEN", "EGFR", "BRAF", "APC", "MYC", "RB1", "NRAS"]
    samples = [f"TCGA-{i:03d}" for i in range(1, 21)]
    classes = ["Missense_Mutation", "Nonsense_Mutation", "Silent", "Frame_Shift_Del", "Splice_Site"]

    rng = random.Random(42)
    weights = {gene: rng.uniform(0.1, 1.0) for gene in genes}
    records = []
    for sample in samples:
        n_mutations = rng.randint(2, 8)
        chosen = rng.choices(genes, weights=[weights[g] for g in genes], k=n_mutations)
        for gene in set(chosen):
            records.append((gene, sample, rng.choice(classes)))
    return records


def summarise(records, top_n):
    gene_sample_hits = defaultdict(set)
    gene_class_counts = defaultdict(Counter)
    for gene, sample, variant_class in records:
        gene_sample_hits[gene].add(sample)
        if variant_class:
            gene_class_counts[gene][variant_class] += 1

    n_samples = len({sample for _, sample, _ in records})
    gene_freq = {
        gene: len(samples) / n_samples if n_samples else 0.0
        for gene, samples in gene_sample_hits.items()
    }
    top_genes = sorted(gene_freq, key=lambda g: (-gene_freq[g], g))[:top_n]
    return top_genes, gene_freq, gene_sample_hits, gene_class_counts, n_samples


def write_gene_summary(top_genes, gene_freq, gene_sample_hits, gene_class_counts, out_prefix, n_samples):
    summary_path = f"{out_prefix}_top_genes.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Gene", "Mutated_Samples", "Total_Samples", "Frequency", "Top_Classification"])
        for gene in top_genes:
            n_hits = len(gene_sample_hits[gene])
            top_class = gene_class_counts[gene].most_common(1)
            top_class_name = top_class[0][0] if top_class else "NA"
            writer.writerow([gene, n_hits, n_samples, f"{gene_freq[gene]:.4f}", top_class_name])
    return summary_path


def write_sample_summary(records, out_prefix):
    sample_counts = Counter(sample for _, sample, _ in records)
    summary_path = f"{out_prefix}_sample_frequencies.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Sample", "Mutation_Count"])
        for sample, count in sorted(sample_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([sample, count])
    return summary_path, sample_counts


def plot_top_genes(top_genes, gene_freq, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [gene_freq[g] * 100 for g in top_genes]
    bars = ax.barh(top_genes[::-1], values[::-1], color="#C44E52")
    ax.set_xlabel("Samples mutated (%)")
    ax.set_title("Most Frequently Mutated Genes")
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = f"{out_prefix}_top_genes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap(records, top_genes, out_prefix):
    samples = sorted({sample for _, sample, _ in records})
    hit_set = {(gene, sample) for gene, sample, _ in records}
    matrix = np.array([[1 if (gene, sample) in hit_set else 0 for sample in samples] for gene in top_genes])

    fig_width = max(6, 0.35 * len(samples))
    fig, ax = plt.subplots(figsize=(fig_width, max(4, 0.4 * len(top_genes))))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes)
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=90, fontsize=6)
    ax.set_title("Gene x Sample Mutation Heatmap")
    fig.colorbar(im, ax=ax, label="Mutated", shrink=0.6)
    fig.tight_layout()
    path = f"{out_prefix}_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Summarise cohort cancer mutations into a gene/sample dashboard.")
    parser.add_argument("--input", type=Path, help="Path to a MAF/CSV/TSV mutation file.")
    parser.add_argument("--gene-column", help="Name of the gene column (auto-detected if omitted).")
    parser.add_argument("--sample-column", help="Name of the sample column (auto-detected if omitted).")
    parser.add_argument("--class-column", help="Name of the variant classification column (optional).")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top mutated genes to report/plot.")
    parser.add_argument("--out-prefix", default="cancer_mutation_dashboard", help="Prefix for output files.")
    parser.add_argument("--demo", action="store_true", help="Run with built-in synthetic demo data instead of an input file.")
    args = parser.parse_args()

    if args.demo:
        records = make_demo_records()
    elif args.input:
        records = load_mutations(args.input, args.gene_column, args.sample_column, args.class_column)
    else:
        parser.error("Provide --input <file> or use --demo.")

    if not records:
        sys.exit("No mutation records found in the input.")

    top_genes, gene_freq, gene_sample_hits, gene_class_counts, n_samples = summarise(records, args.top_n)

    gene_summary_path = write_gene_summary(
        top_genes, gene_freq, gene_sample_hits, gene_class_counts, args.out_prefix, n_samples
    )
    sample_summary_path, sample_counts = write_sample_summary(records, args.out_prefix)
    top_genes_plot = plot_top_genes(top_genes, gene_freq, args.out_prefix)
    heatmap_plot = plot_heatmap(records, top_genes, args.out_prefix)

    print(f"Samples: {n_samples}, mutation records: {len(records)}, genes observed: {len({g for g, _, _ in records})}")
    print(f"Top {len(top_genes)} mutated genes:")
    for gene in top_genes:
        print(f"  {gene:10s}: {len(gene_sample_hits[gene]):4d}/{n_samples} samples ({gene_freq[gene] * 100:5.1f}%)")
    print(f"Gene summary written to {gene_summary_path}")
    print(f"Sample summary written to {sample_summary_path}")
    print(f"Top genes chart written to {top_genes_plot}")
    print(f"Heatmap written to {heatmap_plot}")


if __name__ == "__main__":
    main()
