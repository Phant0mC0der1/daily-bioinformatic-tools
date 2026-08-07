#!/usr/bin/env python3
"""
Gene Expression Lookup

Search for a single gene in an expression matrix and display its
expression values, summary statistics (mean, median), and a violin plot
of its distribution across samples (optionally grouped by a metadata
column, e.g. condition/tissue/subtype).

Input: an expression matrix CSV with genes as rows and samples as columns
(first column = gene identifiers), e.g.:

    gene,sample1,sample2,sample3,sample4
    TP53,8.2,7.9,8.5,8.1
    EGFR,5.1,6.3,5.8,4.9

Optionally, a metadata CSV mapping sample -> group can be supplied to
color/split the violin plot by group:

    sample,group
    sample1,tumor
    sample2,normal
    sample3,tumor
    sample4,normal

Usage:
    python gene_expression_lookup.py --matrix expression.csv --gene TP53
    python gene_expression_lookup.py --matrix expression.csv --gene TP53 \
        --metadata metadata.csv --group-col group \
        --output tp53_violin.png
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Look up a gene's expression in an expression matrix "
        "and plot its distribution across samples."
    )
    parser.add_argument(
        "--matrix", required=True,
        help="Path to expression matrix CSV (genes as rows, samples as columns; "
        "first column is the gene identifier).",
    )
    parser.add_argument(
        "--gene", required=True,
        help="Gene identifier to look up (matched case-insensitively).",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Optional CSV mapping sample names to groups, used to split "
        "the violin plot by group.",
    )
    parser.add_argument(
        "--sample-col", default=None,
        help="Column name in --metadata holding sample names "
        "(default: first column).",
    )
    parser.add_argument(
        "--group-col", default=None,
        help="Column name in --metadata holding group labels "
        "(default: second column).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the violin plot (default: <gene>_violin.png).",
    )
    return parser.parse_args()


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def find_gene_row(matrix, gene):
    lookup = {g.lower(): g for g in matrix.index}
    key = gene.lower()
    if key not in lookup:
        raise SystemExit(
            f"Error: gene '{gene}' not found in expression matrix "
            f"({matrix.shape[0]} genes present)."
        )
    return matrix.loc[lookup[key]]


def load_groups(metadata_path, sample_col, group_col, samples):
    meta = pd.read_csv(metadata_path)
    sample_col = sample_col or meta.columns[0]
    group_col = group_col or meta.columns[1]
    meta = meta.set_index(sample_col)[group_col].astype(str)
    return meta.reindex(samples)


def print_summary(gene, values):
    print(f"Gene: {gene}")
    print(f"Samples: {values.count()}")
    print(f"Mean:    {values.mean():.4f}")
    print(f"Median:  {values.median():.4f}")
    print(f"Std Dev: {values.std():.4f}")
    print(f"Min:     {values.min():.4f}")
    print(f"Max:     {values.max():.4f}")
    print()
    print(values.to_string())


def plot_violin(gene, values, groups, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    if groups is not None and groups.notna().any():
        data = pd.DataFrame({"expression": values, "group": groups})
        data = data.dropna(subset=["group"])
        group_names = sorted(data["group"].unique())
        plot_data = [data.loc[data["group"] == g, "expression"].values for g in group_names]
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(group_names) + 1))
        ax.set_xticklabels(group_names, rotation=30, ha="right")
        ax.set_xlabel("Group")
    else:
        parts = ax.violinplot([values.dropna().values], showmeans=True, showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels([gene])
        ax.set_xlabel("")

    ax.set_ylabel("Expression")
    ax.set_title(f"Expression distribution: {gene}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nViolin plot saved to: {output_path}")


def main():
    args = parse_args()

    matrix = load_matrix(args.matrix)
    values = find_gene_row(matrix, args.gene)
    values = pd.to_numeric(values, errors="coerce")

    groups = None
    if args.metadata:
        groups = load_groups(args.metadata, args.sample_col, args.group_col, values.index)

    print_summary(args.gene, values)

    output_path = args.output or f"{args.gene}_violin.png"
    plot_violin(args.gene, values, groups, output_path)


if __name__ == "__main__":
    main()
