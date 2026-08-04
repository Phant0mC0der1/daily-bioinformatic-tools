#!/usr/bin/env python3
"""
Volcano Plot Creator
=====================
Reads a differential expression (DE) results table and produces a volcano
plot (log2 fold change vs. -log10 p-value), highlighting genes that pass
user-defined significance and fold-change thresholds.

Input:
    A CSV/TSV file of DE results with at least these columns (names are
    configurable via CLI flags):
        gene            - gene identifier
        log2FoldChange  - log2 fold change
        pvalue          - raw p-value
        padj            - adjusted p-value (optional; falls back to pvalue)

Output:
    - A volcano plot image (PNG)
    - A CSV of significant genes (up- and down-regulated)

Usage:
    python volcano_plot_creator.py --input de_results.csv \
        --fc-col log2FoldChange --p-col padj \
        --fc-threshold 1.0 --p-threshold 0.05 \
        --output volcano_plot.png --sig-output significant_genes.csv
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a volcano plot from differential expression results."
    )
    parser.add_argument("--input", required=True, help="Path to DE results file (CSV or TSV).")
    parser.add_argument("--gene-col", default="gene", help="Column name for gene identifiers.")
    parser.add_argument(
        "--fc-col", default="log2FoldChange", help="Column name for log2 fold change."
    )
    parser.add_argument(
        "--p-col",
        default="padj",
        help="Column name for the p-value used for significance (adjusted p-value recommended).",
    )
    parser.add_argument(
        "--fc-threshold",
        type=float,
        default=1.0,
        help="Absolute log2 fold change threshold for significance.",
    )
    parser.add_argument(
        "--p-threshold", type=float, default=0.05, help="P-value threshold for significance."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top significant genes (by p-value) to label on the plot.",
    )
    parser.add_argument(
        "--output", default="volcano_plot.png", help="Output path for the volcano plot image."
    )
    parser.add_argument(
        "--sig-output",
        default="significant_genes.csv",
        help="Output path for the CSV of significant genes.",
    )
    return parser.parse_args()


def load_data(path, gene_col, fc_col, p_col):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)

    missing = [c for c in (gene_col, fc_col, p_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input file is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[fc_col, p_col]).copy()
    df = df[df[p_col] > 0]  # avoid log(0)
    return df


def classify_genes(df, fc_col, p_col, fc_threshold, p_threshold):
    conditions = [
        (df[fc_col] >= fc_threshold) & (df[p_col] < p_threshold),
        (df[fc_col] <= -fc_threshold) & (df[p_col] < p_threshold),
    ]
    choices = ["Up", "Down"]
    df["regulation"] = np.select(conditions, choices, default="Not significant")
    df["neg_log10_p"] = -np.log10(df[p_col])
    return df


def make_plot(df, gene_col, fc_col, fc_threshold, p_threshold, top_n, output_path):
    color_map = {"Up": "#d62728", "Down": "#1f77b4", "Not significant": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(9, 7))
    for label, color in color_map.items():
        subset = df[df["regulation"] == label]
        ax.scatter(
            subset[fc_col],
            subset["neg_log10_p"],
            s=12,
            c=color,
            alpha=0.6,
            label=f"{label} (n={len(subset)})",
            linewidths=0,
        )

    ax.axvline(fc_threshold, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_threshold, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(p_threshold), color="black", linestyle="--", linewidth=0.8)

    significant = df[df["regulation"] != "Not significant"].sort_values("neg_log10_p", ascending=False)
    top_genes = significant.head(top_n)
    for _, row in top_genes.iterrows():
        ax.annotate(
            str(row[gene_col]),
            (row[fc_col], row["neg_log10_p"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano Plot")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    try:
        df = load_data(args.input, args.gene_col, args.fc_col, args.p_col)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    df = classify_genes(df, args.fc_col, args.p_col, args.fc_threshold, args.p_threshold)
    make_plot(
        df,
        args.gene_col,
        args.fc_col,
        args.fc_threshold,
        args.p_threshold,
        args.top_n,
        args.output,
    )

    significant = df[df["regulation"] != "Not significant"].sort_values(
        args.p_col, ascending=True
    )
    significant.to_csv(args.sig_output, index=False)

    n_up = (df["regulation"] == "Up").sum()
    n_down = (df["regulation"] == "Down").sum()
    print(f"Volcano plot saved to: {args.output}")
    print(f"Significant genes saved to: {args.sig_output}")
    print(f"Up-regulated: {n_up} | Down-regulated: {n_down} | Total significant: {n_up + n_down}")


if __name__ == "__main__":
    main()
