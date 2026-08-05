#!/usr/bin/env python3
"""
MA Plot Generator
==================
Classic differential expression visualisation: log2(fold change) vs.
mean expression (the "M" and "A" of an MA plot).

Input
-----
A differential-expression results table (CSV/TSV), such as DESeq2 or
edgeR output, containing at minimum:
    - a mean-expression column   (e.g. "baseMean", "AveExpr", "mean_expr")
    - a log2 fold-change column  (e.g. "log2FoldChange", "logFC")
    - an adjusted p-value column (e.g. "padj", "FDR", "adj.P.Val")

Output
------
- A PNG MA plot with significant genes highlighted (up/down/not significant).
- A CSV listing only the significant genes, sorted by adjusted p-value.

Usage
-----
    python ma_plot_generator.py results.csv \
        --mean-col baseMean --lfc-col log2FoldChange --padj-col padj \
        --padj-threshold 0.05 --lfc-threshold 1.0 \
        --gene-col gene --out-prefix ma_plot

If --mean-col/--lfc-col/--padj-col are omitted, the script tries to guess
them from a list of common column-name aliases.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEAN_ALIASES = ["baseMean", "AveExpr", "mean_expr", "meanExpr", "average_expression", "A"]
LFC_ALIASES = ["log2FoldChange", "logFC", "log2FC", "lfc", "M"]
PADJ_ALIASES = ["padj", "FDR", "adj.P.Val", "qvalue", "q_value", "padj_value"]
GENE_ALIASES = ["gene", "gene_id", "GeneID", "Gene", "symbol", "gene_symbol"]


def guess_column(columns, aliases, kind):
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an MA plot from differential expression results.")
    parser.add_argument("input", help="Path to DE results file (CSV or TSV).")
    parser.add_argument("--mean-col", default=None, help="Column with mean expression values.")
    parser.add_argument("--lfc-col", default=None, help="Column with log2 fold-change values.")
    parser.add_argument("--padj-col", default=None, help="Column with adjusted p-values.")
    parser.add_argument("--gene-col", default=None, help="Column with gene identifiers (optional).")
    parser.add_argument("--padj-threshold", type=float, default=0.05, help="Adjusted p-value cutoff for significance.")
    parser.add_argument("--lfc-threshold", type=float, default=1.0, help="Absolute log2 fold-change cutoff for significance.")
    parser.add_argument("--out-prefix", default="ma_plot", help="Prefix for output files.")
    parser.add_argument("--sep", default=None, help="Field separator. Auto-detected from extension if omitted.")
    return parser.parse_args()


def load_table(path, sep):
    if sep is None:
        sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def main():
    args = parse_args()

    try:
        df = load_table(args.input, args.sep)
    except Exception as exc:
        sys.exit(f"Error reading input file '{args.input}': {exc}")

    mean_col = args.mean_col or guess_column(df.columns, MEAN_ALIASES, "mean expression")
    lfc_col = args.lfc_col or guess_column(df.columns, LFC_ALIASES, "log2 fold change")
    padj_col = args.padj_col or guess_column(df.columns, PADJ_ALIASES, "adjusted p-value")
    gene_col = args.gene_col or guess_column(df.columns, GENE_ALIASES, "gene id")

    missing = [name for name, col in [("mean", mean_col), ("log2FC", lfc_col), ("padj", padj_col)] if col is None]
    if missing:
        sys.exit(
            f"Could not identify column(s) for: {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}. "
            "Specify them explicitly with --mean-col/--lfc-col/--padj-col."
        )

    df = df.dropna(subset=[mean_col, lfc_col, padj_col]).copy()
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df[lfc_col] = pd.to_numeric(df[lfc_col], errors="coerce")
    df[padj_col] = pd.to_numeric(df[padj_col], errors="coerce")
    df = df.dropna(subset=[mean_col, lfc_col, padj_col])

    if df.empty:
        sys.exit("No valid rows remain after cleaning numeric columns.")

    is_sig_up = (df[padj_col] < args.padj_threshold) & (df[lfc_col] >= args.lfc_threshold)
    is_sig_down = (df[padj_col] < args.padj_threshold) & (df[lfc_col] <= -args.lfc_threshold)
    is_sig = is_sig_up | is_sig_down

    df["regulation"] = np.where(is_sig_up, "Up", np.where(is_sig_down, "Down", "Not significant"))

    log_mean = np.log10(df[mean_col].clip(lower=1e-9) + 1)

    fig, ax = plt.subplots(figsize=(9, 7))

    not_sig = ~is_sig
    ax.scatter(log_mean[not_sig], df.loc[not_sig, lfc_col], s=8, color="grey", alpha=0.4, label="Not significant")
    ax.scatter(log_mean[is_sig_up], df.loc[is_sig_up, lfc_col], s=10, color="red", alpha=0.7, label="Up")
    ax.scatter(log_mean[is_sig_down], df.loc[is_sig_down, lfc_col], s=10, color="blue", alpha=0.7, label="Down")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(args.lfc_threshold, color="black", linestyle="--", linewidth=0.6)
    ax.axhline(-args.lfc_threshold, color="black", linestyle="--", linewidth=0.6)

    ax.set_xlabel("log10(mean expression + 1)")
    ax.set_ylabel("log2 fold change")
    ax.set_title(
        f"MA Plot (padj < {args.padj_threshold}, |log2FC| >= {args.lfc_threshold})\n"
        f"Up: {int(is_sig_up.sum())}  Down: {int(is_sig_down.sum())}  "
        f"Not significant: {int(not_sig.sum())}"
    )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    plot_path = f"{args.out_prefix}.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    sig_df = df[is_sig].copy()
    sort_cols = [gene_col] if gene_col else []
    sig_df = sig_df.sort_values(padj_col)
    sig_path = f"{args.out_prefix}_significant_genes.csv"
    sig_df.to_csv(sig_path, index=False)

    print(f"Loaded {len(df)} genes from '{args.input}'.")
    print(f"Columns used -> mean: '{mean_col}', log2FC: '{lfc_col}', padj: '{padj_col}'"
          + (f", gene: '{gene_col}'" if gene_col else ""))
    print(f"Significant genes: {int(is_sig.sum())} (Up: {int(is_sig_up.sum())}, Down: {int(is_sig_down.sum())})")
    print(f"MA plot saved to: {plot_path}")
    print(f"Significant gene table saved to: {sig_path}")


if __name__ == "__main__":
    main()
