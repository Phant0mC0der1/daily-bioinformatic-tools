#!/usr/bin/env python3
"""
Expression Boxplot Tool
========================
Compares gene expression between two conditions using boxplots.

Input:
    - An expression matrix (CSV/TSV) with genes as rows and samples as columns.
    - A sample metadata file (CSV/TSV) with a sample ID column and a condition column,
      assigning each sample to one of exactly two conditions.

Output:
    - A boxplot (with individual sample points overlaid) for each requested gene,
      comparing expression between the two conditions, annotated with a
      Mann-Whitney U test p-value.
    - A summary CSV with test statistics for every gene plotted.

Example:
    python expression_boxplot_tool.py \
        --expression expression_matrix.csv \
        --metadata sample_metadata.csv \
        --sample-col sample_id \
        --condition-col condition \
        --genes TP53 EGFR MYC \
        --outdir results
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare gene expression between two conditions with boxplots."
    )
    parser.add_argument(
        "--expression", required=True,
        help="Path to expression matrix (CSV/TSV), genes as rows, samples as columns."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to sample metadata (CSV/TSV) with sample ID and condition columns."
    )
    parser.add_argument(
        "--sample-col", default="sample_id",
        help="Column name in metadata holding sample IDs (default: sample_id)."
    )
    parser.add_argument(
        "--condition-col", default="condition",
        help="Column name in metadata holding condition labels (default: condition)."
    )
    parser.add_argument(
        "--genes", nargs="*", default=None,
        help="Specific gene names to plot. If omitted, the top-N most variable genes are used."
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of top variable genes to plot when --genes is not given (default: 10)."
    )
    parser.add_argument(
        "--outdir", default="expression_boxplot_output",
        help="Directory to write plots and summary CSV to."
    )
    parser.add_argument(
        "--log2", action="store_true",
        help="Apply log2(x + 1) transform to expression values before plotting/testing."
    )
    return parser.parse_args()


def read_table(path):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, index_col=0 if path == "" else None)


def load_expression(path):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df.index = df.index.astype(str)
    return df


def load_metadata(path, sample_col, condition_col):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    missing = {sample_col, condition_col} - set(df.columns)
    if missing:
        sys.exit(f"Error: metadata file is missing column(s): {sorted(missing)}")
    df[sample_col] = df[sample_col].astype(str)
    return df.set_index(sample_col)[condition_col]


def select_genes(expr, requested_genes, top_n):
    if requested_genes:
        missing = [g for g in requested_genes if g not in expr.index]
        found = [g for g in requested_genes if g in expr.index]
        if missing:
            print(f"Warning: gene(s) not found in expression matrix and skipped: {missing}")
        if not found:
            sys.exit("Error: none of the requested genes were found in the expression matrix.")
        return found

    variances = expr.var(axis=1).sort_values(ascending=False)
    top_genes = variances.head(top_n).index.tolist()
    print(f"No --genes given; using top {len(top_genes)} most variable genes: {top_genes}")
    return top_genes


def plot_gene(gene, values_a, values_b, cond_a, cond_b, pvalue, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))

    data = [values_a, values_b]
    labels = [f"{cond_a}\n(n={len(values_a)})", f"{cond_b}\n(n={len(values_b)})"]

    box = ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(seed=hash(gene) % (2**32))
    for i, values in enumerate(data, start=1):
        jitter = rng.normal(loc=i, scale=0.05, size=len(values))
        ax.scatter(jitter, values, color=colors[i - 1], edgecolor="black",
                   linewidth=0.5, s=30, zorder=3, alpha=0.8)

    ax.set_ylabel("Expression")
    ax.set_title(f"{gene}\nMann-Whitney U p = {pvalue:.3g}")
    fig.tight_layout()

    out_path = os.path.join(outdir, f"{gene}_boxplot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    expr = load_expression(args.expression)
    conditions = load_metadata(args.metadata, args.sample_col, args.condition_col)

    common_samples = [s for s in expr.columns if s in conditions.index]
    if len(common_samples) < len(expr.columns):
        dropped = set(expr.columns) - set(common_samples)
        print(f"Warning: {len(dropped)} sample(s) in expression matrix have no metadata "
              f"and were dropped: {sorted(dropped)}")
    expr = expr[common_samples]
    conditions = conditions.loc[common_samples]

    unique_conditions = conditions.unique().tolist()
    if len(unique_conditions) != 2:
        sys.exit(
            f"Error: expected exactly 2 conditions, found {len(unique_conditions)}: "
            f"{unique_conditions}"
        )
    cond_a, cond_b = unique_conditions

    if args.log2:
        expr = np.log2(expr.clip(lower=0) + 1)

    genes = select_genes(expr, args.genes, args.top_n)

    samples_a = conditions[conditions == cond_a].index
    samples_b = conditions[conditions == cond_b].index

    summary_rows = []
    for gene in genes:
        values_a = expr.loc[gene, samples_a].dropna().astype(float).values
        values_b = expr.loc[gene, samples_b].dropna().astype(float).values

        if len(values_a) < 2 or len(values_b) < 2:
            print(f"Warning: skipping {gene}, not enough data points in one or both conditions.")
            continue

        stat, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        plot_path = plot_gene(gene, values_a, values_b, cond_a, cond_b, pvalue, args.outdir)

        summary_rows.append({
            "gene": gene,
            "condition_a": cond_a,
            "n_a": len(values_a),
            "mean_a": np.mean(values_a),
            "median_a": np.median(values_a),
            "condition_b": cond_b,
            "n_b": len(values_b),
            "mean_b": np.mean(values_b),
            "median_b": np.median(values_b),
            "mannwhitney_u": stat,
            "p_value": pvalue,
            "plot_file": os.path.basename(plot_path),
        })
        print(f"Plotted {gene}: p = {pvalue:.3g} -> {plot_path}")

    if not summary_rows:
        sys.exit("Error: no genes were successfully plotted.")

    summary_df = pd.DataFrame(summary_rows).sort_values("p_value")
    summary_path = os.path.join(args.outdir, "expression_boxplot_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    print(f"{len(summary_rows)} plot(s) written to {args.outdir}/")


if __name__ == "__main__":
    main()
