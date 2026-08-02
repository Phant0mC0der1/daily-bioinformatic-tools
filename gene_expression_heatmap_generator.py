#!/usr/bin/env python3
"""
Gene Expression Heatmap Generator

Takes a gene expression matrix (genes x samples) and produces:
  1. A clustered heatmap (hierarchical clustering of genes and samples)
  2. A list of the top N most variable genes
  3. A CSV export of those top variable genes

Usage:
    python gene_expression_heatmap_generator.py <expression_matrix.csv> [options]

Input format:
    A CSV/TSV file with genes as rows and samples as columns:

        gene,Sample1,Sample2,Sample3,...
        GENE1,5.2,4.9,7.1,...
        GENE2,0.1,0.3,0.2,...

    The first column is treated as the gene identifier (row index).

Example:
    python gene_expression_heatmap_generator.py expression_matrix.csv \\
        --top-n 50 --log2 --zscore --output heatmap.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage


def load_expression_matrix(path: str, sep: str = None) -> pd.DataFrame:
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().all(axis=None):
        raise ValueError("No numeric expression values found after parsing input file.")
    n_dropped = df.isna().any(axis=1).sum()
    if n_dropped:
        print(f"Warning: dropping {n_dropped} gene(s) with missing/non-numeric values.", file=sys.stderr)
    df = df.dropna(axis=0, how="any")
    return df


def select_top_variable_genes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    variance = df.var(axis=1, ddof=1)
    ranked = variance.sort_values(ascending=False)
    top_n = min(top_n, len(ranked))
    top_genes = ranked.index[:top_n]
    result = df.loc[top_genes].copy()
    result.insert(0, "variance", variance.loc[top_genes])
    return result


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0, np.nan)
    scaled = df.sub(mean, axis=0).div(std, axis=0)
    return scaled.fillna(0.0)


def make_heatmap(matrix: pd.DataFrame, output: str, cluster_genes: bool, cluster_samples: bool, cmap: str):
    row_linkage = linkage(matrix.values, method="average", metric="euclidean") if cluster_genes and matrix.shape[0] > 1 else None
    col_linkage = linkage(matrix.values.T, method="average", metric="euclidean") if cluster_samples and matrix.shape[1] > 1 else None

    height = max(6, min(0.25 * matrix.shape[0], 40))
    width = max(6, min(0.6 * matrix.shape[1] + 4, 30))

    grid = sns.clustermap(
        matrix,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_cluster=cluster_genes and matrix.shape[0] > 1,
        col_cluster=cluster_samples and matrix.shape[1] > 1,
        cmap=cmap,
        center=0 if matrix.values.min() < 0 else None,
        figsize=(width, height),
        yticklabels=matrix.shape[0] <= 100,
        xticklabels=True,
        cbar_kws={"label": "Expression (z-score)" if matrix.values.min() < 0 else "Expression"},
    )
    grid.ax_heatmap.set_xlabel("Sample")
    grid.ax_heatmap.set_ylabel("Gene")
    grid.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)


def main():
    parser = argparse.ArgumentParser(description="Generate a clustered heatmap and top-variable-gene list from a gene expression matrix.")
    parser.add_argument("expression_matrix", help="Path to expression matrix CSV/TSV (genes as rows, samples as columns).")
    parser.add_argument("--sep", default=None, help="Field delimiter. Auto-detected from extension if omitted.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of most variable genes to plot/export (default: 50).")
    parser.add_argument("--log2", action="store_true", help="Apply log2(x + 1) transform before analysis.")
    parser.add_argument("--zscore", action="store_true", help="Row-wise z-score normalize before plotting.")
    parser.add_argument("--no-cluster-genes", action="store_true", help="Disable hierarchical clustering of genes.")
    parser.add_argument("--no-cluster-samples", action="store_true", help="Disable hierarchical clustering of samples.")
    parser.add_argument("--cmap", default="vlag", help="Matplotlib/seaborn colormap for the heatmap (default: vlag).")
    parser.add_argument("--output", default="gene_expression_heatmap.png", help="Output image path (default: gene_expression_heatmap.png).")
    parser.add_argument("--top-genes-csv", default="top_variable_genes.csv", help="Output CSV path for top variable genes.")
    args = parser.parse_args()

    print(f"Loading expression matrix from {args.expression_matrix} ...")
    df = load_expression_matrix(args.expression_matrix, sep=args.sep)
    print(f"Loaded {df.shape[0]} genes x {df.shape[1]} samples.")

    if args.log2:
        if (df.values < 0).any():
            print("Warning: negative values present; log2 transform may produce NaNs.", file=sys.stderr)
        df = np.log2(df + 1)

    top_with_variance = select_top_variable_genes(df, args.top_n)
    top_with_variance.to_csv(args.top_genes_csv)
    print(f"Top {top_with_variance.shape[0]} most variable genes written to {args.top_genes_csv}")

    plot_matrix = top_with_variance.drop(columns="variance")
    if args.zscore:
        plot_matrix = zscore_rows(plot_matrix)

    make_heatmap(
        plot_matrix,
        output=args.output,
        cluster_genes=not args.no_cluster_genes,
        cluster_samples=not args.no_cluster_samples,
        cmap=args.cmap,
    )
    print(f"Clustered heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
