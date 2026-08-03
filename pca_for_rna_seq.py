#!/usr/bin/env python3
"""
PCA for RNA-seq

Takes a gene expression matrix (genes x samples) and produces:
  1. A PCA scatter plot (PC1 vs PC2, optionally colored by sample group)
  2. A scree plot of explained variance per principal component
  3. A CSV of per-sample PC scores
  4. Sample clustering (hierarchical dendrogram + flat cluster assignments)

Usage:
    python pca_for_rna_seq.py <expression_matrix.csv> [options]

Input format:
    A CSV/TSV file with genes as rows and samples as columns:

        gene,Sample1,Sample2,Sample3,...
        GENE1,5.2,4.9,7.1,...
        GENE2,0.1,0.3,0.2,...

    The first column is treated as the gene identifier (row index).

Optional metadata file (--metadata) with sample groups for coloring:

        sample,group
        Sample1,Control
        Sample2,Treated
        Sample3,Control

Example:
    python pca_for_rna_seq.py expression_matrix.csv \\
        --metadata sample_metadata.csv --top-n 2000 --log2 \\
        --n-clusters 3 --output-prefix rnaseq_pca
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def load_metadata(path: str, samples: pd.Index) -> pd.Series:
    meta = pd.read_csv(path, index_col=0)
    if meta.shape[1] == 0:
        raise ValueError("Metadata file must have at least one column beyond the sample id.")
    group_col = meta.columns[0]
    meta = meta[group_col].reindex(samples)
    missing = meta[meta.isna()].index.tolist()
    if missing:
        print(f"Warning: no metadata for sample(s) {missing}; will plot as 'Unknown'.", file=sys.stderr)
        meta = meta.fillna("Unknown")
    return meta.astype(str)


def select_top_variable_genes(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if top_n <= 0 or top_n >= df.shape[0]:
        return df
    variance = df.var(axis=1, ddof=1)
    top_genes = variance.sort_values(ascending=False).index[:top_n]
    return df.loc[top_genes]


def run_pca(df: pd.DataFrame, n_components: int, scale: bool):
    sample_by_gene = df.T.values  # rows = samples, cols = genes
    if scale:
        sample_by_gene = StandardScaler().fit_transform(sample_by_gene)
    else:
        sample_by_gene = sample_by_gene - sample_by_gene.mean(axis=0)

    n_components = min(n_components, sample_by_gene.shape[0] - 1, sample_by_gene.shape[1])
    n_components = max(n_components, 1)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(sample_by_gene)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=df.columns, columns=columns)
    return pca, scores_df


def plot_scree(pca: PCA, output: str):
    var_pct = pca.explained_variance_ratio_ * 100
    cum_pct = np.cumsum(var_pct)
    x = np.arange(1, len(var_pct) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, var_pct, color="#4C72B0", label="Explained variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"PC{i}" for i in x], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color="#C44E52", marker="o", label="Cumulative variance")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle("PCA Scree Plot")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(scores_df: pd.DataFrame, pca: PCA, output: str, groups: pd.Series = None, label_samples: bool = False):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    var_pct = pca.explained_variance_ratio_ * 100
    pc1_label = f"PC1 ({var_pct[0]:.1f}% variance)"
    pc2_label = f"PC2 ({var_pct[1]:.1f}% variance)" if len(var_pct) > 1 else "PC2"

    if groups is not None:
        for group_name in sorted(groups.unique()):
            mask = groups == group_name
            ax.scatter(
                scores_df.loc[mask, "PC1"],
                scores_df.loc[mask, "PC2"] if "PC2" in scores_df.columns else np.zeros(mask.sum()),
                label=group_name,
                s=70,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.legend(title="Group", loc="best")
    else:
        ax.scatter(
            scores_df["PC1"],
            scores_df["PC2"] if "PC2" in scores_df.columns else np.zeros(len(scores_df)),
            s=70,
            alpha=0.85,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )

    if label_samples:
        for sample, row in scores_df.iterrows():
            y = row["PC2"] if "PC2" in scores_df.columns else 0.0
            ax.annotate(str(sample), (row["PC1"], y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.set_title("PCA of RNA-seq Samples")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cluster_samples(scores_df: pd.DataFrame, n_clusters: int, output_dendrogram: str):
    link = linkage(scores_df.values, method="average", metric="euclidean")

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(scores_df)), 5))
    dendrogram(link, labels=scores_df.index.tolist(), ax=ax, leaf_rotation=90)
    ax.set_title("Hierarchical Clustering of Samples (PCA space)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_dendrogram, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_clusters = max(1, min(n_clusters, len(scores_df)))
    cluster_ids = fcluster(link, t=n_clusters, criterion="maxclust")
    return pd.Series(cluster_ids, index=scores_df.index, name="cluster")


def main():
    parser = argparse.ArgumentParser(description="Run PCA on an RNA-seq expression matrix: scatter plot, explained variance, and sample clustering.")
    parser.add_argument("expression_matrix", help="Path to expression matrix CSV/TSV (genes as rows, samples as columns).")
    parser.add_argument("--sep", default=None, help="Field delimiter. Auto-detected from extension if omitted.")
    parser.add_argument("--metadata", default=None, help="Optional CSV mapping sample id (first column) to a group label (second column), used to color the PCA scatter plot.")
    parser.add_argument("--top-n", type=int, default=2000, help="Restrict PCA to the N most variable genes (default: 2000; use 0 to use all genes).")
    parser.add_argument("--log2", action="store_true", help="Apply log2(x + 1) transform before analysis.")
    parser.add_argument("--no-scale", action="store_true", help="Disable per-gene standardization (z-score) before PCA; only mean-centers.")
    parser.add_argument("--n-components", type=int, default=10, help="Number of principal components to compute (default: 10).")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of flat clusters to cut from the sample dendrogram (default: 3).")
    parser.add_argument("--label-samples", action="store_true", help="Annotate each point in the PCA scatter plot with its sample name.")
    parser.add_argument("--output-prefix", default="pca_rnaseq", help="Prefix for all output files (default: pca_rnaseq).")
    args = parser.parse_args()

    print(f"Loading expression matrix from {args.expression_matrix} ...")
    df = load_expression_matrix(args.expression_matrix, sep=args.sep)
    print(f"Loaded {df.shape[0]} genes x {df.shape[1]} samples.")

    if df.shape[1] < 3:
        raise ValueError("PCA and clustering require at least 3 samples.")

    if args.log2:
        if (df.values < 0).any():
            print("Warning: negative values present; log2 transform may produce NaNs.", file=sys.stderr)
        df = np.log2(df + 1)

    groups = None
    if args.metadata:
        groups = load_metadata(args.metadata, df.columns)

    df_top = select_top_variable_genes(df, args.top_n)
    print(f"Using {df_top.shape[0]} genes for PCA (top-n={args.top_n}).")

    pca, scores_df = run_pca(df_top, n_components=args.n_components, scale=not args.no_scale)
    print(f"Computed {scores_df.shape[1]} principal components.")
    print("Explained variance (%): " + ", ".join(f"{v:.1f}" for v in pca.explained_variance_ratio_ * 100))

    scores_path = f"{args.output_prefix}_scores.csv"
    scores_out = scores_df.copy()
    if groups is not None:
        scores_out.insert(0, "group", groups)
    scores_out.to_csv(scores_path)
    print(f"PC scores written to {scores_path}")

    scatter_path = f"{args.output_prefix}_scatter.png"
    plot_pca_scatter(scores_df, pca, scatter_path, groups=groups, label_samples=args.label_samples)
    print(f"PCA scatter plot saved to {scatter_path}")

    scree_path = f"{args.output_prefix}_scree.png"
    plot_scree(pca, scree_path)
    print(f"Scree plot saved to {scree_path}")

    dendrogram_path = f"{args.output_prefix}_dendrogram.png"
    clusters = cluster_samples(scores_df, n_clusters=args.n_clusters, output_dendrogram=dendrogram_path)
    print(f"Sample dendrogram saved to {dendrogram_path}")

    clusters_path = f"{args.output_prefix}_clusters.csv"
    clusters_out = clusters.to_frame()
    if groups is not None:
        clusters_out.insert(0, "group", groups)
    clusters_out.to_csv(clusters_path)
    print(f"Cluster assignments written to {clusters_path}")


if __name__ == "__main__":
    main()
