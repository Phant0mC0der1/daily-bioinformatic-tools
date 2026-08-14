#!/usr/bin/env python3
"""
Sample Quality Checker
-----------------------
Reports on RNA-seq / general count-matrix sample quality:
  - missing values
  - zero counts
  - outliers (per-sample library size, via IQR / z-score)
  - sequencing depth (total counts per sample)

Input: a tab- or comma-separated expression/count matrix with genes as rows
and samples as columns (first column = gene ID).

Usage:
    python sample_quality_checker.py counts.csv [--sep ,] [--out report_dir]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_matrix(path, sep):
    df = pd.read_csv(path, sep=sep, index_col=0)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df


def missing_value_report(df):
    missing_per_sample = df.isna().sum()
    missing_pct = (missing_per_sample / len(df) * 100).round(2)
    return pd.DataFrame({
        "missing_count": missing_per_sample,
        "missing_pct": missing_pct,
    })


def zero_count_report(df):
    zero_per_sample = (df == 0).sum()
    zero_pct = (zero_per_sample / len(df) * 100).round(2)
    return pd.DataFrame({
        "zero_count": zero_per_sample,
        "zero_pct": zero_pct,
    })


def sequencing_depth_report(df):
    depth = df.sum(skipna=True)
    return pd.DataFrame({"total_counts": depth})


def detect_outliers(depth_series):
    values = depth_series.values.astype(float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)

    result = pd.DataFrame({
        "total_counts": depth_series,
        "z_score": np.round(z_scores, 3),
        "iqr_outlier": (values < lower) | (values > upper),
        "zscore_outlier": np.abs(z_scores) > 3,
    }, index=depth_series.index)
    result["outlier"] = result["iqr_outlier"] | result["zscore_outlier"]
    return result, lower, upper


def plot_depth_barplot(depth, outlier_mask, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(depth) * 0.4), 5))
    colors = ["#d62728" if o else "#1f77b4" for o in outlier_mask]
    ax.bar(depth.index.astype(str), depth.values, color=colors)
    ax.set_ylabel("Total counts (sequencing depth)")
    ax.set_xlabel("Sample")
    ax.set_title("Sequencing depth per sample (outliers in red)")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_missing_zero(missing_df, zero_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(missing_df) * 0.6), 5))

    axes[0].bar(missing_df.index.astype(str), missing_df["missing_pct"], color="#9467bd")
    axes[0].set_title("Missing values (%) per sample")
    axes[0].set_ylabel("% missing")
    axes[0].tick_params(axis="x", rotation=90)

    axes[1].bar(zero_df.index.astype(str), zero_df["zero_pct"], color="#2ca02c")
    axes[1].set_title("Zero counts (%) per sample")
    axes[1].set_ylabel("% zero")
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Check RNA-seq / count matrix sample quality.")
    parser.add_argument("input", help="Path to expression/count matrix (genes x samples).")
    parser.add_argument("--sep", default=None, help="Field separator (default: auto-detect from extension).")
    parser.add_argument("--out", default="sample_quality_report", help="Output directory for report + plots.")
    args = parser.parse_args()

    sep = args.sep
    if sep is None:
        sep = "\t" if args.input.lower().endswith((".tsv", ".txt")) else ","

    os.makedirs(args.out, exist_ok=True)

    df = load_matrix(args.input, sep)
    if df.shape[1] == 0:
        sys.exit("No sample columns found in input matrix.")

    missing_df = missing_value_report(df)
    zero_df = zero_count_report(df)
    depth_df = sequencing_depth_report(df)
    outlier_df, lower_bound, upper_bound = detect_outliers(depth_df["total_counts"])

    summary = missing_df.join(zero_df).join(outlier_df)
    summary_path = os.path.join(args.out, "sample_quality_summary.csv")
    summary.to_csv(summary_path)

    plot_depth_barplot(depth_df["total_counts"], outlier_df["outlier"], os.path.join(args.out, "sequencing_depth.png"))
    plot_missing_zero(missing_df, zero_df, os.path.join(args.out, "missing_and_zero.png"))

    n_outliers = int(outlier_df["outlier"].sum())
    print("=== Sample Quality Checker ===")
    print(f"Samples: {df.shape[1]} | Genes: {df.shape[0]}")
    print(f"IQR outlier bounds for sequencing depth: [{lower_bound:.0f}, {upper_bound:.0f}]")
    print(f"Samples flagged as outliers: {n_outliers}")
    if n_outliers:
        print(list(outlier_df.index[outlier_df['outlier']]))
    print(f"\nFull report written to: {summary_path}")
    print(f"Plots written to: {args.out}/sequencing_depth.png and {args.out}/missing_and_zero.png")


if __name__ == "__main__":
    main()
