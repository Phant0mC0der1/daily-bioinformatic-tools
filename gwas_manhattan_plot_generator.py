"""
GWAS Manhattan Plot Generator
==============================

Reads GWAS summary statistics and produces a Manhattan plot, highlighting
genome-wide significant loci.

Expected input: a CSV or whitespace-delimited text file containing at least
the following columns (case-insensitive, common aliases are accepted):

 CHR - chromosome (1-22, X, Y)
 BP - base-pair position on the chromosome
 P - association p-value
 SNP - (optional) variant identifier, used for labeling top hits

Usage:
 python gwas_manhattan_plot_generator.py gwas_results.csv
 python gwas_manhattan_plot_generator.py gwas_results.csv --out manhattan.png
 python gwas_manhattan_plot_generator.py gwas_results.csv --sig 5e-8 --suggestive 1e-5

If no input file is given, a synthetic example dataset is generated so the
script can be run and inspected immediately.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "chr": ["chr", "chrom", "chromosome", "#chrom"],
    "bp": ["bp", "pos", "position", "base_pair_location"],
    "p": ["p", "pval", "pvalue", "p_value", "p.value"],
    "snp": ["snp", "id", "rsid", "variant_id", "marker"],
}

CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]


def resolve_columns(df):
    lower_cols = {c.lower(): c for c in df.columns}
    resolved = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                resolved[canonical] = lower_cols[alias]
                break
    missing = [c for c in ("chr", "bp", "p") if c not in resolved]
    if missing:
        raise ValueError(
            f"Could not find required column(s) {missing} in input file. "
            f"Available columns: {list(df.columns)}"
        )
    return resolved


def load_summary_stats(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Failed to parse input file '{path}': {exc}")

    cols = resolve_columns(df)
    out = pd.DataFrame(
        {
            "CHR": df[cols["chr"]].astype(str).str.upper().str.replace("CHR", "", regex=False),
            "BP": pd.to_numeric(df[cols["bp"]], errors="coerce"),
            "P": pd.to_numeric(df[cols["p"]], errors="coerce"),
        }
    )
    if "snp" in cols:
        out["SNP"] = df[cols["snp"]].astype(str)
    else:
        out["SNP"] = out["CHR"] + ":" + out["BP"].astype(str)

    out = out.dropna(subset=["CHR", "BP", "P"])
    out = out[(out["P"] > 0) & (out["P"] <= 1)]
    return out.reset_index(drop=True)


def make_synthetic_data(n_snps_per_chr=2000, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    for chrom in CHROM_ORDER:
        bp = np.sort(rng.integers(1, 250_000_000, size=n_snps_per_chr))
        p = rng.uniform(1e-8, 1, size=n_snps_per_chr) ** 1.5
        records.append(pd.DataFrame({"CHR": chrom, "BP": bp, "P": p}))

    df = pd.concat(records, ignore_index=True)
    df["SNP"] = df["CHR"] + ":" + df["BP"].astype(str)

    hit_chr = rng.choice(CHROM_ORDER, size=8, replace=True)
    for chrom in hit_chr:
        mask = df["CHR"] == chrom
        idx = rng.choice(df[mask].index, size=1)[0]
        df.loc[idx, "P"] = rng.uniform(1e-15, 5e-9)

    return df.reset_index(drop=True)


def compute_plot_positions(df):
    df = df.copy()
    df["CHR"] = df["CHR"].apply(lambda c: c if c in CHROM_ORDER else c)
    order = [c for c in CHROM_ORDER if c in df["CHR"].unique()] or sorted(df["CHR"].unique())

    offsets = {}
    running = 0
    ticks = []
    for chrom in order:
        chrom_max = df.loc[df["CHR"] == chrom, "BP"].max()
        offsets[chrom] = running
        ticks.append((chrom, running + chrom_max / 2))
        running += chrom_max + 1

    df["POS"] = df.apply(lambda row: row["BP"] + offsets[row["CHR"]], axis=1)
    return df, ticks, order


def plot_manhattan(df, sig_threshold, suggestive_threshold, out_path, title, label_top_n):
    df = df.copy()
    df["-log10P"] = -np.log10(df["P"])
    df, ticks, order = compute_plot_positions(df)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#3B4CC0", "#7A98D1"]

    for i, chrom in enumerate(order):
        chunk = df[df["CHR"] == chrom]
        ax.scatter(
            chunk["POS"],
            chunk["-log10P"],
            s=8,
            color=colors[i % 2],
            alpha=0.8,
            linewidths=0,
        )

    sig_hits = df[df["P"] < sig_threshold]
    if not sig_hits.empty:
        ax.scatter(
            sig_hits["POS"],
            sig_hits["-log10P"],
            s=18,
            color="#D62728",
            label=f"Genome-wide significant (p < {sig_threshold:.1e})",
            zorder=5,
        )

    ax.axhline(
        -np.log10(sig_threshold),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Significance threshold ({sig_threshold:.1e})",
    )
    ax.axhline(
        -np.log10(suggestive_threshold),
        color="blue",
        linestyle=":",
        linewidth=1,
        label=f"Suggestive threshold ({suggestive_threshold:.1e})",
    )

    if label_top_n > 0 and not sig_hits.empty:
        top_hits = sig_hits.nsmallest(label_top_n, "P")
        for _, row in top_hits.iterrows():
            ax.annotate(
                row["SNP"],
                xy=(row["POS"], row["-log10P"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks([pos for _, pos in ticks])
    ax.set_xticklabels([chrom for chrom, _ in ticks], fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return sig_hits


def main():
    parser = argparse.ArgumentParser(description="Generate a GWAS Manhattan plot from summary statistics.")
    parser.add_argument("input", nargs="?", help="Path to GWAS summary statistics file (CSV/TSV). Omit to use synthetic demo data.")
    parser.add_argument("--out", default="manhattan_plot.png", help="Output image path (default: manhattan_plot.png)")
    parser.add_argument("--sig", type=float, default=5e-8, help="Genome-wide significance threshold (default: 5e-8)")
    parser.add_argument("--suggestive", type=float, default=1e-5, help="Suggestive significance threshold (default: 1e-5)")
    parser.add_argument("--label-top", type=int, default=10, help="Number of top significant SNPs to label (default: 10)")
    parser.add_argument("--title", default="GWAS Manhattan Plot", help="Plot title")
    parser.add_argument("--hits-out", default=None, help="Optional CSV path to write significant hits table")
    args = parser.parse_args()

    if args.input:
        print(f"Loading GWAS summary statistics from {args.input} ...")
        df = load_summary_stats(args.input)
    else:
        print("No input file given; generating synthetic demo dataset ...")
        df = make_synthetic_data()

    print(f"Loaded {len(df):,} variants across {df['CHR'].nunique()} chromosomes.")

    sig_hits = plot_manhattan(
        df,
        sig_threshold=args.sig,
        suggestive_threshold=args.suggestive,
        out_path=args.out,
        title=args.title,
        label_top_n=args.label_top,
    )

    print(f"Manhattan plot saved to: {args.out}")
    print(f"Genome-wide significant loci (p < {args.sig:.1e}): {len(sig_hits)}")

    if not sig_hits.empty:
        top = sig_hits.sort_values("P").head(20)
        print("\nTop significant loci:")
        print(top[["SNP", "CHR", "BP", "P"]].to_string(index=False))

    if args.hits_out and not sig_hits.empty:
        sig_hits.sort_values("P").to_csv(args.hits_out, index=False)
        print(f"\nSignificant hits table written to: {args.hits_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
