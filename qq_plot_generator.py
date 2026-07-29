"""
QQ Plot Generator
==================

Reads GWAS summary statistics and produces a quantile-quantile (QQ) plot of
observed vs. expected -log10(p) values, and computes the genomic inflation
factor (Lambda GC) to help assess test-statistic inflation/deflation.

Expected input: a CSV or whitespace-delimited text file containing at least
a p-value column (case-insensitive, common aliases are accepted):

    P - association p-value

Usage:
    python qq_plot_generator.py gwas_results.csv
    python qq_plot_generator.py gwas_results.csv --out qq_plot.png
    python qq_plot_generator.py gwas_results.csv --lambda-percentile 0.5

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
from scipy import stats

COLUMN_ALIASES = {
    "p": ["p", "pval", "pvalue", "p_value", "p.value"],
    "snp": ["snp", "id", "rsid", "variant_id", "marker"],
}


def resolve_columns(df: pd.DataFrame) -> dict:
    lower_cols = {c.lower(): c for c in df.columns}
    resolved = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                resolved[canonical] = lower_cols[alias]
                break

    if "p" not in resolved:
        raise ValueError(
            f"Could not find a p-value column in input file. "
            f"Available columns: {list(df.columns)}"
        )
    return resolved


def load_pvalues(path: str) -> np.ndarray:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read '{path}' as a delimited text file: {exc}")

    resolved = resolve_columns(df)
    pvals = pd.to_numeric(df[resolved["p"]], errors="coerce").dropna()
    pvals = pvals[(pvals > 0) & (pvals <= 1)]

    if pvals.empty:
        raise ValueError("No valid p-values (0 < p <= 1) found in input file.")
    return pvals.to_numpy()


def generate_synthetic_pvalues(n: int = 20000, inflation: float = 1.05, seed: int = 42) -> np.ndarray:
    """Generate p-values from chi-squared(1) statistics with mild inflation."""
    rng = np.random.default_rng(seed)
    chisq = rng.chisquare(df=1, size=n) * inflation
    pvals = stats.chi2.sf(chisq, df=1)
    return np.clip(pvals, np.finfo(float).tiny, 1.0)


def lambda_gc(pvals: np.ndarray, percentile: float = 0.5) -> float:
    """
    Compute the genomic inflation factor (Lambda GC).

    Converts p-values to chi-squared(1) statistics and compares the
    observed median (or other percentile) chi-squared statistic to the
    expected value under the null at that percentile.
    """
    chisq = stats.chi2.isf(pvals, df=1)
    observed_stat = np.quantile(chisq, percentile)
    expected_stat = stats.chi2.ppf(percentile, df=1)
    return observed_stat / expected_stat


def qq_plot(pvals: np.ndarray, out_path: str, lam: float, percentile: float) -> None:
    n = len(pvals)
    observed = -np.log10(np.sort(pvals))
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    max_val = max(observed.max(), expected.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(expected, observed, s=8, alpha=0.5, color="#1f77b4", edgecolors="none")
    ax.plot([0, max_val], [0, max_val], color="red", linewidth=1, linestyle="--")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title("QQ Plot of GWAS p-values")
    ax.text(
        0.05,
        0.95,
        rf"$\lambda_{{GC}}$ ({percentile:g} pct) = {lam:.4f}"
        f"\nN variants = {n:,}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def interpret_lambda(lam: float) -> str:
    if lam < 0.95:
        return "Interpretation: possible deflation of test statistics."
    if lam > 1.10:
        return (
            "Interpretation: notable inflation of test statistics "
            "(check for population stratification, cryptic relatedness, "
            "or genotyping artifacts)."
        )
    return "Interpretation: test statistics look approximately well-calibrated."


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a QQ plot and Lambda GC for GWAS p-values."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Path to GWAS summary statistics file (CSV/TSV/whitespace-delimited). "
            "If omitted, a synthetic dataset is used."
        ),
    )
    parser.add_argument(
        "--out",
        default="qq_plot.png",
        help="Output image path (default: qq_plot.png)",
    )
    parser.add_argument(
        "--lambda-percentile",
        type=float,
        default=0.5,
        help="Percentile (0-1) used to compute Lambda GC (default: 0.5, i.e. median)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not 0 < args.lambda_percentile < 1:
        sys.exit("--lambda-percentile must be between 0 and 1 (exclusive).")

    if args.input:
        try:
            pvals = load_pvalues(args.input)
        except ValueError as exc:
            sys.exit(f"Error: {exc}")
        print(f"Loaded {len(pvals):,} p-values from '{args.input}'.")
    else:
        pvals = generate_synthetic_pvalues()
        print(f"No input file given; using {len(pvals):,} synthetic p-values (mild inflation).")

    lam = lambda_gc(pvals, args.lambda_percentile)
    qq_plot(pvals, args.out, lam, args.lambda_percentile)

    print(f"Lambda GC ({args.lambda_percentile:g} percentile): {lam:.4f}")
    print(interpret_lambda(lam))
    print(f"QQ plot saved to '{args.out}'.")


if __name__ == "__main__":
    main()
