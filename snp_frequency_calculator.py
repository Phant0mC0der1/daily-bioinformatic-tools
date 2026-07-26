"""
SNP Frequency Calculator

Reads a VCF file, calculates allele frequencies for each variant
(using the INFO/AF field when present, otherwise deriving it from
sample genotypes), plots the distribution of allele frequencies,
and flags rare variants below a configurable frequency threshold.

Usage:
    python snp_frequency_calculator.py input.vcf [--rare-threshold 0.01] [--out-prefix results]

Outputs:
    <out-prefix>_frequencies.csv - per-variant allele frequency table
    <out-prefix>_rare_variants.csv - variants below the rare threshold
    <out-prefix>_af_distribution.png - histogram of allele frequency distribution
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def open_vcf(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_af_from_info(info_field):
    for entry in info_field.split(";"):
        if entry.startswith("AF="):
            value = entry[len("AF="):].split(",")[0]
            try:
                return float(value)
            except ValueError:
                return None
    return None


def parse_af_from_genotypes(fmt_field, sample_fields):
    fmt_keys = fmt_field.split(":")
    gt_index = fmt_keys.index("GT") if "GT" in fmt_keys else None
    if gt_index is None:
        return None

    alt_count = 0
    total_count = 0
    for sample in sample_fields:
        parts = sample.split(":")
        if gt_index >= len(parts):
            continue
        gt = parts[gt_index]
        alleles = gt.replace("|", "/").split("/")
        for allele in alleles:
            if allele == "." or allele == "":
                continue
            try:
                allele_int = int(allele)
            except ValueError:
                continue
            total_count += 1
            if allele_int > 0:
                alt_count += 1

    if total_count == 0:
        return None
    return alt_count / total_count


def parse_vcf(path):
    variants = []
    with open_vcf(path) as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, variant_id, ref, alt, qual, filt, info = fields[:8]

            af = parse_af_from_info(info)
            if af is None and len(fields) > 9:
                fmt_field = fields[8]
                sample_fields = fields[9:]
                af = parse_af_from_genotypes(fmt_field, sample_fields)

            if af is None:
                continue

            variants.append({
                "chrom": chrom,
                "pos": pos,
                "id": variant_id,
                "ref": ref,
                "alt": alt,
                "af": af,
            })

    return variants


def write_csv(path, variants, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variants:
            writer.writerow(variant)


def plot_distribution(variants, out_path):
    frequencies = [v["af"] for v in variants]

    plt.figure(figsize=(8, 5))
    plt.hist(frequencies, bins=50, color="#4C72B0", edgecolor="black")
    plt.xlabel("Allele Frequency")
    plt.ylabel("Number of Variants")
    plt.title("SNP Allele Frequency Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate SNP allele frequencies from a VCF file.")
    parser.add_argument("vcf", help="Path to input VCF file (.vcf or .vcf.gz)")
    parser.add_argument("--rare-threshold", type=float, default=0.01,
                         help="Allele frequency threshold below which a variant is considered rare (default: 0.01)")
    parser.add_argument("--out-prefix", default="snp_frequency",
                         help="Prefix for output files (default: snp_frequency)")
    args = parser.parse_args()

    vcf_path = Path(args.vcf)
    if not vcf_path.exists():
        print(f"Error: input VCF not found: {vcf_path}", file=sys.stderr)
        sys.exit(1)

    variants = parse_vcf(vcf_path)
    if not variants:
        print("No variants with computable allele frequency were found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["chrom", "pos", "id", "ref", "alt", "af"]

    freq_csv = f"{args.out_prefix}_frequencies.csv"
    write_csv(freq_csv, variants, fieldnames)

    rare_variants = [v for v in variants if v["af"] < args.rare_threshold]
    rare_csv = f"{args.out_prefix}_rare_variants.csv"
    write_csv(rare_csv, rare_variants, fieldnames)

    plot_path = f"{args.out_prefix}_af_distribution.png"
    plot_distribution(variants, plot_path)

    frequencies = [v["af"] for v in variants]
    print(f"Total variants analysed: {len(variants)}")
    print(f"Mean allele frequency: {sum(frequencies) / len(frequencies):.4f}")
    print(f"Min / Max allele frequency: {min(frequencies):.4f} / {max(frequencies):.4f}")
    print(f"Rare variants (AF < {args.rare_threshold}): {len(rare_variants)}")
    print(f"Wrote: {freq_csv}")
    print(f"Wrote: {rare_csv}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
