#!/usr/bin/env python3
"""
Variant Filter
==============

Filters variants in a VCF file by quality (QUAL), chromosome (CHROM),
allele frequency (INFO/AF), and gene (searched across common annotation
tags: GENE, GENEINFO, ANN (SnpEff), CSQ (VEP)), then exports the
filtered records to a new VCF file.

Pure standard library -- no external dependencies required.

Usage examples
--------------
    # Keep only PASS-quality, chr1 variants with QUAL >= 30
    python variant_filter.py input.vcf output.vcf --chrom chr1 --min-qual 30

    # Keep rare variants (AF <= 0.01) affecting gene BRCA1 or TP53
    python variant_filter.py input.vcf output.vcf --max-af 0.01 --gene BRCA1 TP53

    # Combine filters (all filters are AND-combined)
    python variant_filter.py input.vcf output.vcf \\
        --chrom chr17 chr13 --min-qual 20 --min-af 0.0 --max-af 0.05 --gene BRCA1

Run with -h/--help for the full option list.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_chrom: int = 0
    dropped_qual: int = 0
    dropped_af: int = 0
    dropped_gene: int = 0
    dropped_af_missing: int = 0


@dataclass
class VcfRecord:
    chrom: str
    pos: str
    var_id: str
    ref: str
    alt: str
    qual: str
    filt: str
    info: str
    rest: list = field(default_factory=list)
    raw: str = ""


def parse_info(info_field):
    """Parse a VCF INFO field into a dict of key -> value (value may be None for flags)."""
    parsed = {}
    for entry in info_field.split(";"):
        if not entry:
            continue
        if "=" in entry:
            key, _, value = entry.partition("=")
            parsed[key] = value
        else:
            parsed[entry] = None
    return parsed


def extract_allele_frequencies(info_dict):
    """Return a list of floats parsed from the INFO/AF field, or None if absent/unparseable."""
    af_raw = info_dict.get("AF")
    if af_raw is None:
        return None
    freqs = []
    for token in af_raw.split(","):
        try:
            freqs.append(float(token))
        except ValueError:
            continue
    return freqs or None


def extract_genes(info_dict):
    """
    Pull gene symbols out of common INFO annotation tags:
      - GENE=SYMBOL / GENEINFO=SYMBOL:ID
      - ANN=... (SnpEff: Allele|Annotation|Impact|Gene_Name|Gene_ID|...)
      - CSQ=... (VEP: usually Allele|Consequence|IMPACT|SYMBOL|Gene|...)
    Returns a set of gene symbols (uppercased) found in this record.
    """
    genes = set()

    if "GENE" in info_dict and info_dict["GENE"]:
        genes.update(g.strip().upper() for g in info_dict["GENE"].split(","))

    if "GENEINFO" in info_dict and info_dict["GENEINFO"]:
        # dbSNP-style GENEINFO=SYMBOL1:GeneID1|SYMBOL2:GeneID2
        for chunk in info_dict["GENEINFO"].split("|"):
            symbol = chunk.split(":")[0].strip()
            if symbol:
                genes.add(symbol.upper())

    if "ANN" in info_dict and info_dict["ANN"]:
        # SnpEff ANN: one annotation per comma-separated entry, pipe-delimited fields.
        # Gene_Name is typically field index 3 (0-based).
        for ann_entry in info_dict["ANN"].split(","):
            fields_ = ann_entry.split("|")
            if len(fields_) > 3 and fields_[3]:
                genes.add(fields_[3].strip().upper())

    if "CSQ" in info_dict and info_dict["CSQ"]:
        # VEP CSQ: pipe-delimited, SYMBOL field position varies by config;
        # fall back to scanning all fields for something gene-name-shaped.
        for csq_entry in info_dict["CSQ"].split(","):
            fields_ = csq_entry.split("|")
            for f_ in fields_:
                if re.fullmatch(r"[A-Za-z0-9\-]{2,20}", f_ or ""):
                    genes.add(f_.strip().upper())

    return genes


def passes_filters(record, args, stats):
    # Chromosome filter
    if args.chrom:
        allowed = {c.upper() for c in args.chrom}
        if record.chrom.upper() not in allowed:
            stats.dropped_chrom += 1
            return False

    # Quality filter
    if args.min_qual is not None:
        try:
            qual_value = float(record.qual)
        except ValueError:
            stats.dropped_qual += 1
            return False
        if qual_value < args.min_qual:
            stats.dropped_qual += 1
            return False

    info_dict = parse_info(record.info)

    # Allele frequency filter
    if args.min_af is not None or args.max_af is not None:
        freqs = extract_allele_frequencies(info_dict)
        if freqs is None:
            stats.dropped_af_missing += 1
            return False
        # A record passes if ANY of its (possibly multi-allelic) AF values
        # fall within the requested range.
        in_range = False
        for freq in freqs:
            lo_ok = args.min_af is None or freq >= args.min_af
            hi_ok = args.max_af is None or freq <= args.max_af
            if lo_ok and hi_ok:
                in_range = True
                break
        if not in_range:
            stats.dropped_af += 1
            return False

    # Gene filter
    if args.gene:
        wanted = {g.upper() for g in args.gene}
        found_genes = extract_genes(info_dict)
        if not (wanted & found_genes):
            stats.dropped_gene += 1
            return False

    return True


def filter_vcf(input_path, output_path, args):
    stats = FilterStats()
    header_lines = []

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            line = line.rstrip("\n")

            if line.startswith("##"):
                header_lines.append(line)
                outfile.write(line + "\n")
                continue

            if line.startswith("#CHROM"):
                header_lines.append(line)
                outfile.write(line + "\n")
                continue

            if not line.strip():
                continue

            fields_ = line.split("\t")
            if len(fields_) < 8:
                # Malformed data line; skip but count it.
                stats.total += 1
                continue

            record = VcfRecord(
                chrom=fields_[0],
                pos=fields_[1],
                var_id=fields_[2],
                ref=fields_[3],
                alt=fields_[4],
                qual=fields_[5],
                filt=fields_[6],
                info=fields_[7],
                rest=fields_[8:],
                raw=line,
            )

            stats.total += 1

            if passes_filters(record, args, stats):
                stats.kept += 1
                outfile.write(line + "\n")

    if not header_lines:
        print("Warning: no VCF header lines (##...) found -- is this a valid VCF?", file=sys.stderr)

    return stats


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Filter a VCF file by quality, chromosome, allele frequency, and/or gene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_vcf", help="Path to the input VCF file.")
    parser.add_argument("output_vcf", help="Path to write the filtered VCF file.")
    parser.add_argument(
        "--chrom",
        nargs="+",
        default=None,
        help="Only keep variants on these chromosome(s), e.g. --chrom chr1 chr2.",
    )
    parser.add_argument(
        "--min-qual",
        type=float,
        default=None,
        help="Minimum QUAL score required to keep a variant.",
    )
    parser.add_argument(
        "--min-af",
        type=float,
        default=None,
        help="Minimum allele frequency (INFO/AF) required to keep a variant.",
    )
    parser.add_argument(
        "--max-af",
        type=float,
        default=None,
        help="Maximum allele frequency (INFO/AF) allowed to keep a variant.",
    )
    parser.add_argument(
        "--gene",
        nargs="+",
        default=None,
        help="Only keep variants annotated with these gene symbol(s), e.g. --gene BRCA1 TP53.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    stats = filter_vcf(args.input_vcf, args.output_vcf, args)

    print("Variant Filter Summary")
    print("-----------------------")
    print(f"Input file:          {args.input_vcf}")
    print(f"Output file:         {args.output_vcf}")
    print(f"Total variants read: {stats.total}")
    print(f"Variants kept:       {stats.kept}")
    print(f"Dropped (chrom):     {stats.dropped_chrom}")
    print(f"Dropped (qual):      {stats.dropped_qual}")
    print(f"Dropped (AF range):  {stats.dropped_af}")
    print(f"Dropped (AF missing):{stats.dropped_af_missing}")
    print(f"Dropped (gene):      {stats.dropped_gene}")


if __name__ == "__main__":
    main()
