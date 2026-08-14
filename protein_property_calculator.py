#!/usr/bin/env python3
"""
Protein Property Calculator
============================
Computes key physicochemical properties for protein sequences in a FASTA file:
  - Molecular weight (MW)
  - Isoelectric point (pI)
  - Hydrophobicity (GRAVY - Grand Average of Hydropathy)
  - Aromaticity

Usage:
    python protein_property_calculator.py -i proteins.fasta -o properties.csv
    python protein_property_calculator.py -i proteins.fasta   # prints to stdout only

Requires: biopython (pip install biopython)
"""

import argparse
import csv
import sys

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq):
    """Uppercase the sequence and strip any characters ProtParam can't handle (X, *, gaps, etc.)."""
    seq = str(seq).upper().replace("*", "")
    return "".join(ch for ch in seq if ch in VALID_AA)


def analyze_record(record):
    original_seq = str(record.seq)
    cleaned_seq = clean_sequence(original_seq)

    if not cleaned_seq:
        return {
            "id": record.id,
            "length": len(original_seq),
            "molecular_weight": None,
            "isoelectric_point": None,
            "gravy": None,
            "aromaticity": None,
            "note": "No standard amino acids found; could not analyze.",
        }

    analysis = ProteinAnalysis(cleaned_seq)
    dropped = len(original_seq) - len(cleaned_seq)

    return {
        "id": record.id,
        "length": len(original_seq),
        "molecular_weight": round(analysis.molecular_weight(), 2),
        "isoelectric_point": round(analysis.isoelectric_point(), 2),
        "gravy": round(analysis.gravy(), 4),
        "aromaticity": round(analysis.aromaticity(), 4),
        "note": f"{dropped} non-standard residue(s) ignored" if dropped else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute pI, MW, hydrophobicity (GRAVY), and aromaticity for protein sequences in a FASTA file.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file of protein sequences")
    parser.add_argument("-o", "--output", help="Output CSV file (optional; results are always printed to stdout)")
    args = parser.parse_args()

    try:
        records = list(SeqIO.parse(args.input, "fasta"))
    except FileNotFoundError:
        sys.exit(f"Error: input file not found: {args.input}")

    if not records:
        sys.exit(f"Error: no sequences found in {args.input}")

    results = [analyze_record(r) for r in records]

    fieldnames = ["id", "length", "molecular_weight", "isoelectric_point", "gravy", "aromaticity", "note"]

    print(f"{'ID':<20}{'Length':>8}{'MW (Da)':>12}{'pI':>8}{'GRAVY':>10}{'Aromaticity':>13}")
    print("-" * 75)
    for row in results:
        mw = row["molecular_weight"] if row["molecular_weight"] is not None else "NA"
        pi = row["isoelectric_point"] if row["isoelectric_point"] is not None else "NA"
        gravy = row["gravy"] if row["gravy"] is not None else "NA"
        arom = row["aromaticity"] if row["aromaticity"] is not None else "NA"
        print(f"{row['id']:<20}{row['length']:>8}{mw:>12}{pi:>8}{gravy:>10}{arom:>13}")

    if args.output:
        with open(args.output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
