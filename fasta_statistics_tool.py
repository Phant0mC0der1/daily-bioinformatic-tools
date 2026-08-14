#!/usr/bin/env python3
"""FASTA Statistics Tool.

Computes per-sequence statistics for a FASTA file: length, GC content
(nucleotide sequences), amino acid composition (protein sequences), and
molecular weight. Auto-detects whether each sequence is nucleotide or
protein.

Usage:
    python fasta_statistics_tool.py input.fasta [-o output.csv]
"""

import argparse
import csv
import sys

NUCLEOTIDE_LETTERS = set("ACGTUN")

# Average molecular weights of nucleotide monophosphates (DNA/RNA), g/mol.
DNA_MONO_WEIGHTS = {"A": 313.21, "C": 289.18, "G": 329.21, "T": 304.2, "N": 308.95}
RNA_MONO_WEIGHTS = {"A": 329.2, "C": 305.2, "G": 345.2, "U": 306.2, "N": 321.5}

# Average residue masses of amino acids (monomer in a peptide chain), g/mol.
AA_RESIDUE_WEIGHTS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "E": 129.1155, "Q": 128.1307, "G": 57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_WEIGHT = 18.0153

STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header = None
    seq_chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)


def is_nucleotide(sequence):
    if not sequence:
        return True
    letters = set(sequence.upper())
    return letters.issubset(NUCLEOTIDE_LETTERS)


def gc_content(sequence):
    seq = sequence.upper()
    if not seq:
        return 0.0
    gc = sum(seq.count(base) for base in "GC")
    return 100.0 * gc / len(seq)


def amino_acid_composition(sequence):
    seq = sequence.upper()
    length = len(seq)
    counts = {aa: seq.count(aa) for aa in STANDARD_AMINO_ACIDS}
    percentages = {
        aa: (100.0 * count / length if length else 0.0)
        for aa, count in counts.items()
    }
    return counts, percentages


def nucleotide_molecular_weight(sequence):
    seq = sequence.upper()
    is_rna = "U" in seq and "T" not in seq
    weights = RNA_MONO_WEIGHTS if is_rna else DNA_MONO_WEIGHTS
    total = sum(weights.get(base, weights["N"]) for base in seq)
    if total == 0.0:
        return 0.0
    return total - (len(seq) - 1) * 18.0153 if len(seq) > 1 else total


def protein_molecular_weight(sequence):
    seq = sequence.upper()
    residues = [aa for aa in seq if aa in AA_RESIDUE_WEIGHTS]
    if not residues:
        return 0.0
    return sum(AA_RESIDUE_WEIGHTS[aa] for aa in residues) + WATER_WEIGHT


def analyze_sequence(header, sequence):
    seq_type = "nucleotide" if is_nucleotide(sequence) else "protein"
    result = {
        "id": header.split()[0] if header else "",
        "description": header,
        "type": seq_type,
        "length": len(sequence),
    }
    if seq_type == "nucleotide":
        result["gc_content_percent"] = round(gc_content(sequence), 2)
        result["molecular_weight_daltons"] = round(
            nucleotide_molecular_weight(sequence), 2
        )
    else:
        counts, percentages = amino_acid_composition(sequence)
        result["molecular_weight_daltons"] = round(
            protein_molecular_weight(sequence), 2
        )
        for aa in STANDARD_AMINO_ACIDS:
            result[f"aa_{aa}_percent"] = round(percentages[aa], 2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute FASTA sequence statistics.")
    parser.add_argument("fasta_file", help="Path to the input FASTA file.")
    parser.add_argument("-o", "--output", help="Optional path to write results as CSV.")
    args = parser.parse_args()

    records = list(parse_fasta(args.fasta_file))
    if not records:
        print("No sequences found in the input file.", file=sys.stderr)
        sys.exit(1)

    results = [analyze_sequence(header, seq) for header, seq in records]

    for row in results:
        print(f"\n>{row['description']}")
        print(f"  Type:              {row['type']}")
        print(f"  Length:            {row['length']} residues")
        print(f"  Molecular weight:  {row['molecular_weight_daltons']} Da")
        if row["type"] == "nucleotide":
            print(f"  GC content:        {row['gc_content_percent']}%")
        else:
            top_aa = sorted(
                (aa for aa in STANDARD_AMINO_ACIDS),
                key=lambda aa: row[f"aa_{aa}_percent"],
                reverse=True,
            )[:5]
            composition = ", ".join(f"{aa}={row[f'aa_{aa}_percent']}%" for aa in top_aa)
            print(f"  Top AA composition: {composition}")

    if args.output:
        fieldnames = ["id", "description", "type", "length", "molecular_weight_daltons"]
        if any(r["type"] == "nucleotide" for r in results):
            fieldnames.append("gc_content_percent")
        if any(r["type"] == "protein" for r in results):
            fieldnames.extend(f"aa_{aa}_percent" for aa in STANDARD_AMINO_ACIDS)
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
