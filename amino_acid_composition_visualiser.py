"""
Amino Acid Composition Visualiser

Reads one or more protein sequences from a FASTA file, computes the amino
acid composition of each sequence (and the combined total), and renders
bar chart and pie chart visualisations.

Usage:
    python amino_acid_composition_visualiser.py input.fasta [--outdir OUTDIR] [--per-sequence]

If no input file is given, a small built-in example FASTA is used so the
script runs out of the box.
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

AMINO_ACID_NAMES = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
}

EXAMPLE_FASTA = """\
>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGH
GKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEF
TPAVHASLDKFLASVSTVLTSKYR
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNP
KVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHH
FGKEFTPPVQAAYQKVVAGVANALAHKYH
"""


def parse_fasta(handle):
    header = None
    seq_chunks = []
    for line in handle:
        line = line.rstrip("\n")
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


def composition(sequence):
    sequence = sequence.upper()
    counts = Counter(residue for residue in sequence if residue in AMINO_ACID_NAMES)
    total = sum(counts.values())
    percentages = {aa: (counts.get(aa, 0) / total * 100 if total else 0.0) for aa in STANDARD_AMINO_ACIDS}
    return counts, percentages, total


def plot_composition(percentages, title, out_prefix):
    labels = [f"{aa} ({AMINO_ACID_NAMES[aa]})" for aa in STANDARD_AMINO_ACIDS]
    values = [percentages[aa] for aa in STANDARD_AMINO_ACIDS]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("Percentage of residues (%)")
    ax.set_title(f"Amino Acid Composition (Bar) - {title}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    bar_path = f"{out_prefix}_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    nonzero = [(aa, v) for aa, v in zip(STANDARD_AMINO_ACIDS, values) if v > 0]
    fig, ax = plt.subplots(figsize=(8, 8))
    pie_labels = [aa for aa, _ in nonzero]
    pie_values = [v for _, v in nonzero]
    ax.pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title(f"Amino Acid Composition (Pie) - {title}")
    ax.axis("equal")
    fig.tight_layout()
    pie_path = f"{out_prefix}_pie.png"
    fig.savefig(pie_path, dpi=150)
    plt.close(fig)

    return bar_path, pie_path


def safe_name(text, max_len=40):
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
    return keep[:max_len].strip("_") or "sequence"


def main():
    parser = argparse.ArgumentParser(description="Visualise amino acid composition of protein sequences from a FASTA file.")
    parser.add_argument("fasta", nargs="?", help="Path to input FASTA file. Uses a built-in example if omitted.")
    parser.add_argument("--outdir", default="aa_composition_output", help="Directory to write chart images to.")
    parser.add_argument("--per-sequence", action="store_true", help="Also generate charts for each individual sequence, not just the combined total.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.fasta:
        with open(args.fasta, "r") as handle:
            records = list(parse_fasta(handle))
    else:
        print("No FASTA file given - using built-in example sequences.", file=sys.stderr)
        records = list(parse_fasta(EXAMPLE_FASTA.splitlines(keepends=True)))

    if not records:
        print("No sequences found in input.", file=sys.stderr)
        sys.exit(1)

    combined_counts = Counter()
    for header, seq in records:
        counts, percentages, total = composition(seq)
        combined_counts.update(counts)
        print(f"{header}: {total} residues")
        for aa in STANDARD_AMINO_ACIDS:
            print(f"  {aa} ({AMINO_ACID_NAMES[aa]}): {counts.get(aa, 0):4d}  {percentages[aa]:5.2f}%")

        if args.per_sequence:
            prefix = os.path.join(args.outdir, safe_name(header))
            bar_path, pie_path = plot_composition(percentages, header, prefix)
            print(f"  Saved: {bar_path}, {pie_path}")

    combined_total = sum(combined_counts.values())
    combined_percentages = {aa: (combined_counts.get(aa, 0) / combined_total * 100 if combined_total else 0.0) for aa in STANDARD_AMINO_ACIDS}
    combined_prefix = os.path.join(args.outdir, "combined")
    bar_path, pie_path = plot_composition(combined_percentages, "All Sequences Combined", combined_prefix)
    print(f"\nCombined ({combined_total} residues) charts saved: {bar_path}, {pie_path}")


if __name__ == "__main__":
    main()
