#!/usr/bin/env python3
"""
Sequence Logo Generator

Build a classic Shannon-information sequence logo from a set of aligned
sequences (nucleotide or protein). Column heights are scaled by the
information content of that position (in bits, with the standard
small-sample-size correction), and each letter's height within a column
is proportional to its frequency there.

Input: an alignment in FASTA format where every sequence has the same
length (e.g. the output of a multiple sequence alignment tool such as
MUSCLE or Clustal Omega). Gap characters ('-', '.') are counted towards
each position's depth but do not contribute a plotted letter.

Usage:
    python sequence_logo_generator.py --alignment aligned.fasta
    python sequence_logo_generator.py --alignment aligned.fasta \
        --output logo.png --start 10 --end 40
"""

import argparse
import math
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from Bio import SeqIO
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

GAP_CHARS = {"-", "."}

DNA_LETTERS = set("ACGTU")
PROTEIN_LETTERS = set("ACDEFGHIKLMNPQRSTVWY")

DNA_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728", "U": "#d62728"}
PROTEIN_COLORS = {
    "A": "#5DA5DA", "R": "#F17CB0", "N": "#B2912F", "D": "#FAA43A",
    "C": "#DECF3F", "Q": "#B2912F", "E": "#FAA43A", "G": "#5DA5DA",
    "H": "#F17CB0", "I": "#5DA5DA", "L": "#5DA5DA", "K": "#F17CB0",
    "M": "#5DA5DA", "F": "#5DA5DA", "P": "#B276B2", "S": "#60BD68",
    "T": "#60BD68", "W": "#5DA5DA", "Y": "#60BD68", "V": "#5DA5DA",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Shannon-information sequence logo from an aligned FASTA file."
    )
    parser.add_argument(
        "--alignment", required=True,
        help="Path to an aligned FASTA file (all sequences must be the same length).",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="1-based alignment column to start the logo at (default: 1).",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="1-based alignment column to end the logo at, inclusive (default: last column).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the logo image (default: <alignment>_logo.png).",
    )
    return parser.parse_args()


def load_alignment(path):
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise SystemExit(f"Error: no sequences found in '{path}'.")
    sequences = [str(rec.seq).upper() for rec in records]
    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        raise SystemExit(
            "Error: sequences are not aligned (differing lengths found: "
            f"{sorted(lengths)}). Provide a multiple sequence alignment."
        )
    return sequences


def detect_alphabet(sequences):
    observed = set("".join(sequences)) - GAP_CHARS
    if observed <= DNA_LETTERS:
        return "nucleotide", DNA_COLORS
    return "protein", PROTEIN_COLORS


def small_sample_correction(alphabet_size, n_sequences):
    return (1.0 / math.log(2)) * (alphabet_size - 1) / (2.0 * n_sequences)


def column_heights(column, alphabet_size, n_sequences):
    counts = Counter(c for c in column if c not in GAP_CHARS)
    depth = sum(counts.values())
    if depth == 0:
        return {}

    max_bits = math.log2(alphabet_size)
    correction = small_sample_correction(alphabet_size, n_sequences)

    entropy = 0.0
    freqs = {}
    for letter, count in counts.items():
        freq = count / depth
        freqs[letter] = freq
        entropy -= freq * math.log2(freq)

    information = max(0.0, max_bits - entropy - correction)
    return {letter: freq * information for letter, freq in freqs.items()}


LETTER_FONT = FontProperties(family="monospace", weight="bold")


def add_letter(ax, letter, x, y, height, width, color):
    if height <= 0:
        return
    glyph = TextPath((0, 0), letter, size=1, prop=LETTER_FONT)
    bbox = glyph.get_extents()
    if bbox.width <= 0 or bbox.height <= 0:
        return

    scale_x = width / bbox.width
    scale_y = height / bbox.height
    transform = (
        transforms.Affine2D()
        .translate(-bbox.x0, -bbox.y0)
        .scale(scale_x, scale_y)
        .translate(x - width / 2, y)
        + ax.transData
    )
    ax.add_patch(PathPatch(glyph, transform=transform, facecolor=color, edgecolor="none"))


def plot_logo(sequences, alphabet, colors, start, end, output_path):
    n_sequences = len(sequences)
    alignment_length = len(sequences[0])
    start = 1 if start is None else start
    end = alignment_length if end is None else end
    if start < 1 or end > alignment_length or start > end:
        raise SystemExit(
            f"Error: --start/--end ({start}, {end}) out of range for alignment "
            f"of length {alignment_length}."
        )

    alphabet_size = 4 if alphabet == "nucleotide" else 20
    columns = list(zip(*sequences))[start - 1:end]

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(columns)), 4))
    max_bits = math.log2(alphabet_size)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(0, max_bits)

    for i, column in enumerate(columns):
        heights = column_heights(column, alphabet_size, n_sequences)
        ordered = sorted(heights.items(), key=lambda item: item[1])
        y = 0.0
        for letter, height in ordered:
            color = colors.get(letter, "#7f7f7f")
            add_letter(ax, letter, i + 0.5, y, height, 0.9, color)
            y += height

    tick_positions = np.arange(0.5, len(columns), max(1, len(columns) // 20 or 1))
    tick_labels = [str(start + int(pos)) for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Bits")
    ax.set_title(f"Sequence logo ({alphabet}, n={n_sequences} sequences)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Sequence logo saved to: {output_path}")


def main():
    args = parse_args()

    sequences = load_alignment(args.alignment)
    alphabet, colors = detect_alphabet(sequences)
    print(f"Loaded {len(sequences)} aligned sequences (length {len(sequences[0])}).")
    print(f"Detected alphabet: {alphabet}")

    output_path = args.output or f"{args.alignment.rsplit('.', 1)[0]}_logo.png"
    plot_logo(sequences, alphabet, colors, args.start, args.end, output_path)


if __name__ == "__main__":
    main()
