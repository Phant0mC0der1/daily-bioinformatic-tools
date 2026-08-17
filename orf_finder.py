#!/usr/bin/env python3
"""
ORF Finder
----------
Identifies open reading frames (ORFs) in DNA sequences from a FASTA file.
Scans all 6 reading frames (3 forward, 3 reverse-complement), reports every
ORF bounded by a start codon (ATG) and an in-frame stop codon (TAA/TAG/TGA),
and highlights the longest ORF per sequence.

Usage:
    python orf_finder.py input.fasta [--min-length 100] [-o OUTPUT_PREFIX]

Outputs:
    <prefix>_orfs.csv      - every ORF found, one row per ORF
    <prefix>_longest.fasta - longest ORF (translated protein) per input sequence
"""

import argparse
import csv
import sys
from pathlib import Path

START_CODON = "ATG"
STOP_CODONS = {"TAA", "TAG", "TGA"}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def parse_fasta(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.upper())
    if header is not None:
        yield header, "".join(seq_chunks)


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def translate(sequence):
    """Translate a nucleotide sequence (assumed length is a multiple of 3)."""
    protein = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        protein.append(CODON_TABLE.get(codon, "X"))
    return "".join(protein)


def find_orfs_in_frame(sequence, frame, strand, seq_len):
    """Find all ATG...stop ORFs in a single reading frame of `sequence`."""
    orfs = []
    codons = [sequence[i:i + 3] for i in range(frame, len(sequence) - 2, 3)]

    start_index = None
    for codon_pos, codon in enumerate(codons):
        if start_index is None:
            if codon == START_CODON:
                start_index = codon_pos
        elif codon in STOP_CODONS:
            nt_start = frame + start_index * 3
            nt_end = frame + (codon_pos + 1) * 3  # exclusive, includes stop codon
            orf_seq = sequence[nt_start:nt_end]

            if strand == "+":
                genomic_start = nt_start + 1  # 1-based
                genomic_end = nt_end
            else:
                # Coordinates on the original (forward) sequence.
                genomic_start = seq_len - nt_end + 1
                genomic_end = seq_len - nt_start

            orfs.append({
                "strand": strand,
                "frame": frame + 1,
                "start": genomic_start,
                "end": genomic_end,
                "length_nt": len(orf_seq),
                "protein": translate(orf_seq[:-3]),  # exclude stop codon
            })
            start_index = None
    return orfs


def find_all_orfs(sequence, min_length):
    """Find ORFs across all 6 reading frames for one sequence."""
    seq_len = len(sequence)
    rev_seq = reverse_complement(sequence)

    all_orfs = []
    for frame in range(3):
        all_orfs.extend(find_orfs_in_frame(sequence, frame, "+", seq_len))
    for frame in range(3):
        all_orfs.extend(find_orfs_in_frame(rev_seq, frame, "-", seq_len))

    return [orf for orf in all_orfs if orf["length_nt"] >= min_length]


def main():
    parser = argparse.ArgumentParser(description="Find ORFs (start/stop codons) in DNA sequences")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--min-length", type=int, default=75,
                         help="Minimum ORF length in nucleotides, including stop codon (default: 75)")
    parser.add_argument("-o", "--output-prefix", default="orf_finder", help="Output file prefix")
    args = parser.parse_args()

    if args.min_length < 6:
        sys.exit("--min-length must be at least 6 (start codon + stop codon).")

    records = list(parse_fasta(args.fasta))
    if not records:
        sys.exit(f"No sequences found in {args.fasta}")

    csv_path = Path(f"{args.output_prefix}_orfs.csv")
    longest_path = Path(f"{args.output_prefix}_longest.fasta")

    longest_records = []

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "sequence_id", "strand", "frame", "start", "end",
            "length_nt", "length_aa", "protein",
        ])

        for header, sequence in records:
            seq_id = header.split()[0] if header else "sequence"
            orfs = find_all_orfs(sequence, args.min_length)

            if not orfs:
                print(f"'{seq_id}': no ORFs found with length >= {args.min_length} nt.")
                continue

            orfs.sort(key=lambda o: o["length_nt"], reverse=True)

            for orf in orfs:
                writer.writerow([
                    seq_id, orf["strand"], orf["frame"], orf["start"], orf["end"],
                    orf["length_nt"], len(orf["protein"]), orf["protein"],
                ])

            longest = orfs[0]
            print(
                f"'{seq_id}': {len(orfs)} ORF(s) found; longest = {longest['length_nt']} nt "
                f"(strand {longest['strand']}, frame {longest['frame']}, "
                f"{longest['start']}-{longest['end']})"
            )
            longest_records.append((seq_id, longest))

    with open(longest_path, "w") as fasta_out:
        for seq_id, orf in longest_records:
            fasta_out.write(
                f">{seq_id}_longest_ORF strand={orf['strand']} frame={orf['frame']} "
                f"{orf['start']}-{orf['end']} length_nt={orf['length_nt']}\n"
            )
            fasta_out.write(orf["protein"] + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {longest_path}")


if __name__ == "__main__":
    main()
