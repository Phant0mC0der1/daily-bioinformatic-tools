#!/usr/bin/env python3
"""
Protein Mutation Mapper

Takes a reference protein sequence (FASTA) and a list of point mutations
(e.g. "A123T", "p.Gly12Asp", or three-letter codes) and maps them onto
protein coordinates: validates each mutation against the reference
residue, reports the results, and draws a lollipop-style diagram showing
where the mutations fall along the protein, optionally overlaid on
annotated domains/regions.

Input:
    --fasta      A FASTA file with a single protein sequence.
    --mutations  A text or CSV file with one mutation per line, e.g.:

        A123T
        p.Gly12Asp
        R248Q

    --domains    Optional CSV of protein regions to draw as bands, e.g.:

        name,start,end
        Kinase domain,25,280
        SH2 domain,300,350

Output:
    A CSV report (--report) listing each mutation, its parsed wild-type
    residue, position, mutant residue, and whether it matches the
    reference sequence, plus a PNG lollipop plot (--output) mapping the
    (valid) mutations onto the protein.

Usage:
    python protein_mutation_mapper.py --fasta protein.fasta \
        --mutations mutations.txt --output mutation_map.png

    python protein_mutation_mapper.py --fasta protein.fasta \
        --mutations mutations.csv --domains domains.csv \
        --output mutation_map.png --report mutation_report.csv
"""

import argparse
import csv
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "TER": "*", "STOP": "*",
}

MUTATION_RE_1LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z*])(\d+)([A-Za-z*])$"
)
MUTATION_RE_3LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3})$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Map point mutations onto a reference protein sequence "
        "and visualise their positions."
    )
    parser.add_argument("--fasta", required=True, help="Reference protein FASTA file.")
    parser.add_argument(
        "--mutations", required=True,
        help="File with one mutation per line (or a CSV with a 'mutation' column), "
        "e.g. A123T or p.Gly12Asp.",
    )
    parser.add_argument(
        "--domains", default=None,
        help="Optional CSV with columns name,start,end describing protein regions "
        "to draw as bands behind the mutations.",
    )
    parser.add_argument(
        "--output", default="mutation_map.png",
        help="Output PNG path for the lollipop plot (default: mutation_map.png).",
    )
    parser.add_argument(
        "--report", default="mutation_report.csv",
        help="Output CSV path for the validation report (default: mutation_report.csv).",
    )
    return parser.parse_args()


def read_fasta_sequence(path):
    seq_lines = []
    header = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is None:
                    header = line[1:].strip()
                continue
            seq_lines.append(line)
    if not seq_lines:
        sys.exit(f"Error: no sequence found in FASTA file '{path}'.")
    return header or "protein", "".join(seq_lines).upper()


def normalise_residue(token):
    token = token.strip()
    if len(token) == 1:
        return token.upper()
    if token.upper() in THREE_TO_ONE:
        return THREE_TO_ONE[token.upper()]
    return token.upper()


def parse_mutation(raw):
    """Return (wildtype, position, mutant) for a mutation string, or None if unparsable."""
    text = raw.strip()
    if not text:
        return None

    match = MUTATION_RE_1LETTER.match(text)
    if match:
        wt, pos, mut = match.groups()
        return normalise_residue(wt), int(pos), normalise_residue(mut)

    match = MUTATION_RE_3LETTER.match(text)
    if match:
        wt, pos, mut = match.groups()
        return normalise_residue(wt), int(pos), normalise_residue(mut)

    return None


def load_mutations(path):
    raw_entries = []
    if path.lower().endswith(".csv"):
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and "mutation" in [f.lower() for f in reader.fieldnames]:
                key = next(f for f in reader.fieldnames if f.lower() == "mutation")
                for row in reader:
                    raw_entries.append(row[key])
            else:
                fh.seek(0)
                for row in csv.reader(fh):
                    if row:
                        raw_entries.append(row[0])
    else:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    raw_entries.append(line)
    return raw_entries


def load_domains(path):
    domains = []
    if not path:
        return domains
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                domains.append((
                    row["name"].strip(),
                    int(row["start"]),
                    int(row["end"]),
                ))
            except (KeyError, ValueError):
                continue
    return domains


def validate_mutations(sequence, raw_entries):
    results = []
    seq_len = len(sequence)
    for raw in raw_entries:
        parsed = parse_mutation(raw)
        if parsed is None:
            results.append({
                "mutation": raw, "wildtype": "", "position": "", "mutant": "",
                "valid": False, "note": "could not parse mutation string",
            })
            continue

        wt, pos, mut = parsed
        if pos < 1 or pos > seq_len:
            note = f"position {pos} outside protein length ({seq_len})"
            results.append({
                "mutation": raw, "wildtype": wt, "position": pos, "mutant": mut,
                "valid": False, "note": note,
            })
            continue

        ref_residue = sequence[pos - 1]
        if ref_residue != wt:
            note = f"reference residue at {pos} is {ref_residue}, not {wt}"
            results.append({
                "mutation": raw, "wildtype": wt, "position": pos, "mutant": mut,
                "valid": False, "note": note,
            })
            continue

        results.append({
            "mutation": raw, "wildtype": wt, "position": pos, "mutant": mut,
            "valid": True, "note": "matches reference",
        })
    return results


def write_report(results, path):
    fieldnames = ["mutation", "wildtype", "position", "mutant", "valid", "note"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_mutation_map(protein_name, seq_len, results, domains, output_path):
    fig, ax = plt.subplots(figsize=(max(10, seq_len / 40), 5))

    backbone_y = 0
    ax.plot([1, seq_len], [backbone_y, backbone_y], color="#444444", linewidth=4, zorder=1)

    domain_colors = plt.cm.Pastel1.colors
    for i, (name, start, end) in enumerate(domains):
        color = domain_colors[i % len(domain_colors)]
        ax.add_patch(Rectangle(
            (start, backbone_y - 0.08), end - start, 0.16,
            facecolor=color, edgecolor="black", linewidth=0.5, zorder=2,
        ))
        ax.text((start + end) / 2, backbone_y - 0.22, name,
                ha="center", va="top", fontsize=8, rotation=0)

    valid_mutations = [r for r in results if r["valid"]]
    invalid_mutations = [r for r in results if not r["valid"] and r["position"] != ""]

    for r in valid_mutations:
        pos = r["position"]
        ax.plot([pos, pos], [backbone_y, 0.6], color="#1f77b4", linewidth=1, zorder=3)
        ax.plot(pos, 0.6, "o", color="#1f77b4", markersize=8, zorder=4)
        label = f"{r['wildtype']}{pos}{r['mutant']}"
        ax.text(pos, 0.65, label, ha="center", va="bottom", fontsize=7, rotation=45)

    for r in invalid_mutations:
        pos = r["position"]
        ax.plot([pos, pos], [backbone_y, -0.6], color="#d62728", linewidth=1, zorder=3)
        ax.plot(pos, -0.6, "x", color="#d62728", markersize=8, zorder=4)

    ax.set_xlim(0, seq_len + 1)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Residue position")
    ax.set_title(f"Mutation map: {protein_name} ({seq_len} aa)")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="#1f77b4", linestyle="None",
                    markersize=8, label="Valid mutation"),
        plt.Line2D([0], [0], marker="x", color="#d62728", linestyle="None",
                    markersize=8, label="Invalid / mismatched"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    protein_name, sequence = read_fasta_sequence(args.fasta)
    raw_mutations = load_mutations(args.mutations)
    if not raw_mutations:
        sys.exit(f"Error: no mutations found in '{args.mutations}'.")

    domains = load_domains(args.domains)
    results = validate_mutations(sequence, raw_mutations)
    write_report(results, args.report)
    plot_mutation_map(protein_name, len(sequence), results, domains, args.output)

    valid_count = sum(1 for r in results if r["valid"])
    print(f"Protein: {protein_name} ({len(sequence)} aa)")
    print(f"Mutations processed: {len(results)} ({valid_count} valid, "
          f"{len(results) - valid_count} invalid/mismatched)")
    print(f"Report written to: {args.report}")
    print(f"Mutation map written to: {args.output}")


if __name__ == "__main__":
    main()
