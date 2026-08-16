#!/usr/bin/env python3
"""Restriction Enzyme Finder.

Locates restriction enzyme recognition sites and cut positions in DNA
sequences from a FASTA file. Ships with a library of common palindromic
restriction enzymes (EcoRI, BamHI, HindIII, NotI, etc.); a custom enzyme
can also be supplied on the command line.

Usage:
    python restriction_enzyme_finder.py input.fasta
    python restriction_enzyme_finder.py input.fasta -e EcoRI BamHI NotI
    python restriction_enzyme_finder.py input.fasta -o cut_sites.csv
    python restriction_enzyme_finder.py input.fasta --custom MyEnzyme GA^ATTC
    python restriction_enzyme_finder.py --list-enzymes
"""

import argparse
import csv
import re
import sys

# Common Type II restriction enzymes and their palindromic recognition
# sequences, written with "^" marking the cut position on the top strand.
# Because these sites are palindromic, a single forward-strand search also
# accounts for the site on the reverse strand.
RESTRICTION_ENZYMES = {
    "EcoRI": "G^AATTC",
    "BamHI": "G^GATCC",
    "HindIII": "A^AGCTT",
    "NotI": "GC^GGCCGC",
    "XhoI": "C^TCGAG",
    "PstI": "CTGCA^G",
    "SmaI": "CCC^GGG",
    "SacI": "GAGCT^C",
    "KpnI": "GGTAC^C",
    "SalI": "G^TCGAC",
    "NdeI": "CA^TATG",
    "NcoI": "C^CATGG",
    "XbaI": "T^CTAGA",
    "SpeI": "A^CTAGT",
    "ApaI": "GGGCC^C",
    "BglII": "A^GATCT",
    "ClaI": "AT^CGAT",
    "EcoRV": "GAT^ATC",
    "HaeIII": "GG^CC",
    "PvuII": "CAG^CTG",
    "MluI": "A^CGCGT",
}

IUPAC_TO_REGEX = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}


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


def parse_enzyme_pattern(marked_site):
    """Convert a 'GA^ATTC'-style pattern into (plain_site, cut_offset, regex)."""
    if marked_site.count("^") != 1:
        raise ValueError(
            f"Recognition site '{marked_site}' must contain exactly one '^' "
            "marking the cut position."
        )
    cut_offset = marked_site.index("^")
    plain_site = marked_site.replace("^", "").upper()
    if not plain_site:
        raise ValueError(f"Recognition site '{marked_site}' has no bases.")
    invalid = set(plain_site) - set(IUPAC_TO_REGEX)
    if invalid:
        raise ValueError(
            f"Recognition site '{marked_site}' has unsupported base code(s): "
            f"{', '.join(sorted(invalid))}"
        )
    regex = "".join(IUPAC_TO_REGEX[base] for base in plain_site)
    return plain_site, cut_offset, re.compile(f"(?=({regex}))")


def find_cut_sites(sequence, enzymes):
    """Find every recognition site occurrence for each enzyme in sequence.

    `enzymes` maps enzyme name -> marked recognition site (e.g. "G^AATTC").
    Returns a list of dicts with 1-based recognition-site and cut positions.
    Overlapping matches are all reported (via lookahead scanning).
    """
    seq = sequence.upper()
    hits = []
    for name, marked_site in enzymes.items():
        plain_site, cut_offset, pattern = parse_enzyme_pattern(marked_site)
        for match in pattern.finditer(seq):
            start = match.start()
            hits.append({
                "enzyme": name,
                "recognition_site": marked_site,
                "site_start": start + 1,
                "site_end": start + len(plain_site),
                "cut_position": start + cut_offset + 1,
            })
    hits.sort(key=lambda h: (h["site_start"], h["enzyme"]))
    return hits


def main():
    parser = argparse.ArgumentParser(
        description="Locate restriction enzyme recognition and cut sites in a FASTA file."
    )
    parser.add_argument("fasta_file", nargs="?", help="Path to the input FASTA file.")
    parser.add_argument(
        "-e", "--enzymes", nargs="+", metavar="NAME",
        help="Restrict the search to these enzyme names (default: all known enzymes).",
    )
    parser.add_argument(
        "--custom", nargs=2, action="append", metavar=("NAME", "SITE"),
        default=[],
        help="Add a custom enzyme, e.g. --custom MyEnzyme GA^ATTC. Repeatable.",
    )
    parser.add_argument("-o", "--output", help="Optional path to write hits as CSV.")
    parser.add_argument(
        "--list-enzymes", action="store_true",
        help="Print the built-in enzyme library and exit.",
    )
    args = parser.parse_args()

    if args.list_enzymes:
        for name, site in sorted(RESTRICTION_ENZYMES.items()):
            print(f"  {name:10s} {site}")
        return

    if not args.fasta_file:
        parser.error("fasta_file is required unless --list-enzymes is given.")

    enzyme_library = dict(RESTRICTION_ENZYMES)
    for name, site in args.custom:
        enzyme_library[name] = site

    if args.enzymes:
        missing = [name for name in args.enzymes if name not in enzyme_library]
        if missing:
            print(f"Unknown enzyme(s): {', '.join(missing)}", file=sys.stderr)
            print("Use --list-enzymes to see available enzymes.", file=sys.stderr)
            sys.exit(1)
        enzymes = {name: enzyme_library[name] for name in args.enzymes}
    else:
        enzymes = enzyme_library

    records = list(parse_fasta(args.fasta_file))
    if not records:
        print("No sequences found in the input file.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for header, sequence in records:
        seq_id = header.split()[0] if header else ""
        hits = find_cut_sites(sequence, enzymes)
        print(f"\n>{header}  ({len(sequence)} bp)")
        if not hits:
            print("  No recognition sites found.")
        for hit in hits:
            print(
                f"  {hit['enzyme']:10s} {hit['recognition_site']:12s} "
                f"site {hit['site_start']}-{hit['site_end']}  cut after position {hit['cut_position']}"
            )
            row = dict(hit)
            row["sequence_id"] = seq_id
            row["description"] = header
            all_rows.append(row)

        counts = {}
        for hit in hits:
            counts[hit["enzyme"]] = counts.get(hit["enzyme"], 0) + 1
        if counts:
            summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            print(f"  Summary: {summary}")

    if args.output:
        fieldnames = [
            "sequence_id", "description", "enzyme", "recognition_site",
            "site_start", "site_end", "cut_position",
        ]
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
