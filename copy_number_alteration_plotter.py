#!/usr/bin/env python3
"""
Copy Number Alteration Plotter

Visualises amplifications and deletions from a copy-number segmentation file
(e.g. CNVkit .cns, DNAcopy/GATK-style segment tables). Segments are classified
as amplified, deleted, or copy-neutral from their log2 ratio, plotted across
the whole genome on a single concatenated coordinate axis, and summarised in
a CSV report.

Usage:
    python copy_number_alteration_plotter.py --input sample.cns
    python copy_number_alteration_plotter.py --input segments.csv --log2-column log2 --chrom-column chromosome
    python copy_number_alteration_plotter.py --input segments.tsv --amp-threshold 0.4 --del-threshold -0.4
    python copy_number_alteration_plotter.py --demo

Input file requirements:
    A comma/tab-delimited table with a chromosome column, a segment start
    column, a segment end column, and either a log2 copy-ratio column or an
    absolute copy-number column (from which log2(CN / 2) is derived). Column
    names are auto-detected from common conventions (chromosome/chrom/chr,
    start/loc.start, end/loc.end, log2/log2ratio/seg.mean, copy_number/cn),
    or can be supplied explicitly with --chrom-column/--start-column/
    --end-column/--log2-column/--cn-column.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHROM_ORDER_SPECIAL = {"X": 100, "Y": 101, "MT": 102, "M": 102}

AMP_COLOR = "#C44E52"
DEL_COLOR = "#4C72B0"
NEUTRAL_COLOR = "#999999"


def chrom_sort_key(chrom):
    name = chrom.lower().replace("chr", "").upper()
    if name in CHROM_ORDER_SPECIAL:
        return (CHROM_ORDER_SPECIAL[name], name)
    try:
        return (int(name), name)
    except ValueError:
        return (999, name)


def normalise_chrom(chrom):
    return chrom if chrom.lower().startswith("chr") else f"chr{chrom}"


def sniff_delimiter(path):
    with open(path, newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return "\t"


def find_column(fieldnames, requested, candidates, required=True):
    if requested:
        for name in fieldnames:
            if name.lower() == requested.lower():
                return name
        raise ValueError(f"Column '{requested}' not found. Available columns: {fieldnames}")
    for candidate in candidates:
        for name in fieldnames:
            if name.lower() == candidate:
                return name
    if required:
        raise ValueError(
            f"Could not auto-detect a required column from {candidates}. "
            f"Available columns: {fieldnames}. Pass the matching --*-column option explicitly."
        )
    return None


def load_segments(path, chrom_column, start_column, end_column, log2_column, cn_column):
    delimiter = sniff_delimiter(path)
    segments = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("Input file has no header row.")
        fields = reader.fieldnames
        chrom_col = find_column(fields, chrom_column, ["chromosome", "chrom", "chr"])
        start_col = find_column(fields, start_column, ["start", "loc.start", "start_pos"])
        end_col = find_column(fields, end_column, ["end", "loc.end", "end_pos"])
        log2_col = find_column(fields, log2_column, ["log2", "log2ratio", "log2_ratio", "seg.mean", "segmean"], required=False)
        cn_col = None if log2_col else find_column(fields, cn_column, ["copy_number", "cn", "copynumber"])

        for row in reader:
            try:
                chrom = normalise_chrom(str(row[chrom_col]).strip())
                start = int(float(row[start_col]))
                end = int(float(row[end_col]))
                if log2_col:
                    log2 = float(row[log2_col])
                else:
                    cn = float(row[cn_col])
                    if cn <= 0:
                        continue
                    log2 = __import__("math").log2(cn / 2.0)
            except (ValueError, KeyError):
                continue
            segments.append({"chrom": chrom, "start": start, "end": end, "log2": log2})
    return segments


def make_demo_segments():
    random.seed(42)
    chrom_lengths = {"chr1": 249_000_000, "chr7": 159_000_000, "chr8": 145_000_000,
                      "chr17": 83_000_000, "chrX": 156_000_000}
    events = {
        "chr7": [(50_000_000, 90_000_000, 0.65)],
        "chr8": [(20_000_000, 60_000_000, 0.9)],
        "chr17": [(5_000_000, 25_000_000, -0.85)],
        "chrX": [(1, 156_000_000, -0.55)],
    }
    segments = []
    for chrom, length in chrom_lengths.items():
        cursor = 1
        boundaries = sorted(events.get(chrom, []), key=lambda e: e[0])
        for start, end, log2 in boundaries:
            if cursor < start:
                segments.append({"chrom": chrom, "start": cursor, "end": start - 1,
                                  "log2": random.gauss(0, 0.08)})
            segments.append({"chrom": chrom, "start": start, "end": end,
                              "log2": log2 + random.gauss(0, 0.06)})
            cursor = end + 1
        if cursor < length:
            segments.append({"chrom": chrom, "start": cursor, "end": length,
                              "log2": random.gauss(0, 0.08)})
    return segments


def classify(segments, amp_threshold, del_threshold):
    for seg in segments:
        if seg["log2"] >= amp_threshold:
            seg["call"] = "amplification"
        elif seg["log2"] <= del_threshold:
            seg["call"] = "deletion"
        else:
            seg["call"] = "neutral"
    return segments


def write_summary(segments, out_prefix):
    summary_path = f"{out_prefix}_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["chrom", "start", "end", "length", "log2", "call"])
        for seg in sorted(segments, key=lambda s: (chrom_sort_key(s["chrom"]), s["start"])):
            writer.writerow([seg["chrom"], seg["start"], seg["end"],
                              seg["end"] - seg["start"] + 1, f"{seg['log2']:.4f}", seg["call"]])
    return summary_path


def plot_genome(segments, out_prefix, amp_threshold, del_threshold):
    chroms = sorted({seg["chrom"] for seg in segments}, key=chrom_sort_key)
    chrom_max = {c: max(seg["end"] for seg in segments if seg["chrom"] == c) for c in chroms}

    offsets = {}
    cursor = 0
    for chrom in chroms:
        offsets[chrom] = cursor
        cursor += chrom_max[chrom] + 1

    fig, ax = plt.subplots(figsize=(14, 5))
    color_map = {"amplification": AMP_COLOR, "deletion": DEL_COLOR, "neutral": NEUTRAL_COLOR}

    for seg in segments:
        x0 = offsets[seg["chrom"]] + seg["start"]
        x1 = offsets[seg["chrom"]] + seg["end"]
        ax.plot([x0, x1], [seg["log2"], seg["log2"]],
                color=color_map[seg["call"]], linewidth=2.5, solid_capstyle="butt")

    tick_positions = []
    for chrom in chroms:
        boundary = offsets[chrom]
        ax.axvline(boundary, color="lightgray", linewidth=0.6, zorder=0)
        tick_positions.append(boundary + chrom_max[chrom] / 2)

    ax.axhline(0, color="black", linewidth=0.7, zorder=0)
    ax.axhline(amp_threshold, color=AMP_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axhline(del_threshold, color=DEL_COLOR, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([c.replace("chr", "") for c in chroms], fontsize=9)
    ax.set_xlim(0, cursor)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("log2(copy ratio)")
    ax.set_title("Copy Number Alterations Across the Genome")

    handles = [
        plt.Line2D([0], [0], color=AMP_COLOR, linewidth=2.5, label="Amplification"),
        plt.Line2D([0], [0], color=NEUTRAL_COLOR, linewidth=2.5, label="Neutral"),
        plt.Line2D([0], [0], color=DEL_COLOR, linewidth=2.5, label="Deletion"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    plot_path = f"{out_prefix}_genome.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Visualise amplifications and deletions from a copy-number segment file.")
    parser.add_argument("--input", type=Path, help="Path to a segment table (CSV/TSV, e.g. CNVkit .cns).")
    parser.add_argument("--chrom-column", help="Name of the chromosome column (auto-detected if omitted).")
    parser.add_argument("--start-column", help="Name of the segment start column (auto-detected if omitted).")
    parser.add_argument("--end-column", help="Name of the segment end column (auto-detected if omitted).")
    parser.add_argument("--log2-column", help="Name of the log2 copy-ratio column (auto-detected if present).")
    parser.add_argument("--cn-column", help="Name of an absolute copy-number column, used if no log2 column is found.")
    parser.add_argument("--amp-threshold", type=float, default=0.3, help="log2 ratio at/above which a segment is called amplified (default: 0.3).")
    parser.add_argument("--del-threshold", type=float, default=-0.3, help="log2 ratio at/below which a segment is called deleted (default: -0.3).")
    parser.add_argument("--out-prefix", default="cna_plotter", help="Prefix for output files.")
    parser.add_argument("--demo", action="store_true", help="Run with built-in synthetic demo data instead of an input file.")
    args = parser.parse_args()

    if args.demo:
        segments = make_demo_segments()
    elif args.input:
        segments = load_segments(args.input, args.chrom_column, args.start_column,
                                  args.end_column, args.log2_column, args.cn_column)
    else:
        parser.error("Provide --input <file> or use --demo.")

    if not segments:
        sys.exit("No valid copy-number segments were found in the input.")

    segments = classify(segments, args.amp_threshold, args.del_threshold)
    summary_path = write_summary(segments, args.out_prefix)
    plot_path = plot_genome(segments, args.out_prefix, args.amp_threshold, args.del_threshold)

    counts = {"amplification": 0, "deletion": 0, "neutral": 0}
    for seg in segments:
        counts[seg["call"]] += 1

    print(f"Total segments: {len(segments)}")
    print(f"  Amplifications: {counts['amplification']}")
    print(f"  Deletions:      {counts['deletion']}")
    print(f"  Neutral:        {counts['neutral']}")
    print(f"Summary written to {summary_path}")
    print(f"Genome plot written to {plot_path}")


if __name__ == "__main__":
    main()
