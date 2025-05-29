#!/usr/bin/env python3

import argparse

def add_fasta_headers(input_file, output_file, prefix="seq"):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile, start=1):
            sequence = line.strip()
            if sequence:  # Skip empty lines
                outfile.write(f">{prefix}{idx}\n{sequence}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add FASTA headers to plain sequence file.")
    parser.add_argument("input_file", help="Input file with one sequence per line")
    parser.add_argument("output_file", help="Output FASTA-formatted file")
    parser.add_argument("--prefix", default="seq", help="Prefix for sequence headers (default: 'seq')")

    args = parser.parse_args()
    add_fasta_headers(args.input_file, args.output_file, prefix=args.prefix)

