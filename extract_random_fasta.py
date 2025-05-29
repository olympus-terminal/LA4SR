#!/usr/bin/env python3

import sys
import random
from Bio import SeqIO

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} input.fa output.fa", file=sys.stderr)
    sys.exit(1)

input_fasta = sys.argv[1]
output_fasta = sys.argv[2]
num_records = 15

# Load all records into memory
records = list(SeqIO.parse(input_fasta, "fasta"))

# Error if fewer than 30 records
if len(records) < num_records:
    print(f"Error: Input file contains only {len(records)} records.", file=sys.stderr)
    sys.exit(1)

# Randomly sample without replacement
sampled_records = random.sample(records, num_records)

# Write to output
SeqIO.write(sampled_records, output_fasta, "fasta")

print(f"Wrote {num_records} records to {output_fasta}")

