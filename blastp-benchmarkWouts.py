#!/usr/bin/env python3

import os, time, argparse, subprocess
from datetime import datetime
from pathlib import Path
import sys

def read_fasta(fasta_path):
    """
    Generator that yields (header, sequence) tuples from a FASTA file.
    """
    with open(fasta_path) as f:
        header, seq = None, []
        for line in f:
            if line.startswith(">"):
                if header:
                    yield header, ''.join(seq)
                header = line.strip()
                seq = []
            else:
                seq.append(line.strip())
        if header:
            yield header, ''.join(seq)


def run_blast(query_fasta, db, blast_out, outfmt=6, evalue=1e-5):
    """
    Runs BLASTP on `query_fasta` against `db`, writing output to `blast_out`,
    and returns the elapsed runtime in seconds.
    """
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db,
        "-outfmt", str(outfmt),
        "-evalue", str(evalue),
        "-max_target_seqs", "5",
        "-out", str(blast_out)
    ]
    start = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - start


def main(fasta, db, output):
    # Directory for temporary per-query FASTA files
    tmpdir = Path("blast_tmp")
    tmpdir.mkdir(exist_ok=True)

    # Directory for BLAST result files
    blast_dir = Path("blast_results")
    blast_dir.mkdir(exist_ok=True)

    results = []

    # Iterate over each sequence in the query FASTA
    for i, (header, seq) in enumerate(read_fasta(fasta), 1):
        query_file = tmpdir / f"query_{i}.fa"
        with open(query_file, "w") as f:
            f.write(f"{header}\n{seq}\n")

        # Run BLAST and save output
        blast_out = blast_dir / f"query_{i}.blast.tsv"
        runtime = run_blast(str(query_file), db, blast_out)

        # Record timing
        results.append((header, runtime))
        print(f"{i:04}: {header} -> {runtime:.2f} sec")

    # Write all runtimes to the summary TSV
    with open(output, "w") as f:
        f.write("header\truntime_sec\n")
        for h, t in results:
            f.write(f"{h}\t{t:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="FASTA query file")
    parser.add_argument("-d", "--db", required=True, help="BLAST DB name")
    parser.add_argument(
        "-o", "--output", default="blast_runtime.tsv",
        help="TSV file to log per-query runtimes"
    )
    args = parser.parse_args()

    main(args.query, args.db, args.output)
