## original blast_benchmark.py 
#!/usr/bin/env python3

import os, time, argparse, subprocess
from datetime import datetime
from pathlib import Path

def read_fasta(fasta_path):
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

def run_blast(query_fasta, db, outfmt=6, evalue=1e-5):
    cmd = [
        "blastp",
        "-query", query_fasta,
        "-db", db,
        "-outfmt", str(outfmt),
        "-evalue", str(evalue),
        "-max_target_seqs", "5"
    ]
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - start

def main(fasta, db, output):
    tmpdir = Path("blast_tmp")
    tmpdir.mkdir(exist_ok=True)
    results = []

    for i, (header, seq) in enumerate(read_fasta(fasta), 1):
        query_file = tmpdir / f"query_{i}.fa"
        with open(query_file, "w") as f:
            f.write(f"{header}\n{seq}\n")

        runtime = run_blast(str(query_file), db)
        results.append((header, runtime))
        print(f"{i:04}: {header} -> {runtime:.2f} sec")

    with open(output, "w") as f:
        f.write("header\truntime_sec\n")
        for h, t in results:
            f.write(f"{h}\t{t:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="FASTA query file")
    parser.add_argument("-d", "--db", required=True, help="BLAST DB name")
    parser.add_argument("-o", "--output", default="blast_runtime.tsv")
    args = parser.parse_args()

    main(args.query, args.db, args.output)
