#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_sequences(records_chunk):
    local_hits = []
    for rec in records_chunk:
        header = rec.description.split()
        if len(header) < 3:
            continue
        pfam_acc = header[2].split(";", 1)[0]
        seq = str(rec.seq)

        for match in mega_pat.finditer(seq):
            local_hits.append((match.group(), pfam_acc))
    return local_hits

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: motif_pfam_pandas_parallel.py Algal.tsv Bact.tsv Pfam-A.fasta output.tsv", file=sys.stderr)
        sys.exit(1)

    algal_tsv, bact_tsv, pfam_fasta, out_tsv = sys.argv[1:]

    # 1) Load and concatenate motif files
    df_m = pd.concat([
            pd.read_csv(algal_tsv, sep="\t", usecols=["motif", "avg_score"]),
            pd.read_csv(bact_tsv, sep="\t", usecols=["motif", "avg_score"])
        ], ignore_index=True)

    # 2) Filter pure A–Z motifs and compute mean influence score
    df_m = df_m[df_m["motif"].str.match(r"^[A-Z]+$")]
    df_scores = df_m.groupby("motif", as_index=False)["avg_score"] \
                    .mean().rename(columns={"avg_score": "mean_score"})

    # 3) Precompile a mega-regex (must be global for multiprocessing)
    motifs = df_scores["motif"].tolist()
    mega_pat = re.compile("|".join(re.escape(m) for m in motifs))

    # 4) Load Pfam entries into memory
    print("Loading Pfam sequences...", file=sys.stderr)
    records = list(SeqIO.parse(pfam_fasta, "fasta"))
    print(f"Loaded {len(records)} sequences.", file=sys.stderr)

    # 5) Split into chunks
    #n_cores = min(cpu_count(), 64)  # Use up to 64 cores
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    chunk_size = (len(records) + n_cores - 1) // n_cores
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    # 6) Parallel motif search
    print(f"Scanning with {n_cores} cores...", file=sys.stderr)
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_sequences, chunks), total=len(chunks)))

    # 7) Flatten results
    all_hits = [hit for chunk_hits in results for hit in chunk_hits]

    # 8) Build DataFrame
    if not all_hits:
        print("No motif hits found in Pfam-A.", file=sys.stderr)
        sys.exit(1)

    df_hits = pd.DataFrame(all_hits, columns=["motif", "pfam_acc"])
    df_count = df_hits.groupby(["motif", "pfam_acc"], as_index=False) \
                      .size().rename(columns={"size": "count"})

    # 9) Merge and output
    df_out = df_count.merge(df_scores, on="motif") \
                     .sort_values(["motif", "count"], ascending=[True, False])

    df_out.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {len(df_out)} rows to {out_tsv}")

