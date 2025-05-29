#!/usr/bin/env python3
import sys
import re
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm  # live progress bar

if len(sys.argv) != 5:
    print("Usage: motif_pfam_pandas_fast.py Algal.tsv Bact.tsv Pfam-A.fasta output.tsv", file=sys.stderr)
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

# 3) Precompile a mega-regex
motifs = df_scores["motif"].tolist()
mega_pat = re.compile("|".join(re.escape(m) for m in motifs))

# 4) Scan Pfam-A and record hits
hits = []
seq_iterator = SeqIO.parse(pfam_fasta, "fasta")
seq_iterator = tqdm(seq_iterator, desc="Scanning Pfam-A entries")

for rec in seq_iterator:
    header = rec.description.split()
    if len(header) < 3:
        continue
    pfam_acc = header[2].split(";", 1)[0]
    seq = str(rec.seq)

    for match in mega_pat.finditer(seq):
        hits.append({"motif": match.group(), "pfam_acc": pfam_acc})

# 5) Build DataFrame of counts
if not hits:
    print("No motif hits found in Pfam-A.", file=sys.stderr)
    sys.exit(1)

df_hits = pd.DataFrame(hits)
df_count = df_hits.groupby(["motif", "pfam_acc"], as_index=False) \
                  .size().rename(columns={"size": "count"})

# 6) Merge with mean scores and write output
df_out = df_count.merge(df_scores, on="motif") \
                 .sort_values(["motif", "count"], ascending=[True, False])

df_out.to_csv(out_tsv, sep="\t", index=False)
print(f"Wrote {len(df_out)} rows to {out_tsv}")

