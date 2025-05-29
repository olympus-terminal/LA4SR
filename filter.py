#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load file
df = pd.read_csv("motif_pfam_scores_counts.tsv", sep='\t')

# Group by motif and get max score
def pick_best_pfam(group):
    max_score = group['mean_score'].max()
    top_hits = group[group['mean_score'] == max_score]
    return top_hits.sample(n=1, random_state=42)  # random choice if tie

best_matches = df.groupby('motif', group_keys=False).apply(pick_best_pfam)

# Save output
best_matches.to_csv("motif_best_pfam.csv", index=False)

