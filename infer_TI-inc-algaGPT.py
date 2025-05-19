#!/usr/bin/env python3
"""
la4sr_infer_fasta2tsv.py — drop‑in replacement for the older LA4SR
inference utilities used by **run_la4sr_TI-inc.sh**

Changes vs. the legacy “Sample from a trained model” script
----------------------------------------------------------
1. **Parses a FASTA file** and feeds the collapsed sequence to the model.
2. **Generates up to 14 tokens** per record (defaults identical to old code).
3. **Emits a TSV** with columns: record_id, sequence, model_output.
4. CLI knobs match the wrapper (temperature, top‑k, etc.) plus `-o`.

Python ≥3.6 compatible (removed `from __future__ import annotations`).
"""

import os, sys, argparse, pickle, random
# ---------------------------------------------------------------------------
# Python < 3.7 compatibility: provide a fallback for contextlib.nullcontext
# ---------------------------------------------------------------------------
try:
    from contextlib import nullcontext  # Python ≥3.7
except ImportError:                      # Python 3.6 and older
    class _NullContext:
        def __init__(self, result=None):
            self.result = result
        def __enter__(self):
            return self.result
        def __exit__(self, *exc):
            return False
    nullcontext = _NullContext
from typing import Iterator, Tuple

import torch, tiktoken
from model import GPTConfig, GPT

###############################################################################
#                             FASTA reader                                    #
###############################################################################

def stream_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) tuples, collapsing wrapped lines."""
    header, seq_chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks)

###############################################################################
#                          argument parsing                                   #
###############################################################################

def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LA4SR FASTA→TSV inference script")
    # model/runtime-----------------------------------------------------------
    p.add_argument('--init_from', default='resume',
                   choices=['resume','gpt2','gpt2-medium','gpt2-large'],
                   help='Model source; "resume" = local ckpt.pt')
    p.add_argument('--out_dir', default='out',
                   help='Directory with ckpt.pt if --init_from resume')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='float16',
                   choices=['float32','bfloat16','float16'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--compile', action='store_true')
    # generation knobs--------------------------------------------------------
    p.add_argument('--max_new_tokens', type=int, default=14)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--top_k', type=int, default=10)
    # I/O---------------------------------------------------------------------
    p.add_argument('fasta_in', help='Input FASTA')
    p.add_argument('-o','--tsv_out', help='Output TSV (default: out-algaGPT/<basename>.tsv)')
    return p.parse_args()

args = get_cli()

###############################################################################
#                 reproducibility & autocast context                          #
###############################################################################

torch.manual_seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype_map = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}
ptdtype = ptdtype_map[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

###############################################################################
#                           model loading                                     #
###############################################################################

if args.init_from == 'resume':
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    # strip DDP prefixes if present
    state_dict = {k.replace('_orig_mod.',''):v for k,v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
else:
    model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

model.to(args.device).eval()
if args.compile:
    model = torch.compile(model)

###############################################################################
#                       encoding / decoding setup                             #
###############################################################################

if args.init_from == 'resume' and 'config' in locals().get('checkpoint',{}):
    cfg = checkpoint['config']
    meta_path = os.path.join('data', cfg.get('dataset',''), 'meta.pkl')
else:
    meta_path = ''
# ------------------------------------------------------------------
# Fallback: meta.pkl next to ckpt.pt / in --out_dir
# ------------------------------------------------------------------
if (not meta_path or not os.path.exists(meta_path)) and args.out_dir:
    alt_meta = os.path.join(args.out_dir, 'meta.pkl')
    if os.path.exists(alt_meta):
        meta_path = alt_meta

if meta_path and os.path.exists(meta_path):
    with open(meta_path,'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    #encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]
    UNK_ID = stoi.get('<unk>', 0)   # fall back to 0 if not present
    encode = lambda s: [stoi.get(c, UNK_ID) for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)

else:
    enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

###############################################################################
#                               output path                                   #
###############################################################################

os.makedirs('out-algaGPT', exist_ok=True)
outfile = args.tsv_out or os.path.join('out-algaGPT', f"{os.path.splitext(os.path.basename(args.fasta_in))[0]}.tsv")

###############################################################################
#                              generation loop                                #
###############################################################################

with open(outfile,'w') as tsv, torch.no_grad(), ctx:
    tsv.write('record_id\tsequence\tmodel_output\n')
    for rid, seq in stream_fasta(args.fasta_in):
        if not seq:
            print(f"[WARN] empty sequence for {rid}; skipping", file=sys.stderr)
            continue
        x = torch.tensor(encode(seq), dtype=torch.long, device=args.device).unsqueeze(0)
        try:
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            cont = decode(y[0].tolist())
        except Exception as e:
            print(f"[ERR] generation failed on {rid}: {e}", file=sys.stderr)
            cont = ''
        tsv.write(f"{rid}\t{seq}\t{cont}\n")

print(f"\n✓ Predictions saved to {outfile}\n")

