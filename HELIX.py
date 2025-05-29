#!/usr/bin/env python
# HELIX: Hidden Embedding Layer Information eXplorer
# -------------------------------------------------
# A tool for visualizing protein sequence embeddings across transformer layers.
# Author: David Nelson
# Date: October 2024

import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def setup_visualization_params():
    # Initialize all matplotlib parameters for consistent styling
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 6,
        'axes.linewidth': 0.25,
        'grid.linewidth': 0.25,
        'lines.linewidth': 0.25,
        'xtick.major.width': 0.25,
        'ytick.major.width': 0.25,
        'axes.titlesize': 6,
        'legend.fontsize': 6,
        'figure.titlesize': 6,
        'legend.handlelength': 1,
        'legend.handleheight': 1,
        'legend.frameon': False
    })

def setup_model():
    # Initialize and return the model, tokenizer, and device
    print("Loading model and tokenizer...")
    model_name = "ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")
    safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    model = GPTNeoXForCausalLM(config)
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device).half()
    return model, tokenizer, device


def read_sequences_from_file(file_path, max_sequences=10):
    # Read and return up to max_sequences from file
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file if line.strip()]
    print(f"Found {len(sequences)} sequences in {file_path}")
    return sequences[:max_sequences]


def aa_to_input_ids(sequence, tokenizer, device):
    # Convert amino acid sequence to input IDs
    toks = tokenizer.encode(sequence, add_special_tokens=False)
    return torch.as_tensor(toks).view(1, -1).to(device)


def analyze_sequence(model, input_ids):
    # Get hidden states from model for sequence
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]


def setup_a4_figure(n_sequences):
    # Create figure with A4 dimensions and calculate grid layout
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
    if n_sequences <= 5:
        rows, cols = n_sequences, 1
    else:
        rows = (n_sequences + 1) // 2
        cols = 2
    return fig, rows, cols

def plot_sequence_layers(hidden_states, sequence, ax, sequence_name=""):
    # Plot layer-wise PCA projections, annotating each point with its residue letter.
    all_aa = 'ACDEFGHIKLMNPQRSTVWYX'
    unique_aa = sorted(set(all_aa) | set(sequence))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(unique_aa)))
    aa_to_color = dict(zip(unique_aa, colors))
    aa_to_color['C'] = cmap(0.9)
    aa_to_color['M'] = cmap(0.1)

    pca = PCA(n_components=2)
    n_layers = len(hidden_states)
    n_rows = (n_layers + 3) // 4
    n_cols = min(4, n_layers)

    for layer_idx, state in enumerate(hidden_states):
        # 2D projection
        projected = pca.fit_transform(state)
        xs, ys = projected[:, 0], projected[:, 1]

        # compute bounds with padding
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

        # create inset axes
        x0 = (layer_idx % n_cols) / n_cols
        y0 = 1 - ((layer_idx // n_cols + 1) / n_rows)
        w, h = 0.9 / n_cols, 0.9 / n_rows
        layer_ax = ax.inset_axes([x0, y0, w, h])

        # enforce limits
        layer_ax.set_xlim(x_min - pad_x, x_max + pad_x)
        layer_ax.set_ylim(y_min - pad_y, y_max + pad_y)

        # annotate residues
        for pos, aa in enumerate(sequence):
            x, y = xs[pos], ys[pos]
            layer_ax.text(
                x, y, aa,
                fontsize=6,
                fontweight='bold',
                ha='center', va='center',
                color=aa_to_color.get(aa, 'black'),
                alpha=0.8,
                clip_on=True
            )

        # style layer plot
        layer_ax.set_title(f'Layer {layer_idx}', pad=2, fontsize=6)
        layer_ax.set_xticks([])
        layer_ax.set_yticks([])
        for spine in layer_ax.spines.values():
            spine.set_linewidth(0.25)

    # style main axis
    ax.set_title(sequence_name, pad=10, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def process_sequences(file_paths, max_sequences=10, output_file='helix_output.pdf'):
    # Main function to process sequences and create visualizations
    setup_visualization_params()
    model, tokenizer, device = setup_model()

    with PdfPages(output_file) as pdf:
        for file_path in file_paths:
            print(f"\nProcessing file: {file_path}")
            sequences = read_sequences_from_file(file_path, max_sequences)
            file_name = file_path.split('/')[-1].split('.')[0]

            fig, rows, cols = setup_a4_figure(len(sequences))
            plt.suptitle(f'Sequence Representations - {file_name}', y=0.98, fontsize=6)

            for idx, sequence in enumerate(tqdm(sequences, desc="Processing sequences")):
                ax = plt.subplot(rows, cols, idx + 1)
                input_ids = aa_to_input_ids(sequence, tokenizer, device)
                hidden_states = analyze_sequence(model, input_ids)
                plot_sequence_layers(hidden_states, sequence, ax, f'{file_name}_seq_{idx}')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, dpi=300)
            plt.close()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Completed processing {len(sequences)} sequences")

def main():
    # Entry point: parse arguments and launch processing
    parser = argparse.ArgumentParser(
        description="HELIX: Hidden Embedding Layer Information eXplorer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('file_paths', nargs='+', 
                        help='Paths to files containing amino acid sequences (one per line)')
    parser.add_argument('--max_sequences', type=int, default=10, 
                        help='Maximum number of sequences to process per file')
    parser.add_argument('--output', type=str, default='helix_output.pdf',
                        help='Output PDF file name (default: helix_output.pdf)')
    args = parser.parse_args()

    print("HELIX: Hidden Embedding Layer Information eXplorer")
    print("------------------------------------------------")
    process_sequences(args.file_paths, args.max_sequences, args.output)
    print(f"\nVisualization completed! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
