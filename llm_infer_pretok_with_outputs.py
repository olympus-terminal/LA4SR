import torch
import time
import argparse
import gc
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# === Args ===
parser = argparse.ArgumentParser(description="Run LLM inference on pre-tokenized sequences and collect outputs.")
parser.add_argument("tokenized_path", help="Path to tokenized .pt file")
parser.add_argument("--model", default="ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc", help="Model name or path")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
parser.add_argument("--out", default="inference_outputs.tsv", help="Output .tsv with results")
args = parser.parse_args()

# === Load tokenized input ===
batch = torch.load(args.tokenized_path)
input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]
n_seqs = input_ids.shape[0]

# === Load model config, weights, and tokenizer ===
config = AutoConfig.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")  # Set tokenizer explicitly
safetensors_path = hf_hub_download(repo_id=args.model, filename="model.safetensors")
model = GPTNeoXForCausalLM(config)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)

# === Set device and precision ===
device = torch.device(args.device)
model.to(device).half().eval()
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# === Determine padding token ID
pad_token_id = config.pad_token_id or config.eos_token_id

# === Optional: Warmup
_ = model.generate(input_ids[0:1], max_new_tokens=5, pad_token_id=pad_token_id)

# === Run inference and collect timing + outputs ===
print(f"[INFO] Running inference on {n_seqs} sequences...")
seq_lengths = (input_ids != 0).sum(dim=1).tolist()
results = []

with torch.no_grad():
    for i in range(n_seqs):
        start = time.time()
        output_ids = model.generate(
            input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            max_new_tokens=5,
            pad_token_id=pad_token_id
        )
        end = time.time()
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append((i + 1, seq_lengths[i], end - start, output_text))

# === Write results to file ===
with open(args.out, 'w') as f:
    f.write("Index\tSeq_Length\tTime_sec\tOutput\n")
    for idx, length, duration, text in results:
        f.write(f"{idx}\t{length}\t{duration:.6f}\t{text}\n")

print(f"[✔] Saved inference outputs to: {args.out}")

