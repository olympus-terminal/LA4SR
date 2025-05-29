#!/usr/bin/env python3

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import os

REPO_ID = "EleutherAI/gpt-neo-125m"
LOCAL_DIR = os.path.expanduser(f"~/.cache/huggingface/hub/{REPO_ID.replace('/', '--')}")

# Check if local repo exists
if not os.path.exists(LOCAL_DIR):
    print("Downloading model repo from Hugging Face...")
    repo_path = snapshot_download(repo_id=REPO_ID)
else:
    print(f"Using cached repo at {LOCAL_DIR}")
    repo_path = LOCAL_DIR

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_path)
print("Tokenizer loaded successfully.")

