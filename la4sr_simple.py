#!/usr/bin/env python
# Simple LA4SR sequence classifier based on the original working script

import sys
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_output(input_text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate output
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens)
    
    # Decode the output tokens back to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


def read_fasta(file_path):
    """Read sequences from a FASTA file"""
    sequences = []
    current_id = None
    current_seq = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_id is not None:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def classify_fasta(input_file, output_file, model, tokenizer):
    """Process a FASTA file and classify sequences"""
    # Read sequences
    sequences = read_fasta(input_file)
    
    # Process each sequence
    results = []
    raw_outputs = []
    
    for seq_id, seq in sequences:
        # Clean sequence
        clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper())
        if not clean_seq:
            continue
            
        # Generate output
        output_text = generate_output(clean_seq, model, tokenizer)
        raw_outputs.append((seq_id, clean_seq, output_text))
        
        # Check for classification markers
        generated_part = output_text[len(clean_seq):]
        
        if '@' in generated_part:
            classification = "algal"
            p_algal = 1.0
        elif '!' in generated_part:
            classification = "bacterial"
            p_algal = 0.0
        else:
            # Count ampersands as a backup classification method
            ampersand_count = generated_part.count('&')
            if ampersand_count > 5:  # Arbitrary threshold
                classification = "algal"
                p_algal = 0.8
            else:
                classification = "bacterial"
                p_algal = 0.2
        
        # Calculate confidence
        confidence = 1.0 if ('@' in generated_part or '!' in generated_part) else 0.5
        
        results.append((seq_id, classification, p_algal, confidence))
    
    # Write results
    with open(output_file, 'w') as f:
        f.write("Sequence_ID\tClassification\tP(algal)\tConfidence\n")
        for seq_id, cls, p_algal, conf in results:
            f.write(f"{seq_id}\t{cls}\t{p_algal:.4f}\t{conf:.4f}\n")
    
    # Print summary
    print("\nSample of raw outputs:")
    for seq_id, seq, output in raw_outputs[:3]:
        print(f"{seq_id}:")
        print(f"  Input: {seq[:30]}..." if len(seq) > 30 else f"  Input: {seq}")
        print(f"  Output: {output}")
        print()
    
    # Counts
    total = len(results)
    algal_count = sum(1 for _, cls, _, _ in results if cls == "algal")
    bact_count = sum(1 for _, cls, _, _ in results if cls == "bacterial")
    
    print(f"Classification summary:")
    print(f"  Algal:     {algal_count}/{total} ({algal_count/total*100:.1f}%)")
    print(f"  Bacterial: {bact_count}/{total} ({bact_count/total*100:.1f}%)")
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <model_name_or_path> <input_file> [output_file] [max_tokens]")
        sys.exit(1)
    
    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "results.txt"
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english", use_fast=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, is_decoder=True).to(device)
    model.eval()
    
    # Process the input file
    classify_fasta(input_file, output_file, model, tokenizer)
