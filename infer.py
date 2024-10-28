import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_output(input_text, model, tokenizer):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cuda')  # Move input_ids to GPU

    # Generate output
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(input_ids, max_new_tokens=15)

    # Decode the output tokens back to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name_or_path> <input_file>")
        sys.exit(1)

    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, is_decoder=True).to('cuda')  # Move the model to GPU

    with open(input_file, 'r') as file:
        for line in file:
            input_text = line.strip()
            output_text = generate_output(input_text, model, tokenizer)
            print(output_text)
