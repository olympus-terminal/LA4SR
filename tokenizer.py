import sys
from transformers import AutoTokenizer
from datasets import load_dataset

# Check if the text file path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Please provide the path to the text file as a command-line argument.")
    sys.exit(1)

# Get the text file path from the command-line argument
text_file_path = sys.argv[1]

# Specify the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("lhy/character-level-tokenizer")
tokenizer = AutoTokenizer.from_pretrained('hmbyt5/byt5-small-english')
# Load the dataset
dataset = load_dataset('text', data_files=text_file_path)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=32, remove_columns=["text"])

# Save the tokenized dataset
tokenized_dataset.save_to_disk("byte5-tokenized_dataset")

print("Tokenization completed. Tokenized dataset saved to 'tokenized_dataset' directory.")
