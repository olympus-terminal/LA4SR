import sys
from random import shuffle

def split_and_tag_file(input_file, tag):
  """
  Splits a file into 90%/10% parts and adds a tag to the end of each line in the 90% file.

  Args:
    input_file: The path to the input file.
    tag: The string tag to append to each line in the 90% file.
  """
  with open(input_file, 'r') as infile:
    lines = infile.readlines()
    shuffle(lines)  # Shuffle lines to ensure random distribution

  split_index = int(0.9 * len(lines))
  ninety_percent_lines = lines[:split_index]
  ten_percent_lines = lines[split_index:]

  # Write 90% file with tag
  with open(f"{input_file}.90", 'w') as outfile_90:
    for line in ninety_percent_lines:
      outfile_90.write(line.rstrip() + f" {tag}\n")

  # Write 10% file
  with open(f"{input_file}.10", 'w') as outfile_10:
    outfile_10.writelines(ten_percent_lines)

# Check if input file and tag are provided as arguments
if len(sys.argv) == 3:
  input_file = sys.argv[1]
  tag = sys.argv[2]
  split_and_tag_file(input_file, tag)
else:
  print("Error: Please provide the input file and tag as command-line arguments.")
[drn2@login1 DATA_S2]$ cat break-to100.py 
import sys

def break_sequence_into_lines(input_file, line_length=100):
  """
  Breaks a single-line sequence in a FASTA file into lines of a specified length.

  Args:
    input_file: The path to the input FASTA file.
    line_length: The desired length of each line (default is 100).
  """
  output_file = f"{input_file}.wrapped"
  with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    sequence = infile.read().strip()  # Read entire sequence as a single string
    for i in range(0, len(sequence), line_length):
      outfile.write(sequence[i:i+line_length] + "\n")

# Check if input file is provided as sys.argv[1]
if len(sys.argv) > 1:
  input_file = sys.argv[1]
  break_sequence_into_lines(input_file)
else:
  print("Error: Please provide the input FASTA file as a command-line argument.")
