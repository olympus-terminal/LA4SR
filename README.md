# LA<sub>4</sub>SR: Language modeling with AI for Algal Amino Acid Sequence Representation 🧬

🔬 Illuminating the
             Dark Proteome


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LA4SR (pronounced "laser") 🎯 is a framework for implementing state-of-the-art language models to process microbial genomic data and extract otherwise intractable information. It excels at distinguishing between algal and bacterial sequences with high accuracy and unprecedented speed (16,580x faster than BLAST with ~3x higher recall rate).

## 🧫 Key Features

- **Multiple AI Model Architecture Support** 🏗️
  - Transformer-based: GPT variants, DistilRoBERTa, BLOOM, Mistral
  - State-space models: Mamba
- **Rich Interpretability Tools** 🔍
  - HELIX (Hidden Embedding Layer Information eXplorer)
  - DeepLift LA4SR
  - Deep Motif Miner (DMM)
- **Flexible Processing & GPU Acceleration** 🚄
  - Terminal Information (TI) inclusive/free processing
  - Fully optimized for modern GPU architectures

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/LA4SR.git
cd LA4SR
pip install -r requirements.txt
```

## 🎯 Usage

```python
from la4sr import LA4SRModel

# Initialize model
model = LA4SRModel.from_pretrained('ChlorophyllChampion/LA4SR-Mamba-370m-88F1-45000')

# Classify sequences
sequences = ["MKTLLLTLVV...", "GPRTEINPLL..."]  # Your amino acid sequences
predictions = model.predict(sequences)
```

## 🌊 Performance Benchmarks

| Model | Parameters | F1 Score | Speed (vs BLAST) |
|-------|------------|----------|------------------|
| LA4SR Pythia 70m | 70M | 0.90 | 16,580x |
| LA4SR Mamba 370m | 370M | 0.88 | 15,200x |
| LA4SR Mistral 7B | 7B | 0.88 | 14,800x |

## 📚 Pre-trained Models

Available on Hugging Face:
- [ChlorophyllChampion/Mamba-370m-88F1-45000](https://huggingface.co/ChlorophyllChampion/Mamba-370m-88F1-45000)
- [ChlorophyllChampion/LA4SR-Mamba-2.8b-QLORA-ft](https://huggingface.co/ChlorophyllChampion/LA4SR-Mamba-2.8b-QLORA-ft)

## 📝 Citation

```bibtex
@article{nelson2024la4sr,
  title={LA4SR: illuminating the dark proteome with generative AI},
  author={Nelson, David R. and Jaiswal, Ashish Kumar and Ismail, Noha and Salehi-Ashtiani, Kourosh},
  journal={[Journal TBD]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
🧬 Empowering Microbial Genomics with AI 🧬

Made by the LA4SR Team
</div>
