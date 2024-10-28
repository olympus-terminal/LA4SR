# LA<sub>4</sub>SR: Language modeling with AI for Algal Amino Acid Sequence Representation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LA4SR (pronounced "laser") is a framework for implementing state-of-the-art language models to process microbial genomic data and extract otherwise intractable information. It excels at distinguishing between algal and bacterial sequences with high accuracy and unprecedented speed.

## 🚀 Key Features

- **Multiple AI model Architecture Support**: Compatible with various open-source architectures including:
  - Transformer-based: GPT variants, DistilRoBERTa, BLOOM, Mistral
  - State-space models: Mamba
- **Flexible Processing Options**: 
  - Terminal Information (TI) inclusive processing
  - TI-free processing for improved generalization
- **Rich Interpretability Tools**:
  - HELIX (Hidden Embedding Layer Information eXplorer)
  - DeepLift LA4SR
  - Deep Motif Miner (DMM)
- **GPU Acceleration**: Fully optimized for modern GPU architectures

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Bitsandbytes (for quantization)
- CUDA-capable GPU (recommended)

## 🔧 Installation

```bash
git clone https://github.com/drdavidroynelson/LA4SR.git
cd LA4SR
pip install -r requirements.txt
```

## 📊 Performance Benchmarks

| Model | Parameters | F1 Score | Speed (vs BLAST) |
|-------|------------|----------|------------------|
| LA4SR Pythia 70m | 70M | 0.90 | 16,580x |
| LA4SR Mamba 370m | 370M | 0.88 | 15,200x |
| LA4SR Mistral 7B | 7B | 0.88 | 14,800x |

## 🎯 Usage

### Basic Sequence Classification

```python
from la4sr import LA4SRModel

# Initialize model
model = LA4SRModel.from_pretrained('ChlorophyllChampion/LA4SR-Mamba-370m-88F1-45000')

# Classify sequences
sequences = ["MKTLLLTLVV...", "GPRTEINPLL..."]  # Your amino acid sequences
predictions = model.predict(sequences)
```

### Training a New Model

```python
from la4sr import LA4SRTrainer

# Initialize trainer
trainer = LA4SRTrainer(
    model_name="pythia-70m",
    batch_size=32,
    learning_rate=1e-4
)

# Train model
trainer.train(
    train_data="path/to/training/data",
    eval_data="path/to/eval/data",
    epochs=1
)
```

## 🔍 Interpretability Tools

### Using HELIX

```python
from la4sr.explainers import HELIX

explainer = HELIX(model)
layer_representations = explainer.analyze_sequence("MKTLLLTLVV...")
```

### Using Deep Motif Miner

```python
from la4sr.explainers import DeepMotifMiner

dmm = DeepMotifMiner(model)
motifs = dmm.find_motifs(sequence, window_size=5)
```

## 📚 Pre-trained Models

Available models on Hugging Face:
- [ChlorophyllChampion/Mamba-370m-88F1-45000](https://huggingface.co/ChlorophyllChampion/Mamba-370m-88F1-45000)
- [ChlorophyllChampion/LA4SR-Mamba-2.8b-QLORA-ft](https://huggingface.co/ChlorophyllChampion/LA4SR-Mamba-2.8b-QLORA-ft)
...
## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📝 Citation

If you use LA4SR in your research, please cite:

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

## 🙏 Acknowledgments

- The authors thank their respective institutions for support
- Built with PyTorch and Hugging Face Transformers
- GPU resources provided by NVIDIA

## 📧 Contact

For questions and support, please open an issue or contact the maintainers.
