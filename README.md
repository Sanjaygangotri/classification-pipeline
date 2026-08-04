# Topic Classification Pipeline

This repository contains a highly efficient, custom-built topic classification pipeline designed to classify text data into 24 predefined topics. 

Per the strict constraints of the project, **no pretrained models or fine-tuned pretrained weights were used**. The architecture and tokenizer were built entirely from scratch.

## Project Constraints Adherence
* **Parameter Count Limit:** The global average pooling text classifier contains exactly **3,859,608 parameters**, remaining well below the 5 Billion (5B) parameter constraint.
* **Originality:** The tokenizer is a custom Byte-Pair Encoding (BPE) model trained from scratch. The classification architecture utilizes standard PyTorch `nn.Module` layers without loading external weights.
* **Hardware Efficiency:** Implements a streaming data pipeline using `pyarrow.parquet` to train on 10 million rows (4GB) iteratively, completely bypassing Out-Of-Memory (OOM) limitations on standard hardware.

---

## Repository Structure
```text
project/
├── src/                 # Final production scripts
│   ├── train.py         # Training loop with dynamic inverse class weighting
│   ├── inference.py     # Evaluation and metric generation script
│   ├── model.py         # Custom PyTorch model architecture
│   └── utils.py         # BPE Tokenizer training script
├── experiments/         # Iterative baselines, EDA, and early models
├── final_models/        # Saved weights (.pth), tokenizer, and mappings
├── report.pdf           # Detailed approach, iteration, and error analysis
├── requirements.txt     # Environment dependencies
└── README.md