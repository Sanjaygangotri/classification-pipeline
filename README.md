# Topic Classification Pipeline

This repository contains a highly efficient, custom-built topic classification pipeline designed to classify text data into 24 predefined topics. 

Per the strict constraints of the project, **no pretrained models or fine-tuned pretrained weights were used**. The architecture and tokenizer were built entirely from scratch.

## Project Constraints Adherence
* **Parameter Count Limit:** The global average pooling text classifier contains exactly **3,859,608 parameters**, remaining well below the 5 Billion (5B) parameter constraint.
* **Originality:** The tokenizer is a custom Byte-Pair Encoding (BPE) model trained from scratch. The classification architecture utilizes standard PyTorch `nn.Module` layers without loading external weights.

---

## 1. Setup Instructions

### Environment Setup
Ensure you have Python 3.9+ installed on your system. It is recommended to use a virtual environment.

### Dependencies Installation
Install the required libraries using the provided requirements file:
```bash
pip install -r requirements.txt