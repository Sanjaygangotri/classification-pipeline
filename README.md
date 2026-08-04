# Topic Classification Pipeline

This repository contains a highly efficient, custom-built topic classification pipeline designed to classify text data into 24 predefined topics. Per the strict constraints of the project, no pretrained models or fine-tuned pretrained weights were used. The architecture and tokenizer were built entirely from scratch, keeping the parameter count strictly below the 5 Billion (5B) limit at just 3,859,608 parameters.

---

## a. Setup Instructions

### Environment Setup
It is highly recommended to isolate the dependencies using a virtual environment to prevent version conflicts. Ensure you have Python 3.9 or higher installed on your system.

Create and activate a virtual environment:
* Windows: `python -m venv venv` followed by `venv\Scripts\activate`
* Mac/Linux: `python3 -m venv venv` followed by `source venv/bin/activate`

### Dependencies Installation
All required libraries are listed in the `requirements.txt` file. Install them using the following command:
`pip install -r requirements.txt`

Important Note: Ensure the raw data file `dataset_10M.parquet` is placed directly in the root directory of this project before proceeding to execution.

---

## b. Training Instructions

The training pipeline streams the 4GB dataset directly from disk using `pyarrow` to maintain an extremely low memory footprint. It utilizes Polars to dynamically compute inverse class frequencies, passing them as weights to the loss function to handle severe class imbalance. 

To execute the end-to-end training pipeline, run:
`python src/train.py`

This script will automatically:
* Lock all random seeds for strict reproducibility.
* Stream the data and train the 3.8M parameter model.
* Save the resulting weights as `final_model.pth` inside the `final_models/` directory.

*(Note: The BPE tokenizer and mapping files are already provided in the `final_models/` folder. If you wish to retrain the tokenizer from scratch, run `python src/utils.py` prior to training).*

---

## c. Inference Instructions

The inference script evaluates the trained model on a sequential validation sample. It loads the saved production weights and custom tokenizer from the `final_models/` directory.

To run the evaluation, execute:
`python src/inference.py`

This script will output:
* The macro Evaluation Metrics (Accuracy, Precision, Recall, and F1 Score).
* A detailed per-class Classification Report showing precision and recall for all 24 individual topics.

---

## d. Input/Output Schema

The pipeline expects structured tabular data with specific column headers and data types.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **DATA** | String | The unstructured text document or article sequence to be classified. |
| **TOPIC** | String | The target categorical label representing one of 24 defined topics (e.g., `software`, `industrial`, `games`). |

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