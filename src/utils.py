from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import time

print("Starting Phase 2: Training Custom BPE Tokenizer...")
start_time = time.time()

# 1. Initialize an empty BPE tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. Use standard whitespace splitting before BPE merging
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Configure the Trainer (Vocab size: 30,000 to keep embedding layer small)
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
)

# 4. Train the tokenizer purely on your extracted sample data
tokenizer.train(files=["raw_text_sample.txt"], trainer=trainer)

# 5. Save the vocabulary and merge rules locally
tokenizer.save("final_models/custom_tokenizer.json")

print(f"✅ Custom Tokenizer trained and saved in {round(time.time() - start_time, 2)} seconds!")
print(f"Final Vocabulary Size: {tokenizer.get_vocab_size()}")