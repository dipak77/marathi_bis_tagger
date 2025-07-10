# Marathi POS Tagger: Console Training + Evaluation with BIS Tags (Hugging Face BERT)
import os
import pandas as pd
import unicodedata
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, Trainer,
                          TrainingArguments, DataCollatorForTokenClassification)
import torch
from sklearn.metrics import classification_report

# === CONFIGURATION ===
FOLD_FILES = [f"input_pos_files/fold_5_{i}.txt" for i in range(5)]  # All 5 folds
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "marathi_pos_model_bis"
LABEL_LIST = []  # Will populate dynamically

# === Load and Parse Tagged BIS Format ===
def parse_bis_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.read().strip()
    tokens = []
    tags = []
    sentence_tokens = []
    sentence_tags = []
    for pair in data.split():
        if '|' not in pair:
            continue
        token, tag = pair.rsplit('|', 1)
        token = unicodedata.normalize("NFC", token.strip())
        tag = tag.strip()
        sentence_tokens.append(token)
        sentence_tags.append(tag)
        if tag == "PUNC" and token in ["।", ".", "?"]:
            tokens.append(sentence_tokens)
            tags.append(sentence_tags)
            sentence_tokens = []
            sentence_tags = []
    if sentence_tokens:
        tokens.append(sentence_tokens)
        tags.append(sentence_tags)
    return tokens, tags

# === Combine All Folds into Training/Test Split ===
all_tokens = []
all_tags = []
for path in FOLD_FILES:
    sents, bis_tags = parse_bis_file(path)
    all_tokens.extend(sents)
    all_tags.extend(bis_tags)

LABEL_LIST = sorted(set(tag for sent in all_tags for tag in sent))
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}

train_tokens, test_tokens, train_tags, test_tags = train_test_split(all_tokens, all_tags, test_size=0.2, random_state=42)

# === Tokenizer and Label Alignment ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id[example["tags"][word_idx]])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# === Build Datasets ===
train_dataset = Dataset.from_dict({"tokens": train_tokens, "tags": train_tags}).map(tokenize_and_align_labels)
test_dataset = Dataset.from_dict({"tokens": test_tokens, "tags": test_tags}).map(tokenize_and_align_labels)

# === Load and Train Model ===
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id)

data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\n🚀 Training model on BIS POS tags...")
trainer.train()
print("✅ Training completed. Model saved to:", OUTPUT_DIR)

# === Evaluation Report ===
print("\n🔍 Evaluating model...")
preds = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(preds.predictions), dim=-1).numpy()
true_labels = preds.label_ids

final_preds = []
final_truth = []
for pred, true, tokens in zip(pred_labels, true_labels, test_dataset):
    for p, t in zip(pred, true):
        if t != -100:
            final_preds.append(id2label[p])
            final_truth.append(id2label[t])

report = classification_report(final_truth, final_preds, output_dict=False, digits=3)
print("\n📊 Classification Report (BIS):\n")
print(report)

# Save Report to File
with open(os.path.join(OUTPUT_DIR, "bis_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(f"✅ Report saved to: {OUTPUT_DIR}/bis_classification_report.txt")
