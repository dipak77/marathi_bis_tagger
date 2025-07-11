# POS Tagging Evaluation and Prediction Script (Post-Training)
# Uses trained model from marathi_pos_model_bis/

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
import unicodedata
import os
from tqdm import tqdm
import argparse

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Evaluate POS model on test data")
parser.add_argument("--limit", type=int, default=100, help="Number of test sentences to evaluate")
args = parser.parse_args()

# === LOAD MODEL ===
MODEL_PATH = "marathi_pos_model_bis"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
label_list = list(model.config.id2label.values())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# === UTILS ===
def normalize(text):
    return unicodedata.normalize("NFC", text.strip())

def predict_tags(sentence):
    words = sentence.strip().split()
    tokens = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = tokens.word_ids()

    results = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        tag = id2label.get(predictions[idx], "UNK")
        results.append((words[word_idx], tag))
        previous_word_idx = word_idx
    return results

# === 3. CONFUSION MATRIX FROM fold_5_0.txt ===
def parse_bis_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.read().strip()
    token_tag_pairs = data.split()
    sentences = []
    current = []
    for pair in token_tag_pairs:
        if '|' not in pair: continue
        token, tag = pair.rsplit('|', 1)
        token = normalize(token)
        tag = tag.strip()
        current.append((token, tag))
        if tag == "PUNC" and token in ["।", ".", "?"]:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences

def evaluate_on_test_file(test_file, sample_size=100):
    print("\n📊 Evaluating on test file:", test_file)
    all_sentences = parse_bis_file(test_file)
    test_sentences = all_sentences[:sample_size]

    y_true = []
    y_pred = []
    sentence_rows = []

    for sid, sentence in enumerate(tqdm(test_sentences, desc=f"Evaluating {sample_size} Sentences"), 1): 
        tokens = [t for t, _ in sentence]
        gold_tags = [p for _, p in sentence]
        predicted = predict_tags(" ".join(tokens))
        pred_tags = [t for _, t in predicted]
        if len(gold_tags) == len(pred_tags):
            correct = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
            accuracy = round((correct / len(gold_tags)) * 100, 2)
            y_true.extend(gold_tags)
            y_pred.extend(pred_tags)
            sentence_rows.append({
                "Sentence #": sid,
                "Tokens": " ".join(tokens),
                "Truth": " ".join(gold_tags),
                "Predicted": " ".join(pred_tags),
                "IsMatch": gold_tags == pred_tags,
                "Sentence Accuracy": accuracy
            })

    df_sentences = pd.DataFrame(sentence_rows)
    df_sentences.to_csv("evaluation_sentences.csv", index=False, encoding="utf-8-sig")
    print("📄 Per-sentence evaluation saved to: evaluation_sentences.csv")

    mismatches_df = df_sentences[df_sentences["IsMatch"] == False]
    mismatches_df.to_csv("sentence_mismatches.csv", index=False, encoding="utf-8-sig")
    print("📄 Mismatched sentences saved to: sentence_mismatches.csv")

    total = len(df_sentences)
    matches = df_sentences["IsMatch"].sum()
    match_pct = round((matches / total) * 100, 2)
    avg_accuracy = round(df_sentences["Sentence Accuracy"].mean(), 2)
    print(f"✅ Sentence Match Summary: {matches}/{total} sentences matched exactly ({match_pct}%)")
    print(f"✅ Average Sentence Accuracy: {avg_accuracy}%")

    html_rows = []
    for row in sentence_rows:
        tokens = row['Tokens'].split()
        gold = row['Truth'].split()
        pred = row['Predicted'].split()
        tokens_html = []
        for t, g, p in zip(tokens, gold, pred):
            color = 'green' if g == p else 'red'
            tokens_html.append(f"<b style='color:{color}'>{t}|{p}</b>")
        row_html = f"<tr><td>{row['Sentence #']}</td><td>{' '.join(tokens_html)}</td><td>{' '.join(gold)}</td></tr>"
        html_rows.append(row_html)

    html_table = """<html><head><meta charset='utf-8'><style>
table { border-collapse: collapse; width: 100%%; font-family: sans-serif; }
th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
th { background: #f2f2f2; }
</style></head><body>
<h2>Per-Token POS Tagging Report</h2>
<p><b>Average Sentence Accuracy:</b> %.2f%%</p>
<table>
<tr><th>Sentence #</th><th>Tagged Prediction (output)</th><th>Truth Tags(input)</th></tr>
%s
</table></body></html>""" % (avg_accuracy, "\n".join(html_rows))

    with open("tagging_report.html", "w", encoding="utf-8") as f:
        f.write(html_table)
    print("📄 HTML token report saved to: tagging_report.html")

    print("\n📈 Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=3))

   
   # Generate the confusion matrix
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xlabel("Predicted", labelpad=20)
    plt.ylabel("Actual", labelpad=20)

    # Move y-axis labels to right
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')

    # Adjust layout and save
    plt.title(f"Confusion Matrix: Trained vs BIS ({sample_size} Sentences)", pad=40)
    plt.tight_layout()
    plt.savefig("confusion_matrix_eval_fold5_0.png")
    print("✅ Saved: confusion_matrix_eval_fold5_0.png")

# Run it on fold_5_0.txt — evaluate top N
EVAL_TEST_FILE = "input_pos_files/fold_5_4.txt"
if os.path.exists(EVAL_TEST_FILE):
    print(f"🔍 Evaluating {args.limit} sentences from fold_5_0.txt")
    evaluate_on_test_file(EVAL_TEST_FILE, sample_size=args.limit)
else:
    print("⚠️ fold_5_0.txt not found — skipping evaluation")
