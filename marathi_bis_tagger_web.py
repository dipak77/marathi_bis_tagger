# === Marathi POS Tagger Web App with Custom BERT Model (HuggingFace Integration) ===
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import unicodedata
import os

# === CONFIG ===
MODEL_PATH = "custom_pos_model"  # Path to your fine-tuned model
TAGSET = "BIS"  # You can make this dynamic too

tag_id2label = {
    0: "NN", 1: "NNP", 2: "PRP", 3: "VM", 4: "VAUX", 5: "JJ", 6: "RB", 7: "PSP",
    8: "CCD", 9: "CCS", 10: "DMR", 11: "RPD", 12: "QTC", 13: "SYM", 14: "PUNC",
    15: "INJ", 16: "UNK"
}

# === Load Model and Tokenizer ===
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

def normalize(text):
    return unicodedata.normalize("NFC", text.strip())

def predict_tags(sentence):
    words = sentence.strip().split()
    tokens = tokenizer(words, return_tensors="pt", is_split_into_words=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = tokens.word_ids()

    results = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        label_id = predictions[idx]
        tag = tag_id2label.get(label_id, "UNK")
        results.append((words[word_idx], tag))
        previous_word_idx = word_idx
    return results

# === Flask App ===
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="mr">
<head>
  <meta charset="UTF-8">
  <title>Custom Marathi POS Tagger</title>
  <style>
    body { font-family: sans-serif; background: #f4f4f4; padding: 2rem; }
    form, .output { background: white; padding: 1rem; border-radius: 8px; max-width: 700px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    input[type=text] { width: 100%; padding: 0.5rem; margin-bottom: 1rem; font-size: 1.1rem; }
    input[type=submit] { background: #3498db; color: white; padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; }
    pre { background: #eee; padding: 1rem; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h2 style="text-align:center">🗣️ Marathi POS Tagger using BERT</h2>
  <form method="post">
    <label>Enter a Marathi sentence:</label>
    <input type="text" name="sentence" placeholder="उदाहरण: जयपुर हे प्रसिद्ध शहर आहे">
    <input type="submit" value="Tag">
  </form>
  {% if result %}
    <div class="output">
      <h3>Tagged Output:</h3>
      <pre>{{ result }}</pre>
    </div>
  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        sentence = request.form['sentence']
        tags = predict_tags(sentence)
        result = " ".join([f"{normalize(tok)}|{tag}" for tok, tag in tags])
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
