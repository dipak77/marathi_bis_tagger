# Marathi POS Tagger - Model Training & Evaluation Documentation

## 🪄 Overview

This documentation outlines the end-to-end architecture, technology stack, training and evaluation pipeline, and application workflow for the Marathi Part-of-Speech (POS) tagging system developed using a custom-trained transformer-based model.

---

## 🚀 Tech Stack

### Programming Language:

* **Python 3.10+**

### Libraries & Frameworks:

* **Transformers** (HuggingFace) - Model training & inference
* **Datasets** (HuggingFace) - Dataset preparation
* **Sklearn** - Evaluation metrics
* **Matplotlib, Seaborn** - Visualizations
* **Pandas** - Data handling
* **Torch (PyTorch)** - Deep learning engine
* **Stanza** (optional) - External POS tagging baseline
* **Unicodedata** - Unicode normalization for Marathi

### Tools:

* **CUDA (optional)** - GPU acceleration
* **Flask (optional)** - Web application layer

---

## 💡 Architecture Overview

```text
+----------------------------+
|      Marathi Dataset       |
+-------------+-------------+
              |
              v
+----------------------------+
|   Preprocessing Pipeline   | (Token|Tag split, NFC normalization)
+----------------------------+
              |
              v
+----------------------------+
|      Custom Dataset        | (Split into train/test folds)
+----------------------------+
              |
              v
+----------------------------+
| Transformers Trainer (HF) |
|  BERT / DistilBERT Model   |
+----------------------------+
              |
              v
+----------------------------+
|    Evaluation & Reports    |
+-------------+--------------+
              |
              v
+----------------------------+
| Web App / Console Scripts  |
+----------------------------+
```

---

## 🔄 Training Pipeline (Model Creation)

### Input Data Format:

* Format: `token|POS` separated by space, sentence ends with PUNC (`.`, `।`, `?`)

### Steps:

1. **Data Collection:**

   * 5-fold BIS-annotated `.txt` files (`fold_5_0.txt`, ..., `fold_5_4.txt`)

2. **Preprocessing:**

   * Unicode NFC normalization
   * Sentence boundary detection on punctuation

3. **Dataset Conversion:**

   * Converted into HuggingFace `datasets.Dataset` format
   * Train/validation split (e.g., 80/20 or cross-validation)

4. **Model Training:**

   * Using `AutoModelForTokenClassification`
   * Tokenized via `AutoTokenizer`
   * Optimized using AdamW optimizer and scheduler

5. **Model Saving:**

   * Stored under `marathi_pos_model_bis/`

---

## ✅ Evaluation Pipeline

### Input File:

* `fold_5_0.txt` (test set)

### Evaluation Script Features:

* Sentence-by-sentence tagging using trained model
* Outputs:

  * `evaluation_sentences.csv` with token/gold/prediction and accuracy
  * `sentence_mismatches.csv`
  * `tagging_report.html` with colored tokens (green=match, red=wrong)
  * `confusion_matrix_eval_fold5_0.png`
  * Sklearn classification report (F1, precision, recall, support)

### Additional Metrics:

* Sentence-level accuracy
* Total average sentence accuracy
* Per-token confusion matrix

---

## ⚖️ How to Run

### Train the Model:

```bash
python train_marathi_pos_model.py
```

### Evaluate the Model:

```bash
python evaluate_bis_tag_On_trained_data.py --limit 100
```

### Predict from Console:

```bash
# Input single sentence
Enter Marathi sentence: जयपूर हे सुंदर शहर आहे.
```

### Bulk Prediction:

1. Create `predict_input.txt`
2. Run script
3. Output: `predicted_tags_output.csv`

---

## 📄 Application Workflow

```text
User Input (Sentence)
        |
        v
 Tokenize + Normalize
        |
        v
  Predict Tags (Trained Model)
        |
        v
 Highlight + Save
     - CSV (tags)
     - HTML (visual)
     - Plots (Confusion matrix)
```

---

## 🔖 Notes & Tips

* All Unicode text must be normalized using `unicodedata.normalize("NFC", text)`
* Model supports GPU acceleration via `torch.device('cuda')`
* Can integrate with Flask for UI-based usage
* To retrain with more folds, simply merge all train files

---

## 📅 Future Enhancements

* LSTM or BERTCRF for higher accuracy
* Subword-based error correction
* Integration with grammar checker
* Multi-tagset inference (IITB, LDCIL, UPOS switching via flag)

---

## 📁 Output Files Summary

| File                                | Description                             |
| ----------------------------------- | --------------------------------------- |
| `evaluation_sentences.csv`          | Token-wise predictions + sentence match |
| `sentence_mismatches.csv`           | Only incorrect rows                     |
| `tagging_report.html`               | Color-coded report (green/red)          |
| `confusion_matrix_eval_fold5_0.png` | POS class matrix                        |
| `predicted_tags_output.csv`         | Tags for raw sentences                  |

---

## 🚀 Final Thoughts

This Marathi POS system provides a flexible, trainable framework for linguistic research and NLP applications in Indian languages. With proper dataset curation and tokenization logic, the system achieves high accuracy and supports extendable workflows for annotation, tagging, and evaluation.
