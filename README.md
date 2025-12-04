

# 🌐 **TruthLens — Semantic Analysis of Political Bias in News Articles**

### **Authors:**

* **Arth Sindekar**
* **Sheshang Ramesh**

---

## 📌 **Overview**

**TruthLens** is an end-to-end machine learning project that automatically predicts political bias — **Left**, **Right**, or **Center** — from news articles.
Using a combination of **classical NLP models** (Logistic Regression, SVM, Decision Tree) and a **fine-tuned BERT transformer**, the system analyzes linguistic cues, framing patterns, vocabulary, and semantic structure to determine ideological leaning.

This project aims to promote **media transparency**, improve **misinformation analysis**, and develop tools for understanding **ideological framing** in real-world news data.

---

## 📰 **Why Political Bias Detection?**

Media shapes public opinion.
But subtle bias often goes unnoticed:

* Selective framing
* Loaded language
* Emphasis or omission
* Tone and sentiment differences

TruthLens explores whether machine learning can **quantify these patterns**, enabling:

✔ Bias-aware media consumption
✔ Fact-checking support tools
✔ Academic research in media studies
✔ Automated large-scale bias analysis

---

## 📂 **Project Structure**

```
truth-lens-media-bias/
│
├── logistic_regression.py       # TF-IDF + Logistic Regression model
├── svm_model.py                 # TF-IDF + Support Vector Machine
├── Tree_Classifier.py           # Decision Tree Model with feature importance
├── BERT_model.py                # Fine-tuned BERT classifier
│
├── data/                        # Parquet dataset splits
│   ├── train-00000-of-00001.parquet
│   ├── valid-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
│
└── README.md                    # You are here
```

---

## 📊 **Dataset**

The project uses two openly available datasets from HuggingFace:

* **Article Bias Prediction Media Splits**
* **BABE (Balanced Annotated Bias Evaluation)**

Total articles: **~34,000**
Labels:

* **0 — Left**
* **1 — Right**
* **2 — Center**

### ⚠️ Validation Fix

The provided validation split was **heavily imbalanced** (~70% Left), resulting in misleading performance.

➡️ We created a **10% stratified validation split** to ensure balanced evaluation.

---

## 🧠 **Models Implemented**

### ✔ **1. Logistic Regression (TF–IDF)**

* Fast, simple, works well for sparse features
* Achieves **0.77 validation accuracy**, **0.55 test accuracy**
* Pipeline includes 100k TF-IDF vocabulary + bigrams

### ✔ **2. Support Vector Machine**

* Linear SVM with `class_weight="balanced"`
* Achieves **0.786 in-distribution**, **0.548 out-of-distribution**
* Robust baseline for TF-IDF models

### ✔ **3. Decision Tree**

* Provides **feature importance** analysis
* Lower accuracy due to overfitting
* Reveals which political terms influence predictions

### ✔ **4. BERT Transformer**

* Fine-tuned `bert-base-uncased`
* Achieves **0.743 validation**, **0.49 test**
* Performs best semantically but needs more tuning for domain generalization

---





## 📈 **Results Summary**

### **Validation Results (In-Distribution)**

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | **0.7698** |
| SVM                 | **0.7868** |
| Decision Tree       | **0.6728** |
| BERT                | **0.7432** |

### **Test Results (Out-of-Distribution)**

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | **0.5461** |
| SVM                 | **0.5485** |
| Decision Tree       | **0.4500** |
| BERT                | **0.4900** |

---

## 🔍 **Key Findings**

* Models perform **much better on in-distribution validation** than on unseen test data.
* The main issue: **domain shift** — publishers use different vocabulary and linguistic styles.
* TF-IDF models struggle with unseen words.
* BERT captures semantics but needs **more training and deeper fine-tuning**.
* Decision Trees show interpretable political keywords:

  * *“donald john”, “mr”, “fox”, “npr”, “president obama”, “reuters”*, etc.

---

## 🛠 **Technologies Used**

* **Python 3.10+**
* **scikit-learn**
* **PyTorch**
* **Hugging Face Transformers**
* **pandas / numpy / tqdm**
* **Matplotlib / Seaborn (optional visualization)**

---

## 📘 **Future Improvements**

* Use larger transformer architectures (RoBERTa, DeBERTa, Longformer)
* Add metadata-aware models (publisher, topic, year)
* Apply adversarial domain adaptation
* Perform more aggressive text augmentation
* Build explainability dashboards (Grad-CAM for BERT)

---

## ❤️ **Acknowledgements**

This project was built for:
**CS5100 – Foundations of Artificial Intelligence**

Special thanks to the dataset contributors and the open-source NLP community.



