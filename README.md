# Media Bias Classification from News Articles

This project builds and compares multiple text classification models to predict **media bias** from news articles.  
Each article is represented by its **title** and **content**, and the target label is a discrete `bias` class.

We implement and evaluate:

- **Classical ML baselines** using TF–IDF features:
  - Logistic Regression
  - Linear SVM
  - Decision Tree
- A **BERT-based classifier** using Hugging Face Transformers and PyTorch.

---

## 1. Dataset

The dataset is stored locally as Parquet files in the `data/` directory:

```text
data/
  train-00000-of-00001.parquet
  valid-00000-of-00001.parquet
  test-00000-of-00001.parquet
