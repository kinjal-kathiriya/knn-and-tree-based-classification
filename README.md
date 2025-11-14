## knn-and-tree-based-classification


# 🧠 Text & Wine Classification with Classical Machine Learning

This project implements and compares several classical machine learning models on two different types of data:

1. **Text classification** using a subset of the 20 Newsgroups dataset (Hockey vs Windows)  
2. **Multiclass classification** on the Wine dataset from scikit-learn  

The work includes:

- A custom implementation of **K-Nearest Neighbors (KNN)** for text
- Experiments with **multiple distance metrics** (Euclidean, Cosine, Manhattan, Pearson)
- Use of **TF–IDF** representations
- A custom **Rocchio (nearest centroid)** classifier and scikit-learn’s NearestCentroid
- KNN, Decision Trees, Naive Bayes, and Linear Discriminant Analysis on the Wine dataset
- Analysis of **model performance, overfitting, and bias–variance trade-offs**

---

## 📂 Datasets

### 1. Newsgroups Text Data (Hockey vs Windows)

A subset of the classic 20 Newsgroups dataset:

- ~1000 documents  
- 2 classes:
  - **1 → Hockey**
  - **0 → Microsoft Windows**
- Data provided as:
  - Term-by-document matrices for **train** and **test**
  - Separate label files for train/test
  - Vocabulary file with ~5,500 terms
- Values are **raw term counts** (already tokenized, stopwords removed, and stemmed)

This dataset is used for:
- Custom KNN classifier (no scikit-learn for this part)
- Distance metric comparison
- TF–IDF vs raw term-frequency comparison
- Custom Rocchio (nearest centroid) classifier
- scikit-learn NearestCentroid classifier

---

### 2. Wine Dataset (from scikit-learn)

Loaded using:

```python
from sklearn.datasets import load_wine
wine = load_wine()
