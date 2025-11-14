# 🧠 Text & Wine Classification with Classical Machine Learning

This project explores classical machine learning techniques for **text classification** and **numeric tabular data classification**. It includes a custom K-Nearest Neighbors (KNN) implementation for text, distance metric comparisons, TF–IDF experiments, Rocchio classification, and scikit-learn models applied to the Wine dataset.

---

## 📂 Datasets

### **1. Newsgroups Text Data (Hockey vs Windows)**
- ~1000 documents  
- 2 classes: Hockey (1), Windows (0)  
- Term-document matrices with raw term counts  
- Preprocessed with tokenization, stopword removal, and stemming  
- Contains:
  - Training matrix  
  - Test matrix  
  - Train/test labels  
  - Vocabulary file  

Used for:
- Custom KNN implementation  
- Distance metric comparison  
- TF–IDF vs raw term-frequency analysis  
- Rocchio (nearest centroid) classification  
- scikit-learn NearestCentroid comparison  

---

### **2. Wine Dataset (from scikit-learn)**  
Loaded via `load_wine()` and converted to a Pandas DataFrame.

Used for:
- KNN classification with normalization  
- Hyperparameter tuning  
- Decision Trees  
- Naive Bayes (Gaussian)  
- Linear Discriminant Analysis (LDA)  
- Cross-validation and overfitting analysis  

---

## ✅ Implemented Tasks

### 🔹 **Part 1 — KNN Classification on Text Data**

#### **1(a). Data Loading & Term Frequency Analysis**
- Loaded training/test matrices and label files  
- Computed total term frequencies  
- Displayed top 20 most frequent terms  
- Plotted long-tail frequency distribution  

#### **1(b). Custom KNN Classifier**
Implemented without scikit-learn.  
Supports four distance metrics:
- Euclidean  
- Cosine  
- Manhattan  
- Pearson  

Returns:
- Predicted class  
- Indices of top-K neighbors  

Tested on first two test instances.

#### **1(c). Evaluation Function**
- Computes classification accuracy across all test instances  
- Uses the custom KNN classifier  
- Accuracy = correct / total  

#### **1(d). Accuracy vs K (5 to 100)**
For each K:
- Evaluated Euclidean, Cosine, Manhattan, Pearson  
- Plotted all four accuracy curves in one figure  

#### **1(e). TF–IDF Experiments**
- Converted raw term counts into TF–IDF  
- Re-evaluated Cosine-based KNN for K=5…100  
- Compared TF–IDF vs raw count performance in a single plot  

#### **1(f). Rocchio / Nearest Centroid Classifier (Custom)**
- Computed centroid vector per class  
- Classified by cosine similarity  
- Compared performance to best KNN configuration  
- Works for any number of classes  

#### **1(g). scikit-learn Nearest Centroid**
- Evaluated on the same text data  
- Compared accuracy with custom Rocchio implementation  

---

### 🔹 **Part 2 — Predictive Modeling on Wine Data**

#### **2(a). Data Preparation**
- Loaded dataset  
- Ensured all features were numeric  
- Split into 80% train / 20% test (random_state=111)  

#### **2(b). KNN Classification**

**(i) Baseline Model (K=10)**
- Normalized features to [0,1]  
- Generated:  
  - Confusion matrix  
  - Classification report  
  - Train/test accuracy  

**(ii) Hyperparameter Exploration**
- Tested K=5…100  
- Compared `weights="uniform"` vs `weights="distance"`  
- Plotted accuracy curves  
- Selected best K + weighting 

**(iii) Overfitting Analysis (Uniform Only)**
- Plot of training vs test accuracy across K  
- Identified overfitting & underfitting regions  

---

#### **2(c). Decision Trees**
**Model 1 — Default Tree**
- Trained on non-normalized data  
- Generated confusion matrix, classification report  
- Compared training vs test accuracy (bias–variance interpretation)

**Model 2 — Constrained Tree**
- `criterion="gini"`, `min_samples_split=10`, `max_depth=4`  
- Reported updated accuracy  
- Plotted tree visualization  

---

#### **2(d). Naive Bayes & LDA**
For both models:
- Performed **10-fold cross-validation** on training data  
- Reported mean CV accuracy  
- Trained final model + evaluated on test set  
- Compared:
  - CV accuracy  
  - Training accuracy  
  - Test accuracy  

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  

---


---

## ▶️ How to Run

pip install -r requirements.txt

jupyter notebook notebooks/KNN.ipynb
