# 🧠 Customer Segmentation System 

## 📌 Overview
This project segments customers into different groups using unsupervised machine learning techniques.

The goal is to help businesses understand customer behavior and improve marketing strategies.

---

## 🎯 Problem
Businesses often treat all customers the same, which leads to inefficient marketing.

This project solves that by meaningfull segments.

---

## 🧠 Techniques Used
- KMeans Clustering
- DBSCAN Clustering
- PCA (Dimensionality Reduction)
- Feature Scaling (StandardScaler)

---

## 📊 Process 
1. Data preprocessing and scaling
2. Finding optimal clusters using Elbow Method
3. Applying KMeans and DBSCAN
4. Visualizing clusters using PCA
5. Extracting business insight

---

## 📈 Results 

### KMeans Clustering
- Cluster 0 `n High income, high spending (VIP customers)
- Cluster 1 `n Medium-value customers
- Cluster 2 `n Low-value customers

### BDSCAN 
- Identified outliers (unusual customers)

---

## 💡 Business Value
- Identify high-value customers for premium targeting
- improve marketing strategies
- Detect unusual customer behavior
- Increase customer retention

---

## ⚙️ Tools
- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib

  ---

  ## 📁 Project Structure

  customer-segmentation/ |--- data/ |--- notebook/ |--- src/ |--- models/ |--- README.md |--- requirements.txt

  ---

  ## ▶️ How to Run
  ```bash
  pip install -r requirements.txt
  python src/data.py
  python notebook/project.py
