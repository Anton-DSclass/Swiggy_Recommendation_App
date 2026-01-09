Streamlit Code

---

# 🍽️ Swiggy Restaurant Recommendation System

A **Machine Learning–based restaurant recommendation system** built using **Python, Streamlit, and Scikit-learn**, inspired by Swiggy.
This app recommends restaurants based on **city, cuisine preference, minimum rating, cost, and delivery time** using a **content-based filtering approach**.

---

## 🚀 Features

* 🏙️ **City-based filtering**
* 🍱 **Cuisine-based recommendation**
* ⭐ **Minimum rating selection**
* 📊 **Similarity matching using ML**
* 🔄 **Smart fallback logic** when data is limited
* 🎨 **Custom Streamlit UI** with Swiggy-style theme
* ⚡ **Fast loading using caching**

---

## 🧠 Recommendation Logic (How it Works)

1. **User Inputs**

   * City
   * Cuisine
   * Minimum Rating
   * Number of Recommendations

2. **Filtering Steps**

   * City + Cuisine + Rating
   * If insufficient → City + Rating
   * If still insufficient → Global top-rated restaurants

3. **Machine Learning**

   * Uses **StandardScaler** for normalization
   * Applies **Nearest Neighbors (Cosine Similarity)**
   * Recommends similar restaurants based on:

     * Rating
     * Cost
     * Delivery Time

---

## 🛠️ Tech Stack

| Category      | Tools                                 |
| ------------- | ------------------------------------- |
| Frontend      | Streamlit                             |
| Backend       | Python                                |
| ML            | Scikit-learn                          |
| Data Handling | Pandas                                |
| Styling       | HTML + CSS                            |
| Model         | Nearest Neighbors (Cosine Similarity) |

---

## 📂 Project Structure

```
Project-04/
│
├── Streamlitapp/
│   ├── app.py                  # Main Streamlit app
│   ├── Cleaned.pkl             # Cleaned restaurant data
│   ├── Encoded.pkl             # One-hot encoded features
│   ├── Swiggy-logo.png         # App logo
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── feature_encoding.ipynb
│
├── README.md
└── requirements.txt
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/swiggy-recommendation-system.git
cd swiggy-recommendation-system
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📄 Dataset Description

### `Cleaned.pkl`

Contains:

* Restaurant Name
* City
* Cuisine
* Rating
* Cost
* Delivery Time

### `Encoded.pkl`

Contains:

* Numerical features
* One-hot encoded cuisines
* Used for similarity calculations

---

## 🎯 Use Cases

* Food delivery platforms
* Recommendation systems projects
* Machine learning portfolios
* Academic final-year projects
* Viva & interview demonstrations

---

## 📸 App Preview

> Swiggy-style UI with sidebar preferences and restaurant cards
> *(Add screenshots here if you want)*

---

## 🔮 Future Enhancements

* 🔐 User login & personalization
* 🗺️ Location-based distance filtering
* 🤖 Hybrid recommendation (content + collaborative)
* 📈 Popularity trends
* 🌐 Cloud deployment

---

## 🙌 Author

**Anton Sam**
📧 *Add email if needed*
💼 *Machine Learning | Data Science | Python*

---

## ⭐ Acknowledgements

* Inspired by **Swiggy**
* Streamlit Community
* Scikit-learn Documentation

---
*****************************************************************************************************************************************************************************

Main Code

# 📊 Data Preprocessing, Feature Engineering & Clustering Pipeline

This module focuses on **cleaning raw restaurant data**, **preparing features for machine learning**, **performing clustering**, and **saving processed datasets** for downstream use in a **recommendation system and ML models**.

---

## 🧩 Overview

The goal of this pipeline is to:

* Clean noisy real-world data
* Remove irrelevant and redundant columns
* Prepare numerical & encoded features
* Apply **unsupervised clustering**
* Store optimized datasets for **Streamlit deployment**

---

## 🛠️ Libraries Used

| Category         | Libraries                       |
| ---------------- | ------------------------------- |
| Data Handling    | `pandas`, `numpy`               |
| Visualization    | `matplotlib`, `seaborn`         |
| Statistics       | `scipy.stats`                   |
| Machine Learning | `scikit-learn`, `xgboost`       |
| Models           | Linear, Tree, Boosting, SVM, GP |
| Clustering       | `KMeans`                        |
| Serialization    | `pickle`                        |

---

## 🧹 Step 1: Data Cleaning

### ✔️ Remove Invalid Column Names

```python
df = df.loc[:, df.columns.notna()]
df = df.loc[:, df.columns.str.strip() != ""]
```

* Removes **empty** or **blank** column names
* Prevents pipeline failures

---

### ✔️ Remove Auto-Generated Columns

```python
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
```

* Eliminates unnecessary index columns from CSV imports

---

### ✔️ Remove Time-Range Columns

```python
df = df.loc[:, ~df.columns.str.contains("To|PM|Pm")]
```

* Removes time-based textual columns not useful for ML
* Ensures numeric-only feature space

---

### ✔️ Drop Non-ML Columns

```python
cols_to_remove = ["name", "lic_no", "link", "address", "menu"]
df.drop(columns=cols_to_remove, inplace=True)
```

These columns:

* Are **identifiers or text-heavy**
* Do not contribute to prediction or similarity

---

## 🧠 Step 2: Machine Learning Models Imported

This project supports **multiple regression models** for experimentation:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting
* AdaBoost
* XGBoost
* Support Vector Regression (SVR)
* Gaussian Process Regressor

📌 *Models are imported for benchmarking and performance comparison.*

---

## 🔧 Step 3: Feature Scaling & Pipelines

* **StandardScaler** used for normalization
* **OneHotEncoder** & **MultiLabelBinarizer** for categorical data
* **ColumnTransformer + Pipeline** for clean ML workflows

This ensures:

* Consistent preprocessing
* Reproducibility
* No data leakage

---

## 🧩 Step 4: Clustering (Unsupervised Learning)

### 🔹 KMeans Clustering

```python
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
```

✔️ Groups similar restaurants based on numeric features
✔️ Helps in:

* Restaurant segmentation
* Faster recommendations
* Exploratory analysis

---

### 📏 Cluster Evaluation

```python
score = silhouette_score(X, df['cluster'])
```

**Silhouette Score** measures:

* How well each restaurant fits within its cluster
* Value close to **1 → better clustering**

---

## 💾 Step 5: Dataset Serialization (Pickle)

```python
cleaned.to_pickle("Cleaned.pkl")
encoded.to_pickle("Encoded.pkl")
```

### Why Pickle?

* Faster loading than CSV
* Preserves data types
* Ideal for Streamlit apps

---

## 📂 Output Files

| File          | Description                           |
| ------------- | ------------------------------------- |
| `Cleaned.pkl` | Human-readable restaurant details     |
| `Encoded.pkl` | ML-ready numerical & encoded features |

These files are later used in:

* Recommendation engine
* Similarity calculations
* Streamlit deployment

---

## 🔄 Workflow Summary

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Scaling & Encoding
   ↓
Clustering
   ↓
Pickle Serialization
   ↓
Streamlit App
```

---

## 🎯 Use Cases

* Restaurant recommendation systems
* Customer segmentation
* ML portfolio projects
* Final year / capstone projects
* Interview & viva demonstrations

---

## 🔮 Future Enhancements

* Automatic optimal `k` selection
* Cluster visualization dashboard
* Hybrid recommendation system
* Model performance comparison UI

---

## 👨‍💻 Author

**Anton Sam**
Machine Learning | Data Science | Python





