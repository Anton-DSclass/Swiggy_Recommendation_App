Below is a **clean, professional GitHub README** tailored exactly for your **Swiggy Restaurant Recommendation System (Streamlit + ML)** project.
You can **copy–paste directly into `README.md`** ✅

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


