import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data():
    cleaned = pd.read_pickle(
        r"C:\Users\ANTON\Documents\VS Code\Project-04\Streamlitapp\Cleaned.pkl"
    )
    encoded = pd.read_pickle(
        r"C:\Users\ANTON\Documents\VS Code\Project-04\Streamlitapp\Encoded.pkl"
    )
    return cleaned, encoded

cleaned_df, encoded_df = load_data()

st.title("🍽️ Swiggy Restaurant Recommendation System")

# ----------------------------------
# Cuisine Columns (from Encoded.csv)
# ----------------------------------
cuisine_cols = [c for c in encoded_df.columns if c not in
                ['rating', 'cost', 'delivery_time', 'cluster']]

if not cuisine_cols:
    st.error("No cuisine columns found in Encoded.csv")
    st.stop()

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("User Preferences")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cleaned_df['city'].dropna().unique())
)

cuisine = st.sidebar.selectbox(
    "Select Cuisine",
    sorted(cuisine_cols)
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0, 5.0, 3.5, 0.1
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    3, 10, 5
)

# ----------------------------------
# STEP 1: City + Cuisine + Rating
# ----------------------------------
mask = (
    (cleaned_df['city'] == city) &
    (cleaned_df['rating'] >= min_rating) &
    (encoded_df[cuisine] == 1)
)

cleaned_filtered = cleaned_df.loc[mask].reset_index(drop=True)
encoded_filtered = encoded_df.loc[mask].reset_index(drop=True)

# ----------------------------------
# STEP 2: City + Rating fallback
# ----------------------------------
if len(cleaned_filtered) < 2:
    st.info("Not enough cuisine matches. Showing top restaurants in city.")

    mask = (
        (cleaned_df['city'] == city) &
        (cleaned_df['rating'] >= min_rating)
    )

    cleaned_filtered = cleaned_df.loc[mask].reset_index(drop=True)
    encoded_filtered = encoded_df.loc[mask].reset_index(drop=True)

# ----------------------------------
# STEP 3: Global fallback
# ----------------------------------
if len(cleaned_filtered) < 2:
    st.info("Not enough restaurants. Showing top rated overall.")

    cleaned_filtered = cleaned_df.sort_values(
        'rating', ascending=False
    ).head(50).reset_index(drop=True)

    encoded_filtered = encoded_df.loc[
        cleaned_filtered.index
    ].reset_index(drop=True)

# ----------------------------------
# Safety Check
# ----------------------------------
if len(cleaned_filtered) < 2:
    st.error("Not enough restaurants to recommend.")
    st.stop()

st.write(f"🔍 Found {len(cleaned_filtered)} restaurants")

# ----------------------------------
# Similarity (ENCODED numeric features)
# ----------------------------------
feature_cols = ['rating', 'cost', 'delivery_time']
feature_cols = [c for c in feature_cols if c in encoded_filtered.columns]

X = encoded_filtered[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# Nearest Neighbors
# ----------------------------------
nn = NearestNeighbors(
    n_neighbors=min(top_n + 1, len(X_scaled)),
    metric="cosine"
)
nn.fit(X_scaled)

distances, indices = nn.kneighbors(X_scaled[0].reshape(1, -1))
recommend_indices = indices[0][1:]

recommendations = cleaned_filtered.iloc[recommend_indices]

# ----------------------------------
# Display Results (FROM CLEANED.CSV)
# ----------------------------------
st.subheader("🍴 Recommended Restaurants")

for _, row in recommendations.iterrows():
    st.markdown(f"""
    ### 🍽️ {row['name']}
    ⭐ Rating: **{row['rating']}**
    🍱 Cuisine: **{cuisine}**
    📍 City: {row['city']}
    🏠 Address: {row.get('address_clean', 'N/A')}
    💰 Cost: {row.get('cost', 'N/A')}
    ⏱️ Delivery Time: {row.get('delivery_time', 'N/A')}
    ---
    """)
