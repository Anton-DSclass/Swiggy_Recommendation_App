import streamlit as st
import pandas as pd
from PIL import Image

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


#-----------------------------------------------------------------------------------------
#backround Colour
#------------------------------------------------------------------------------------------

st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(
        135deg,
        #fff3e0,
        #ffe0b2,
        #ffffff
    );
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #fc8019,
        #ff9f43
    );
}

/* Sidebar text color */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Slider & select label */
.css-16idsys p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

#----------------------------------------------------------------------------------
#logo
#-------------------------------------------------------------------

logo = Image.open(r"C:\Users\ANTON\Documents\VS Code\Project-04\Streamlitapp\Swiggy-logo.png")  # logo path



col1, col2 = st.columns([1, 6])


logo = Image.open("Swiggy-logo.png")
logo = logo.resize((300, 300))   # (width, height)

with col1:
    st.image(logo)



with col2:
    st.markdown("""
    <style>
    .swiggy-title {
        font-family: 'Illuma';
        font-size: 38px;
        font-weight: 900;
        color: #FC8019;   /* Swiggy orange */
        letter-spacing: 0.5px;
    }
    </style>

    <div class="swiggy-title">
        🍽️ Swiggy Restaurant Recommendation System
    </div>
    """, unsafe_allow_html=True)

#with col2:
    #st.markdown("## 🍽️ **Swiggy Restaurant Recommendation System**")


#st.title("🍽️ **Swiggy Restaurant Recommendation System**")

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
st.sidebar.header("**User Preferences**")

st.sidebar.markdown("🏙️ **Select City**")

st.markdown("""
<style>
div[data-baseweb="select"] > div {
    border: 2px solid #4CAF50;   /* stroke color */
    border-radius: 6px;         /* rounded corner */
}
</style>
""", unsafe_allow_html=True)

city = st.sidebar.selectbox(
    "",
    sorted(cleaned_df['city'].dropna().unique())
)


cuisine = st.sidebar.selectbox(
    "**Select Cuisine**",
    sorted(cuisine_cols)
)

min_rating = st.sidebar.slider(
    "**Minimum Rating**",
    0.0, 5.0, 3.5, 0.1
)

top_n = st.sidebar.slider(
    "***Number of Recommendations***",
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
    <div style="
        border:2px solid #ddd;
        padding:5px;
        border-radius:3px;
        margin-bottom: 2px;
        font-size:15px;
        line-height:1.5;
    ">
        <h4 style="margin:0;">🍽️ {row['name']}</h4>
        ⭐ Rating: <b>{row['rating']}</b><br>
        🍱 Cuisine: <b>{row['cuisine']}</b><br>
        📍 City: {row['city']}<br>
        💰 Cost: {row.get('cost', 'N/A')}<br>
       
    </div>
    """, unsafe_allow_html=True)
