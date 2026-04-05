import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ====== CSS Styling ======
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #161A21 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main text */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* Button style */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ====== Load Best Model ======
model = joblib.load("rf_model.pkl")

# ====== Title ======
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house prices using Machine Learning")

# ====== Sidebar ======
st.sidebar.header("📊 Model Info")
st.sidebar.write("""
This app uses **Random Forest Regression**, which provides high accuracy for predicting house prices.
""")

# ====== Input Section ======
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("🌍 Longitude")
    latitude = st.number_input("📍 Latitude")
    housing_median_age = st.number_input("🏚 House Age")
    total_rooms = st.number_input("🛏 Total Rooms")

with col2:
    total_bedrooms = st.number_input("🛌 Bedrooms")
    population = st.number_input("👨‍👩‍👧 Population")
    households = st.number_input("🏠 Households")
    median_income = st.number_input("💰 Median Income")

# Input array
input_data = np.array([[longitude, latitude, housing_median_age,
                        total_rooms, total_bedrooms,
                        population, households, median_income]])

# ====== Prediction ======
if st.button("🔍 Predict Price"):

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

# ====== Footer ======
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Bhairav Thakare")
