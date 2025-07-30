import streamlit as st
import numpy as np
import joblib
import os

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="🧠 Customer Segmentation",
    layout="centered",
    page_icon="🧠"
)

# ---------------- BRIGHT MODERN CSS ----------------
st.markdown("""
<style>
/* === Global Styling === */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f6d365, #fda085);
    color: #1f1f1f;
}

/* === Header === */
h1 {
    text-align: center;
    font-size: 50px !important;
    font-weight: 900;
    color: #2e2e2e;
    padding: 40px 0 10px 0;
}

/* === Form Card === */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    padding: 40px;
    margin-top: 30px;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}

/* === Labels & Selectbox === */
label, .stSelectbox label {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #333 !important;
    margin-top: 15px;
}
.stSelectbox > div > div,
.stSlider > div > div {
    background-color: #ffffffaa !important;
    border-radius: 12px;
    color: #111 !important;
    font-weight: bold;
}

/* === Slider Labels === */
.css-1emrehy, .css-14xtw13 {
    font-size: 18px !important;
    color: #000 !important;
    font-weight: 600 !important;
}

/* === Button Styling === */
.stButton > button {
    background-color: #ff6f61;
    color: white;
    font-size: 20px;
    font-weight: 900;
    padding: 12px 24px;
    border-radius: 10px;
    margin-top: 30px;
    width: 100%;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background-color: #ff3d2e;
    transform: scale(1.05);
    cursor: pointer;
}

/* === Output Styling === */
.stAlert, .stSuccess, .stInfo {
    font-size: 18px !important;
    font-weight: bold;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model_path = "customer_segmentation_model.pkl"
    try:
        if not os.path.exists(model_path):
            st.error("❌ Model file not found.")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None

model_data = load_model()

# ---------------- App Title ----------------
st.markdown("<h1>🧠 Customer Segmentation</h1>", unsafe_allow_html=True)

# ---------------- Main Content ----------------
if model_data:
    try:
        model = model_data.get("model")
        scaler = model_data.get("scaler")
        feature_names = model_data.get("feature_names", [])
        optimal_k = model_data.get("optimal_k", 0)

        if model is None or scaler is None:
            st.error("⚠️ Model or scaler is missing.")
            st.stop()

        # ---------------- Input Form ----------------
        with st.form(key="customer_form"):
            st.markdown("### 👤 Fill in customer details:")

            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 18, 80, 30)
            with col2:
                annual_income = st.slider("Annual Income ($k)", 10, 150, 60)
                spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)

            submit = st.form_submit_button("🔍 Predict Segment")

        # ---------------- Prediction Logic ----------------
        if submit:
            gender_num = 1 if gender == "Male" else 0
            input_data = np.array([gender_num, age, annual_income, spending_score])

            try:
                scaled_input = scaler.transform([input_data])
                cluster = model.predict(scaled_input)[0]
                distances = model.transform(scaled_input)[0]
                confidence = max(distances) / sum(distances)

                st.success(f"🎯 Segment Prediction: Cluster {cluster}")
                st.info(f"📊 Confidence Level: `{confidence:.2%}`")

                with st.expander("📍 Distances to All Cluster Centers"):
                    for i, dist in enumerate(distances):
                        st.markdown(f"- Cluster `{i}`: `{dist:.4f}`")

            except Exception as e:
                st.error(f"❌ Prediction Failed: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
else:
    st.stop()
