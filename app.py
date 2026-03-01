import streamlit as st
import pandas as pd
import joblib
import kagglehub
import os
import requests
from io import StringIO

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # These files were created by your Training Script
    model = joblib.load('attrition_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

model, encoders = load_assets()

# --- 2. UI HEADER ---
st.set_page_config(page_title="Universal HR Predictor", layout="wide")
st.title("🛡️ Universal Employee Risk Scanner")
st.write("Analyze data from **Kaggle Slugs** or **Direct CSV URLs**.")

# --- 3. DYNAMIC INPUT ---
user_input = st.text_input(
    "Enter Kaggle Slug or CSV URL:", 
    placeholder="e.g., username/dataset-name OR https://raw.github...data.csv"
)

if st.button("Download & Process"):
    try:
        with st.spinner("Fetching data..."):
            # --- LOGIC: URL vs KAGGLE ---
            if user_input.startswith(("http://", "https://")):
                # Handle Direct CSV Link
                df = pd.read_csv(user_input)
                source = "Direct URL"
            else:
                # Handle Kaggle Slug
                path = kagglehub.dataset_download(user_input)
                files = [f for f in os.listdir(path) if f.endswith('.csv')]
                df = pd.read_csv(os.path.join(path, files[0]))
                source = f"Kaggle ({files[0]})"

        st.success(f"Successfully loaded data from {source}!")

        # --- 4. PRE-PROCESSING (The Translator) ---
        X = df.copy()
        # Drop non-feature columns
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

        # Use saved encoders to turn Text into Numbers
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = X[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0)

        # --- 5. PREDICTION (The Brain) ---
        # Get probability of leaving (class 1)
        risk_probs = model.predict_proba(X)[:, 1]
        df['Risk_Score'] = risk_probs
        df['Status'] = df['Risk_Score'].apply(
            lambda x: "🚨 High Risk" if x > 0.7 else ("⚠️ Warning" if x > 0.4 else "✅ Stable")
        )

        # --- 6. DISPLAY RESULTS ---
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Employees Analyzed", len(df))
        c2.metric("Avg Risk Score", f"{df['Risk_Score'].mean():.1%}")
        c3.info(f"Source: {source}")

        st.subheader("Detailed Risk Report")
        # Highlight high risk employees at the top
        st.dataframe(df.sort_values(by='Risk_Score', ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")