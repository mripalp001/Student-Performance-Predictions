import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import requests
import os

# === Load model ===
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")

model_url = "https://github.com/mripalp001/Student-Performance-Predictions/raw/refs/heads/main/model.joblib"
model_filename = "model.joblib"


# Fungsi untuk mengunduh model dari GitHub
def download_model(url, filename):
    if not os.path.isfile(filename):
        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)


# Unduh model jika belum ada
download_model(model_url, model_filename)

# Memuat model dari file

model = load(model_filename)

try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error(
        "Model tidak memiliki 'feature_names_in_'. Latih ulang model dengan DataFrame."
    )
    st.stop()


def predict(df):
    pred_proba = model.predict_proba(df)[:, 1]
    pred_label = model.predict(df)
    return pred_label, pred_proba


st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Unggah file CSV data mahasiswa untuk memprediksi kemungkinan dropout.")

# === Upload CSV ===
st.subheader("📁 Upload CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("🗂️ Data yang Diupload:")
        st.dataframe(df.head())

        # Validasi kolom
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            st.error("❌ File CSV tidak memiliki semua kolom yang dibutuhkan model.")
            st.write("Kolom yang hilang:", missing)
        else:
            if st.button("🔍 Prediksi"):
                pred, proba = predict(df[feature_names])
                df["Prediksi Dropout"] = np.where(pred == 1, "Ya", "Tidak")
                df["Probabilitas"] = proba.round(2)
                st.success("✅ Prediksi selesai!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil",
                    data=csv,
                    file_name="hasil_prediksi_dropout.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
