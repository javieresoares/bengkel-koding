import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Masukkan data berikut untuk memprediksi status obesitas:")

# Input user
age = st.number_input("Usia", min_value=1, max_value=100, value=25)
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.70)
weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=300.0, value=70.0)
calc = st.selectbox("Konsumsi Alkohol (CALC)", ['no', 'Sometimes', 'Frequently', 'Always'])
fcvc = st.slider("Frekuensi Konsumsi Sayur (FCVC)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.slider("Jumlah Makan Per Hari (NCP)", min_value=1.0, max_value=4.0, value=3.0)
ch2o = st.slider("Konsumsi Air Per Hari (CH2O)", min_value=1.0, max_value=3.0, value=2.0)
family_history = st.selectbox("Riwayat Keluarga Obesitas", ['yes', 'no'])
faf = st.slider("Frekuensi Aktivitas Fisik (FAF)", min_value=0.0, max_value=3.0, value=1.0)
tue = st.slider("Waktu Layar (TUE)", min_value=0.0, max_value=2.0, value=1.0)
caec = st.selectbox("Kebiasaan Makan Berlebih (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])

# Map kategori ke label encoded
def label_encode_input(value, mapping_dict):
    return mapping_dict.get(value)

# Buat mapping dari LabelEncoder terlatih
def get_encoder_mapping(encoder, classes):
    return {label: idx for idx, label in enumerate(classes)}

# Ambil nilai mapping dari encoder
gender_map = get_encoder_mapping(label_encoder, label_encoder.classes_)  # perlu disesuaikan

# Karena kamu melakukan `le.fit_transform` per kolom, sebaiknya simpan dict custom sendiri:
manual_maps = {
    "Gender": {'Female': 0, 'Male': 1},
    "CALC": {'no': 2, 'Sometimes': 3, 'Frequently': 0, 'Always': 1},
    "family_history_with_overweight": {'no': 0, 'yes': 1},
    "CAEC": {'no': 2, 'Sometimes': 3, 'Frequently': 1, 'Always': 0},
}

# Masukkan input ke dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": manual_maps["Gender"][gender],
    "Height": height,
    "Weight": weight,
    "CALC": manual_maps["CALC"][calc],
    "FCVC": fcvc,
    "NCP": ncp,
    "CH2O": ch2o,
    "family_history_with_overweight": manual_maps["family_history_with_overweight"][family_history],
    "FAF": faf,
    "TUE": tue,
    "CAEC": manual_maps["CAEC"][caec]
}])

# Normalisasi jika menggunakan scaler
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(input_scaled)
    label = label_encoder.inverse_transform(pred)[0]

    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Obesitas: **{label}**")
