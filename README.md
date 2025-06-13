# Obesity Prediction using Machine Learning

Proyek ini merupakan tugas akhir dari mata kuliah **Bengkel Koding**, yang bertujuan untuk memprediksi tingkat obesitas seseorang berdasarkan data gaya hidup dan pola makan menggunakan model machine learning.

## 📁 Struktur Repository

| File/Folder                | Deskripsi |
|---------------------------|-----------|
| `ObesityDataSet.csv`      | Dataset utama yang digunakan dalam proyek |
| `UAS_BengKod.ipynb`       | Notebook utama yang berisi eksplorasi data, preprocessing, modelling, evaluasi |
| `UAS_BengKod.ipynb - Colab.pdf` | Versi PDF dari notebook untuk dokumentasi |
| `app.py`                  | File Streamlit untuk deployment model |
| `model.pkl`               | Model Random Forest terbaik yang telah dilatih dan disimpan |
| `label_encoder.pkl`       | Encoder untuk fitur kategorikal |
| `scaler.pkl`              | Scaler untuk normalisasi data (jika digunakan) |
| `requirements.txt`        | Daftar dependensi Python untuk menjalankan aplikasi |
| `UAS_BENGKOD_A11202214732.pdf` | Laporan tugas akhir |

## ⚙️ Teknologi yang Digunakan

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (untuk deployment)
- GitHub (untuk version control)

## 📊 Proses yang Dilakukan

1. **Data Preprocessing**
   - Imputasi missing value
   - Encoding fitur kategorikal
   - Penskalaan fitur numerik (jika diperlukan)
   - Deteksi dan analisis outlier

2. **Pemodelan & Evaluasi**
   - Menggunakan model:
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Random Forest (terbaik)
     - Support Vector Machine (SVM)
     - Logistic Regression
   - Evaluasi dengan:
     - Confusion Matrix
     - Accuracy, Precision, Recall, F1-Score
     - Waktu pelatihan (training time)

3. **Hyperparameter Tuning**
   - Menggunakan GridSearchCV & RandomizedSearchCV
   - Perbandingan performa sebelum dan sesudah tuning

4. **Deployment**
   - Menggunakan Streamlit
   - Aplikasi menerima input dari user dan memberikan prediksi tingkat obesitas

## 🚀 Menjalankan Aplikasi Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
