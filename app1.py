import streamlit as st
import numpy as np
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Klasifikasi Gerakan Tangan sEMG",
    layout="wide"
)

# =========================
# LOAD MODEL DAN ENCODER
# =========================
model = joblib.load('saved_models/uas/best_model.pkl')
le = joblib.load('saved_models/uas/label_encoder.pkl')

# Mapping label ke nama gerakan
gerakan_map = {
    1: "Cylindrical",
    2: "Hook",
    3: "Tip",
    4: "Palmar",
    5: "Spherical",
    6: "Lateral"
}

# =========================
# SIDEBAR INFORMASI PROYEK
# =========================
st.sidebar.title("Informasi Proyek")
st.sidebar.write("""
Nama Proyek:
Klasifikasi Gerakan Tangan berbasis sEMG

Dataset:
SemgHandMovementCh2
(UCI Machine Learning Repository)

Metodologi:
CRISP-DM

Model:
1-NN Euclidean
(Akurasi sekitar 60%)

Jumlah Kelas:
6 Gerakan Tangan
""")

st.sidebar.write("Demo UAS Proyek Sains Data")
st.sidebar.write("By Muhammad Hanif")

# =========================
# JUDUL UTAMA
# =========================
st.title("Klasifikasi Gerakan Tangan Berdasarkan Sinyal sEMG")

st.write("""
Aplikasi ini digunakan untuk memprediksi jenis gerakan tangan
berdasarkan sinyal surface electromyography (sEMG) berbentuk time series.
""")

st.markdown("---")

# =========================
# PENJELASAN KELAS GERAKAN
# =========================
st.subheader("Kelas Gerakan Tangan")

st.write("""
Dataset SemgHandMovementCh2 terdiri dari enam jenis gerakan tangan,
yang masing-masing direpresentasikan oleh pola sinyal sEMG yang berbeda:
""")

st.write("""
1. Cylindrical (Genggaman Silinder)  
   Gerakan menggenggam objek berbentuk silinder seperti botol atau gelas.

2. Hook (Genggaman Kait)  
   Gerakan menggenggam tanpa dominasi ibu jari, seperti membawa tas.

3. Tip (Genggaman Ujung Jari)  
   Gerakan presisi menggunakan ujung jari dan ibu jari, misalnya mengambil koin.

4. Palmar (Genggaman Telapak Tangan)  
   Gerakan menggenggam atau menekan objek menggunakan seluruh telapak tangan.

5. Spherical (Genggaman Bola)  
   Gerakan menggenggam objek berbentuk bulat seperti bola atau apel.

6. Lateral (Genggaman Samping)  
   Gerakan menjepit objek menggunakan ibu jari dan sisi jari telunjuk,
   seperti memegang kunci.
""")

st.markdown("---")

# =========================
# LAYOUT INPUT DAN OUTPUT
# =========================
col1, col2 = st.columns([1, 1])

# =========================
# INPUT DATA
# =========================
with col1:
    st.subheader("Input Data Sinyal")
    st.write("""
    Upload file .txt yang berisi:
    - 1 baris data
    - 1500 nilai numerik
    - Dipisahkan dengan spasi
    """)

    uploaded_file = st.file_uploader("Upload file sinyal sEMG", type="txt")

# =========================
# OUTPUT PREDIKSI
# =========================
with col2:
    st.subheader("Hasil Prediksi")

    if uploaded_file is not None:
        try:
            data = np.loadtxt(uploaded_file).reshape(1, 1500)

            pred_encoded = model.predict(data)
            pred_label = le.inverse_transform(pred_encoded)[0]
            gerakan = gerakan_map[pred_label]

            st.write("Hasil Prediksi Gerakan Tangan:")
            st.write(f"Gerakan: {gerakan}")
            st.write(f"Label Kelas: {pred_label}")

        except Exception as e:
            st.error(
                "Terjadi kesalahan. Pastikan file berisi 1500 nilai numerik "
                "dalam satu baris dan dipisahkan oleh spasi."
            )

# =========================
# VISUALISASI SINYAL
# =========================
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("Visualisasi Sinyal sEMG")
    st.line_chart(data.flatten())

# =========================
# PENJELASAN MODEL
# =========================
st.markdown("---")
st.subheader("Penjelasan Model")

st.write("""
Model yang digunakan adalah 1-Nearest Neighbor (1-NN) dengan jarak Euclidean.
Setiap sinyal input dibandingkan dengan data latih untuk mencari pola sinyal
yang paling mirip, kemudian kelas dari tetangga terdekat digunakan sebagai
hasil prediksi.

Model ini digunakan sebagai baseline dan pembanding terhadap model lanjutan
seperti DTW dan CNN dalam proyek ini.
""")

st.markdown("---")
st.write("Repository GitHub: https://munif26.github.io/proyeksaindata/ujianakhirsemesterpsd00.html")
