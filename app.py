import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('saved_models/uas/best_model.pkl')
le = joblib.load('saved_models/uas/label_encoder.pkl')

# Mapping label ke nama gerakan (sesuai dataset: 1=Cylindrical, 2=Hook, dll.)
gerakan_map = {
    1: "Cylindrical",
    2: "Hook",
    3: "Tip",
    4: "Palmar",
    5: "Spherical",
    6: "Lateral"
}

st.title("Klasifikasi Gerakan Tangan berdasarkan Sinyal sEMG")
st.write("Upload file sinyal time series (.txt, 1 baris dengan 1500 nilai) untuk prediksi gerakan tangan.")
st.write("Dataset: SemgHandMovementCh2 (6 kelas gerakan). Model: 1-NN Euclidean (Akurasi ~60%).")

# Input: Upload file
uploaded_file = st.file_uploader("Upload file .txt", type="txt")

if uploaded_file is not None:
    try:
        # Load data (asumsikan 1 sampel, 1500 titik, format space-separated seperti dataset)
        data = np.loadtxt(uploaded_file).reshape(1, 1500)  # Reshape ke (1, 1500)
        
        # Prediksi
        pred_encoded = model.predict(data)
        pred_label = le.inverse_transform(pred_encoded)[0]  # Decode ke label asli (1-6)
        gerakan = gerakan_map[pred_label]  # Map ke nama gerakan
        
        st.success(f"Prediksi Gerakan: **{gerakan}** (Label: {pred_label})")
        
        # Visualisasi sinyal
        st.subheader("Visualisasi Sinyal Input")
        st.line_chart(data.flatten())
        
    except Exception as e:
        st.error(f"Error: Pastikan file berisi 1500 nilai numerik (space-separated). Detail: {str(e)}")

st.write("---")
st.write("**Penjelasan:** Model menggunakan jarak Euclidean untuk klasifikasi time series. Demo UAS mengikuti CRISP-DM.")
st.write("**Link GitHub:** [Tambahkan link repo kamu di sini]")
