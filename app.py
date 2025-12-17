import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Judul aplikasi
st.title("Deteksi Objek dengan YOLOv8 dari Kamera dan Estimasi Kelayakan")

# Load model YOLOv8 (versi nano)
@st.cache_resource  # Cache model agar tidak reload setiap kali
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Fungsi simulasi untuk menghitung "Persentase Kelayakan"
def hitung_estimasi_layak(detections):
    score = 0
    for r in detections:
        # Mengambil ID kelas objek yang terdeteksi
        classes = r.boxes.cls.tolist()

        # Logika: Jika terdeteksi 'person' (ID 0)
        if 0 in classes:
            score += 60  # Skor dasar jika ada orang

    # Simulasi tambahan: Analisis elemen lingkungan sederhana
    score += 25

    return min(score, 100)  # Batas maksimal 100%

# Capture gambar dari kamera
st.write("Klik tombol di bawah untuk capture gambar dari kamera:")
captured_image = st.camera_input("Capture Gambar")

if captured_image is not None:
    # Baca gambar yang di-capture
    image = Image.open(captured_image)
    img_array = np.array(image)
    
    # Tampilkan gambar asli
    st.image(image, caption="Gambar yang Di-Capture", use_column_width=True)
    
    # Jalankan deteksi
    with st.spinner("Menjalankan deteksi..."):
        results = model(img_array)
    
    # Ambil hasil deteksi (gambar dengan bounding box)
    annotated_img = results[0].plot()  # Plot hasil deteksi
    
    # Tampilkan gambar hasil deteksi
    st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)
    
    # Hitung dan tampilkan estimasi kelayakan
    estimasi = hitung_estimasi_layak(results)
    st.write(f"**Persentase Kelayakan:** {estimasi}%")
    
    # Tampilkan detail deteksi (opsional)
    st.subheader("Detail Deteksi:")
    for r in results:
        st.write(f"Objek terdeteksi: {r.boxes.cls.tolist()}")  # ID kelas
        st.write(f"Koordinat bounding box: {r.boxes.xyxy.tolist()}")  # Koordinat
else:
    st.write("Capture gambar dari kamera untuk memulai deteksi.")