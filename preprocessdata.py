import cv2
import os
from PIL import Image
import numpy as np

# Path ke folder dataset (input) dan folder hasil preprocessing (output)
input_folder = "dataset_raw"
output_folder = "dataset_preprocessed"

# Pastikan folder output ada
os.makedirs(output_folder, exist_ok=True)

# Load model deteksi wajah dari OpenCV (pre-trained Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk preprocessing 1 gambar
def preprocess_image(image_path, output_path, target_size=(128, 128)):
    # Baca gambar
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Kalau ada wajah terdeteksi
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]  # Crop wajah
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert ke RGB untuk PIL
        
        pil_image = Image.fromarray(face)
        pil_image = pil_image.resize(target_size)  # Resize ke ukuran target
        
        # Simpan gambar hasil preprocessing
        pil_image.save(output_path)
        return True  # Sukses memproses
    
    return False  # Tidak ada wajah terdeteksi

# Loop semua file gambar di input_folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        success = preprocess_image(input_path, output_path)
        
        if success:
            print(f"Berhasil proses {filename}")
        else:
            print(f"Tidak ada wajah terdeteksi di {filename}")
