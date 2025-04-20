import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk melakukan resize dan normalisasi gambar
def preprocess_images(image_folder, target_size=(200, 200)):
    images = []
    labels = []
    image_paths = []
    
    # Loop melalui semua file gambar dalam folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.pgm'):  # Hanya file .pgm
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Baca gambar dalam grayscale
            img = cv2.resize(img, target_size)  # Resize gambar ke target_size
            
            # Normalisasi pixel (ubah rentang 0-255 menjadi 0-1)
            img = img / 255.0  # Normalisasi
            
            images.append(img)
            image_paths.append(img_path)
    
    return np.array(images), image_paths

# Fungsi untuk preprocessing data CSV (encoding label)
def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Encode label ID (misalnya, ID user jadi angka)
    label_encoder = LabelEncoder()
    df['ID'] = label_encoder.fit_transform(df['ID'])
    
    # Extract features (age, gender, timestamp) untuk model training
    df_features = df[['age', 'gender', 'timestamp']].copy()
    
    # Encode gender menjadi angka (M -> 1, F -> 0)
    df_features['gender'] = df_features['gender'].map({'M': 1, 'F': 0})

    return df_features, label_encoder

# Path ke folder gambar dan file CSV
image_folder = './data/at/j'  # Sesuaikan path jika perlu
csv_path = './data/at/j/features.csv'

# Preprocess gambar
images, image_paths = preprocess_images(image_folder)

# Preprocess CSV untuk fitur tambahan
features, label_encoder = preprocess_csv(csv_path)

# Cek beberapa hasil preprocessing
print(f"Jumlah gambar yang diproses: {len(images)}")
print(f"Beberapa fitur: {features.head()}")

# Simpan hasil preprocessing gambar dan fitur ke file atau variabel lain jika perlu
# Misalnya, kita bisa simpan hasil fitur dan label encoder untuk nanti.
