import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Fungsi preprocessing gambar dengan histogram equalization
def preprocess_images_with_histogram_equalization(image_folder, target_size=(200, 200)):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.pgm'):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            
            # Histogram Equalization untuk meningkatkan kontras
            img_eq = cv2.equalizeHist(img)
            
            # Normalisasi pixel 0-1
            img_eq = img_eq / 255.0
            
            images.append(img_eq)
            labels.append(filename)
    
    return np.array(images), labels

# Fungsi preprocessing CSV (bisa sama seperti sebelumnya)
def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Encode label ID
    label_encoder = LabelEncoder()
    df['ID'] = label_encoder.fit_transform(df['ID'])
    
    # Encode gender
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    
    # Isi missing value jika ada
    if df['age'].isnull().sum() > 0:
        df['age'] = df['age'].fillna(df['age'].median())
    if df['timestamp'].isnull().sum() > 0:
        df['timestamp'] = df['timestamp'].fillna(df['timestamp'].mean())

    return df

# Path ke folder dan CSV
image_folder = './data/at/j'
csv_path = './data/at/j/features.csv'

# Preprocessing dengan histogram equalization
images, labels = preprocess_images_with_histogram_equalization(image_folder)

# Preprocessing CSV
features = preprocess_csv(csv_path)

# Cek hasil
print(f"Jumlah gambar setelah histogram equalization: {len(images)}")
print(f"Beberapa data fitur: {features.head()}")
