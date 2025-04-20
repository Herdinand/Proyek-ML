import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Path folder dataset hasil preprocessing
dataset_folder = "dataset_preprocessed"

# Step 1: Load Data
X = []
y = []

for filename in os.listdir(dataset_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        label = filename.split('_')[0]  # Ambil label dari nama file sebelum "_"
        img_path = os.path.join(dataset_folder, filename)
        
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.0  # Normalisasi pixel value
        img = img.flatten()  # Flatten gambar ke 1D
        
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Shape data: {X.shape}, Label unik: {np.unique(y)}")

# Step 2: Encode Label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 4: Train SVM
model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)  # Kernel linear untuk data besar
model.fit(X_train, y_train)

# Step 5: Evaluasi Model
y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
