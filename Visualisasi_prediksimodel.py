import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ===================================
# Load Data
# ===================================
# Load fitur dan label
features = pd.read_csv('./data/at/j/features.csv')

# Load hasil preprocessing image
images = np.load('./data/processed_images.npy')  # Pastikan ini hasil preprocessing

# X dan y
X = images.reshape(len(images), -1)  # Flatten 200x200 ke 40000 fitur
y = features['ID'].values

# ===================================
# Split Data
# ===================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================================
# Train Model (contoh dengan Random Forest)
# ===================================
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ===================================
# Visualisasi Hasil Prediksi
# ===================================
# Menampilkan gambar asli dan hasil prediksi untuk beberapa sampel
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel()

# Ambil beberapa gambar uji dan tampilkan prediksi
for i in np.arange(9):
    ax = axes[i]
    
    # Ambil gambar asli dari X_test dan reshape ke bentuk 200x200
    img = X_test[i].reshape(200, 200)
    
    # Tampilkan gambar
    ax.imshow(img, cmap='gray')
    
    # Prediksi ID menggunakan model
    predicted_label = y_pred_rf[i]
    true_label = y_test[i]
    
    # Set judul dengan ID yang diprediksi vs yang sebenarnya
    ax.set_title(f"Pred: {predicted_label}\nTrue: {true_label}")
    
    ax.axis('off')

plt.tight_layout()
plt.show()
