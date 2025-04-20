import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===================================
# Load Data (ganti path sesuai dataset kamu)
# ===================================
# Misal fitur dan label sudah dari preprocessing sebelumnya
features = pd.read_csv('./data/at/j/features.csv')

# Contoh load images
images = np.load('./data/processed_images.npy')  # Anggap kamu simpan hasil preprocessing image ke sini

# X dan y
X = images.reshape(len(images), -1)  # Flatten gambar 200x200 jadi 40000 fitur
y = features['ID'].values

# ===================================
# Split Data
# ===================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Ukuran Train: {X_train.shape}, Ukuran Test: {X_test.shape}")

# ===================================
# Training Model Random Forest
# ===================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf.predict(X_test)

# Evaluasi Awal
print("\n=== Evaluasi Random Forest Sebelum Tuning ===")
print(classification_report(y_test, y_pred_rf))

# ===================================
# Tuning Model Random Forest (Grid Search)
# ===================================
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

cv = StratifiedKFold(n_splits=5)

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)

print("\n=== Hasil Grid Search Random Forest ===")
print(f"Best Parameters: {grid_search_rf.best_params_}")
print(f"Best Cross-Validation Score: {grid_search_rf.best_score_}")

# ===================================
# Evaluasi Model Terbaik Setelah Tuning
# ===================================
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\n=== Evaluasi Random Forest Setelah Tuning ===")
print(classification_report(y_test, y_pred_best_rf))

# ===================================
# Visualisasi Confusion Matrix
# ===================================
cm_rf = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(10,8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix Random Forest Setelah Tuning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
