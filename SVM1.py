import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ===================================
# Load Data (ganti path sesuai dataset kamu)
# ===================================
# Load fitur dan label
features = pd.read_csv('./data/at/j/features.csv')

# Load hasil preprocessing image
images = np.load('./data/processed_images.npy')  # Harusnya ini hasil preprocessing

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
# Training Model SVM
# ===================================
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Prediksi
y_pred_svm = svm.predict(X_test)

# Evaluasi Awal
print("\n=== Evaluasi SVM Sebelum Tuning ===")
print(classification_report(y_test, y_pred_svm))

# ===================================
# Tuning Model SVM (Grid Search)
# ===================================
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

cv = StratifiedKFold(n_splits=5)

grid_search_svm = GridSearchCV(
    SVC(random_state=42),
    param_grid_svm,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search_svm.fit(X_train, y_train)

print("\n=== Hasil Grid Search SVM ===")
print(f"Best Parameters: {grid_search_svm.best_params_}")
print(f"Best Cross-Validation Score: {grid_search_svm.best_score_}")

# ===================================
# Evaluasi Model Terbaik Setelah Tuning
# ===================================
best_svm = grid_search_svm.best_estimator_
y_pred_best_svm = best_svm.predict(X_test)

print("\n=== Evaluasi SVM Setelah Tuning ===")
print(classification_report(y_test, y_pred_best_svm))

# ===================================
# Visualisasi Confusion Matrix
# ===================================
cm_svm = confusion_matrix(y_test, y_pred_best_svm)
plt.figure(figsize=(10,8))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix SVM Setelah Tuning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
