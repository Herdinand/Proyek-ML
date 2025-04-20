import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

print(f"Ukuran Train: {X_train.shape}, Ukuran Test: {X_test.shape}")

# ===================================
# Training Model KNN
# ===================================
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Prediksi
y_pred_knn = knn.predict(X_test)

# Evaluasi Awal
print("\n=== Evaluasi KNN Sebelum Tuning ===")
print(classification_report(y_test, y_pred_knn))

# ===================================
# Tuning Model KNN (Grid Search)
# ===================================
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

cv = StratifiedKFold(n_splits=5)

grid_search_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search_knn.fit(X_train, y_train)

print("\n=== Hasil Grid Search KNN ===")
print(f"Best Parameters: {grid_search_knn.best_params_}")
print(f"Best Cross-Validation Score: {grid_search_knn.best_score_}")

# ===================================
# Evaluasi Model Terbaik Setelah Tuning
# ===================================
best_knn = grid_search_knn.best_estimator_
y_pred_best_knn = best_knn.predict(X_test)

print("\n=== Evaluasi KNN Setelah Tuning ===")
print(classification_report(y_test, y_pred_best_knn))

# ===================================
# Visualisasi Confusion Matrix
# ===================================
cm_knn = confusion_matrix(y_test, y_pred_best_knn)
plt.figure(figsize=(10,8))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix KNN Setelah Tuning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
