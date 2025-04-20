import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Asumsikan X (fitur) dan y (label) sudah diproses dari preprocessing sebelumnya
# Contoh:
# X = images.reshape(len(images), -1)  # flatten 200x200 gambar menjadi 40000 features
# y = features['ID'].values

# Membagi data menjadi train dan test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Ukuran Train: {X_train.shape}, Ukuran Test: {X_test.shape}")

# ===================
# Model Training (Logistic Regression biasa)
# ===================
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Prediksi
y_pred = logreg.predict(X_test)

# Evaluasi
print("\n=== Evaluasi Logistic Regression Sebelum Tuning ===")
print(classification_report(y_test, y_pred))

# ===================
# Model Tuning (Grid Search untuk Logistic Regression)
# ===================
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'saga']
}

cv = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n=== Hasil Grid Search ===")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# ===================
# Evaluasi Model Terbaik dari Grid Search
# ===================
best_logreg = grid_search.best_estimator_
y_pred_best = best_logreg.predict(X_test)

print("\n=== Evaluasi Logistic Regression Setelah Tuning ===")
print(classification_report(y_test, y_pred_best))

# ===================
# (Opsional) Visualisasi Confusion Matrix
# ===================
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Logistic Regression Setelah Tuning')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
