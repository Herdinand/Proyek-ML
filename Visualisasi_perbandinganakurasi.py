import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
# Train Semua Model
# ===================================
# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# ===================================
# Visualisasi Perbandingan Akurasi
# ===================================
# Membuat Bar Chart
models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
accuracies = [acc_logreg, acc_rf, acc_svm, acc_knn]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])

# Menambahkan labels dan title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi Model')

# Menampilkan nilai akurasi di atas bar
for i in range(len(accuracies)):
    plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()
