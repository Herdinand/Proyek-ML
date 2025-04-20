import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data CSV
csv_path = './data/at/j/features.csv'  # Sesuaikan path kalau berbeda
df = pd.read_csv(csv_path)

# Cek beberapa data awal
print(df.head())

# Set style for better looking plots
sns.set(style="whitegrid")

# --- Visualisasi 1: Histogram Sebaran Usia ---
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], kde=True, color='skyblue', bins=15)

plt.title('Distribusi Usia dalam Dataset', fontsize=14)
plt.xlabel('Usia', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.tight_layout()
plt.show()

# --- Visualisasi 2: Boxplot Usia Berdasarkan Gender ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='age', data=df, palette='Set2')

plt.title('Penyebaran Usia Berdasarkan Gender', fontsize=14)
plt.xlabel('Gender (M/F)', fontsize=12)
plt.ylabel('Usia', fontsize=12)
plt.tight_layout()
plt.show()
