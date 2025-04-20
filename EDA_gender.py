import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data CSV
csv_path = './data/at/j/features.csv'  # Sesuaikan path kalau berbeda
df = pd.read_csv(csv_path)

# Cek beberapa data awal
print(df.head())

# Visualisasi distribusi gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df, palette='Set2')

plt.title('Distribusi Gender pada Dataset Wajah', fontsize=14)
plt.xlabel('Gender (M/F)', fontsize=12)
plt.ylabel('Jumlah Foto', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
