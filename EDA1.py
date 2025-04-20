import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv('./data/at/j/features.csv')

# Visualisasi jumlah data per gender
gender_counts = data['gender'].value_counts()

plt.figure(figsize=(6,4))
gender_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Jumlah Data Wajah per Gender')
plt.xlabel('Gender')
plt.ylabel('Jumlah Wajah')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Kesimpulan
print("KESIMPULAN:")
print("Grafik menunjukkan jumlah wajah yang berhasil ditangkap berdasarkan gender.\n"
      "Jika salah satu gender lebih banyak, berarti saat pengumpulan data, user dari gender tersebut lebih aktif atau lebih banyak wajah yang terdeteksi.")
