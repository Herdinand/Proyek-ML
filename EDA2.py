import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv('./data/at/j/features.csv')

# Scatter plot ukuran wajah
plt.figure(figsize=(6,6))
plt.scatter(data['w'], data['h'], alpha=0.7, color='green')
plt.title('Scatter Plot Ukuran Wajah (Width vs Height)')
plt.xlabel('Width (Lebar)')
plt.ylabel('Height (Tinggi)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Kesimpulan
print("KESIMPULAN:")
print("Scatter plot menunjukkan hubungan antara lebar dan tinggi wajah yang terdeteksi.\n"
      "Jika titik-titik tersebar rapat sepanjang garis diagonal, berarti proporsi wajah (width dan height) cenderung konsisten.\n"
      "Jika ada sebaran yang sangat lebar, berarti ada variasi ukuran wajah (bisa karena jarak dari kamera atau perbedaan postur).")
