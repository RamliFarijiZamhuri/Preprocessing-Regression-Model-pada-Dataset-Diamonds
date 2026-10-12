# Penjelasan Kode & Tujuan Proyek (Diamonds Dataset)

Dokumen ini berisi penjelasan lengkap mengenai alur kerja, tujuan, dan rincian setiap bagian kode yang digunakan dalam analisis serta penyiapan data (*data preprocessing*) pada dataset harga berlian (`train_Diamonds.csv`).

---

## Untuk Apa Kode Ini?

Kode ini digunakan untuk **Data Preprocessing (Pembersihan & Penyiapan Data)**. 

Tujuan utamanya adalah mengubah data mentah (*raw data*) berlian—yang mungkin berisi data hilang (*missing values*), pencilan (*outliers*), data duplikat, serta format teks (kategorikal)—menjadi **data yang bersih, terstruktur, dan siap dipakai untuk melatih model Machine Learning** (seperti prediksi harga berlian berdasarkan spesifikasi fisik dan kualitasnya).

---

## Penjelasan Detail Alur dan Fungsi Kode

### 1. Import Library
* **Fungsi Kode:** `import pandas as pd`, `import numpy as np`, `import seaborn as sns`, `import matplotlib.pyplot as plt`, `from sklearn.preprocessing import StandardScaler`
* **Penjelasan:** Mengimpor semua pustaka (*library*) utama yang dibutuhkan:
  * **Pandas & NumPy:** Untuk manipulasi dan pengolahan data tabel/matriks.
  * **Seaborn & Matplotlib:** Untuk visualisasi data dan pembuatan grafik.
  * **StandardScaler (Scikit-Learn):** Untuk menyamakan skala numerik pada variabel data.

---

### 2. Memuat & Memeriksa Data (*Exploratory Data Analysis*)
* **Fungsi Kode:** `pd.read_csv()`, `train.head()`, `train.info()`, `train.describe()`
* **Penjelasan:**
  * Membaca file dataset `train_Diamonds.csv`.
  * `head()` menampilkan 5 baris data teratas untuk gambaran awal.
  * `info()` mengecek tipe data setiap kolom dan mendeteksi ketersediaan data.
  * `describe()` melihat statistik deskriptif dasar (seperti rata-rata, nilai minimum, dan maksimum) dari variabel numerik maupun kategorikal.

---

### 3. Penanganan Data Hilang (*Missing Values*)
* **Fungsi Kode:** `train.isnull().sum()`, `fillna()`
* **Penjelasan:**
  * Mendeteksi kolom mana saja yang memiliki nilai kosong/hilang.
  * **Variabel Numerik:** Diisi menggunakan nilai tengah (*median*) agar tidak terpengaruh oleh angka ekstrem.
  * **Variabel Kategorikal:** Diisi menggunakan nilai yang paling sering muncul (*modus*).

---

### 4. Deteksi & Penanganan Pencilan (*Outliers*)
* **Fungsi Kode:** `sns.boxplot()`, perhitungan **IQR (Interquartile Range)**
* **Penjelasan:**
  * **Boxplot** digunakan untuk memvisualisasikan angka-angka yang berada jauh di luar rentang wajar (pencilan).
  * Menghitung rentang IQR:
    $$ \text{IQR} = Q_3 - Q_1 $$
  * Data ekstrem di luar rentang $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ dibuang agar tidak membuat model Machine Learning menjadi bias saat memprediksi harga.

---

### 5. Normalisasi / Standardisasi Skala (*Feature Scaling*)
* **Fungsi Kode:** `StandardScaler().fit_transform()`
* **Penjelasan:**
  * Fitur numerik memiliki rentang yang berbeda jauh (misalnya `carat` bernilai `0.2 - 5.0`, sedangkan `price` bernilai `300 - 18000`).
  * `StandardScaler` mengonversi angka-angka tersebut agar memiliki rata-rata $\mu = 0$ dan standar deviasi $\sigma = 1$.
  * Ini membantu algoritma Machine Learning memproses data lebih stabil dan cepat.

---

### 6. Penanganan Data Duplikat
* **Fungsi Kode:** `train.duplicated().sum()`, `train.drop_duplicates()`
* **Penjelasan:**
  * Mendeteksi dan menghapus baris data yang identik/berulang.
  * Mencegah model mengalami *overfitting* akibat mempelajari data yang sama berulang kali.

---

### 7. Conversi Teks ke Angka (*One-Hot Encoding*)
* **Fungsi Kode:** `pd.get_dummies(train, columns=['cut', 'color', 'clarity'])`
* **Penjelasan:**
  * Algoritma Machine Learning hanya dapat memproses data berupa angka.
  * `pd.get_dummies()` mengonversi kolom kategorikal/teks (seperti kualitas pemotongan, warna, dan kejernihan) menjadi kolom biner berisikan nilai `0` atau `1`.
