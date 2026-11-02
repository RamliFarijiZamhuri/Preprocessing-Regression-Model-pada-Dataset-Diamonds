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

  # Prediksi Harga Berlian: Preprocessing & Regression Model

Notebook: `Preprocessing-dan-Regression-Model-pada-Dataset-Diamonds.ipynb`

Proyek ini membangun model regresi untuk memprediksi **harga berlian (`price`)** dari dataset Diamonds. Tiga model dibandingkan: **Random Forest**, **Decision Tree**, dan **KNN Regressor**, dengan metrik evaluasi **MAE**, **RMSE**, dan **R²**.

---

## 1. Hasil Evaluasi

| Model         | MAE      | RMSE     | R² Score |
|---------------|----------|----------|----------|
| Random Forest | 0.072904 | 0.134204 | 0.982307 |
| Decision Tree | 0.099421 | 0.184361 | 0.966610 |
| KNN Regressor | 0.112351 | 0.197151 | 0.961816 |

> **Catatan penting:** kolom `price` ikut distandardisasi (`StandardScaler`), sehingga **MAE dan RMSE di atas dalam satuan standar deviasi harga, bukan dolar**. MAE 0.0729 berarti rata-rata prediksi meleset sekitar 0,07 standar deviasi harga.

---

## 2. Alur Kode

### 2.1 Preprocessing

| Tahap | Apa yang dilakukan | Tujuan |
|-------|--------------------|--------|
| Load data | `pd.read_csv("train_Diamonds.csv")` | Membaca dataset (cell pertama menamai variabelnya `test`, padahal file yang dibaca sama dengan `train`) |
| Missing values | Kolom dengan missing < 1000 diisi (median untuk numerik, modus untuk kategorik); kolom dengan missing ≥ 1000 dibuang | Menghindari error model dan menjaga kualitas data |
| Outlier | Baris di luar [Q1 − 1,5·IQR, Q3 + 1,5·IQR] pada kolom numerik dihapus | Mengurangi pengaruh nilai ekstrem |
| Standardisasi | `StandardScaler` pada kolom numerik (mean 0, std 1) | Menyamakan skala fitur, penting untuk KNN |
| Duplikat | `drop_duplicates()` | Menghindari data yang sama muncul di train dan test |
| Encoding | One-Hot (`df_one_hot`) dan Label Encoding (`df_lencoder`) | Mengubah fitur kategorik (cut, color, clarity) menjadi angka |

Label Encoding dipakai untuk EDA dan korelasi. Model memakai hasil **One-Hot**.

### 2.2 Modeling

1. Pisahkan fitur `X` dan target `y = price`.
2. Bagi data 80% latih dan 20% uji (`random_state=42`).
3. Latih tiga model:
   - `DecisionTreeRegressor(random_state=42)`
   - `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`
   - `KNeighborsRegressor(n_neighbors=5)`
4. Hitung MAE, RMSE (`sqrt(MSE)`), dan R² pada data uji.
5. Visualisasi: scatter plot aktual vs prediksi (Random Forest) dan bar chart R² tiap model.

---

## 3. Penjelasan Metrik

### MAE (Mean Absolute Error)

```
MAE = (1/n) · Σ |yᵢ − ŷᵢ|
```

Rata-rata selisih absolut antara nilai asli dan prediksi. Mudah ditafsirkan dan tidak terlalu sensitif terhadap outlier. **Semakin kecil semakin baik** (ideal = 0).

### RMSE (Root Mean Squared Error)

```
RMSE = √[ (1/n) · Σ (yᵢ − ŷᵢ)² ]
```

Selisih dikuadratkan sebelum dirata-rata, sehingga **kesalahan besar dihukum lebih berat**. Selalu ≥ MAE. **Semakin kecil semakin baik** (ideal = 0).

Rasio RMSE/MAE pada ketiga model ≈ 1,75–1,85. Artinya ada sebagian prediksi yang meleset cukup jauh (kemungkinan pada berlian berharga tinggi), bukan hanya error kecil yang merata.

### R² (Koefisien Determinasi)

```
R² = 1 − SS_res / SS_tot
```

Proporsi variasi `price` yang berhasil dijelaskan model. Ideal = 1; nilai 0 berarti setara dengan selalu menebak rata-rata.

### Hubungan ketiganya

Karena target distandardisasi (varians ≈ 1):

```
R² ≈ 1 − RMSE²
Random Forest: 1 − 0,1342² ≈ 0,982   ✔ cocok dengan tabel
```

Selisih R² antar model terlihat kecil, tetapi bagian variasi yang **tidak** terjelaskan sangat berbeda:

| Model | Tidak terjelaskan (1 − R²) |
|-------|---------------------------|
| Random Forest | ≈ 1,8% |
| Decision Tree | ≈ 3,3% |
| KNN | ≈ 3,8% |

Error variance KNN lebih dari dua kali lipat Random Forest.

---

## 4. Perbandingan dengan Referensi

MAE dan RMSE bergantung skala, jadi tidak punya ambang universal. Untuk membandingkannya dipakai **RSR** (RMSE dibagi standar deviasi data) dari Moriasi et al. (2007). Karena target sudah distandardisasi, RSR ≈ RMSE ≈ √(1 − R²).

### Ambang R²

| Sumber | Kategori "substantial" |
|--------|------------------------|
| Hair et al. (2011) | R² ≥ 0,75 (moderat 0,50–0,75; lemah 0,25–0,50) |
| Chin (1998) | R² ≥ 0,67 (moderat 0,33–0,67; lemah 0,19–0,33) |
| Cohen (1988), regresi linear | R² ≥ 0,26 |

### Ambang RSR dan NSE (Moriasi et al., 2007)

| Rating | RSR | NSE (≈ R² pada data uji) |
|--------|-----|--------------------------|
| Very good | 0,00 – 0,50 | 0,75 – 1,00 |
| Good | 0,50 – 0,60 | 0,65 – 0,75 |_
| Satisfactory | 0,60 – 0,70 | 0,50 – 0,65 |
| Unsatisfactory | > 0,70 | < 0,50 |

### Posisi ketiga model

| Model | R² | RSR ≈ √(1−R²) | Rating |
|-------|----|---------------|--------|
| Random Forest | 0,982 | ≈ 0,133 | Very good |
| Decision Tree | 0,967 | ≈ 0,183 | Very good |
| KNN | 0,962 | ≈ 0,195 | Very good |

**Kesimpulan:** ketiga model jauh lebih dekat ke nilai ideal daripada ke batas "kurang baik". Urutan kedekatan ke ideal: **Random Forest > Decision Tree > KNN**.

> **Peringatan:** ambang di atas berasal dari bidang lain (hidrologi dan PLS-SEM/riset pemasaran). Gunakan sebagai pembanding kasar, bukan standar mutlak untuk harga berlian.

---

## 5. Keterbatasan & Perbaikan yang Disarankan

| Masalah | Dampak | Perbaikan |
|---------|--------|-----------|
| Outlier dihapus dari semua kolom numerik, termasuk `price` | Berlian mahal tidak ikut diuji, hasil terlalu optimis | Hapus outlier hanya pada fitur, atau evaluasi juga pada data tanpa filter |
| Standardisasi dilakukan **sebelum** split | Data leakage kecil | `fit` scaler hanya pada data latih, lalu `transform` data uji (gunakan `Pipeline`) |
| Evaluasi hanya satu kali split | Hasil bisa kebetulan | Gunakan k-fold cross-validation |
| Metrik dalam satuan terstandardisasi | Sulit dijelaskan ke non-teknis | Kembalikan ke dolar: `mae_dolar = mae * scaler.scale_[idx_price]` |
| Cell terakhir memakai `GBR` yang tidak terdefinisi | Error saat menyimpan model | Ganti dengan `best_model` |
| Tidak ada perbandingan skor train vs test | Overfitting (terutama Decision Tree) tidak terdeteksi | Hitung metrik pada data latih juga |
| Tidak ada baseline | Sulit menilai seberapa bagus model | Bandingkan dengan `DummyRegressor` dan regresi linear |

---

## 6. Referensi

- Fajar, skripsi UIN Sunan Gunung Djati Bandung, Random Forest vs KNN vs AdaBoost, abstrak: https://digilib.uinsgd.ac.id/73390/2/2_abstrak.pdf