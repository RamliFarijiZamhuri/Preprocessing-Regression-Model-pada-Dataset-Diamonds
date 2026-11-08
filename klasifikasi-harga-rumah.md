# Laporan Analisis dan Penentuan Metode Machine Learning: Prediksi Harga Rumah (Ames Housing)

Dokumen ini berisi penjelasan komprehensif mengenai penentuan jenis metode *Machine Learning*, rujukan teori, bedah alur kode pada file notebook [`latihan.ipynb`](file:///d:/Kuliah/Semester%205/Machine%20Learning/FIle-Koding/latihan.ipynb), analisis performa algoritma, serta catatan kritis untuk keperluan tugas/laporan praktikum.

---

## 1. Penentuan Paradigma dan Jenis Metode Machine Learning

### A. Paradigma: Supervised Learning (Pembelajaran Terawasi)
- **Alasan**: Dataset memiliki label target yang jelas, yaitu variabel **`SalePrice`** (harga jual rumah), serta sejumlah variabel prediktor/fitur independen ($X$) seperti luas tanah, jumlah kamar, kualitas material, dan tahun pembuatan.
- **Mekanisme**: Model mempelajari fungsi pemetaan matematis $f(X) \rightarrow y$ berdasarkan data historis yang telah memiliki pasangan nilai input dan output terverifikasi.
- **Bukan Unsupervised**: Karena analisis ini tidak ditujukan untuk pengelompokan tanpa label (*clustering*) atau reduksi dimensi tanpa target (*unsupervised dimensionality reduction*).

### B. Sub-Metode: Regresi (Bukan Klasifikasi)
- **Karakteristik Target**: Kolom `SalePrice` bertipe data **numerik kontinu** (nilai angka riil dalam rentang kontinu, misalnya \$140.000, \$181.500, \$250.000).
- **Perbedaan Mendasar**:
  - **Regresi**: Memprediksi kuantitas riil/kontinu ($y \in \mathbb{R}$). Kasus pada notebook ini murni kasus regresi.
  - **Klasifikasi**: Memprediksi label kategori/diskrit ($y \in \{C_1, C_2, \dots, C_k\}$), misalnya jika target dibagi menjadi kelas: `[Murah, Sedang, Mahal]`.
- **Metrik Evaluasi yang Dipakai**: Metrik khusus model regresi, yaitu:
  - **MAE** (*Mean Absolute Error*)
  - **MSE** (*Mean Squared Error*)
  - **$R^2$ Score** (*Coefficient of Determination*)

---

## 2. Landasan Teori dan Rujukan Akademik

1. **Definisi Supervised Learning**:
   > *Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.*  
   > Menyatakan bahwa *Supervised Learning* adalah proses induksi fungsi dari data pelatihan berlabel, di mana tujuannya adalah menemukan hipotesis yang mampu memprediksi output yang benar untuk contoh data baru.

2. **Perbedaan Regresi dan Klasifikasi**:
   > *Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer.*  
   > Menjelaskan bahwa jika variabel respon berupa kuantitatif/kontinu, permasalahan disebut **masalah regresi**; sedangkan jika variabel respon berupa kualitatif (faktor/kelas diskrit), disebut **masalah klasifikasi**.

3. **Metrik Evaluasi Model Regresi**:
   - **MAE**: Mengukur rata-rata magnitudo kesalahan prediksi tanpa memperhatikan arahnya:
     $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - **MSE**: Memberikan penalti kuadrat terhadap kesalahan yang lebih besar:
     $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - **$R^2$ Score**: Proporsi variansi variabel terikat yang dapat dijelaskan oleh variabel-variabel bebas:
     $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

---

## 3. Bedah Alur Kerja Kode (Machine Learning Pipeline)

Tahapan pada [`latihan.ipynb`](file:///d:/Kuliah/Semester%205/Machine%20Learning/FIle-Koding/latihan.ipynb) mencakup alur standar *Data Science Lifecycle*:

```
[1. Load Data] ➔ [2. Data Cleaning & Imputation] ➔ [3. Outlier Handling (IQR)]
                      │
                      ▼
[6. Modeling]  [5. Train-Test Split]  [4. Scaling & Encoding]
      │
      ▼
[7. Model Evaluation] ➔ [8. Model Serialization (.joblib / .pkl)]
```

### 1. Data Ingestion & Exploration (Cell 0 – 4)
- Mengimpor library: `pandas`, `seaborn`, `matplotlib`, dan `sklearn`.
- Memuat dataset: `train.csv` (1460 baris, 81 kolom) dan `test.csv` (1459 baris, 80 kolom).
- Analisis struktur: `train.info()` dan ringkasan statistik `train.describe(include="all")`.

### 2. Penanganan Nilai Hilang / Missing Values (Cell 5 – 10)
- **Pembersihan Kolom Kosong Masif**: Kolom dengan nilai kosong $\ge 1000$ baris (`PoolQC`, `MiscFeature`, `Alley`, `Fence`) dihapus menggunakan `train.drop(columns=over)`.
- **Imputasi Kolom Numerik**: Nilai yang hilang diisi dengan nilai **Median** untuk meminimalkan pengaruh pencilan (*robust to outliers*):
  ```python
  train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())
  ```
- **Imputasi Kolom Kategorikal**: Nilai yang hilang diisi dengan nilai yang paling sering muncul (**Modus**):
  ```python
  train[column] = train[column].fillna(train[column].mode()[0])
  ```

### 3. Penanganan Pencilan / Outlier Handling (Cell 11 – 15)
- Menggunakan aturan **Interquartile Range (IQR)**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Batas Bawah} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Batas Atas} = Q_3 + 1.5 \times \text{IQR}$$
- Baris data yang memiliki nilai di luar batas IQR pada fitur numerik difilter/dihapus dengan kondisi `.any(axis=1)`.

### 4. Transformasi Fitur (Scaling & Encoding) (Cell 16 – 24)
- **Standardisasi**: Menggunakan `StandardScaler` dari Scikit-Learn agar setiap fitur numerik memiliki rata-rata $\mu = 0$ dan deviasi standar $\sigma = 1$:
  $$z = \frac{x - \mu}{\sigma}$$
- **Encoding Variabel Kategorikal**: Mengonversi fitur teks/objek menjadi nilai integer menggunakan `LabelEncoder`.

### 5. Exploratory Data Analysis & Korelasi (Cell 25 – 29)
- Menghitung matriks korelasi Pearson (`df_lencoder.corr()`).
- Mengevaluasi hubungan variabel independen terhadap target `SalePrice` untuk memahami fitur mana yang paling dominan mempengaruhi harga.

### 6. Pembagian Data / Train-Test Split (Cell 30 – 32)
- Memisahkan matriks fitur prediktor ($X$) dan vektor target ($y = \text{SalePrice}$).
- Membagi data menjadi **80% Data Latih** dan **20% Data Uji** dengan `random_state=1`:
  ```python
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  ```

### 7. Pelatihan Model (Cell 33)
Melatih 3 varian model regresi:
1. **Least Angle Regression (LARS)**: `linear_model.Lars(n_nonzero_coefs=1)`
2. **Linear Regression (Ordinary Least Squares)**: `LinearRegression()`
3. **Gradient Boosting Regressor**: `GradientBoostingRegressor(random_state=184)`

### 8. Evaluasi Performa Model (Cell 34 – 36)
Berikut adalah hasil evaluasi pada data uji (`x_test`):

| Algoritma | MAE | MSE | $R^2$ Score | Keterangan |
| :--- | :---: | :---: | :---: | :--- |
| **LARS** | $0.720054$ | $0.817625$ | $-0.018517$ | *Underfitting* akibat pembatasan koefisien |
| **Linear Regression** | $0.237356$ | $0.095778$ | $0.880690$ | Performa linier sangat baik |
| **Gradient Boosting Regressor** | **$0.217787$** | **$0.089663$** | **$0.888307$** | **Model Terbaik (Juara)** |

### 9. Penyimpanan Model / Model Persistence (Cell 37 – 38)
Model terbaik (`GBR`) disimpan ke dalam bentuk biner agar siap digunakan dalam lingkungan produksi tanpa perlu melatih ulang:
- `joblib.dump(GBR, 'gbr_model.joblib')`
- `pickle.dump(GBR, open('gbr_model.pkl', 'wb'))`

---

## 4. Analisis Teknis Mendalam & Catatan Kritis untuk Ujian/Laporan

Saat mempresentasikan notebook ini di hadapan dosen penguji atau asisten lab, perhatikan 3 catatan teknis krusial berikut:

1. **Mengapa Nilai MAE dan MSE Bernilai Pecahan Kecil ($\approx 0.21$)?**
   - **Fakta**: Nilai asli harga rumah adalah ratusan ribu dollar.
   - **Penyebab**: Di Cell 16, kolom target `SalePrice` ikut terstandarisasi oleh `StandardScaler` bersama fitur numerik lainnya.
   - **Dampak**: Prediksi model berada dalam skala satuan standar ($Z$-score). Untuk mengembalikan ke satuan dollar asli, hasil prediksi harus dikalikan deviasi standar ($\sigma$) lalu ditambah rata-rata ($\mu$) dari data asli (`inverse_transform`).

2. **Penyebab Kegagalan Model LARS ($R^2 < 0$):**
   - Kode menggunakan `n_nonzero_coefs=1`. Ini membatasi algoritma hanya menggunakan **1 variabel prediktor tunggal** dan membuang puluhan variabel lainnya. Akibatnya, model mengalami bias tinggi (*extreme underfitting*).

3. **Keunggulan Gradient Boosting Regressor:**
   - Hubungan antara atribut rumah (misal: luas bangunan, kualitas bahan, lokasi) dan harga sering kali bersifat non-linier dan memiliki interaksi kompleks.
   - GBR menggabungkan banyak *decision trees* secara berurutan (*boosting*), di mana setiap pohon baru memperbaiki residual kesalahan pohon sebelumnya, sehingga menghasilkan akurasi tertinggi ($R^2 = 88.83\%$).

---

## 5. Ringkasan Jawaban Singkat untuk Dosen

- **Metode yang digunakan**: **Supervised Learning - Regresi**.
- **Alasan**: Memprediksi variabel dependen bertipe data kontinu/numerik (`SalePrice`) berdasarkan sekumpulan variabel independen berlabel.
- **Algoritma yang diuji**: *Least Angle Regression (LARS)*, *Linear Regression*, dan *Gradient Boosting Regressor*.
- **Algoritma terpilih**: **Gradient Boosting Regressor** dengan skor $R^2 \approx 0.888$ dan $MAE \approx 0.217$.

---

## 6. Daftar Pustaka dan Referensi Lengkap (Format Sitasi Akademik & Tautan Sumber)

Berikut adalah daftar referensi lengkap yang dapat langsung disalin ke Bab **Daftar Pustaka** pada laporan kuliah atau makalah:

### A. Buku Teks Utama (Textbooks)
1. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.  
   🌐 Tautan Resmi: [AIMA Official Website (UC Berkeley)](http://aima.cs.berkeley.edu/)  
   *(Rujukan utama untuk konsep dasar Machine Learning, Supervised Learning, dan fungsi pemetaan input-output).*
2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer New York.  
   🌐 Tautan Unduh Resmi (Stanford University): [The Elements of Statistical Learning (Free PDF)](https://hastie.su.domains/ElemStatLearn/)  
   *(Rujukan standar untuk klasifikasi vs regresi, model linier OLS, dan metode ensemble boosting).*
3. **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.  
   🌐 Repositori Resmi GitHub: [ageron/handson-ml3](https://github.com/ageron/handson-ml3)  
   *(Panduan praktis implementasi pipeline machine learning, data cleaning, train-test split, dan evaluasi model regresi).*

### B. Paper Ilmiah Algoritma & Dataset Asli
4. **Dataset Rumah (Ames Housing Dataset):**  
   De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3).  
   🌐 Tautan Paper Resmi (DOI): [Taylor & Francis Online - Paper Ames Housing](https://doi.org/10.1080/10691898.2011.11889627)  
   🌐 Halaman Kompetisi Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
   *(Paper asli pembuat dataset `train.csv` dan `test.csv` yang Anda gunakan di notebook ini, yang diterbitkan khusus sebagai proyek regresi).*
5. **Algoritma Gradient Boosting:**  
   Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics, 29(5), 1189–1232.  
   🌐 Tautan Paper Asli: [Project Euclid - Greedy Function Approximation (PDF)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
6. **Algoritma LARS (Least Angle Regression):**  
   Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). *Least angle regression*. Annals of Statistics, 32(2), 407–499.  
   🌐 Tautan Paper Asli: [Project Euclid - Least Angle Regression (PDF)](https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-2/Least-angle-regression/10.1214/009053604000000067.full)

### C. Dokumentasi Resmi Scikit-Learn (Library Documentation)
7. **Scikit-Learn Regression Guide:**  
   Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825-2830.  
   - [Scikit-Learn: Supervised Learning Overview](https://scikit-learn.org/stable/supervised_learning.html)
   - [Scikit-Learn: Linear Regression & LARS Module](https://scikit-learn.org/stable/modules/linear_model.html)
   - [Scikit-Learn: Gradient Tree Boosting Documentation](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
   - [Scikit-Learn: Regression Metrics (MAE, MSE, $R^2$)](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
   - [Scikit-Learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
   - [Scikit-Learn: LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

---

### D. Tabel Ringkasan Tautan

| No | Sumber Referensi | Deskripsi | Tautan Langsung |
| :---: | :--- | :--- | :---: |
| 1 | **Paper Ames Housing** | Paper asli pembuat dataset `train.csv` | [Buka Paper (DOI)](https://doi.org/10.1080/10691898.2011.11889627) |
| 2 | **Kaggle Housing Competition** | Halaman kompetisi dataset Ames Housing | [Buka Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 3 | **Buku Hastie et al. (Stanford)** | Buku teks pembeda Regresi vs Klasifikasi | [Unduh PDF Gratis](https://hastie.su.domains/ElemStatLearn/) |
| 4 | **Buku Russell & Norvig** | Buku pegangan dasar Supervised Learning | [Buka Situs AIMA](http://aima.cs.berkeley.edu/) |
| 5 | **Paper Gradient Boosting (Friedman)** | Paper asli algoritma Gradient Boosting | [Buka Paper Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) |
| 6 | **Paper LARS (Efron et al.)** | Paper asli algoritma Least Angle Regression | [Buka Paper Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-2/Least-angle-regression/10.1214/009053604000000067.full) |
| 7 | **Scikit-Learn Docs** | Dokumentasi resmi fungsi regresi & metrik | [Buka Dokumentasi](https://scikit-learn.org/stable/supervised_learning.html) |

