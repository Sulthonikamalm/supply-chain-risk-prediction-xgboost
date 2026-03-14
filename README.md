<div align="center">

# Prediksi Risiko Keterlambatan Pengiriman

### Machine Learning untuk Supply Chain: Deteksi Dini Paket Terlambat

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.8426-success?style=flat-square)](/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)](/)

**Memprediksi keterlambatan pengiriman saat order masuk, bukan setelah terlambat**

[Masalah](#masalah-yang-diselesaikan) • [Solusi](#solusi-yang-ditawarkan) • [Proses](#apa-yang-dikerjakan) • [Kendala](#problem-yang-dihadapi) • [Hasil](#hasil-akhir)

</div>

---

## Masalah yang Diselesaikan

### Konteks Bisnis

Perusahaan e-commerce menangani ribuan order setiap hari. Dari dataset DataCo Global Supply Chain yang berisi 180,519 transaksi, **54.8% paket terlambat sampai ke pelanggan**.

Dampaknya:
- Komplain pelanggan meningkat
- Biaya kompensasi membengkak  
- Rating dan reputasi turun
- Operasional tidak efisien

### Pertanyaan Utama

**Bisakah kita tahu lebih awal kalau sebuah paket akan terlambat?**

Bukan setelah terlambat. Bukan saat sudah di jalan. Tapi **saat order baru masuk** - sebelum barang dikemas, sebelum kurir dijadwalkan.

### Kenapa Ini Penting?

Kalau bisa diprediksi sejak awal, tim operasional bisa:

| Aksi | Manfaat |
|------|---------|
| Prioritaskan order berisiko tinggi | Proses lebih cepat, hindari keterlambatan |
| Pilih carrier alternatif | Gunakan kurir lebih reliable untuk rute tertentu |
| Komunikasi proaktif ke pelanggan | Set ekspektasi realistis sejak awal |
| Alokasi resource lebih baik | Fokus effort di order yang memang butuh |

---

## Solusi yang Ditawarkan

### Model Machine Learning dengan Constraint Realistis

Kebanyakan model di Kaggle mencapai akurasi 95%+ dengan menggunakan fitur seperti:
- `shipping date` (tanggal pengiriman aktual)
- `Days for shipping (real)` (durasi pengiriman aktual)
- `Delivery Status` (status pengiriman)

**Masalahnya:** Fitur-fitur ini hanya tersedia **setelah barang dikirim**. Tidak bisa dipakai untuk prediksi saat order masuk.

### Pendekatan Proyek Ini

✅ **Hanya gunakan fitur yang tersedia saat order placement**
- Informasi produk (kategori, harga, berat)
- Data pelanggan (region, segmen, metode pembayaran)
- Jadwal pengiriman yang direncanakan
- Moda transportasi yang dipilih

✅ **F1-Score 0.84** - Lebih rendah dari 0.95, tapi **bisa dijalankan di production**

✅ **Model yang explainable** - Bisa jelaskan ke stakeholder kenapa prediksi A atau B

---

## Apa yang Dikerjakan

Ini bukan sekadar training model lalu selesai. Proyek ini dikerjakan secara iteratif dengan eksperimen sistematis.

### 1. Baseline: 10 Model Berbeda

Mulai dengan 10 algoritma sebagai patokan awal:
- Logistic Regression
- Random Forest  
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- Extra Trees
- AdaBoost
- Histogram Gradient Boosting
- SVM

**Hasil awal:** F1-Score rata-rata 0.70

### 2. Target Encoding untuk Fitur Kategorikal

**Kenapa?** Label encoding (0, 1, 2...) tidak membawa makna. Angka 2 tidak lebih besar dari 1 untuk kategori seperti "Jakarta" vs "Surabaya".

**Solusi:** Target encoding - ganti kategori dengan rata-rata risiko historis per grup.

**Implementasi:** K-Fold Leave-One-Out untuk mencegah overfitting.

**Hasil:** F1-Score naik ke 0.76

### 3. Hyperparameter Tuning dengan Optuna

**Kenapa Optuna?**
- Grid search terlalu lambat untuk 10 model
- Random search tidak belajar dari trial sebelumnya
- Optuna pakai Tree-structured Parzen Estimator yang adaptif

**Konfigurasi:** 3-fold CV, 15 trials per model, 20% data subset

**Waktu:** ~25 menit (vs estimasi 4-8 jam dengan grid search)

### 4. Custom Loss Functions

Implementasi 3 loss function dari paper akademis:

| Loss Function | Paper | Hasil |
|---------------|-------|-------|
| Focal Loss | Lin et al. (2017) | Di bawah baseline |
| Asymmetric Loss | Ben-Baruch et al. (2021) | Di bawah baseline |
| Adaptive Weighted Error | arxiv 2407.14381 | Di bawah baseline |

**Kesimpulan:** Custom loss dirancang untuk extreme imbalance (1:1000). DataCo hampir balanced (1:1.2), jadi XGBoost standar lebih baik.

### 5. Ensemble Methods

Stacking dengan 2 arsitektur berbeda:
- Meta-learner: Logistic Regression
- Meta-learner: XGBoost

**Hasil:** XGBoost sebagai meta-learner memberikan performa lebih baik.

### 6. Probability Calibration

Isotonic regression untuk memperbaiki distribusi probabilitas.

**Trade-off:**
- F1-Score naik: 0.8375 → 0.8426 ✅
- ECE (calibration error) naik: 0.097 → 0.1048 ❌

**Keputusan:** Pakai model calibrated karena F1 lebih prioritas untuk use case ini.

### 7. Threshold Optimization

Default threshold 0.5 hampir tidak pernah optimal.

| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 0.50 | 0.84 | 0.81 | 0.8375 |
| **0.33** | **0.81** | **0.87** | **0.8426** |

**Impact:** +3-5% F1 tanpa perlu training ulang.

### 8. Explainability & Business Metrics

- SHAP values untuk interpretasi model
- Dynamic threshold per segmen pengiriman
- Kalkulasi ROI dan Expected Monetary Value
- Cost-sensitive decision matrix

---

## Problem yang Dihadapi

Ini bagian paling penting yang jarang ditulis orang. Dokumentasi kendala dan solusinya.


### Problem #1: F1-Score Stuck di 0.70 Selama Berjam-jam

**Situasi:** Pipeline sudah lengkap, preprocessing jalan, tuning jalan, semua model running tanpa error. Tapi F1-Score tidak bergerak dari 0.70.

**Investigasi:** Cek ulang preprocessing, feature engineering, hyperparameter. Semua terlihat benar.

**Root Cause:** Satu baris kode yang salah:

```python
# ❌ SALAH - Index tidak sejajar
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

Setelah target encoding, index DataFrame bergeser. Tanpa `index=X.index`, fitur dan label tidak sejajar. Model belajar dari pasangan yang salah.

**Kenapa Berbahaya:** Tidak ada error, tidak ada warning. Hanya performa yang aneh.

**Solusi:**

```python
# ✅ BENAR - Preserve index
X_scaled = pd.DataFrame(
    scaler.fit_transform(X), 
    columns=X.columns, 
    index=X.index
)
```

**Impact:** F1-Score langsung naik ke 0.76 setelah fix.

**Lesson Learned:** Selalu preserve index saat transform DataFrame, terutama setelah operasi yang mengubah urutan baris.

---

### Problem #2: Custom Loss Function Tidak Bekerja

**Situasi:** Implementasi Focal Loss, Asymmetric Loss, dan Adaptive Weighted Error berdasarkan paper. Matematikanya sudah dibuktikan, kode sudah dioptimasi dengan Optuna.

**Hasil:** Semua custom loss di bawah XGBoost standar.

**Analisis:** 

| Aspek | Custom Loss Dirancang Untuk | DataCo Reality |
|-------|------------------------------|----------------|
| Imbalance Ratio | 1:100 hingga 1:1000 | 1:1.2 (hampir balanced) |
| Use Case | Object detection, medical diagnosis | Supply chain prediction |
| Complexity | High (focal modulation, asymmetric weighting) | Adds noise for balanced data |

**Kenapa Focal Loss Gagal:**
- Dirancang untuk kasus seperti deteksi objek di foto satelit
- Kelas minoritas sangat jarang (1 objek di 1000 pixel)
- DataCo: 54.8% late vs 45.2% on-time - hampir balanced

**Keputusan:** Tetap sertakan di notebook sebagai dokumentasi eksperimen, tapi tidak jadi model utama.

**Lesson Learned:** Teknik advanced tidak selalu lebih baik. Pahami asumsi di balik algoritma sebelum implementasi.

---


### Problem #3: GPU Crash Saat Stacking Ensemble

**Situasi:** `StackingClassifier` dengan `n_jobs=-1` untuk paralel processing. XGBoost dan CatBoost dikonfigurasi pakai GPU.

**Error:** Process crash tanpa error message yang jelas.

**Root Cause:** 
- `n_jobs=-1` spawn multiple processes secara paralel
- Setiap process coba akses GPU bersamaan
- XGBoost dan CatBoost tidak bisa share single GPU dari process berbeda

**Solusi yang Dicoba:**

1. **Attempt 1:** Buat instance CPU untuk setiap model
   - Result: CatBoost error karena parameter tidak bisa diubah setelah fit

2. **Attempt 2:** Gunakan `n_jobs=1` (sequential execution)
   - Result: ✅ Berhasil - GPU dipakai bergantian, tidak rebutan

**Implementasi:**

```python
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=XGBClassifier(tree_method='gpu_hist'),
    n_jobs=1  # Sequential, not parallel
)
```

**Trade-off:** Lebih lambat (sequential vs parallel), tapi stabil dan tidak crash.

**Lesson Learned:** Parallelisme tidak selalu lebih baik. GPU resource sharing butuh koordinasi yang tidak otomatis.

---

### Problem #4: Hyperparameter Tuning Terlalu Lama

**Situasi:** Optuna tuning berjalan 20+ menit tanpa output apapun.

**Konfigurasi Awal:**
- 5-fold cross-validation
- 30 trials per model
- 40% data (72,000 rows)
- 10 models

**Estimasi Waktu:**
```
72,000 rows × 5 folds × 30 trials × 10 models = 10,800,000 fits
Gradient Boosting: ~25 detik per fit
Total: 4-8 jam
```

**Analisis:** Terlalu banyak kombinasi untuk eksperimen iteratif.

**Optimasi:**

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| CV Folds | 5 | 3 | 3-fold cukup untuk dataset besar |
| Trials | 30 | 15 | Diminishing returns setelah 15 trials |
| Data Subset | 40% | 20% | 36K rows masih representatif |

**Hasil:** Waktu turun dari 4-8 jam ke **25 menit** tanpa loss kualitas signifikan.

**Lesson Learned:** Untuk dataset besar (>100K rows), subset 20% sudah cukup representatif untuk tuning.

---


### Problem #5: scale_pos_weight yang Kontraintuitif

**Situasi:** XGBoost punya parameter `scale_pos_weight` untuk handle imbalanced data.

**Formula Umum:**
```python
scale_pos_weight = negative_samples / positive_samples
```

**Untuk DataCo:**
```python
negative = 81,636  # On-time deliveries
positive = 98,883  # Late deliveries
scale_pos_weight = 81,636 / 98,883 = 0.82
```

**Problem:** Nilai 0.82 (< 1) justru **menurunkan** bobot kelas positif, bukan menaikkan.

**Kenapa Ini Terjadi:**
- Formula dirancang untuk kasus minoritas class adalah positive
- Di DataCo, late delivery (positive) justru mayoritas (54.8%)
- Formula otomatis jadi kontraproduktif

**Solusi:** Manual tuning dengan eksperimen:

| scale_pos_weight | F1-Score | Keterangan |
|------------------|----------|------------|
| 0.82 (auto) | 0.8156 | Dari formula |
| 1.0 (balanced) | 0.8289 | Neutral weight |
| 1.5 | 0.8351 | Better |
| **1.85** | **0.8375** | Optimal |
| 2.0 | 0.8342 | Mulai turun |

**Lesson Learned:** Jangan blind trust formula. Untuk near-balanced dataset, manual tuning lebih reliable.

---

### Problem #6: Kalibrasi Malah Memperburuk ECE

**Situasi:** Isotonic regression dipasang untuk improve probability calibration.

**Hasil:**

| Metric | Raw Model | Calibrated Model |
|--------|-----------|------------------|
| F1-Score | 0.8375 | 0.8426 ✅ |
| ECE (Expected Calibration Error) | 0.097 | 0.1048 ❌ |

**Analisis:**
- Isotonic regression butuh sebagian data training sebagai calibration set
- Mengurangi data available untuk training inti
- Untuk dataset dengan weak signal, kehilangan data lebih merugikan

**Keputusan:** Pakai calibrated model karena:
1. F1-Score lebih prioritas untuk use case ini
2. Probabilitas dipakai untuk ranking, bukan absolute value
3. Improvement 0.8375 → 0.8426 signifikan untuk production

**Lesson Learned:** Calibration bukan always win. Ada trade-off antara classification performance vs probability accuracy.

---

## Hasil Akhir

### Model Performance

| Model | F1-Score | Precision | Recall | Keterangan |
|-------|----------|-----------|--------|------------|
| Baseline (avg 10 models) | 0.70 | - | - | Setelah index alignment fix |
| + Target Encoding | 0.76 | - | - | K-Fold LOO, 5 fitur kategorikal |
| + Optuna Tuning | 0.81 | - | - | 15 trials, 3-fold CV |
| XGBoost V4 | 0.8375 | 0.84 | 0.84 | scale_pos_weight=1.85 |
| **XGBoost V4 Calibrated** | **0.8426** | **0.84** | **0.84** | Isotonic regression |

### Kenapa Tidak 95%+ Seperti Notebook Lain?

**Data Leakage Detection:** Notebook dengan akurasi 95%+ menggunakan fitur yang hanya ada setelah shipment:


| Fitur Leaked | Kenapa Ini Leakage | Impact |
|--------------|-------------------|--------|
| `shipping date (DateOrders)` | Hanya diketahui setelah barang dikirim | Korelasi langsung dengan target |
| `Days for shipping (real)` | Durasi pengiriman aktual | Secara matematis menentukan target |
| `Delivery Status` | Berisi label "Late delivery" | Bentuk lain dari target variable |

**Bukti:** Semua notebook 95%+ menunjukkan train accuracy ~99% dengan gap train-test besar → overfitting/memorization.

**Proyek Ini:** F1-Score 0.84 adalah **batas realistis** dengan fitur pre-shipment. 60% penyebab keterlambatan adalah faktor eksternal yang tidak ada di dataset (cuaca, kapasitas carrier, customs, traffic).

### Top 5 Fitur Paling Penting (SHAP)

1. **Days for shipment (scheduled)** - Jendela waktu pengiriman yang direncanakan
2. **Order Region** - Destinasi geografis
3. **Shipping Mode** - Metode transportasi (air, sea, road)
4. **Product Category** - Jenis produk
5. **Customer Segment** - B2B vs B2C

### Business Impact

**Threshold Optimization:**

| Threshold | Precision | Recall | F1 | Use Case |
|-----------|-----------|--------|-----|----------|
| 0.50 | 0.84 | 0.81 | 0.8375 | Balanced |
| **0.33** | **0.81** | **0.87** | **0.8426** | Prioritize recall (catch more late orders) |
| 0.60 | 0.87 | 0.76 | 0.8112 | Prioritize precision (reduce false alarms) |

**ROI Calculation:**
- Cost of late delivery: kompensasi, reputasi, customer churn
- Cost of false positive: resource allocation yang tidak perlu
- Optimal threshold tergantung cost matrix bisnis

---

## Dataset

**DataCo Global Supply Chain for Big Data Analysis**

| Metric | Value |
|--------|-------|
| Total Transaksi | 180,519 |
| Late Delivery Rate | 54.8% |
| Fitur yang Digunakan | 24 (pre-shipment only) |
| Periode | 2015-2018 |
| Coverage | 5 negara |

**Sumber:**
- Kaggle: [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- Mendeley: [Original Dataset](https://data.mendeley.com/datasets/8gx2fvg2k6/5)
- Paper: Constante, F., Silva, F., & Pereira, A. (2019). DOI: [10.17632/8gx2fvg2k6.5](https://doi.org/10.17632/8gx2fvg2k6.5)

---

## Tech Stack

**Core Libraries:**
```
xgboost==1.7.0          # Model utama
lightgbm==3.3.5         # Baseline & ensemble
catboost==1.2           # Baseline & ensemble
optuna==3.1.0           # Hyperparameter optimization
scikit-learn==1.2.2     # Preprocessing & metrics
shap==0.41.0            # Model explainability
pandas==1.5.3           # Data manipulation
numpy==1.23.5           # Numerical computing
matplotlib==3.7.1       # Visualization
seaborn==0.12.2         # Statistical plots
```

**Environment:**
- Python 3.8+
- Google Colab with T4 GPU
- Runtime: ~55-65 menit

---


## Cara Menjalankan

### Prerequisites

- Google Account (untuk Colab)
- Kaggle Account (untuk download dataset)

### Langkah-langkah

1. **Download Dataset**
   - Buka [Kaggle Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
   - Download `DataCoSupplyChainDataset.csv`

2. **Upload ke Google Drive**
   ```
   /MyDrive/DATASET/DataCoSupplyChainDataset.csv
   ```

3. **Buka Notebook di Colab**
   - Upload `supply_chain_FINAL_COMPLETE (4).ipynb` ke Colab
   - Atau buka langsung dari GitHub

4. **Set Runtime**
   - Runtime → Change runtime type → GPU → T4

5. **Run All Cells**
   - Runtime → Run all
   - Estimasi waktu: 55-65 menit

### Struktur File

```
supply-chain-delivery-risk-ml/
├── supply_chain_FINAL_COMPLETE (4).ipynb    # Notebook utama
├── DATASET.zip                               # Raw data (download terpisah)
├── model_output_final.zip                    # Trained models & results
└── README.md                                 # Dokumentasi ini
```

---

## Referensi Paper

### Papers yang Diimplementasikan

| Paper | Aplikasi | Link |
|-------|----------|------|
| Chen & Guestrin (2016) | XGBoost model utama | [arxiv:1603.02754](https://arxiv.org/abs/1603.02754) |
| Ke et al. (2017) | LightGBM baseline | [NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) |
| Lin et al. (2017) | Focal Loss | [arxiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| Akiba et al. (2019) | Optuna tuning | [arxiv:1907.10902](https://arxiv.org/abs/1907.10902) |
| Ben-Baruch et al. (2021) | Asymmetric Loss | [arxiv:2009.14119](https://arxiv.org/abs/2009.14119) |
| arxiv 2407.14381 (2024) | Adaptive Weighted Error | [arxiv:2407.14381](https://arxiv.org/abs/2407.14381) |
| Micci-Barreca (2001) | Target encoding | [ACM SIGKDD](https://doi.org/10.1145/507533.507538) |
| Lundberg & Lee (2017) | SHAP explainability | [arxiv:1705.07874](https://arxiv.org/abs/1705.07874) |
| Platt (1999) | Isotonic calibration | [CiteSeer](https://citeseerx.ist.psu.edu/doc/10.1.1.41.1639) |
| He & Garcia (2009) | Cost-sensitive learning | [IEEE TKDE](https://doi.org/10.1109/TKDE.2008.239) |

### Additional Resources

- Provost & Fawcett (2013). Data Science for Business. [O'Reilly](https://www.oreilly.com/library/view/data-science-for/9781449374273/)
- Sculley et al. (2015). Hidden Technical Debt in ML Systems. [NeurIPS](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- Hernandez-Orallo et al. (2022). Calibration in ML. [Springer](https://link.springer.com/article/10.1007/s10994-021-06026-4)

---

## Key Takeaways

### Untuk Data Scientists

1. **Index alignment matters** - Selalu preserve index saat transform DataFrame
2. **Advanced ≠ Better** - Teknik sophisticated tidak selalu cocok untuk semua kasus
3. **Understand assumptions** - Pahami asumsi di balik algoritma sebelum implementasi
4. **Manual tuning sometimes wins** - Formula otomatis tidak selalu optimal
5. **Trade-offs are real** - Calibration, parallelism, data size - semua ada trade-off

### Untuk ML Engineers

1. **Production viability > Benchmark scores** - Model 0.84 yang bisa jalan > 0.95 yang tidak bisa
2. **Feature availability is critical** - Pastikan fitur benar-benar ada saat inference
3. **GPU resource management** - Parallelism butuh koordinasi, sequential kadang lebih stabil
4. **Hyperparameter tuning efficiency** - Subset data besar sudah cukup representatif

### Untuk Business Stakeholders

1. **Realistic expectations** - 60% faktor keterlambatan di luar dataset
2. **Threshold is business decision** - Precision vs recall tergantung cost matrix
3. **Explainability matters** - SHAP values bantu komunikasi ke non-technical team
4. **ROI calculation** - Translate F1-Score ke business metrics

---

## License

MIT License - Lihat [LICENSE](LICENSE) untuk detail

---

<div align="center">

[⬆ Kembali ke Atas](#prediksi-risiko-keterlambatan-pengiriman)

</div>
