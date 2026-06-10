# Dashboard Komparasi JST: Prediksi Kepuasan Pelanggan

Aplikasi web berbasis Streamlit untuk memprediksi tingkat kepuasan pelanggan dan membandingkan dua metode Jaringan Syaraf Tiruan:

1. TensorFlow/Keras Sequential
2. Scikit-Learn MLPClassifier

Dataset utama yang digunakan berasal dari Kaggle Customer Satisfaction 10K, tetapi aplikasi juga dapat membaca dataset CSV lain dari folder `data/`.

## Fitur Utama

- Upload CSV dan simpan otomatis ke folder `data/`.
- Membaca seluruh file CSV dari folder `data/`.
- Mode pilih satu dataset atau gabungkan beberapa CSV jika struktur kolom sama.
- Preview dataset, tipe data, missing value, jumlah baris, dan jumlah kolom.
- Deteksi target kepuasan otomatis seperti `Satisfaction Score`, `CSAT Score`, `Satisfaction`, atau `Rating`.
- Transformasi skor 1-5 menjadi tiga kelas: Tidak Puas, Netral, dan Puas.
- Preprocessing otomatis: hapus ID, tangani missing value, encode kategori, normalisasi numerik, dan konversi kolom waktu.
- Batas maksimal baris training agar aman untuk Docker/laptop.
- Evaluasi model dengan accuracy, precision, recall, F1-score, dan classification report.
- Grafik training accuracy/loss Keras, confusion matrix, dan heatmap korelasi numerik.
- Form prediksi pelanggan baru.
- Download hasil evaluasi CSV.
- Terjemahan tampilan offline melalui `local_translator.py`.

## Struktur File

```text
app.py
local_translator.py
requirements.txt
README.md
Dockerfile
.streamlit/config.toml
data/
```

## Cara Menjalankan Lokal

Gunakan Python 3.11 agar kompatibel dengan TensorFlow.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka aplikasi di:

```text
http://localhost:8501
```

## Cara Menjalankan dengan Docker

Build image:

```bash
docker build -t prediksi-kepuasan-jst .
```

Jalankan container:

```bash
docker run --rm -p 8501:8501 -v "%cd%/data:/app/data" prediksi-kepuasan-jst
```

Volume `data` membuat dataset yang diupload tetap tersimpan di folder lokal `data/`.

## Penggunaan Dataset

Ada dua cara menambahkan dataset:

1. Letakkan file `.csv` langsung ke folder `data/`.
2. Upload melalui aplikasi pada bagian **Upload Dataset Baru**.

Setelah upload, file akan disimpan ke folder `data/` dan otomatis muncul pada pilihan dataset aktif.

Jika ingin mengunggah dataset ke GitHub, pastikan file CSV yang dibutuhkan berada di folder `data/`.

## Modul Penerjemah Lokal

File `local_translator.py` digunakan untuk menerjemahkan sebagian tampilan ke Bahasa Indonesia secara offline. Data asli tetap dipakai untuk training; yang diterjemahkan hanya nama kolom, nilai kategori umum, label target, dan hasil prediksi.

Untuk menambah kata baru:

- `PHRASE_TRANSLATIONS`: frasa atau nama kolom lengkap.
- `VALUE_TRANSLATIONS`: nilai kategori seperti `Yes`, `No`, `Satisfied`, `Inbound`.
- `WORD_TRANSLATIONS`: kata kunci umum untuk fuzzy keyword mapping.

## Alur Preprocessing

1. Dataset dibaca dari folder `data/`.
2. Kolom target kepuasan dipilih atau dideteksi otomatis.
3. Kolom ID seperti `Customer ID`, `Unique id`, dan `Order_id` dihapus.
4. Kolom tanggal/jam diubah menjadi fitur numerik seperti bulan, hari, jam, dan hari dalam minggu.
5. Kolom teks bebas atau kategori yang terlalu unik tidak dipakai untuk menjaga stabilitas training.
6. Missing value numerik diisi median.
7. Missing value kategorikal diisi `Tidak Diketahui`.
8. Fitur kategorikal diubah dengan OneHotEncoder.
9. Fitur numerik dinormalisasi dengan StandardScaler.
10. Data dibagi 80% training dan 20% testing dengan `random_state=42`.

## Cara Membaca Hasil

**Accuracy** menunjukkan persentase prediksi benar dari seluruh data testing.

**Precision** menunjukkan ketepatan prediksi model pada kelas tertentu.

**Recall** menunjukkan kemampuan model menemukan data yang benar-benar termasuk kelas tertentu.

**F1-score** menunjukkan keseimbangan antara precision dan recall.

**Training accuracy/loss** membantu melihat apakah model Keras belajar stabil atau mulai overfitting.

**Confusion matrix** menunjukkan perbandingan kelas aktual dan kelas hasil prediksi.

**Heatmap korelasi** menunjukkan hubungan antarfitur numerik.

## Deploy ke Streamlit Community Cloud

Upload file berikut ke GitHub:

- `app.py`
- `local_translator.py`
- `requirements.txt`
- `README.md`
- `.streamlit/config.toml`
- folder `data/` jika dataset ingin ikut tersedia saat deploy

Saat deploy, gunakan:

```text
Python version: 3.11
Main file path: app.py
```

Jika aplikasi lambat saat training, kurangi `Epoch Keras`, `Max Iteration MLPClassifier`, atau `Maksimal Baris Training` dari sidebar.
