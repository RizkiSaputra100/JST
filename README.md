# Aplikasi Komparasi JST: Prediksi Kepuasan Pelanggan

Aplikasi ini dibuat dengan Python Streamlit untuk membandingkan dua metode Jaringan Syaraf Tiruan pada studi kasus prediksi kepuasan pelanggan. Dataset yang digunakan adalah **Customer Satisfaction 10K** dari Kaggle: <https://www.kaggle.com/datasets/ahmedaliraja/customer-satisfaction-10k>.

## Latar Belakang

Kepuasan pelanggan merupakan indikator penting untuk mengevaluasi kualitas layanan. Dengan data historis pelanggan, model machine learning dapat membantu memprediksi apakah pelanggan cenderung **Tidak Puas**, **Netral**, atau **Puas** berdasarkan atribut layanan yang tersedia di dataset.

## Alasan Memilih JST

Jaringan Syaraf Tiruan cocok digunakan karena mampu mempelajari pola non-linear dari banyak fitur, baik numerik maupun kategorikal. Pada kasus kepuasan pelanggan, hubungan antarfitur layanan sering kali tidak sederhana, sehingga pendekatan JST dapat menjadi pilihan yang relevan untuk klasifikasi.

## Metode yang Dikomparasikan

1. **TensorFlow/Keras Sequential**
   - Menggunakan model neural network berlapis `Dense`.
   - Menggunakan `validation_split` agar grafik training accuracy dan loss dapat ditampilkan.

2. **Scikit-Learn MLPClassifier**
   - Menggunakan implementasi multilayer perceptron dari Scikit-Learn.
   - Cocok sebagai pembanding karena memiliki konsep JST yang mirip, tetapi API dan proses training berbeda.

## Alur Preprocessing

1. Upload dataset CSV melalui aplikasi.
2. Aplikasi menampilkan preview, jumlah baris dan kolom, nama kolom, missing value, dan tipe data.
3. Kolom ID seperti `Customer ID` dihapus agar tidak digunakan sebagai fitur training.
4. Missing value ditangani otomatis:
   - Fitur numerik diisi dengan median.
   - Fitur kategorikal diisi dengan nilai paling sering muncul.
5. Fitur kategorikal diubah menjadi numerik menggunakan OneHotEncoder.
6. Fitur numerik dinormalisasi menggunakan StandardScaler.
7. Kolom target menggunakan `Satisfaction Score`.
8. Jika `Satisfaction Score` berupa angka 1-5, nilainya diubah menjadi:
   - 1-2 = Tidak Puas
   - 3 = Netral
   - 4-5 = Puas
9. Data dibagi menjadi 80% training dan 20% testing dengan `random_state=42`.

## Cara Instalasi

Pastikan Python sudah terpasang, lalu jalankan perintah berikut dari folder proyek:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Aplikasi

Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

Setelah itu, browser akan membuka aplikasi Streamlit. Upload file CSV dataset, lalu klik tombol **Mulai Training Model**.

## Cara Menjalankan dengan Docker

Docker dapat digunakan agar aplikasi berjalan dengan versi Python dan library yang konsisten, terutama karena TensorFlow membutuhkan versi Python yang kompatibel.

Build image Docker:

```bash
docker build -t prediksi-kepuasan-jst .
```

Jalankan container:

```bash
docker run --rm -p 8501:8501 prediksi-kepuasan-jst
```

Buka aplikasi di browser:

```text
http://localhost:8501
```

## Cara Deploy ke Streamlit Community Cloud

Cara ini direkomendasikan jika aplikasi ingin dibuka teman melalui link web tanpa menjalankan Docker atau Google Colab.

1. Buat repository baru di GitHub.
2. Upload file proyek berikut ke repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - folder `.streamlit/config.toml`
   - `Dockerfile` opsional, tidak wajib untuk Streamlit Cloud
3. Buka <https://share.streamlit.io>.
4. Login menggunakan akun GitHub.
5. Klik **Create app**.
6. Pilih repository GitHub yang berisi proyek ini.
7. Isi konfigurasi deploy:
   - Branch: `main`
   - Main file path: `app.py`
8. Buka **Advanced settings**.
9. Pilih Python version: `3.11`.
10. Klik **Deploy**.

Setelah deploy selesai, aplikasi akan mendapat link publik dengan domain `streamlit.app`. Link tersebut bisa dibagikan ke teman untuk mencoba aplikasi langsung dari browser.

Catatan penting:

- Jangan upload folder `.venv` ke GitHub.
- Dataset tidak perlu dimasukkan ke repository karena aplikasi sudah menyediakan fitur upload CSV.
- Jika deploy gagal pada tahap instalasi TensorFlow, pastikan Python version di Advanced settings adalah `3.11`.
- Jika aplikasi lambat saat training, kurangi jumlah epoch Keras dari sidebar.

## Cara Membaca Hasil

**Accuracy** menunjukkan persentase prediksi yang benar dari seluruh data testing. Semakin tinggi accuracy, semakin baik performa umum model.

**Precision** menunjukkan ketepatan model saat memprediksi suatu kelas. Nilai precision tinggi berarti prediksi pada kelas tersebut lebih sedikit salah.

**Recall** menunjukkan kemampuan model menemukan data yang benar-benar termasuk dalam suatu kelas. Nilai recall tinggi berarti model lebih sedikit melewatkan data dari kelas tersebut.

**F1-score** adalah rata-rata harmonis dari precision dan recall. Metrik ini berguna ketika ingin melihat keseimbangan antara precision dan recall.

**Training accuracy Keras** memperlihatkan perkembangan akurasi model Keras selama proses training. Jika training accuracy meningkat tetapi validation accuracy turun, model mungkin mengalami overfitting.

**Training loss Keras** memperlihatkan besarnya error selama training. Loss yang semakin turun biasanya menunjukkan model semakin baik dalam mempelajari pola data.

**Confusion matrix** menampilkan perbandingan antara kelas aktual dan kelas hasil prediksi. Nilai diagonal menunjukkan prediksi benar, sedangkan nilai di luar diagonal menunjukkan kesalahan klasifikasi.

**Heatmap korelasi fitur numerik** menunjukkan hubungan antarfitur numerik. Nilai mendekati 1 berarti korelasi positif kuat, nilai mendekati -1 berarti korelasi negatif kuat, dan nilai mendekati 0 berarti hubungan linear lemah.

## Fitur Aplikasi

- Upload dataset CSV.
- Preview dataset.
- Informasi dataset lengkap.
- Preprocessing otomatis.
- Training model TensorFlow/Keras Sequential.
- Training model Scikit-Learn MLPClassifier.
- Evaluasi dengan accuracy, precision, recall, f1-score, dan confusion matrix.
- Grafik training accuracy dan training loss Keras.
- Heatmap confusion matrix dan korelasi fitur numerik.
- Tabel komparasi hasil kedua metode.
- Form prediksi data pelanggan baru.
- Download hasil evaluasi dalam format CSV.
- Terjemahan tampilan ke Bahasa Indonesia untuk nama kolom dan nilai kategori umum, tanpa mengubah nama kolom asli yang dipakai model.
