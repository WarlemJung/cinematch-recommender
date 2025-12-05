# 🎬 Sistem Rekomendasi Film

Selamat datang di **Sistem Rekomendasi Film**! Proyek ini adalah aplikasi Python tangguh yang dirancang untuk menyarankan film berdasarkan teknik analisis data tingkat lanjut. Baik Anda mencari film hits terbesar sepanjang masa atau permata tersembunyi yang mirip dengan film favorit Anda, sistem ini siap membantu Anda.

## 🚀 Fitur

### 1. 📈 Penyaringan Demografis (Top Charts)
Sama seperti Top 250 IMDB, sistem ini mengidentifikasi film terbaik dengan menghitung **Peringkat Tertimbang (Weighted Rating - WR)**. Ini mempertimbangkan peringkat rata-rata dan jumlah suara untuk memastikan bahwa film yang direkomendasikan populer dan sangat diakui.

*   **Algoritma**: Rumus Peringkat Tertimbang IMDB.
*   **Tujuan**: Merekomendasikan film yang disukai secara universal.

### 2. 🧠 Penyaringan Berbasis Konten
Mencari sesuatu yang spesifik? Fitur ini merekomendasikan film yang mirip dengan film tertentu yang Anda sukai. Ini menganalisis **ringkasan plot (overviews)** film untuk menemukan kesamaan.

*   **Algoritma**: TF-IDF (Term Frequency-Inverse Document Frequency) & Kesamaan Kosinus.
*   **Tujuan**: Merekomendasikan film dengan alur cerita dan tema yang serupa.

---

## 🛠️ Teknologi yang Digunakan

*   **Python 3.13.7**: Bahasa pemrograman inti.
*   **Pandas**: Untuk manipulasi dan analisis data yang efisien.
*   **Scikit-learn**: Untuk algoritma pembelajaran mesin (TF-IDF Vectorizer, Linear Kernel).
*   **Flask**: Untuk membangun backend web.

---

## 📦 Instalasi & Pengaturan

1.  **Kloning repositori** (atau unduh file).
2.  **Instal pustaka yang diperlukan**:
    Buka terminal Anda dan jalankan:
    ```bash
    pip install pandas scikit-learn flask flask-cors
    ```
3.  **Unduh Dataset**:
    Pastikan Anda memiliki file `movies_metadata.csv` di direktori yang sama dengan skrip.
    [Unduh dataset di sini](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv)

---

## 🎮 Cara Menggunakan

### 🖥️ Antarmuka Baris Perintah (CLI)

1.  Buka terminal atau command prompt Anda.
2.  Arahkan ke direktori proyek.
3.  Jalankan skrip:
    ```bash
    python recommender.py
    ```

### 🌐 Menjalankan Antarmuka Web (UI Ultramodern)

1.  **Instal Flask**:
    ```bash
    pip install flask flask-cors
    ```
2.  **Jalankan Aplikasi**:
    ```bash
    python app.py
    ```
3.  **Buka di Browser**:
    Buka `http://localhost:5000` untuk melihat antarmuka baru!

---

## 🔍 Ikhtisar Kode

### Kelas `MovieRecommenderSystem`

*   **`__init__`**: Memuat data, membersihkannya, dan menyiapkannya untuk analisis.
*   **`get_demographic_recommendations`**: Menghitung skor tertimbang untuk setiap film dan mengembalikan hasil teratas.
*   **`compute_content_similarity`**: Membangun matriks TF-IDF dari ikhtisar film dan menghitung kernel kesamaan kosinus.
*   **`get_content_recommendations`**: Mengambil judul film sebagai input dan mengembalikan daftar film yang paling mirip berdasarkan skor yang dihitung.

---

## 📄 Lisensi

Proyek ini open-source dan tersedia untuk tujuan pendidikan. Jangan ragu untuk memodifikasi dan memperbaikinya!

---

*Dibuat dengan ❤️ oleh Warlem Jung*
