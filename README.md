# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal dengan reputasi lulusannya yang berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi tantangan serius berupa angka _dropout_ (putus studi) mahasiswa yang cukup tinggi. Fenomena ini tidak hanya memengaruhi citra institusi, tetapi juga berdampak pada efisiensi operasional, perencanaan akademik, dan keuangan.

### Permasalahan Bisnis

Meskipun telah meluluskan banyak mahasiswa berprestasi, Jaya Jaya Institut sedang menghadapi tantangan serius terkait tingginya angka mahasiswa yang tidak menyelesaikan studinya (_dropout_). Tingkat _dropout_ yang tinggi ini menjadi permasalahan penting yang berdampak langsung pada berbagai aspek strategis institusi, antara lain:

- Penurunan tingkat kelulusan dan retensi karena setiap mahasiswa yang keluar sebelum lulus mencerminkan hilangnya potensi akademik dan pendapatan jangka panjang.

- Menurunnya reputasi institusi diakibatkan tingginya _dropout_ _rate_ dapat memengaruhi persepsi publik dan kepercayaan calon mahasiswa.

- Beban operasional dan sumber daya dari mahasiswa yang tidak menyelesaikan studi yang menyerap sumber daya pengajaran dan administrasi.

- Ancaman terhadap akreditasi dan evaluasi eksternal terhadap mutu pendidikan yang seringkali mempertimbangkan angka _dropout_ dalam proses evaluasi dan akreditasi institusi.

Karena itu, Jaya Jaya Institut memandang penting untuk melakukan deteksi dini terhadap risiko _dropout_, agar mahasiswa dengan risiko tinggi bisa diberikan intervensi dan pendampingan secara proaktif.

### Cakupan Proyek

- **Pengumpulan Data:** Mengumpulkan data dari berbagai sumber yang berisi informasi terkait mahasiswa, termasuk jalur akademik, demografi, sosial ekonomi, dan performa akademik.
- **Data Understanding:** Melakukan eksplorasi data untuk memahami pola, tren, dan hubungan antar fitur dalam dataset, serta mengidentifikasi variabel-variabel yang berpotensi mempengaruhi keberhasilan akademik dan risiko dropout.
- **Data Preparation**: Melakukan pembersihan data, penanganan missing values, transformasi fitur, dan encoding untuk memastikan data siap digunakan dalam pengembangan model machine learning.
- **Pengembangan Model:** Membangun model machine learning menggunakan teknik yang sesuai untuk memprediksi risiko _dropout_ dan keberhasilan akademik mahasiswa, serta melakukan tuning hyperparameter untuk meningkatkan performa model.
- **Evaluasi:** Mengukur kinerja model yang dikembangkan menggunakan metrik evaluasi yang relevan (seperti akurasi, presisi, dan recall), dan melakukan analisis lebih lanjut untuk memastikan model memenuhi kebutuhan bisnis dan akurasi yang diharapkan.

### Persiapan

Sumber data: [Student Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup Environment - Anaconda or Shell/Terminal

```bash
# Anancoda
conda create --name main-ds python=3.11.11
conda activate main-ds
```

```bash
# Shell/Terminal
mkdir proyek_student_analysis
cd proyek_student_analysis
pipenv install
pipenv shell
pip install -r requirements.txt
```

### Data Understanding

- **Dataset students performance:** Dataset yang digunakan adalah tentang performa mahasiswa dengan jumlah data sebanyak 4424 baris dan 37 kolom yang terbagi menjadi 36 kolom independen yang akan digunakan untuk melatih model dan 1 kolom yaitu kolom Status sebagai targetnya.

### Data Visualisation

**Distribusi Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/b9a4f8b8-bada-482b-8f1a-6e24bae98d17)

Berdasarkan plot pie chart di atas menunjukkan bahwa proporsi mahasiswa graduate sebesar 49.9% dengan jumlah mahasiswa yang _dropout_ cukup banyak yaitu 32.1%

**Gender dan Status Mahasiwa**

![Image](https://github.com/user-attachments/assets/395ca39a-b71c-4408-8bf3-e2dc53186455)

Berdasarkan plot di atas mahasiswa laki-laki memiliki persen _dropout_ lebih tinggi (45.1%) dibandingkan dengan perempuan (25.1%). Lalu untuk tingkat gradutenya mahasiswa laki-laki memiliki tingkat kelulusan yang rendah (35.2%) di bandingkan perempuan (57.9%).

**Marital Status dan Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/d750fbf1-f6e5-442e-ae43-9b67af9b145a)

Dapat dilihat bahwa kebanyakan mahasiswa masih status single dan beberapa lainnya sudah menikah atau cerai. Selain itu, mahasiswa yang sudah menikah atau cerai lebih banyak yang _dropout_ dibandingkan graduate.

**Usia (Age) dan Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/72f7ddb2-7eef-4f7c-807e-36e2210fc108)

Dapat dilihat mahasiswa yang rata-rata umur 25 tahun keatas pada saat pendaftaran cenderung _dropout_, dibandingkan dengan mahasiswa yang berumur dibawah 25 tahun.

**Course dan Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/3a64baf3-94ff-4d9a-b86e-2e7f9cd21e23)

Status mahasiswa juga dipengaruhi oleh course yang dijalani, dapat dilihat persebaran data pada grafik diatas sangat beragam, course nursing sendiri memiliki status graduate tertinggi, lalu Management (evening attendance) dan management memiliki tingkat _dropout_ yang tinggi. Dengan demikian, status mahasiswa dapat dipengaruhi dengan course apa yang dipilih.

**Educational Special Needs dan Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/df80ee71-6933-4995-9776-91500aa8a01f)

Mahasiswa yang lulus dan tidak lulus hampir semuanya tidak memiliki kebutuhan khusus.

**Debtor dan Status Mahasiswa**

![Image](https://github.com/user-attachments/assets/7165ccf8-fe75-4d5e-a4ed-87612c4a635b)

Banyak mahasiswa yang tidak memiliki hutang kuliah baik yang terdaftar maupun lulus, sementara hanya sedikit mahasiswa yang memiliki hutang kuliah untuk yang terdaftar dan putus kuliah.

**Beasiswa dan Status**

![Image](https://github.com/user-attachments/assets/f9d6b8e9-4f23-4edd-b126-1bac93d8cdab)

Berdasarkan grafik dia atas, Penerima beasiswa cenderung berstatus graduate dan tidak _dropout_ dibandingkan dengan mahasiswa yang tidak menerima beasiswa.

## Business Dashboard

Dashboard ini memberikan visualisasi untuk memantau prediksi _dropout_ mahasiswa dan keberhasilan akademik mereka. Melalui dashboard ini, pihak institusi dapat melihat tren _dropout_ dan mengidentifikasi kelompok mahasiswa yang membutuhkan perhatian lebih.

**Menjalankan Dashboard**

Untuk melihat isi dashboard secara langsung, dapat menggunakan **metabase** dengan bantuan Docker.

- Jalankan perintah berikut:
  ```
  docker pull metabase/metabase:v0.54.6
  ```
- Jalankan container Metabase menggunakan perintah:
  ```
  docker run -p 3000:3000 --name metabase metabase/metabase
  ```
- Login ke Metabase menggunakan username dan password berikut:
  ```
  username: root@mail.com
  password: root123
  ```

![Image](https://github.com/user-attachments/assets/d99d3ebc-767d-40d1-9588-005a48dd9591)

## Menjalankan Sistem Machine Learning

1. Clone the repository or download the source code.

```bash
https://github.com/mripalp001/Student-Performance-Predictions.git
```

2. Run the Streamlit app using:

```bash
streamlit run dashboard/app.py
```

4. Buka tautan yang disediakan oleh Streamlit untuk mengakses dashboard di browser web.

   [Student Performance Prediction](https://student-performance-predictions-9w4yussqhg8eeuxmkupzkx.streamlit.app/)

5. Upload test data untuk mencoba aplikasi machine learning.

## Conclusion

Berdasarkan analisis data dan insight yang telah diperoleh, beberapa faktor kunci yang dapat memprediksi potensi mahasiswa untuk putus kuliah (_dropout_) sejak semester pertama adalah sebagai berikut:

- **Jenis Kelamin:** mahasiswa laki-laki memiliki risiko putus kuliah yang lebih tinggi dibandingkan perempuan. Ini terlihat dari tingkat _dropout_ laki-laki sebesar 45.1% sementara perempuan hanya 25.1%. Sebaliknya, tingkat kelulusan perempuan lebih tinggi (57.9%) dibandingkan laki-laki (35.2%).
- **Status Pernikahan:** mahasiswa yang sudah menikah atau bercerai cenderung lebih berisiko untuk putus kuliah dibandingkan dengan mahasiswa yang masih lajang. Hal ini bisa disebabkan oleh tanggung jawab tambahan yang mereka hadapi, seperti tanggungan keluarga.
- **Usia Pendaftaran:** Mahasiswa yang rata-rata umur 25 tahun keatas pada saat pendaftaran cenderung _dropout_, dibandingkan dengan mahasiswa yang berumur dibawah 25 tahun.. Ini bisa menjadi indikasi bahwa mahasiswa yang lebih tua mungkin memiliki komitmen lain di luar perkuliahan yang mempengaruhi kinerja akademis mereka.
- **Pilihan Jurusan:** Jurusan yang diambil juga sangat mempengaruhi status mahasiswa. Jurusan seperti nursing memiliki tingkat kelulusan tertinggi, sedangkan jurusan management (evening attendance) dan management memiliki tingkat _dropout_ yang lebih tinggi. Ini menunjukkan bahwa tingkat kesulitan atau pola pembelajaran dari masing-masing program studi mungkin mempengaruhi hasil akademis mahasiswa.
- **Berkebutuhan Khusus:** Mahasiswa yang lulus dan tidak lulus hampir semuanya tidak memiliki kebutuhan khusus.
- **Hutang:** Banyak mahasiswa yang tidak memiliki hutang kuliah baik yang terdaftar maupun lulus, sementara hanya sedikit mahasiswa yang memiliki hutang kuliah untuk yang terdaftar dan putus kuliah.
- **Beasiswa:** Mahasiswa yang menerima beasiswa lebih cenderung untuk lulus dibandingkan dengan mereka yang tidak mendapatkan beasiswa. Dukungan finansial tampaknya berperan besar dalam mempertahankan mahasiswa agar tetap melanjutkan studi hingga lulus.
- **Penggunaan Model dalam Pengambilan Keputusan:** Model XGBoost terbukti sebagai model terbaik dengan akurasi sebesar 76.9%. Model ini bisa diintegrasikan dalam sistem administrasi akademik untuk membantu institusi memonitor mahasiswa dan memberikan peringatan awal (early warning system).

## Rekomendasi Action Items

- **Intervensi Awal:** Mengidentifikasi mahasiswa yang berisiko tinggi untuk _dropout_ sejak semester pertama, terutama berdasarkan jenis kelamin, status pernikahan, dan usia pendaftaran, sehingga intervensi seperti konseling atau bimbingan tambahan dapat diberikan.
- **Dukungan Keuangan:** Memberikan lebih banyak kesempatan beasiswa kepada mahasiswa yang berpotensi, karena ini dapat meningkatkan peluang mereka untuk lulus.
- **Pendekatan Khusus Berdasarkan Jurusan:** Menyesuaikan program dukungan atau bimbingan akademik berdasarkan jurusan yang memiliki tingkat _dropout_ lebih tinggi, seperti jurusan management.
- **Fokus pada Nilai Akademik:** Menyediakan program bimbingan belajar atau tutor tambahan bagi mahasiswa dengan nilai akademik rendah agar dapat meningkatkan performa akademik dan mengurangi risiko putus kuliah.

Dengan memahami faktor-faktor yang mempengaruhi tingkat _dropout_, institusi pendidikan dapat mengambil langkah-langkah proaktif untuk mendukung mahasiswa dalam menyelesaikan studi mereka.

```

```
