# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal dengan reputasi lulusannya yang berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi tantangan serius berupa angka _dropout_ (putus studi) mahasiswa yang cukup tinggi. Fenomena ini tidak hanya memengaruhi citra institusi, tetapi juga berdampak pada efisiensi operasional, perencanaan akademik, dan keuangan. Tingginya tingkat _dropout_ dapat menyebabkan turunnya akreditasi dan reputasi institusi, penurunan jumlah pendaftar baru akibat persepsi negatif, pemborosan sumber daya (dosen, fasilitas, beasiswa) untuk mahasiswa yang tidak menyelesaikan studi, dan ketidakefisienan dalam perencanaan akademik dan kapasitas kelas.

### Permasalahan Bisnis

- Mengindentifikasi banyaknya mahasiswa yang _dropout_ berdasarkan jenis kelamin
- Mengidentifikasi _course_ yang memiliki banyak mahasiswanya yang _dropout_
- Apakah status pernikahan memiliki pengaruh terhadap _dropout_ nya mahasiswa
- Mengidentifikasi usia rata-rata mahasiswa yang _dropout_ dibandingkan dengan tingkatan usia lain.

### Cakupan Proyek

- **Pengumpulan Data:** Mengumpulkan data dari berbagai sumber yang berisi informasi terkait mahasiswa, termasuk jalur akademik, demografi, sosial ekonomi, dan performa akademik.
- **Data Understanding:** Melakukan eksplorasi data untuk memahami pola, tren, dan hubungan antar fitur dalam dataset, serta mengidentifikasi variabel-variabel yang berpotensi mempengaruhi keberhasilan akademik dan risiko dropout.
- **Data Preparation**: Melakukan pembersihan data, penanganan missing values, transformasi fitur, dan encoding untuk memastikan data siap digunakan dalam pengembangan model machine learning.
- **Pengembangan Model:** Membangun model machine learning menggunakan teknik yang sesuai untuk memprediksi risiko dropout dan keberhasilan akademik mahasiswa, serta melakukan tuning hyperparameter untuk meningkatkan performa model.
- **Evaluasi:** Mengukur kinerja model yang dikembangkan menggunakan metrik evaluasi yang relevan (seperti akurasi, presisi, dan recall), dan melakukan analisis lebih lanjut untuk memastikan model memenuhi kebutuhan bisnis dan akurasi yang diharapkan.

### Persiapan

Sumber data: [Student Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup environment:

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn import preprocessing

# Algoritma
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# Deployment
import joblib
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
```

### Data Understanding

- **Dataset students performance:** Dataset yang digunakan adalah tentang performa mahasiswa dengan jumlah data sebanyak 4424 baris dan 37 kolom yang terbagi menjadi 36 kolom independen yang akan digunakan untuk melatih model dan 1 kolom yaitu kolom Status sebagai targetnya.

### Data Visualisation

**Distribusi Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/01.%20Distribusi%20Status%20Mahasiswa.png)

Berdasarkan plot pie chart di atas menunjukkan bahwa proporsi mahasiswa graduate sebesar 49.9% dengan jumlah mahasiswa yang dropout cukup banyak yaitu 32.1%

**Gender dan Status Mahasiwa**

![image](https://github.com/mripalp001/Student-Performance-Predictions/blob/main/02.%20Distribusi%20Gender%20Mahasiswa.png?raw=true)

Berdasarkan plot di atas mahasiswa laki-laki memiliki persen dropout lebih tinggi (45.1%) dibandingkan dengan perempuan (25.1%). Lalu untuk tingkat gradutenya mahasiswa laki-laki memiliki tingkat kelulusan yang rendah (35.2%) di bandingkan perempuan (57.9%).

**Marital Status dan Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/03.%20Marital%20Status%20dan%20Status%20Mahasiswa.png)

Dapat dilihat bahwa kebanyakan mahasiswa masih status single dan beberapa lainnya sudah menikah atau cerai. Selain itu, mahasiswa yang sudah menikah atau cerai lebih banyak yang dropout dibandingkan graduate.

**Usia (Age) dan Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/04.%20Age%20dan%20Status%20Mahasiswa.png)

Dapat dilihat mahasiswa yang rata-rata umur 25 tahun keatas pada saat pendaftaran cenderung dropout, dibandingkan dengan mahasiswa yang berumur dibawah 25 tahun.

**Course dan Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/05.%20Course%20dan%20Status%20Mahasiswa.png)

Status mahasiswa juga dipengaruhi oleh course yang dijalani, dapat dilihat persebaran data pada grafik diatas sangat beragam, course nursing sendiri memiliki status graduate tertinggi, lalu Management (evening attendance) dan management memiliki tingkat dropout yang tinggi. Dengan demikian, status mahasiswa dapat dipengaruhi dengan course apa yang dipilih.

**Educational Special Needs dan Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/06.%20Educational%20Special%20Needs%20dan%20Status%20Mahasiswa.png)

Mahasiswa yang lulus dan tidak lulus hampir semuanya tidak memiliki kebutuhan khusus.

**Debtor dan Status Mahasiswa**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/07.%20Debtor%20dan%20Status%20Mahasiswa.png)

Banyak mahasiswa yang tidak memiliki hutang kuliah baik yang terdaftar maupun lulus, sementara hanya sedikit mahasiswa yang memiliki hutang kuliah untuk yang terdaftar dan putus kuliah.

**Beasiswa dan Status**

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/08.%20Scholarship%20Holder%20dan%20Status%20Mahasiswa.png)

Berdasarkan grafik dia atas, Penerima beasiswa cenderung berstatus graduate dan tidak dropout dibandingkan dengan mahasiswa yang tidak menerima beasiswa.

## Business Dashboard

Dashboard ini memberikan visualisasi untuk memantau prediksi dropout mahasiswa dan keberhasilan akademik mereka. Melalui dashboard ini, pihak institusi dapat melihat tren dropout dan mengidentifikasi kelompok mahasiswa yang membutuhkan perhatian lebih.

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

![image](https://raw.githubusercontent.com/mripalp001/Student-Performance-Predictions/refs/heads/main/mohamad_ripal-dashboard.png)

## Menjalankan Sistem Machine Learning

### How to Run

1. Clone the repository or download the source code.

```bash
https://github.com/mripalp001/Student-Performance-Predictions.git
```

2. Setup Environment - Anaconda

```bash
conda create --name proyek_student_analysis python=3.11.11
conda activate proyek_student_analysis
```

3. Install the required Python packages

```bash
cd dashboard
pip install -r requirements.txt
```

4. Run the Streamlit app using:

```bash
streamlit run app.py
```

##

4. Buka tautan yang disediakan oleh Streamlit untuk mengakses dasbor di browser web Anda.

   [Streamlit Student Performance Prediction](https://student-performance-predictions-es3u6bcxr7cpfda3zxcpqp.streamlit.app/)

5. upload test data untuk mencoba aplikasi machine learning.

## Conclusion

Berdasarkan analisis data dan insight yang telah diperoleh, beberapa faktor kunci yang dapat memprediksi potensi mahasiswa untuk putus kuliah (dropout) sejak semester pertama adalah sebagai berikut:

- **Jenis Kelamin:** mahasiswa laki-laki memiliki risiko putus kuliah yang lebih tinggi dibandingkan perempuan. Ini terlihat dari tingkat dropout laki-laki sebesar 45.1% sementara perempuan hanya 25.1%. Sebaliknya, tingkat kelulusan perempuan lebih tinggi (57.9%) dibandingkan laki-laki (35.2%).
- **Status Pernikahan:** mahasiswa yang sudah menikah atau bercerai cenderung lebih berisiko untuk putus kuliah dibandingkan dengan mahasiswa yang masih lajang. Hal ini bisa disebabkan oleh tanggung jawab tambahan yang mereka hadapi, seperti tanggungan keluarga.
- **Usia Pendaftaran:** Mahasiswa yang rata-rata umur 25 tahun keatas pada saat pendaftaran cenderung dropout, dibandingkan dengan mahasiswa yang berumur dibawah 25 tahun.. Ini bisa menjadi indikasi bahwa mahasiswa yang lebih tua mungkin memiliki komitmen lain di luar perkuliahan yang mempengaruhi kinerja akademis mereka.
- **Pilihan Jurusan:** Jurusan yang diambil juga sangat mempengaruhi status mahasiswa. Jurusan seperti nursing memiliki tingkat kelulusan tertinggi, sedangkan jurusan management (evening attendance) dan management memiliki tingkat dropout yang lebih tinggi. Ini menunjukkan bahwa tingkat kesulitan atau pola pembelajaran dari masing-masing program studi mungkin mempengaruhi hasil akademis mahasiswa.
- **Berkebutuhan Khusus:** Mahasiswa yang lulus dan tidak lulus hampir semuanya tidak memiliki kebutuhan khusus.
- **Hutang:** Banyak mahasiswa yang tidak memiliki hutang kuliah baik yang terdaftar maupun lulus, sementara hanya sedikit mahasiswa yang memiliki hutang kuliah untuk yang terdaftar dan putus kuliah.
- **Beasiswa:** Mahasiswa yang menerima beasiswa lebih cenderung untuk lulus dibandingkan dengan mereka yang tidak mendapatkan beasiswa. Dukungan finansial tampaknya berperan besar dalam mempertahankan mahasiswa agar tetap melanjutkan studi hingga lulus.
- **Penggunaan Model dalam Pengambilan Keputusan:** Model XGBoost terbukti sebagai model terbaik dengan akurasi sebesar 76.9%. Model ini bisa diintegrasikan dalam sistem administrasi akademik untuk membantu institusi memonitor mahasiswa dan memberikan peringatan awal (early warning system).

### Rekomendasi Action Items

- **Intervensi Awal:** Mengidentifikasi mahasiswa yang berisiko tinggi untuk dropout sejak semester pertama, terutama berdasarkan jenis kelamin, status pernikahan, dan usia pendaftaran, sehingga intervensi seperti konseling atau bimbingan tambahan dapat diberikan.
- **Dukungan Keuangan:** Memberikan lebih banyak kesempatan beasiswa kepada mahasiswa yang berpotensi, karena ini dapat meningkatkan peluang mereka untuk lulus.
- **Pendekatan Khusus Berdasarkan Jurusan:** Menyesuaikan program dukungan atau bimbingan akademik berdasarkan jurusan yang memiliki tingkat dropout lebih tinggi, seperti jurusan management.
- **Fokus pada Nilai Akademik:** Menyediakan program bimbingan belajar atau tutor tambahan bagi mahasiswa dengan nilai akademik rendah agar dapat meningkatkan performa akademik dan mengurangi risiko putus kuliah.

Dengan memahami faktor-faktor yang mempengaruhi tingkat dropout, institusi pendidikan dapat mengambil langkah-langkah proaktif untuk mendukung mahasiswa dalam menyelesaikan studi mereka.
