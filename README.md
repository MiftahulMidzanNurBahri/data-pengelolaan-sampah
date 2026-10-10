# Analisis Kuesioner dan Persiapan PLS-SEM

## Cakupan

Skrip Python membaca CSV, memvalidasi 15 indikator Likert, melaporkan kualitas data, mengekspor indikator individual yang telah dibersihkan, serta menghitung statistik deskriptif dan visualisasi konstruk. Skrip ini **bukan estimator PLS-SEM**: korelasi Pearson dan skor rata-rata deskriptif tidak menggantikan evaluasi outer model, inner model, atau pengujian hipotesis.

## Konstruk dan indikator

| Konstruk | Indikator |
| --- | --- |
| X1 - System Quality & Features | Q1, Q2, Q3 |
| X2 - Reward & Incentive Structure | Q4, Q5, Q6 |
| X3 - Drop-Point / Pickup Accessibility | Q7, Q8, Q9 |
| M - Environmental Attitude & Engagement | Q10, Q11, Q12 |
| Y - Actual Recycling Behavior | Q13, Q14, Q15 |

Mapping ini konfigurasi awal proyek; pastikan nama, arah, dan isi indikator cocok dengan instrumen final. Tidak ada reverse coding otomatis.

## Persiapan dan penggunaan

Gunakan Python 3.10 atau lebih baru. Dari PowerShell pada folder proyek:

```powershell
python -m pip install -r requirements.txt
python .\analisis_sem_pls.py .\data_kuisioner_15_item.csv
```

Rentang Likert default adalah 1 sampai 5 dan dapat diubah:

```powershell
python .\analisis_sem_pls.py .\data.csv --likert-min 1 --likert-max 7
```

Nilai respons harus integer dalam rentang yang dipilih; pecahan seperti `2.5` ditolak. `--missing-warning-threshold` mengatur proporsi missing yang diberi tanda pada laporan responden (default `0.5`). Ambang tersebut hanya peringatan screening dan tidak menghapus data.

CSV wajib memuat satu kolom untuk setiap Q1 sampai Q15. Kapitalisasi Q/q dinormalisasi. Kolom tambahan diperbolehkan; kolom ID umum seperti `Respondent_ID` dipertahankan pada ekspor, tetapi tidak dipakai sebagai indikator. Delimiter koma, titik koma, tab, dan pipe dideteksi; encoding UTF-8/UTF-8-BOM dan Windows-1252 didukung.

Contoh header:

```csv
Respondent_ID,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15
R01,4,4,4,1,2,1,5,2,3,3,4,3,3,2,3
```

Nilai teks Likert, nilai di luar rentang, dan sel kosong tidak dikonversi menjadi jawaban valid. Semuanya menjadi missing pada salinan bersih dan dirinci di laporan kualitas data. Tidak ada imputasi atau penghapusan baris. Skor konstruk hanya dihitung jika ketiga indikator konstruk valid; total berkisar 3-15 dan mean berkisar sesuai rentang Likert.

## Output

Secara default output ditulis ke `hasil_analisis/`; gunakan `--output-dir` untuk memilih folder lain.

- `data_indikator_bersih.csv`: ID opsional dan setiap indikator Q1-Q15 sebagai kolom terpisah. Missing direpresentasikan sebagai sel kosong. Gunakan file ini untuk impor indikator ke SmartPLS; jangan tetapkan kolom ID sebagai indikator/model variable.
- `laporan_missing_values.csv`: laporan kualitas per indikator, termasuk missing asli/pasca-cleaning, nilai nonnumerik, desimal, di luar rentang, jumlah valid, persentase missing, dan jumlah kategori valid. Frekuensi jawaban ada di statistik deskriptif.
- `laporan_kualitas_responden.csv`: jumlah/persentase missing per responden serta flag missing tinggi, pola seragam, jawaban duplikat, dan ID duplikat. Flag hanya untuk pemeriksaan manual.
- `ringkasan_pemrosesan.csv`: jumlah baris awal, dipertahankan, dan dikeluarkan. Skrip saat ini tidak menghapus baris otomatis.
- `skor_konstruk_per_responden.csv`: skor mean dan total per responden; bukan pengganti data indikator untuk estimasi SEM.
- `statistik_deskriptif.csv`: statistik indikator dan konstruk; indikator juga memuat frekuensi dan persentase tiap nilai skala.
- `statistik_deskriptif_konstruk.csv`: output konstruk saja untuk kompatibilitas dengan format lama.
- `boxplot_skor_konstruk.png` dan `heatmap_korelasi_konstruk.png`: grafik 300 DPI. Heatmap memakai korelasi Pearson deskriptif dan bukan hasil uji jalur.

## Alur SmartPLS yang masih manual

Belum ada SmartPLS atau engine PLS-SEM di environment ini, dan proyek tidak mengotomasi antarmuka/API SmartPLS. Untuk estimasi artikel, gunakan perangkat lunak PLS-SEM yang tervalidasi:

1. Impor `data_indikator_bersih.csv`; keluarkan `Respondent_ID` dari model pengukuran. Missing tetap kosong, jadi tetapkan kebijakan penanganannya berdasarkan metodologi sebelum estimasi.
2. Definisikan lima konstruk dan indikator sesuai tabel di atas. Tentukan reflective/formative berdasarkan teori dan instrumen; jenis pengukuran belum ditentukan oleh file proyek.
3. Gambar jalur langsung X1, X2, X3 ke Y; X1, X2, X3 ke M; serta M ke Y.
4. Jalankan bootstrapping 5.000 subsampel untuk koefisien jalur, interval kepercayaan, signifikansi, dan specific indirect effects X1/X2/X3 -> M -> Y.
5. Evaluasi outer model dengan outer loadings, AVE, Cronbach's alpha, Composite Reliability, Fornell-Larcker, dan HTMT sesuai tipe model serta pedoman metodologi yang dirujuk.
6. Evaluasi inner model dengan R-squared, f-squared, dan Q-squared bila didukung prosedur yang dipilih. Dokumentasikan versi perangkat lunak, pilihan missing data, seed/pengaturan bootstrap, dan keputusan eksklusi.

Proyek saat ini tidak menghitung metrik tersebut. Jangan menafsirkan mean/korelasi sebagai bukti validitas, signifikansi, mediasi kausal, atau penerimaan hipotesis. Desain cross-sectional mendukung interpretasi asosiasi/mediasi statistik, bukan bukti temporal kausal. Pastikan konstruk Y mengukur perilaku daur ulang aktual, bukan niat.

## Validasi lokal

```powershell
python -m unittest discover -s tests -v
```

Data sintetis hanya untuk menguji pipeline, bukan temuan empiris. Kecukupan sampel perlu dinilai terhadap kompleksitas model dan rancangan analisis; jumlah baris saja tidak membuktikannya.