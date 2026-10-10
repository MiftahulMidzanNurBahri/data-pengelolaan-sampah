from __future__ import annotations  # Mengaktifkan evaluasi anotasi tipe yang lebih fleksibel.

import argparse  # Menyediakan antarmuka argumen baris perintah.
import csv  # Mendeteksi delimiter dan memvalidasi header mentah sebelum pandas memodifikasinya.
from pathlib import Path  # Menangani lokasi file dengan aman lintas sistem operasi.

import matplotlib.pyplot as plt  # Membuat dan menyimpan visualisasi.
import pandas as pd  # Membaca, membersihkan, dan merangkum data tabular.
import seaborn as sns  # Membuat visualisasi statistik seperti boxplot dan heatmap.

CONSTRUCTS = {  # Memetakan butir pertanyaan ke konstruk laten penelitian.
    "X1 - System Quality & Features": ("Q1", "Q2", "Q3"),  # Konstruk kualitas sistem.
    "X2 - Reward & Incentive Structure": ("Q4", "Q5", "Q6"),  # Konstruk reward dan insentif.
    "X3 - Drop-Point / Pickup Accessibility": ("Q7", "Q8", "Q9"),  # Konstruk aksesibilitas.
    "M - Environmental Attitude & Engagement": ("Q10", "Q11", "Q12"),  # Konstruk sikap dan keterlibatan.
    "Y - Actual Recycling Behavior": ("Q13", "Q14", "Q15"),  # Konstruk perilaku daur ulang.
}  # Mengakhiri definisi pemetaan konstruk.


def parse_args() -> argparse.Namespace:  # Membaca dan memvalidasi argumen pengguna.
    parser = argparse.ArgumentParser(description="Analisis deskriptif kuesioner SEM-PLS dari CSV.")  # Menyiapkan parser CLI.
    parser.add_argument("csv_file", type=Path, help="Path ke file CSV masukan.")  # Mewajibkan path CSV masukan.
    parser.add_argument("--output-dir", type=Path, default=Path("hasil_analisis"), help="Folder keluaran (default: hasil_analisis).")  # Mengatur folder keluaran.
    parser.add_argument("--likert-min", type=int, default=1, help="Nilai minimum Skala Likert integer (default: 1).")  # Mengatur batas bawah skala.
    parser.add_argument("--likert-max", type=int, default=5, help="Nilai maksimum Skala Likert integer (default: 5).")  # Mengatur batas atas skala.
    parser.add_argument("--missing-warning-threshold", type=float, default=0.5, help="Proporsi missing untuk peringatan responden (default: 0.5; tidak menghapus responden).")  # Mengatur batas peringatan, bukan eksklusi.
    return parser.parse_args()  # Mengembalikan argumen yang telah dibaca.


def detect_csv_format(csv_file: Path) -> tuple[str, str, list[str]]:  # Mendeteksi encoding, delimiter, dan header CSV.
    for encoding in ("utf-8-sig", "cp1252"):  # Mendukung ekspor UTF-8 dan CSV Windows yang umum.
        try:  # Mencoba membaca header menggunakan encoding saat ini.
            with csv_file.open("r", encoding=encoding, newline="") as file:  # Membuka file tanpa mengubah newline CSV.
                sample = file.read(8192)  # Membaca sampel untuk mendeteksi delimiter.
                if not sample.strip():  # Memastikan file berisi teks.
                    raise ValueError("File CSV kosong.")  # Memberikan pesan yang jelas untuk file kosong.
                try:  # Mendeteksi koma, titik koma, tab, atau pipe.
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter  # Memilih delimiter berdasarkan isi file.
                except csv.Error:  # Menangani sampel yang tidak cukup untuk deteksi otomatis.
                    delimiter = ","  # Menggunakan koma sebagai fallback CSV standar.
                file.seek(0)  # Mengembalikan posisi baca ke awal file.
                headers = next(csv.reader(file, delimiter=delimiter))  # Membaca header mentah sebelum pandas mengubah duplikat.
            return encoding, delimiter, headers  # Mengembalikan format yang terdeteksi.
        except UnicodeDecodeError:  # Beralih encoding bila byte tidak valid pada percobaan ini.
            continue  # Mencoba encoding berikutnya.
    raise ValueError("Encoding CSV tidak dikenali; simpan ulang sebagai UTF-8 atau CSV Windows.")  # Menolak file dengan encoding yang tidak didukung.


def normalize_headers(headers: list[str]) -> list[str]:  # Merapikan header dan menyeragamkan nama indikator.
    question_lookup = {question.casefold(): question for questions in CONSTRUCTS.values() for question in questions}  # Membuat nama baku untuk Q1 sampai Q15.
    cleaned_headers = [header.strip().lstrip("\ufeff").strip() for header in headers]  # Menghapus spasi dan BOM dari setiap nama kolom.
    return [question_lookup.get(header.casefold(), header) for header in cleaned_headers]  # Menyeragamkan kapitalisasi indikator yang dikenali.


def read_csv(csv_file: Path) -> pd.DataFrame:  # Membaca CSV dan memeriksa struktur dasarnya.
    if not csv_file.is_file():  # Memastikan file masukan benar-benar tersedia.
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_file}")  # Menghentikan proses dengan pesan yang jelas.
    try:  # Menangani kesalahan pembacaan CSV secara informatif.
        encoding, delimiter, raw_headers = detect_csv_format(csv_file)  # Mendeteksi format dan membaca nama kolom mentah.
        normalized_headers = normalize_headers(raw_headers)  # Menormalkan header sebelum membaca tabel ke pandas.
        duplicated_headers = pd.Index(normalized_headers)[pd.Index(normalized_headers).duplicated()].unique().tolist()  # Menangkap duplikat termasuk beda kapitalisasi.
        if duplicated_headers:  # Memastikan setiap indikator memiliki satu kolom yang tidak ambigu.
            raise ValueError(f"Nama kolom duplikat ditemukan setelah normalisasi: {duplicated_headers}")  # Menjelaskan kolom yang bentrok.
        data = pd.read_csv(csv_file, sep=delimiter, encoding=encoding, keep_default_na=False, skip_blank_lines=False)  # Membaca semua respons termasuk baris kosong untuk audit.
    except (OSError, csv.Error, pd.errors.ParserError, UnicodeDecodeError) as error:  # Menangkap masalah file, format, atau encoding.
        raise ValueError(f"CSV tidak dapat dibaca: {error}") from error  # Menyampaikan penyebab kegagalan.
    data.columns = normalized_headers  # Menetapkan nama header yang sudah divalidasi.
    if len(data.columns) != len(raw_headers):  # Memeriksa jumlah kolom konsisten dengan header.
        raise ValueError("Jumlah kolom pada baris data tidak sesuai dengan header CSV.")  # Menolak struktur tabel yang tidak konsisten.
    if data.empty:  # Memastikan file memiliki setidaknya satu baris respons.
        raise ValueError("CSV tidak memiliki baris data.")  # Menghentikan analisis tanpa data.
    return data  # Mengembalikan tabel yang sudah memiliki header bersih.


def identifier_columns(data: pd.DataFrame) -> list[str]:  # Menemukan kolom ID yang umum tanpa menganggapnya sebagai indikator.
    recognized_names = {"id", "respondentid", "responseid", "participantid"}  # Nama ID yang dikenali secara eksplisit.
    return [  # Mengembalikan kolom identitas yang tersedia.
        column  # Mempertahankan nama kolom asli untuk output.
        for column in data.columns  # Memeriksa setiap kolom input.
        if "".join(character for character in column.casefold() if character.isalnum()) in recognized_names  # Mencocokkan nama tanpa spasi atau tanda baca.
    ]  # Menyelesaikan pencarian kolom identitas.


def clean_likert_responses(  # Mengonversi jawaban menjadi angka dan mencatat nilai yang tidak valid.
    data: pd.DataFrame,  # Tabel CSV yang telah dibaca.
    likert_min: float,  # Batas minimum nilai Skala Likert yang diperbolehkan.
    likert_max: float,  # Batas maksimum nilai Skala Likert yang diperbolehkan.
) -> tuple[pd.DataFrame, pd.DataFrame]:  # Mengembalikan jawaban numerik dan laporan missing.
    question_columns = [question for questions in CONSTRUCTS.values() for question in questions]  # Mengumpulkan 15 kolom indikator wajib.
    missing_columns = [question for question in question_columns if question not in data.columns]  # Mencari indikator wajib yang belum tersedia.
    if missing_columns:  # Memastikan setiap konstruk memiliki ketiga indikatornya.
        raise ValueError(f"Kolom pertanyaan wajib tidak ditemukan: {missing_columns}")  # Menampilkan indikator yang perlu ditambahkan.
    numeric_data = pd.DataFrame(index=data.index)  # Menyediakan tabel keluaran numerik dengan indeks respons yang sama.
    report_rows: list[dict[str, int | str]] = []  # Menyimpan hitungan missing dan jawaban tidak valid per butir.
    for question in question_columns:  # Memproses setiap kolom pertanyaan satu per satu.
        raw_values = data[question]  # Mengambil jawaban mentah dari kolom saat ini.
        text_values = raw_values.astype("string").str.strip()  # Menangani angka yang tersimpan sebagai teks dan spasi kosong.
        is_missing = (raw_values.isna() | text_values.eq("").fillna(False)).fillna(False)  # Mendeteksi NaN, sel kosong, dan teks berisi spasi.
        converted = pd.to_numeric(raw_values, errors="coerce")  # Mengonversi jawaban numerik dan menandai teks tak terbaca sebagai NaN.
        invalid_text = (~is_missing) & converted.isna()  # Menghitung jawaban nonkosong yang bukan angka.
        decimal_values = converted.notna() & converted.mod(1).ne(0)  # Menandai nilai pecahan karena instrumen memakai Likert integer.
        out_of_range = converted.notna() & ~converted.between(likert_min, likert_max)  # Menghitung angka di luar rentang Likert.
        invalid_values = is_missing | invalid_text | decimal_values | out_of_range  # Menggabungkan alasan nilai tidak boleh digunakan sebagai jawaban.
        numeric_data[question] = converted.mask(invalid_values)  # Mengubah jawaban tak valid menjadi missing.
        report_rows.append(  # Menambahkan ringkasan kualitas data untuk butir ini.
            {  # Membentuk satu baris laporan.
                "kolom": question,  # Menyimpan nama butir pertanyaan.
                "missing_asli": int(is_missing.sum()),  # Menghitung missing sebelum konversi.
                "jawaban_non_numerik": int(invalid_text.sum()),  # Menghitung jawaban nonangka.
                "jawaban_desimal": int(decimal_values.sum()),  # Menghitung jawaban pecahan yang tidak sesuai skala Likert integer.
                "di_luar_rentang_likert": int(out_of_range.sum()),  # Menghitung angka di luar rentang yang dipilih.
                "jumlah_respons_valid": int(numeric_data[question].notna().sum()),  # Menghitung jawaban valid pada indikator.
                "jumlah_nilai_unik_valid": int(numeric_data[question].nunique()),  # Mendeteksi indikator tanpa variasi yang dapat dianalisis.
                "missing_setelah_pembersihan": int(numeric_data[question].isna().sum()),  # Menghitung total missing setelah pembersihan.
                "persentase_missing_setelah_pembersihan": float(numeric_data[question].isna().mean() * 100),  # Menghitung proporsi missing setelah cleaning.
            }  # Mengakhiri baris laporan.
        )  # Menyelesaikan penambahan laporan butir.
    missing_report = pd.DataFrame(report_rows)  # Mengubah daftar laporan menjadi tabel.
    return numeric_data, missing_report  # Mengembalikan data bersih dan laporan kualitas data.


def build_respondent_quality_report(  # Menghitung missing dan tanda pemeriksaan per responden tanpa menghapus baris.
    data: pd.DataFrame,  # Tabel masukan mentah.
    numeric_data: pd.DataFrame,  # Indikator numerik setelah validasi.
    missing_warning_threshold: float,  # Ambang peringatan missing sebagai proporsi antara nol dan satu.
) -> pd.DataFrame:  # Mengembalikan laporan kualitas per baris responden.
    question_columns = list(numeric_data.columns)  # Mengambil indikator yang digunakan dalam pemeriksaan.
    valid_counts = numeric_data.notna().sum(axis=1)  # Menghitung jumlah jawaban Likert valid per responden.
    missing_counts = numeric_data.isna().sum(axis=1)  # Menghitung jumlah jawaban hilang/tidak valid setelah cleaning.
    report = pd.DataFrame(index=data.index)  # Menyediakan tabel laporan dengan satu baris per respons.
    id_columns = identifier_columns(data)  # Menemukan ID eksplisit bila tersedia.
    if id_columns:  # Mempertahankan ID untuk penelusuran responden.
        report[id_columns] = data[id_columns]  # Menyalin ID ke laporan kualitas.
    else:  # Memberi label baris bila input tidak punya ID.
        report["baris_responden"] = range(1, len(data) + 1)  # Menghasilkan nomor urut tanpa mengubah data sumber.
    report["jumlah_respons_valid"] = valid_counts  # Menyimpan jumlah item valid per responden.
    report["jumlah_missing_setelah_pembersihan"] = missing_counts  # Menyimpan jumlah item kosong/tidak valid per responden.
    report["persentase_missing_setelah_pembersihan"] = missing_counts.div(len(question_columns)).mul(100)  # Menghitung persentase missing dari 15 item.
    report["peringatan_missing_tinggi"] = missing_counts.div(len(question_columns)).ge(missing_warning_threshold)  # Menandai ambang screening tanpa eksklusi.
    report["peringatan_pola_jawaban_seragam"] = valid_counts.ge(2) & numeric_data.nunique(axis=1, dropna=True).eq(1)  # Menandai straight-lining tanpa menganggapnya otomatis invalid.
    report["peringatan_baris_jawaban_duplikat"] = data[question_columns].duplicated(keep=False)  # Menandai pola jawaban identik tanpa menghapus responden.
    if id_columns:  # Memeriksa ID yang berulang bila ID tersedia.
        report["peringatan_id_duplikat"] = data[id_columns].duplicated(keep=False)  # Menandai ID duplikat untuk pemeriksaan manual.
    return report.reset_index(drop=True)  # Mengembalikan satu baris laporan untuk setiap responden.


def save_cleaned_indicators(data: pd.DataFrame, numeric_data: pd.DataFrame, output_dir: Path) -> Path:  # Mengekspor indikator individual yang sudah dibersihkan.
    id_columns = identifier_columns(data)  # Memisahkan ID yang dikenal dari kolom indikator.
    export_data = pd.concat([data[id_columns], numeric_data], axis=1)  # Mempertahankan ID opsional dan semua indikator Q1-Q15.
    output_file = output_dir / "data_indikator_bersih.csv"  # Menentukan nama output untuk penggunaan lanjutan.
    export_data.to_csv(output_file, index=False, encoding="utf-8-sig", na_rep="")  # Menyimpan missing sebagai sel kosong tanpa imputasi.
    return output_file  # Mengembalikan lokasi output indikator bersih.


def calculate_construct_scores(numeric_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Menghitung total dan rata-rata tiga indikator per konstruk.
    mean_scores = pd.DataFrame(index=numeric_data.index)  # Menyediakan tabel skor rata-rata per responden.
    total_scores = pd.DataFrame(index=numeric_data.index)  # Menyediakan tabel skor total per responden.
    for construct, questions in CONSTRUCTS.items():  # Memproses lima konstruk dengan tepat tiga indikator masing-masing.
        responses = numeric_data[list(questions)]  # Mengambil jawaban untuk ketiga indikator konstruk.
        complete_responses = responses.notna().all(axis=1)  # Menandai responden yang menjawab ketiga indikator secara valid.
        mean_scores[construct] = responses.mean(axis=1).where(complete_responses)  # Menghitung rata-rata 1-5 hanya dari tiga jawaban valid.
        total_scores[f"{construct} - total"] = responses.sum(axis=1, min_count=len(questions))  # Menghitung total 3-15 dan membiarkan jawaban tidak lengkap kosong.
    return mean_scores, total_scores  # Mengembalikan skor rata-rata dan total.


def save_descriptive_statistics(scores: pd.DataFrame, output_dir: Path) -> pd.DataFrame:  # Menghitung statistik dan menyimpan tabel CSV.
    statistics = scores.agg(["mean", "std", "min", "max"]).T  # Menghitung rerata, simpangan baku sampel, minimum, dan maksimum.
    statistics.insert(0, "jumlah_respons_valid", scores.count())  # Menambahkan jumlah skor responden yang valid.
    statistics.index.name = "konstruk"  # Memberi nama indeks agar jelas saat disimpan sebagai CSV.
    statistics.to_csv(output_dir / "statistik_deskriptif_konstruk.csv", encoding="utf-8-sig")  # Menyimpan statistik dengan encoding ramah spreadsheet.
    return statistics  # Mengembalikan tabel statistik untuk ditampilkan.


def save_comprehensive_descriptive_statistics(  # Menyimpan statistik indikator dan konstruk dalam satu tabel.
    numeric_data: pd.DataFrame,  # Indikator Likert setelah cleaning.
    mean_scores: pd.DataFrame,  # Skor mean konstruk dengan aturan tiga indikator lengkap.
    likert_min: int,  # Batas integer minimum skala.
    likert_max: int,  # Batas integer maksimum skala.
    output_dir: Path,  # Folder untuk file keluaran.
) -> Path:  # Mengembalikan lokasi statistik gabungan.
    records: list[dict[str, int | float | str]] = []  # Menampung statistik per item dan per konstruk.
    frequency_columns = [f"frekuensi_{value}" for value in range(likert_min, likert_max + 1)]  # Menyediakan kolom frekuensi tiap kategori Likert.
    percentage_columns = [f"persentase_{value}" for value in range(likert_min, likert_max + 1)]  # Menyediakan persentase tiap kategori Likert.
    for question in numeric_data.columns:  # Menghitung statistik tiap indikator satu per satu.
        values = numeric_data[question].dropna()  # Menggunakan hanya jawaban valid untuk statistik.
        record: dict[str, int | float | str] = {  # Menyiapkan ringkasan dasar indikator.
            "tingkat": "indikator",  # Membedakan indikator dari skor konstruk.
            "variabel": question,  # Menamai indikator.
            "jumlah_respons_valid": int(values.count()),  # Menyimpan jumlah jawaban valid.
            "jumlah_missing": int(numeric_data[question].isna().sum()),  # Menyimpan jumlah missing setelah cleaning.
            "mean": float(values.mean()) if not values.empty else float("nan"),  # Menghitung mean atau NaN jika tidak ada nilai.
            "std": float(values.std()) if len(values) > 1 else float("nan"),  # Menghitung sample SD bila minimal dua jawaban tersedia.
            "min": float(values.min()) if not values.empty else float("nan"),  # Menghitung nilai minimum.
            "max": float(values.max()) if not values.empty else float("nan"),  # Menghitung nilai maksimum.
        }  # Menyelesaikan statistik dasar indikator.
        counts = values.value_counts().reindex(range(likert_min, likert_max + 1), fill_value=0)  # Menghitung frekuensi jawaban tiap kategori skala.
        for value in range(likert_min, likert_max + 1):  # Menambahkan frekuensi dan persentase per kategori.
            record[f"frekuensi_{value}"] = int(counts.loc[value])  # Menyimpan jumlah jawaban pada kategori.
            record[f"persentase_{value}"] = round(float(counts.loc[value] / len(values) * 100), 2) if len(values) else float("nan")  # Menyimpan persentase valid dengan dua desimal.
        records.append(record)  # Menambahkan ringkasan indikator.
    for construct in mean_scores.columns:  # Menghitung statistik skor mean per konstruk.
        values = mean_scores[construct].dropna()  # Menggunakan responden dengan tiga indikator lengkap.
        record = {  # Menyiapkan ringkasan konstruk.
            "tingkat": "konstruk",  # Membedakan konstruk dari indikator.
            "variabel": construct,  # Menamai konstruk.
            "jumlah_respons_valid": int(values.count()),  # Menyimpan jumlah skor konstruk valid.
            "jumlah_missing": int(mean_scores[construct].isna().sum()),  # Menyimpan jumlah skor konstruk yang tidak dapat dihitung.
            "mean": float(values.mean()) if not values.empty else float("nan"),  # Menghitung mean atau NaN jika tidak ada nilai.
            "std": float(values.std()) if len(values) > 1 else float("nan"),  # Menghitung sample SD bila minimal dua skor tersedia.
            "min": float(values.min()) if not values.empty else float("nan"),  # Menghitung minimum.
            "max": float(values.max()) if not values.empty else float("nan"),  # Menghitung maksimum.
        }  # Menyelesaikan statistik konstruk.
        record.update({column: float("nan") for column in frequency_columns + percentage_columns})  # Frekuensi kategori hanya berlaku pada indikator Likert.
        records.append(record)  # Menambahkan ringkasan konstruk.
    statistics = pd.DataFrame(records)  # Membentuk tabel statistik gabungan.
    output_file = output_dir / "statistik_deskriptif.csv"  # Menetapkan nama output baru untuk tabel gabungan.
    statistics.to_csv(output_file, index=False, encoding="utf-8-sig")  # Menyimpan tanpa indeks DataFrame.
    return output_file  # Mengembalikan path file statistik gabungan.


def save_processing_summary(row_count: int, output_dir: Path) -> Path:  # Mencatat jumlah baris tanpa eksklusi otomatis.
    summary = pd.DataFrame(  # Menyusun hitungan sampel untuk audit.
        [{"baris_awal": row_count, "baris_dipertahankan": row_count, "baris_dikeluarkan": 0, "kebijakan": "Tidak ada baris yang dihapus otomatis."}]  # Menegaskan tidak ada eksklusi.
    )  # Menyelesaikan tabel ringkasan.
    output_file = output_dir / "ringkasan_pemrosesan.csv"  # Menetapkan nama file ringkasan.
    summary.to_csv(output_file, index=False, encoding="utf-8-sig")  # Menyimpan tanpa indeks tambahan.
    return output_file  # Mengembalikan lokasi ringkasan.


def create_boxplot(scores: pd.DataFrame, output_dir: Path) -> Path:  # Membuat boxplot distribusi skor konstruk.
    plot_data = scores.melt(var_name="Konstruk", value_name="Skor rata-rata").dropna()  # Mengubah skor ke format panjang dan membuang skor kosong.
    if plot_data.empty:  # Memeriksa apakah ada skor yang dapat divisualisasikan.
        raise ValueError("Tidak ada skor konstruk valid untuk dibuat boxplot.")  # Menghindari gambar kosong yang menyesatkan.
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)  # Membuat kanvas dengan ukuran yang mudah dibaca.
    sns.boxplot(data=plot_data, x="Konstruk", y="Skor rata-rata", hue="Konstruk", palette="Set2", legend=False, dodge=False, ax=axis)  # Menggambar distribusi skor tanpa peringatan palette.
    axis.set_title("Distribusi Skor Rata-rata Konstruk", fontsize=15, weight="bold")  # Memberi judul yang informatif.
    axis.set_xlabel("Konstruk laten")  # Menamai sumbu horizontal.
    axis.set_ylabel("Skor rata-rata Likert")  # Menamai sumbu vertikal.
    axis.tick_params(axis="x", rotation=15)  # Memiringkan label agar nama konstruk tidak bertumpuk.
    output_file = output_dir / "boxplot_skor_konstruk.png"  # Menetapkan nama file gambar boxplot.
    figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan gambar dengan resolusi 300 DPI.
    plt.close(figure)  # Menutup kanvas untuk membebaskan memori.
    return output_file  # Mengembalikan lokasi gambar yang dibuat.


def create_correlation_heatmap(scores: pd.DataFrame, output_dir: Path) -> Path:  # Menggambar korelasi Pearson antar-konstruk.
    correlation = scores.corr(method="pearson", min_periods=2)  # Menghitung korelasi hanya jika sedikitnya dua pasangan skor tersedia.
    if not correlation.notna().to_numpy().any():  # Memastikan tersedia setidaknya satu korelasi yang dapat dihitung.
        raise ValueError("Korelasi tidak dapat dihitung: periksa jumlah respons valid dan variasi skor konstruk.")  # Menjelaskan kondisi data yang menghalangi heatmap.
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)  # Membuat kanvas heatmap yang proporsional.
    sns.heatmap(  # Menggambar matriks korelasi dengan skala warna yang terpusat pada nol.
        correlation,  # Menggunakan matriks korelasi konstruk.
        annot=True,  # Menampilkan nilai korelasi pada setiap sel.
        fmt=".2f",  # Membatasi angka anotasi hingga dua desimal.
        cmap="vlag",  # Memakai palet divergen untuk korelasi negatif dan positif.
        vmin=-1,  # Menetapkan batas bawah skala korelasi.
        vmax=1,  # Menetapkan batas atas skala korelasi.
        center=0,  # Menempatkan korelasi nol di tengah skala warna.
        square=True,  # Membuat sel heatmap berbentuk persegi.
        linewidths=0.5,  # Memberi pemisah tipis antarsel.
        cbar_kws={"label": "Korelasi Pearson"},  # Memberi label pada colorbar.
        ax=axis,  # Menggambar pada kanvas yang telah disiapkan.
    )  # Menyelesaikan heatmap.
    axis.set_title("Korelasi Antar-Konstruk Laten", fontsize=15, weight="bold")  # Memberi judul heatmap.
    output_file = output_dir / "heatmap_korelasi_konstruk.png"  # Menetapkan nama file gambar heatmap.
    figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan heatmap dengan resolusi 300 DPI.
    plt.close(figure)  # Menutup kanvas untuk membebaskan memori.
    return output_file  # Mengembalikan lokasi gambar yang dibuat.


def main() -> None:  # Menjalankan seluruh alur analisis dari CLI.
    args = parse_args()  # Membaca path masukan, folder keluaran, dan batas Likert.
    if args.likert_min >= args.likert_max:  # Memastikan rentang Likert memiliki urutan yang masuk akal.
        raise SystemExit("--likert-min harus lebih kecil daripada --likert-max.")  # Menghentikan proses dengan petunjuk yang jelas.
    if not 0 <= args.missing_warning_threshold <= 1:  # Memastikan ambang peringatan berupa proporsi yang valid.
        raise SystemExit("--missing-warning-threshold harus berada antara 0 dan 1.")  # Menghentikan proses dengan petunjuk yang jelas.
    try:  # Menangani kesalahan data agar pengguna menerima pesan yang dapat ditindaklanjuti.
        data = read_csv(args.csv_file)  # Membaca respons dari file CSV.
        numeric_data, missing_report = clean_likert_responses(data, args.likert_min, args.likert_max)  # Membersihkan dan memvalidasi jawaban.
        mean_scores, total_scores = calculate_construct_scores(numeric_data)  # Menghitung skor rata-rata dan total konstruk per responden.
        respondent_quality = build_respondent_quality_report(data, numeric_data, args.missing_warning_threshold)  # Merangkum missing dan tanda pemeriksaan tiap responden.
        args.output_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder keluaran bila belum tersedia.
        cleaned_indicators_file = save_cleaned_indicators(data, numeric_data, args.output_dir)  # Menyimpan seluruh indikator bersih untuk analisis lanjutan.
        missing_report.to_csv(args.output_dir / "laporan_missing_values.csv", index=False, encoding="utf-8-sig")  # Menyimpan laporan missing dan nilai tak valid.
        respondent_quality.to_csv(args.output_dir / "laporan_kualitas_responden.csv", index=False, encoding="utf-8-sig")  # Menyimpan missing dan flag pemeriksaan per responden.
        processing_summary_file = save_processing_summary(len(data), args.output_dir)  # Mencatat jumlah awal/dipertahankan/dikeluarkan.
        id_columns = identifier_columns(data)  # Mengambil ID responden bila tersedia.
        respondent_scores = pd.concat([data[id_columns], mean_scores.add_suffix(" - rata-rata"), total_scores], axis=1)  # Menggabungkan ID dan skor konstruk.
        if not id_columns:  # Menyediakan identifikasi baris bila CSV tidak memuat kolom ID.
            respondent_scores.insert(0, "baris_responden", range(1, len(data) + 1))  # Memberi nomor urut satu-based untuk tiap baris.
        respondent_scores.to_csv(args.output_dir / "skor_konstruk_per_responden.csv", index=False, encoding="utf-8-sig")  # Menyimpan rekap tanpa indeks pandas.
        statistics = save_descriptive_statistics(mean_scores, args.output_dir)  # Menghitung statistik deskriptif dari skor rata-rata.
        comprehensive_statistics_file = save_comprehensive_descriptive_statistics(numeric_data, mean_scores, args.likert_min, args.likert_max, args.output_dir)  # Menyimpan statistik indikator dan konstruk.
        try:  # Mengizinkan pipeline data tetap selesai ketika grafik tidak dapat dihitung.
            boxplot_file = create_boxplot(mean_scores, args.output_dir)  # Menghasilkan boxplot skor rata-rata konstruk.
        except ValueError as error:  # Menangani kondisi tanpa skor konstruk valid.
            boxplot_file = None  # Menandai boxplot tidak tersedia.
            boxplot_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
        try:  # Mengizinkan pipeline data tetap selesai ketika korelasi tidak terdefinisi.
            heatmap_file = create_correlation_heatmap(mean_scores, args.output_dir)  # Menghasilkan heatmap korelasi skor rata-rata.
        except ValueError as error:  # Menangani korelasi yang tidak dapat dihitung.
            heatmap_file = None  # Menandai heatmap tidak tersedia.
            heatmap_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
    except (FileNotFoundError, OSError, ValueError) as error:  # Menangkap kesalahan file, konfigurasi, atau kualitas data.
        raise SystemExit(f"Analisis gagal: {error}") from error  # Mengakhiri program dengan pesan ringkas.
    print("Statistik deskriptif skor konstruk:")  # Menampilkan ringkasan statistik ke terminal.
    print(statistics.to_string(float_format=lambda value: f"{value:.3f}"))  # Memformat angka agar mudah dibaca.
    print(f"\nBaris awal/dipertahankan/dikeluarkan: {len(data)}/{len(data)}/0")  # Menegaskan semua baris dipertahankan.
    unused_columns = [column for column in data.columns if column not in numeric_data.columns and column not in identifier_columns(data)]  # Menemukan kolom tambahan yang tidak dianalisis.
    if unused_columns:  # Memberi peringatan atas kolom tambahan tanpa menghapusnya dari input.
        print(f"Kolom tambahan tidak dianalisis (tetap ada di sumber): {unused_columns}")  # Menampilkan kolom yang tidak dipakai.
    print(f"Responden dengan missing tinggi: {int(respondent_quality['peringatan_missing_tinggi'].sum())}")  # Merangkum flag screening missing.
    print(f"Pola jawaban seragam: {int(respondent_quality['peringatan_pola_jawaban_seragam'].sum())}")  # Merangkum straight-lining flag.
    print(f"Baris jawaban duplikat: {int(respondent_quality['peringatan_baris_jawaban_duplikat'].sum())}")  # Merangkum pola jawaban duplikat.
    print(f"\nLaporan missing: {args.output_dir / 'laporan_missing_values.csv'}")  # Menampilkan lokasi laporan kualitas data.
    print(f"Kualitas responden: {args.output_dir / 'laporan_kualitas_responden.csv'}")  # Menampilkan laporan kualitas per responden.
    print(f"Ringkasan pemrosesan: {processing_summary_file}")  # Menampilkan jumlah baris input/output.
    print(f"Indikator bersih: {cleaned_indicators_file}")  # Menampilkan lokasi data indikator untuk SmartPLS.
    print(f"Statistik indikator+konstruk: {comprehensive_statistics_file}")  # Menampilkan tabel statistik lengkap.
    print(f"Boxplot: {boxplot_file or boxplot_warning}")  # Menampilkan grafik atau alasan tidak dibuat.
    print(f"Heatmap: {heatmap_file or heatmap_warning}")  # Menampilkan grafik atau alasan tidak dibuat.
    print(f"Statistik CSV: {args.output_dir / 'statistik_deskriptif_konstruk.csv'}")  # Menampilkan lokasi tabel statistik.
    print(f"Skor per responden: {args.output_dir / 'skor_konstruk_per_responden.csv'}")  # Menampilkan lokasi rekap total dan rata-rata.


if __name__ == "__main__":  # Memastikan main hanya dijalankan ketika skrip dieksekusi langsung.
    main()  # Memulai proses analisis.