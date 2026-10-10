from __future__ import annotations  # Mengaktifkan evaluasi anotasi tipe yang lebih fleksibel.

import argparse  # Menyediakan antarmuka argumen baris perintah.
import csv  # Mendeteksi delimiter dan memvalidasi header mentah sebelum pandas memodifikasinya.
from itertools import combinations_with_replacement  # Menghitung jumlah pasangan valid untuk korelasi deskriptif.
from pathlib import Path  # Menangani lokasi file dengan aman lintas sistem operasi.

import matplotlib  # Memilih backend headless sebelum pyplot diimpor.
matplotlib.use("Agg")  # CLI hanya menulis file gambar dan tidak memerlukan GUI/Tcl-Tk.
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
MAX_RESPONDENTS = 150  # Batas jumlah baris respons nonkosong yang dapat diproses.


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
        normalized_header_keys = ["".join(character for character in header.casefold() if character.isalnum()) for header in normalized_headers]  # Menyamakan kapitalisasi, spasi, dan tanda baca.
        duplicated_headers = pd.Index(normalized_headers)[pd.Index(normalized_header_keys).duplicated()].unique().tolist()  # Menangkap nama header duplikat setelah normalisasi.
        if duplicated_headers:  # Memastikan setiap indikator memiliki satu kolom yang tidak ambigu.
            raise ValueError(f"Nama kolom duplikat ditemukan setelah normalisasi: {duplicated_headers}")  # Menjelaskan kolom yang bentrok.
        with csv_file.open("r", encoding=encoding, newline="") as file:  # Memeriksa jumlah field sebelum pandas mengisi field yang hilang.
            reader = csv.reader(file, delimiter=delimiter)  # Membaca record CSV dengan aturan quote yang sama.
            next(reader, None)  # Melewati header yang sudah diperiksa.
            respondent_count = 0  # Menghitung record dengan setidaknya satu field terisi.
            for row in reader:  # Memeriksa semua record data.
                if row and len(row) != len(raw_headers):  # Mengizinkan baris fisik kosong tetapi menolak record yang lebarnya salah.
                    raise ValueError(f"Baris CSV {reader.line_num} memiliki {len(row)} kolom; header memiliki {len(raw_headers)} kolom.")  # Menjelaskan lokasi dan lebar record.
                if row and any(value.strip() for value in row):  # Baris kosong sepenuhnya tidak dihitung sebagai responden.
                    respondent_count += 1  # Mempertahankan setiap record respons hingga batas.
                    if respondent_count > MAX_RESPONDENTS:  # Menghentikan input sebelum memuat dataset berlebih.
                        raise ValueError(f"Jumlah responden melebihi batas maksimum {MAX_RESPONDENTS}; file memiliki setidaknya {respondent_count} responden. Data tidak dipotong.")  # Menjelaskan batas dan aturan tanpa eksklusi diam-diam.
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


def nonempty_response_rows(data: pd.DataFrame) -> pd.Series:  # Menandai baris dengan setidaknya satu sel terisi.
    if data.empty:  # Menjaga tipe mask konsisten untuk file tanpa baris.
        return pd.Series(False, index=data.index, dtype=bool)  # Tidak ada baris respons pada DataFrame kosong.
    populated_cells = data.astype("string").fillna("").apply(lambda column: column.str.strip().ne(""))  # Memperlakukan null dan whitespace sebagai kosong.
    return populated_cells.any(axis=1)  # Baris seluruhnya kosong tetap dicatat terpisah dari respons.


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
        non_finite = converted.isin([float("inf"), float("-inf")])  # Menandai infinity sebagai angka yang tidak berada pada rentang Likert.
        decimal_values = converted.notna() & ~non_finite & converted.mod(1).ne(0)  # Menandai nilai pecahan finite karena instrumen memakai Likert integer.
        out_of_range = converted.notna() & ~converted.between(likert_min, likert_max) & ~decimal_values  # Menghitung hanya integer di luar rentang agar kategori tidak tumpang tindih.
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
    has_response = nonempty_response_rows(data)  # Membedakan baris kosong dari baris respons.
    report = pd.DataFrame(index=data.index)  # Menyediakan tabel laporan dengan satu baris per respons.
    id_columns, has_id, missing_id, partial_id, duplicate_ids, _ = identifier_summary(data)  # Mengambil semua status ID dari aturan yang sama.
    if id_columns:  # Mempertahankan ID untuk penelusuran responden.
        report[id_columns] = data[id_columns]  # Menyalin ID ke laporan kualitas.
    else:  # Memberi label baris bila input tidak punya ID.
        report["baris_responden"] = range(1, len(data) + 1)  # Menghasilkan nomor urut tanpa mengubah data sumber.
    report["jumlah_respons_valid"] = valid_counts  # Menyimpan jumlah item valid per responden.
    report["jumlah_missing_setelah_pembersihan"] = missing_counts  # Menyimpan jumlah item kosong/tidak valid per responden.
    report["persentase_missing_setelah_pembersihan"] = missing_counts.div(len(question_columns)).mul(100)  # Menghitung persentase missing dari 15 item.
    report["baris_kosong_sepenuhnya"] = ~has_response  # Menandai baris tanpa isi di semua kolom input.
    report["peringatan_missing_tinggi"] = has_response & missing_counts.div(len(question_columns)).ge(missing_warning_threshold)  # Menandai ambang hanya pada baris respons.
    report["peringatan_pola_jawaban_seragam"] = valid_counts.ge(2) & numeric_data.nunique(axis=1, dropna=True).eq(1)  # Menandai straight-lining tanpa menganggapnya otomatis invalid.
    report["peringatan_baris_jawaban_duplikat"] = has_response & data[question_columns].duplicated(keep=False)  # Mengabaikan baris kosong dari pemeriksaan pola duplicate.
    if id_columns:  # Menambahkan flag identitas hanya bila kolom ID dikenali.
        report["peringatan_id_kosong"] = missing_id  # Menandai ID tunggal kosong atau semua komponen kosong.
        report["peringatan_id_tidak_lengkap"] = partial_id  # Menandai composite ID dengan komponen kosong.
        report["peringatan_id_duplikat"] = duplicate_ids  # Menandai ID duplikat untuk pemeriksaan manual.
    return report.reset_index(drop=True)  # Mengembalikan satu baris laporan untuk setiap responden.


def identifier_summary(  # Merangkum ID kosong, composite ID lengkap, duplikat, dan unik.
    data: pd.DataFrame,  # Data masukan dengan kolom ID opsional.
) -> tuple[list[str], pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None, int | None]:  # Mengembalikan kolom dan mask ID.
    id_columns = identifier_columns(data)  # Mencari nama kolom ID yang dikenal.
    if not id_columns:  # Menghindari klaim keunikan bila tidak ada kolom identitas.
        return [], None, None, None, None, None  # Menandai seluruh metrik ID sebagai tidak tersedia.
    id_values = data[id_columns].astype("string").fillna("").apply(lambda column: column.str.strip())  # Menormalkan sel ID untuk mengabaikan spasi tepi.
    filled_components = id_values.ne("")  # Menandai komponen composite ID yang terisi.
    has_id = filled_components.all(axis=1)  # ID berlaku hanya jika satu-satunya/semua komponennya terisi.
    has_response = nonempty_response_rows(data)  # Mencegah baris kosong sepenuhnya dihitung sebagai responden tanpa ID.
    missing_id = has_response & ~filled_components.any(axis=1)  # Menandai record respons tanpa komponen ID terisi.
    partial_id = filled_components.any(axis=1) & ~has_id  # Menandai composite ID yang hanya terisi sebagian.
    id_keys = id_values.apply(lambda row: tuple(row.tolist()), axis=1)  # Membentuk kunci komposit bila beberapa kolom ID tersedia.
    duplicate_rows = has_id & id_keys.duplicated(keep=False)  # Menguji duplikasi hanya pada ID yang lengkap.
    unique_count = int(id_keys[has_id].nunique())  # Menghitung ID unik nonkosong.
    return id_columns, has_id, missing_id, partial_id, duplicate_rows, unique_count  # Mengembalikan metrik identitas tanpa mengeksklusi baris.


def duplicate_identifier_rows(data: pd.DataFrame) -> pd.Series | None:  # Mengembalikan flag baris dengan ID duplikat bila tersedia.
    return identifier_summary(data)[4]  # Menggunakan satu implementasi aturan ID untuk semua laporan.


def calculate_correlation_analysis(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Menghasilkan korelasi dan pairwise N dengan aturan seragam.
    correlation = scores.corr(method="pearson", min_periods=2)  # Pearson hanya dihitung dari minimal dua pasangan valid.
    pairwise_counts = pd.DataFrame(0, index=scores.columns, columns=scores.columns, dtype="int64")  # Menyiapkan matriks jumlah pasangan valid.
    for first, second in combinations_with_replacement(scores.columns, 2):  # Menghitung setiap pasangan sekali.
        pair_count = int(scores[[first, second]].dropna().shape[0])  # Menggunakan observasi yang valid pada kedua konstruk.
        pairwise_counts.loc[first, second] = pair_count  # Menyimpan jumlah pasangan di bagian atas matriks.
        pairwise_counts.loc[second, first] = pair_count  # Mencerminkan jumlah pasangan agar matriks simetris.
    return correlation, pairwise_counts  # Mengembalikan nilai korelasi aktual dan N pasangan.


def build_data_quality_summary(  # Menghitung agregat kualitas dengan total invalid berdasarkan sel setelah cleaning.
    data: pd.DataFrame,  # Data CSV yang telah dibaca.
    numeric_data: pd.DataFrame,  # Jawaban Likert setelah cleaning.
    indicator_report: pd.DataFrame,  # Hitungan kualitas per indikator.
    respondent_report: pd.DataFrame,  # Flag kualitas per responden.
    missing_warning_threshold: float,  # Ambang peringatan missing responden.
) -> pd.DataFrame:  # Mengembalikan laporan kualitas berbentuk kategori/metrik/nilai/keterangan.
    question_count = len(numeric_data.columns)  # Menghitung indikator wajib yang benar-benar dianalisis.
    row_count = len(data)  # Menghitung baris respons yang dipertahankan.
    has_response = nonempty_response_rows(data)  # Membedakan baris kosong dari respons yang terisi.
    blank_row_count = int((~has_response).sum())  # Menghitung baris yang seluruh kolomnya kosong.
    nonempty_row_count = int(has_response.sum())  # Menghitung baris dengan setidaknya satu nilai terisi.
    total_cells = row_count * question_count  # Menghitung seluruh sel indikator yang diperiksa.
    valid_cells = int(numeric_data.notna().to_numpy().sum())  # Menghitung sel yang lolos seluruh validasi.
    original_missing = int(indicator_report["missing_asli"].sum())  # Menjumlahkan sel kosong asli.
    nonnumeric_cells = int(indicator_report["jawaban_non_numerik"].sum())  # Menjumlahkan respons nonnumerik.
    decimal_cells = int(indicator_report["jawaban_desimal"].sum())  # Menjumlahkan angka pecahan yang tidak sesuai skala.
    out_of_range_cells = int(indicator_report["di_luar_rentang_likert"].sum())  # Menjumlahkan integer di luar rentang Likert.
    invalid_cells = int(numeric_data.isna().to_numpy().sum())  # Menghitung sel invalid sekali dari hasil cleaning, bukan penjumlahan kategori.
    complete_respondents = int((numeric_data.notna().all(axis=1) & has_response).sum())  # Menghitung respons terisi dengan semua indikator valid.
    incomplete_respondents = int((~numeric_data.notna().all(axis=1) & has_response).sum())  # Menghitung respons terisi dengan sedikitnya satu indikator invalid.
    high_missing_respondents = int(respondent_report["peringatan_missing_tinggi"].sum())  # Menghitung baris yang melewati ambang screening.
    no_variation_indicators = int(indicator_report["jumlah_nilai_unik_valid"].le(1).sum())  # Menghitung item kosong total atau hanya satu nilai unik.
    straightline_respondents = int(respondent_report["peringatan_pola_jawaban_seragam"].sum())  # Menghitung flag pola seragam.
    duplicate_answer_rows = int(respondent_report["peringatan_baris_jawaban_duplikat"].sum())  # Menghitung semua baris yang termasuk kelompok pola duplikat.
    duplicate_ids = int(respondent_report["peringatan_id_duplikat"].sum()) if "peringatan_id_duplikat" in respondent_report else None  # Menghitung baris ID duplikat atau menandainya tidak tersedia.
    missing_ids = int(respondent_report["peringatan_id_kosong"].sum()) if "peringatan_id_kosong" in respondent_report else None  # Menghitung ID kosong bila kolom tersedia.
    partial_ids = int(respondent_report["peringatan_id_tidak_lengkap"].sum()) if "peringatan_id_tidak_lengkap" in respondent_report else None  # Menghitung ID komposit parsial bila tersedia.
    def cell_percentage(count: int) -> float | None:  # Menghitung proporsi sel tanpa membagi nol.
        return round(count / total_cells * 100, 2) if total_cells else None  # Seluruh sel indikator menjadi penyebut.

    metric_rows: list[dict[str, int | float | str | None]] = []  # Menyiapkan tabel ringkasan tanpa angka responden sintetis.

    def add_metric(metric: str, value: int | float | str | None, note: str) -> None:  # Menambahkan satu baris metrik ke ringkasan.
        metric_rows.append({"kategori": "ringkasan", "metrik": metric, "nilai": value, "keterangan": note})  # Menjaga format laporan konsisten.

    add_metric("jumlah_baris_respons_awal", row_count, "Jumlah baris data yang dibaca; semua baris dipertahankan.")  # Mencatat jumlah responden/baris awal.
    add_metric("jumlah_baris_kosong_sepenuhnya", blank_row_count, "Baris tanpa nilai pada seluruh kolom; tetap dipertahankan dan tidak dihitung sebagai responden incomplete.")  # Mencatat blank rows.
    add_metric("jumlah_baris_respons_non_kosong", nonempty_row_count, "Baris dengan setidaknya satu sel terisi, termasuk ID atau metadata.")  # Mencatat nonempty rows.
    add_metric("jumlah_indikator_wajib", question_count, "Indikator Q1-Q15 yang divalidasi.")  # Mencatat banyak indikator wajib.
    add_metric("jumlah_sel_jawaban_diperiksa", total_cells, "Jumlah baris awal dikali jumlah indikator wajib.")  # Mencatat penyebut seluruh sel.
    add_metric("jumlah_sel_valid", valid_cells, "Sel integer dalam rentang Likert yang lolos cleaning.")  # Mencatat sel valid.
    add_metric("persentase_sel_valid", cell_percentage(valid_cells), "Jumlah sel valid dibagi seluruh sel indikator, dikali 100; kosong jika tidak ada sel.")  # Mencatat persentase valid.
    add_metric("jumlah_sel_missing_asli", original_missing, "Sel kosong pada data masukan sebelum cleaning.")  # Mencatat missing asli.
    add_metric("persentase_sel_missing_asli", cell_percentage(original_missing), "Missing asli dibagi seluruh sel indikator, dikali 100.")  # Mencatat persentase missing asli.
    add_metric("jumlah_sel_nonnumerik", nonnumeric_cells, "Sel nonkosong yang tidak dapat dikonversi ke angka.")  # Mencatat teks invalid.
    add_metric("persentase_sel_nonnumerik", cell_percentage(nonnumeric_cells), "Sel nonnumerik dibagi seluruh sel indikator, dikali 100.")  # Mencatat persentase nonnumerik.
    add_metric("jumlah_sel_desimal_tidak_valid", decimal_cells, "Sel numerik pecahan; kategori ini terpisah dari integer di luar rentang.")  # Mencatat pecahan invalid.
    add_metric("persentase_sel_desimal_tidak_valid", cell_percentage(decimal_cells), "Sel desimal invalid dibagi seluruh sel indikator, dikali 100.")  # Mencatat persentase desimal.
    add_metric("jumlah_sel_di_luar_rentang", out_of_range_cells, "Integer yang berada di luar batas Likert.")  # Mencatat out of range secara eksklusif.
    add_metric("persentase_sel_di_luar_rentang", cell_percentage(out_of_range_cells), "Integer di luar rentang dibagi seluruh sel indikator, dikali 100.")  # Mencatat persentase out-of-range.
    add_metric("jumlah_sel_tidak_valid_setelah_pembersihan", invalid_cells, "Jumlah sel kosong pada matriks bersih; setiap sel dihitung satu kali.")  # Mencatat total invalid tanpa double count.
    add_metric("persentase_sel_tidak_valid", cell_percentage(invalid_cells), "Total sel invalid dibagi seluruh sel indikator, dikali 100.")  # Mencatat persentase invalid.
    add_metric("jumlah_responden_lengkap", complete_respondents, "Baris dengan seluruh indikator valid.")  # Mencatat complete cases.
    add_metric("jumlah_responden_tidak_lengkap", incomplete_respondents, "Baris dengan setidaknya satu indikator missing/tidak valid.")  # Mencatat incomplete cases.
    add_metric("jumlah_responden_peringatan_missing", high_missing_respondents, f"Proporsi missing >= {missing_warning_threshold:.2f}; flag tidak menghapus baris.")  # Mencatat responden yang melewati ambang.
    add_metric("jumlah_indikator_tanpa_variasi", no_variation_indicators, "Indikator dengan nol atau satu nilai unik valid.")  # Mencatat item tanpa variasi.
    add_metric("jumlah_responden_pola_jawaban_seragam", straightline_respondents, "Responden dengan minimal dua jawaban valid dan hanya satu nilai unik.")  # Mencatat straight-line flags.
    add_metric("jumlah_baris_pola_jawaban_duplikat", duplicate_answer_rows, "Jumlah baris yang terlibat dalam pola Q1-Q15 identik.")  # Mencatat baris duplikat.
    add_metric("jumlah_responden_id_kosong", missing_ids, "Semua kolom ID kosong; kosong bila tidak ada kolom ID.")  # Mencatat ID kosong.
    add_metric("jumlah_responden_id_tidak_lengkap", partial_ids, "Sebagian komponen ID komposit kosong; tidak dipakai untuk uji keunikan.")  # Mencatat ID parsial.
    add_metric("jumlah_baris_terlibat_id_duplikat", duplicate_ids, "Jumlah baris dengan ID komposit nonkosong berulang; kosong bila ID tidak tersedia.")  # Mencatat ID duplikat.

    missing_or_invalid = original_missing + nonnumeric_cells + decimal_cells + out_of_range_cells > 0  # Menentukan masalah format/nilai dari kategori eksklusif.
    respondent_flags = high_missing_respondents + straightline_respondents + duplicate_answer_rows > 0 or (duplicate_ids or 0) > 0 or (missing_ids or 0) > 0 or (partial_ids or 0) > 0  # Menentukan sinyal pemeriksaan manual.
    statuses = []  # Mengumpulkan status berbasis hasil pemeriksaan aktual.
    if missing_or_invalid:  # Menandai missing atau nilai invalid.
        statuses.append("Terdapat data hilang atau jawaban tidak valid")  # Menjelaskan kondisi data yang ditemukan.
    if no_variation_indicators:  # Menandai item tanpa variasi.
        statuses.append("Terdapat indikator tanpa variasi")  # Menjelaskan kondisi variasi.
    if respondent_flags:  # Menandai pola responden yang perlu dilihat manual.
        statuses.append("Terdapat pola responden untuk ditinjau manual")  # Menghindari tuduhan terhadap responden.
    if total_cells == 0 or nonempty_row_count == 0:  # Menghindari status tanpa temuan saat tidak ada respons untuk dinilai.
        statuses.insert(0, "Tidak ada respons nonkosong untuk diperiksa")  # Menyatakan keterbatasan pemeriksaan secara eksplisit.
    final_status = "; ".join(statuses) if statuses else "Tidak ditemukan masalah format atau nilai indikator"  # Menetapkan status ringkas yang tidak mengklaim validitas instrumen.
    add_metric("status_kualitas_data", final_status, "Flag adalah sinyal pemeriksaan, bukan dasar otomatis menghapus responden atau menyatakan instrumen valid.")  # Mencatat status akhir.
    return pd.DataFrame(metric_rows)  # Mengembalikan laporan tabular.


def build_sample_size_report(  # Menghitung sampel awal, ID, kelengkapan, dan jumlah analisis yang benar-benar tersedia.
    data: pd.DataFrame,  # Data masukan.
    numeric_data: pd.DataFrame,  # Matriks indikator bersih.
    mean_scores: pd.DataFrame,  # Skor mean konstruk dengan complete-three-item rule.
    respondent_report: pd.DataFrame,  # Flag missing per responden.
    correlation: pd.DataFrame | None = None,  # Matriks korelasi yang benar-benar dipakai heatmap.
    pairwise_counts: pd.DataFrame | None = None,  # Jumlah pasangan valid untuk setiap sel korelasi.
) -> pd.DataFrame:  # Mengembalikan laporan jumlah sampel long-form.
    if correlation is None or pairwise_counts is None:  # Menjaga penggunaan fungsi lama tanpa menduplikasi aturan korelasi.
        correlation, pairwise_counts = calculate_correlation_analysis(mean_scores)  # Menghitung matriks dan pairwise N dengan aturan yang sama.
    id_columns, has_id, missing_ids, partial_ids, duplicate_ids, unique_id_count = identifier_summary(data)  # Mengambil metrik ID dengan aturan composite eksplisit.
    has_response = nonempty_response_rows(data)  # Memisahkan baris kosong sepenuhnya dari respons.
    complete_mask = numeric_data.notna().all(axis=1) & has_response  # Mendefinisikan respons lengkap di seluruh Q1-Q15.
    incomplete_mask = ~numeric_data.notna().all(axis=1) & has_response  # Baris kosong penuh dilaporkan terpisah.
    metric_rows: list[dict[str, int | float | str | None]] = []  # Menyimpan nilai atau None bila tidak tersedia.

    def add_metric(metric: str, value: int | float | str | None, note: str) -> None:  # Menambahkan metrik jumlah sampel.
        metric_rows.append({"metrik": metric, "nilai": value, "keterangan": note})  # Menjaga kolom output konsisten.

    add_metric("jumlah_baris_awal", len(data), "Baris respons hasil parsing; baris kosong fisik yang terbaca tetap dihitung.")  # Mencatat N awal.
    add_metric("jumlah_baris_kosong_sepenuhnya", int((~has_response).sum()), "Baris kosong dipertahankan; tidak dianggap sebagai responden.")  # Mencatat baris kosong.
    add_metric("jumlah_baris_respons_non_kosong", int(has_response.sum()), "Baris dengan setidaknya satu nilai input terisi.")  # Mencatat record respons.
    add_metric("jumlah_responden_dengan_id", int(has_id.sum()) if has_id is not None else None, "Baris dengan ID terisi; tidak tersedia bila tidak ada kolom ID yang dikenali.")  # Mencatat ID nonkosong.
    add_metric("jumlah_responden_id_kosong", int(missing_ids.sum()) if missing_ids is not None else None, "Semua komponen ID kosong; tidak tersedia bila kolom ID tidak ada.")  # Mencatat ID kosong.
    add_metric("jumlah_responden_id_tidak_lengkap", int(partial_ids.sum()) if partial_ids is not None else None, "Composite ID dengan sebagian komponen kosong; tidak tersedia bila ID tidak ada.")  # Mencatat ID parsial.
    add_metric("jumlah_id_unik", unique_id_count, "ID komposit unik nonkosong; tidak membuktikan responden unik di luar kolom ID.")  # Mencatat unique ID.
    add_metric("jumlah_id_duplikat", int(duplicate_ids.sum()) if duplicate_ids is not None else None, "Jumlah baris yang terlibat dalam ID duplikat; tidak otomatis dihapus.")  # Mencatat duplicate IDs.
    add_metric("jumlah_responden_lengkap", int(complete_mask.sum()), "Baris dengan 15 indikator valid.")  # Mencatat full-item complete cases.
    add_metric("jumlah_responden_tidak_lengkap", int(incomplete_mask.sum()), "Baris nonkosong dengan setidaknya satu indikator invalid setelah cleaning; baris kosong dihitung terpisah.")  # Mencatat incomplete cases.
    add_metric("jumlah_responden_peringatan_missing", int(respondent_report["peringatan_missing_tinggi"].sum()), "Baris yang memenuhi ambang missing-warning; tidak otomatis dikeluarkan.")  # Mencatat missing warning.
    add_metric("jumlah_baris_dipertahankan", len(data), "Tidak ada kebijakan eksklusi otomatis di pipeline.")  # Mencatat retained rows.
    add_metric("jumlah_baris_dikeluarkan", 0, "Tidak ada baris yang dikeluarkan otomatis.")  # Mencatat excluded rows.
    for construct in mean_scores.columns:  # Melaporkan jumlah skor valid secara terpisah per konstruk.
        construct_count = int(mean_scores[construct].count())  # Menghitung skor konstruk yang tersedia.
        add_metric("jumlah_responden_untuk_skor_konstruk", construct_count, construct)  # Mencatat valid N tiap konstruk.
        add_metric("jumlah_responden_untuk_analisis", construct_count, f"Statistik deskriptif skor konstruk: {construct}; ketiga indikator lengkap.")  # Mencatat N konstruk yang benar-benar digunakan.
    for indicator in numeric_data.columns:  # Melaporkan N statistik deskriptif setiap indikator.
        add_metric("jumlah_responden_untuk_analisis", int(numeric_data[indicator].count()), f"Statistik deskriptif indikator: {indicator}.")  # Mencatat N item.
    for first, second in combinations_with_replacement(mean_scores.columns, 2):  # Menghitung N pairwise untuk setiap sel korelasi.
        pair_count = int(pairwise_counts.loc[first, second])  # Menggunakan pairwise N dari objek yang juga digunakan untuk korelasi.
        correlation_value = correlation.loc[first, second]  # Mengambil status numerik korelasi aktual.
        detail = f"Korelasi Pearson deskriptif: {first} dengan {second}; pairwise complete."  # Menjelaskan pasangan variabel.
        add_metric("jumlah_responden_untuk_analisis", pair_count, detail)  # Mencatat N pasangan valid.
        if pd.notna(correlation_value):  # Status berhasil hanya jika korelasi aktual berupa angka.
            correlation_status = "dapat_dihitung"  # Menandai korelasi yang terdefinisi.
            status_detail = f"{detail} Korelasi Pearson tersedia."  # Menjelaskan hasil status.
        elif pair_count < 2:  # Korelasi Pearson memerlukan minimal dua pasangan.
            correlation_status = "tidak_terdefinisi"  # Menandai N terlalu kecil.
            status_detail = f"{detail} N pasangan kurang dari dua."  # Menjelaskan penyebab korelasi kosong.
        else:  # Pada N memadai, korelasi kosong terjadi karena salah satu skor tidak bervariasi.
            correlation_status = "tidak_terdefinisi"  # Menandai variabel konstan.
            status_detail = f"{detail} Salah satu konstruk tidak memiliki variasi."  # Menjelaskan penyebab korelasi kosong.
        add_metric("status_korelasi_pearson", correlation_status, status_detail)  # Membedakan pair count dari status korelasi.
    add_metric("jumlah_responden_untuk_analisis", None, "Estimasi SmartPLS/PLS-SEM tidak dijalankan oleh program Python ini.")  # Mencegah klaim N SmartPLS yang tidak diverifikasi.
    return pd.DataFrame(metric_rows)  # Mengembalikan laporan jumlah sampel.


def save_quality_summary(summary: pd.DataFrame, output_dir: Path) -> Path:  # Menyimpan agregat kualitas data dalam format spreadsheet.
    output_file = output_dir / "laporan_kualitas_data.csv"  # Menetapkan nama laporan kualitas agregat.
    summary.to_csv(output_file, index=False, encoding="utf-8-sig")  # Menyimpan kategori/metrik/nilai/keterangan tanpa indeks.
    return output_file  # Mengembalikan lokasi file output.


def save_sample_size_report(report: pd.DataFrame, output_dir: Path) -> Path:  # Menyimpan laporan jumlah sampel dan N analisis.
    output_file = output_dir / "laporan_jumlah_sampel.csv"  # Menetapkan nama laporan jumlah sampel.
    report.to_csv(output_file, index=False, encoding="utf-8-sig")  # Menyimpan metrik tanpa indeks DataFrame.
    return output_file  # Mengembalikan lokasi file output.


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


def save_processing_summary(  # Mencatat jumlah baris tanpa eksklusi otomatis.
    row_count: int,  # Jumlah record hasil parsing.
    output_dir: Path,  # Folder output.
    blank_row_count: int = 0,  # Jumlah baris yang seluruh kolomnya kosong.
    nonempty_row_count: int | None = None,  # Jumlah baris yang memuat setidaknya satu nilai.
) -> Path:  # Mengembalikan lokasi ringkasan.
    summary = pd.DataFrame(  # Menyusun hitungan sampel untuk audit.
        [{"baris_awal": row_count, "baris_kosong_sepenuhnya": blank_row_count, "baris_respons_non_kosong": nonempty_row_count if nonempty_row_count is not None else row_count - blank_row_count, "baris_dipertahankan": row_count, "baris_dikeluarkan": 0, "kebijakan": "Semua baris dipertahankan; baris sepenuhnya kosong dilaporkan terpisah."}]  # Menegaskan tidak ada eksklusi.
    )  # Menyelesaikan tabel ringkasan.
    output_file = output_dir / "ringkasan_pemrosesan.csv"  # Menetapkan nama file ringkasan.
    summary.to_csv(output_file, index=False, encoding="utf-8-sig")  # Menyimpan tanpa indeks tambahan.
    return output_file  # Mengembalikan lokasi ringkasan.


def create_boxplot(scores: pd.DataFrame, output_dir: Path) -> Path:  # Membuat boxplot distribusi skor konstruk.
    plot_data = scores.melt(var_name="Konstruk", value_name="Skor rata-rata").dropna()  # Mengubah skor ke format panjang dan membuang skor kosong.
    if plot_data.empty:  # Memeriksa apakah ada skor yang dapat divisualisasikan.
        raise ValueError("Tidak ada skor konstruk valid untuk dibuat boxplot.")  # Menghindari gambar kosong yang menyesatkan.
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)  # Membuat kanvas dengan ukuran yang mudah dibaca.
    try:  # Memastikan figure selalu ditutup termasuk ketika plot atau save gagal.
        sns.boxplot(data=plot_data, x="Konstruk", y="Skor rata-rata", hue="Konstruk", palette="Set2", legend=False, dodge=False, ax=axis)  # Menggambar distribusi skor.
        axis.set_title("Distribusi Skor Rata-rata Konstruk", fontsize=15, weight="bold")  # Memberi judul yang informatif.
        axis.set_xlabel("Konstruk laten")  # Menamai sumbu horizontal.
        axis.set_ylabel("Skor rata-rata Likert")  # Menamai sumbu vertikal.
        axis.tick_params(axis="x", rotation=15)  # Memiringkan label agar nama konstruk tidak bertumpuk.
        output_file = output_dir / "boxplot_skor_konstruk.png"  # Menetapkan nama file gambar boxplot.
        figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan gambar dengan resolusi 300 DPI.
        return output_file  # Mengembalikan lokasi gambar yang dibuat.
    finally:  # Membebaskan resource walau plotting atau penyimpanan melempar error.
        plt.close(figure)  # Menutup kanvas untuk membebaskan memori.


def create_correlation_heatmap(  # Menggambar korelasi Pearson antar-konstruk.
    scores: pd.DataFrame,  # Skor konstruk dengan missing dipertahankan.
    output_dir: Path,  # Folder output gambar.
    correlation: pd.DataFrame | None = None,  # Matriks prahitung opsional agar N dan heatmap memakai objek yang sama.
) -> Path:  # Mengembalikan lokasi file gambar.
    if correlation is None:  # Menjaga kompatibilitas pemanggilan lama.
        correlation, _ = calculate_correlation_analysis(scores)  # Menghitung matriks melalui aturan bersama.
    if not correlation.notna().to_numpy().any():  # Memastikan tersedia setidaknya satu korelasi yang dapat dihitung.
        raise ValueError("Korelasi tidak dapat dihitung: periksa jumlah respons valid dan variasi skor konstruk.")  # Menjelaskan kondisi data yang menghalangi heatmap.
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)  # Membuat kanvas heatmap yang proporsional.
    try:  # Memastikan figure selalu ditutup termasuk ketika plot atau save gagal.
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
        return output_file  # Mengembalikan lokasi gambar yang dibuat.
    finally:  # Membebaskan resource walau plotting atau penyimpanan melempar error.
        plt.close(figure)  # Menutup kanvas untuk membebaskan memori.


def create_construct_mean_bar_chart(scores: pd.DataFrame, output_dir: Path) -> Path:  # Membandingkan mean skor antar-konstruk.
    means = scores.mean().dropna()  # Menggunakan skor konstruk valid yang sudah dihitung.
    if means.empty:  # Memastikan setidaknya satu konstruk memiliki skor.
        raise ValueError("Tidak ada skor konstruk valid untuk dibuat grafik batang.")  # Menjelaskan kondisi tanpa gambar.
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)  # Menyiapkan kanvas perbandingan.
    try:  # Memastikan figure ditutup ketika pembuatan atau penyimpanan gagal.
        bars = axis.bar(means.index, means.values, color=sns.color_palette("Set2", n_colors=len(means)))  # Menggambar mean setiap konstruk.
        axis.bar_label(bars, fmt="%.2f", padding=3)  # Menampilkan nilai mean di atas batang.
        axis.set_title("Rata-rata Skor per Konstruk", fontsize=15, weight="bold")  # Memberi judul grafik.
        axis.set_xlabel("Konstruk")  # Menamai sumbu konstruk.
        axis.set_ylabel("Skor rata-rata")  # Menamai sumbu nilai.
        axis.tick_params(axis="x", rotation=15)  # Memiringkan label untuk mengurangi tumpang tindih.
        axis.set_ylim(bottom=0)  # Mempertahankan baseline nol untuk perbandingan batang.
        output_file = output_dir / "grafik_batang_mean_konstruk.png"  # Menetapkan nama file keluaran.
        figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan grafik beresolusi tinggi.
        return output_file  # Mengembalikan lokasi grafik.
    finally:  # Membebaskan resource walau plotting atau penyimpanan melempar error.
        plt.close(figure)  # Menutup kanvas.


def create_indicator_mean_line_chart(numeric_data: pd.DataFrame, output_dir: Path) -> Path:  # Menampilkan pola mean antarindikator.
    means = numeric_data.mean().dropna()  # Menggunakan jawaban valid per indikator.
    if means.empty:  # Memastikan ada indikator dengan jawaban valid.
        raise ValueError("Tidak ada jawaban indikator valid untuk dibuat grafik garis.")  # Menjelaskan kondisi tanpa gambar.
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)  # Menyiapkan kanvas grafik indikator.
    try:  # Memastikan figure ditutup ketika pembuatan atau penyimpanan gagal.
        axis.plot(means.index, means.values, marker="o", linewidth=2, color="#287A70")  # Menghubungkan mean sesuai urutan indikator, bukan waktu.
        axis.set_title("Rata-rata Skor Tiap Indikator", fontsize=15, weight="bold")  # Memberi judul grafik.
        axis.set_xlabel("Indikator")  # Menamai sumbu indikator.
        axis.set_ylabel("Skor rata-rata")  # Menamai sumbu nilai.
        axis.grid(axis="y", linestyle="--", alpha=0.35)  # Membantu membandingkan nilai.
        output_file = output_dir / "grafik_garis_mean_indikator.png"  # Menetapkan nama file keluaran.
        figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan grafik beresolusi tinggi.
        return output_file  # Mengembalikan lokasi grafik.
    finally:  # Membebaskan resource walau plotting atau penyimpanan melempar error.
        plt.close(figure)  # Menutup kanvas.


def create_respondent_completeness_pie_chart(  # Membandingkan respons lengkap dengan tidak lengkap.
    data: pd.DataFrame,  # Tabel sumber untuk mengenali baris kosong sepenuhnya.
    numeric_data: pd.DataFrame,  # Jawaban indikator setelah validasi.
    output_dir: Path,  # Folder keluaran grafik.
) -> Path:  # Mengembalikan lokasi file gambar.
    has_response = nonempty_response_rows(data)  # Mengecualikan baris kosong sepenuhnya dari denominator.
    complete_count = int((has_response & numeric_data.notna().all(axis=1)).sum())  # Menghitung respons terisi dengan semua indikator valid.
    incomplete_count = int((has_response & ~numeric_data.notna().all(axis=1)).sum())  # Menghitung respons terisi yang belum lengkap.
    counts = [complete_count, incomplete_count]  # Menjaga urutan kategori tetap konsisten.
    if sum(counts) == 0:  # Pie chart tidak bermakna tanpa respons.
        raise ValueError("Tidak ada baris respons untuk dibuat grafik pie kelengkapan.")  # Menjelaskan kondisi tanpa gambar.
    visible = [(label, count) for label, count in zip(("Lengkap", "Tidak lengkap"), counts) if count > 0]  # Menghindari irisan nol.
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)  # Menyiapkan kanvas proporsional.
    try:  # Memastikan figure ditutup ketika pembuatan atau penyimpanan gagal.
        axis.pie(  # Menggambar proporsi kelengkapan seluruh respons nonkosong.
            [count for _, count in visible],  # Menggunakan hitungan yang sudah dihitung.
            labels=[f"{label} (n={count})" for label, count in visible],  # Menampilkan kategori dan jumlah.
            autopct="%1.1f%%",  # Menampilkan persentase tiap kategori.
            startangle=90,  # Memulai irisan dari arah atas.
            colors=sns.color_palette("Set2", n_colors=len(visible)),  # Membedakan kategori dengan palet konsisten.
        )  # Menyelesaikan pie chart.
        axis.set_title("Kelengkapan Respons", fontsize=15, weight="bold")  # Memberi judul grafik.
        axis.axis("equal")  # Menjaga lingkaran tetap bulat.
        output_file = output_dir / "grafik_pie_kelengkapan_responden.png"  # Menetapkan nama file keluaran.
        figure.savefig(output_file, dpi=300, bbox_inches="tight")  # Menyimpan grafik beresolusi tinggi.
        return output_file  # Mengembalikan lokasi grafik.
    finally:  # Membebaskan resource walau plotting atau penyimpanan melempar error.
        plt.close(figure)  # Menutup kanvas.


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
        quality_summary = build_data_quality_summary(data, numeric_data, missing_report, respondent_quality, args.missing_warning_threshold)  # Menghitung agregat kualitas tanpa double count.
        correlation_matrix, pairwise_counts = calculate_correlation_analysis(mean_scores)  # Menghitung korelasi/N sekali agar laporannya konsisten.
        sample_size_report = build_sample_size_report(data, numeric_data, mean_scores, respondent_quality, correlation_matrix, pairwise_counts)  # Menghitung N awal, ID, konstruk, dan analisis.
        args.output_dir.mkdir(parents=True, exist_ok=True)  # Membuat folder keluaran bila belum tersedia.
        cleaned_indicators_file = save_cleaned_indicators(data, numeric_data, args.output_dir)  # Menyimpan seluruh indikator bersih untuk analisis lanjutan.
        missing_report.to_csv(args.output_dir / "laporan_missing_values.csv", index=False, encoding="utf-8-sig")  # Menyimpan laporan missing dan nilai tak valid.
        respondent_quality.to_csv(args.output_dir / "laporan_kualitas_responden.csv", index=False, encoding="utf-8-sig")  # Menyimpan missing dan flag pemeriksaan per responden.
        quality_summary_file = save_quality_summary(quality_summary, args.output_dir)  # Menyimpan ringkasan kualitas agregat.
        sample_size_file = save_sample_size_report(sample_size_report, args.output_dir)  # Menyimpan laporan jumlah sampel aktual.
        response_mask = nonempty_response_rows(data)  # Menghitung record respons sekali untuk ringkasan pemrosesan.
        processing_summary_file = save_processing_summary(len(data), args.output_dir, int((~response_mask).sum()), int(response_mask.sum()))  # Mencatat baris awal, kosong, respons, dipertahankan, dan dikeluarkan.
        id_columns = identifier_columns(data)  # Mengambil ID responden bila tersedia.
        respondent_scores = pd.concat([data[id_columns], mean_scores.add_suffix(" - rata-rata"), total_scores], axis=1)  # Menggabungkan ID dan skor konstruk.
        if not id_columns:  # Menyediakan identifikasi baris bila CSV tidak memuat kolom ID.
            respondent_scores.insert(0, "baris_responden", range(1, len(data) + 1))  # Memberi nomor urut satu-based untuk tiap baris.
        respondent_scores.to_csv(args.output_dir / "skor_konstruk_per_responden.csv", index=False, encoding="utf-8-sig")  # Menyimpan rekap tanpa indeks pandas.
        statistics = save_descriptive_statistics(mean_scores, args.output_dir)  # Menghitung statistik deskriptif dari skor rata-rata.
        comprehensive_statistics_file = save_comprehensive_descriptive_statistics(numeric_data, mean_scores, args.likert_min, args.likert_max, args.output_dir)  # Menyimpan statistik indikator dan konstruk.
        try:  # Mengizinkan pipeline data tetap selesai ketika grafik tidak dapat dihitung.
            boxplot_file = create_boxplot(mean_scores, args.output_dir)  # Menghasilkan boxplot skor rata-rata konstruk.
        except (ValueError, OSError) as error:  # Menangani data tidak memadai atau kegagalan tulis gambar.
            boxplot_file = None  # Menandai boxplot tidak tersedia.
            boxplot_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
        try:  # Mengizinkan pipeline data tetap selesai ketika korelasi tidak terdefinisi.
            heatmap_file = create_correlation_heatmap(mean_scores, args.output_dir, correlation_matrix)  # Menggunakan matriks yang sama dengan laporan N.
        except (ValueError, OSError) as error:  # Menangani data tidak memadai atau kegagalan tulis gambar.
            heatmap_file = None  # Menandai heatmap tidak tersedia.
            heatmap_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
        try:  # Menambahkan grafik pembanding mean konstruk tanpa mengubah skor.
            construct_bar_file = create_construct_mean_bar_chart(mean_scores, args.output_dir)  # Membuat grafik batang dari skor yang tersedia.
        except (ValueError, OSError) as error:  # Menjaga ekspor tabel tetap selesai bila grafik gagal.
            construct_bar_file = None  # Menandai grafik batang tidak tersedia.
            construct_bar_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
        try:  # Menambahkan grafik pola mean indikator.
            indicator_line_file = create_indicator_mean_line_chart(numeric_data, args.output_dir)  # Membuat grafik garis dari indikator bersih.
        except (ValueError, OSError) as error:  # Menjaga ekspor tabel tetap selesai bila grafik gagal.
            indicator_line_file = None  # Menandai grafik garis tidak tersedia.
            indicator_line_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
        try:  # Menambahkan ringkasan proporsi kelengkapan respons.
            completeness_pie_file = create_respondent_completeness_pie_chart(data, numeric_data, args.output_dir)  # Mengabaikan baris kosong sepenuhnya.
        except (ValueError, OSError) as error:  # Menjaga ekspor tabel tetap selesai bila grafik gagal.
            completeness_pie_file = None  # Menandai pie chart tidak tersedia.
            completeness_pie_warning = str(error)  # Menyimpan alasan grafik tidak dibuat.
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
    print(f"\nStatus kualitas data: {quality_summary.loc[quality_summary['metrik'].eq('status_kualitas_data'), 'nilai'].iloc[0]}")  # Menampilkan status berdasarkan temuan aktual.
    print(f"\nLaporan missing: {args.output_dir / 'laporan_missing_values.csv'}")  # Menampilkan lokasi laporan kualitas data.
    print(f"Kualitas responden: {args.output_dir / 'laporan_kualitas_responden.csv'}")  # Menampilkan laporan kualitas per responden.
    print(f"Kualitas agregat: {quality_summary_file}")  # Menampilkan laporan kualitas data baru.
    print(f"Jumlah sampel: {sample_size_file}")  # Menampilkan laporan jumlah sampel baru.
    print(f"Ringkasan pemrosesan: {processing_summary_file}")  # Menampilkan jumlah baris input/output.
    print(f"Indikator bersih: {cleaned_indicators_file}")  # Menampilkan lokasi data indikator untuk SmartPLS.
    print(f"Statistik indikator+konstruk: {comprehensive_statistics_file}")  # Menampilkan tabel statistik lengkap.
    print(f"Boxplot: {boxplot_file or boxplot_warning}")  # Menampilkan grafik atau alasan tidak dibuat.
    print(f"Heatmap: {heatmap_file or heatmap_warning}")  # Menampilkan grafik atau alasan tidak dibuat.
    print(f"Grafik batang mean konstruk: {construct_bar_file or construct_bar_warning}")  # Menampilkan grafik batang atau alasannya.
    print(f"Grafik garis mean indikator: {indicator_line_file or indicator_line_warning}")  # Menampilkan grafik garis atau alasannya.
    print(f"Grafik pie kelengkapan responden: {completeness_pie_file or completeness_pie_warning}")  # Menampilkan pie chart atau alasannya.
    print(f"Statistik CSV: {args.output_dir / 'statistik_deskriptif_konstruk.csv'}")  # Menampilkan lokasi tabel statistik.
    print(f"Skor per responden: {args.output_dir / 'skor_konstruk_per_responden.csv'}")  # Menampilkan lokasi rekap total dan rata-rata.


if __name__ == "__main__":  # Memastikan main hanya dijalankan ketika skrip dieksekusi langsung.
    main()  # Memulai proses analisis.