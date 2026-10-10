import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import analisis_sem_pls as analysis


class CsvValidationTests(unittest.TestCase):
    def test_normalizes_case_bom_and_semicolon_delimiter(self):
        headers = ["Respondent_ID", *(f"q{number}" for number in range(1, 16))]
        row = ["R01", *(["4"] * 15)]
        content = ";".join(headers) + "\n" + ";".join(row) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "responses.csv"
            csv_file.write_text("\ufeff" + content, encoding="utf-8")

            data = analysis.read_csv(csv_file)

        self.assertEqual(data.columns[1:].tolist(), [f"Q{number}" for number in range(1, 16)])
        self.assertEqual(data.loc[0, "Respondent_ID"], "R01")

    def test_detects_comma_semicolon_and_tab_delimiters(self):
        headers = ["Respondent_ID", *(f"Q{number}" for number in range(1, 16))]
        row = ["R01", *("4" for _ in range(15))]

        for delimiter in (",", ";", "\t"):
            with self.subTest(delimiter=repr(delimiter)), tempfile.TemporaryDirectory() as temporary_directory:
                csv_file = Path(temporary_directory) / "responses.csv"
                csv_file.write_text(delimiter.join(headers) + "\n" + delimiter.join(row) + "\n", encoding="utf-8")

                data = analysis.read_csv(csv_file)

            self.assertEqual(data.shape, (1, 16))

    def test_reads_windows_1252_encoding(self):
        headers = ["Respondent_ID", *(f"Q{number}" for number in range(1, 16))]
        row = ["R\u00e901", *("4" for _ in range(15))]

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "windows.csv"
            csv_file.write_text(",".join(headers) + "\n" + ",".join(row) + "\n", encoding="cp1252")

            data = analysis.read_csv(csv_file)

        self.assertEqual(data.loc[0, "Respondent_ID"], "R\u00e901")

    def test_rejects_data_row_with_inconsistent_field_count(self):
        headers = [f"Q{number}" for number in range(1, 16)]
        content = ",".join(headers) + "\n" + ",".join(["4"] * 14) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "wrong_width.csv"
            csv_file.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Baris CSV 2 memiliki 14 kolom"):
                analysis.read_csv(csv_file)

    def test_rejects_empty_and_header_only_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "empty.csv"
            csv_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kosong"):
                analysis.read_csv(csv_file)

            csv_file.write_text(",".join(f"Q{number}" for number in range(1, 16)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baris data"):
                analysis.read_csv(csv_file)

    def test_blank_physical_row_is_preserved_but_not_flagged_as_respondent(self):
        headers = [f"Q{number}" for number in range(1, 16)]
        content = ",".join(headers) + "\n\n" + ",".join(["4"] * 15) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "blank_row.csv"
            csv_file.write_text(content, encoding="utf-8")

            data = analysis.read_csv(csv_file)
            cleaned, _ = analysis.clean_likert_responses(data, 1, 5)
            report = analysis.build_respondent_quality_report(data, cleaned, 0.5)

        self.assertEqual(len(data), 2)
        self.assertEqual(report.loc[0, "jumlah_missing_setelah_pembersihan"], 15)
        self.assertTrue(report.loc[0, "baris_kosong_sepenuhnya"])
        self.assertFalse(report.loc[0, "peringatan_missing_tinggi"])
        self.assertFalse(report.loc[0, "peringatan_baris_jawaban_duplikat"])
        self.assertFalse(report.loc[0, "peringatan_pola_jawaban_seragam"])

    def test_rejects_duplicate_headers_after_case_normalization(self):
        headers = ["Q1", "q1", *(f"Q{number}" for number in range(2, 16))]
        content = ",".join(headers) + "\n" + ",".join(["1"] * 16) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "duplicate.csv"
            csv_file.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplikat"):
                analysis.read_csv(csv_file)

    def test_rejects_duplicate_identifier_headers_after_case_normalization(self):
        headers = ["Respondent_ID", "respondent id", *(f"Q{number}" for number in range(1, 16))]
        content = ",".join(headers) + "\n" + ",".join(["R01", "R01", *("4" for _ in range(15))]) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "duplicate_id_headers.csv"
            csv_file.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplikat"):
                analysis.read_csv(csv_file)

    def test_rejects_missing_required_indicator(self):
        data = pd.DataFrame({f"Q{number}": [4] for number in range(1, 15)})

        with self.assertRaisesRegex(ValueError, "Q15"):
            analysis.clean_likert_responses(data, 1, 5)

    def test_reports_missing_text_and_out_of_range_values(self):
        data = pd.DataFrame([[4] * 15, [4] * 15, [4] * 15, [4] * 15], columns=[f"Q{number}" for number in range(1, 16)], dtype=object)
        data["Q1"] = [None, "tidak diketahui", 6, float("inf")]

        cleaned, report = analysis.clean_likert_responses(data, 1, 5)
        q1_report = report.set_index("kolom").loc["Q1"]

        self.assertEqual(q1_report["missing_asli"], 1)
        self.assertEqual(q1_report["jawaban_non_numerik"], 1)
        self.assertEqual(q1_report["di_luar_rentang_likert"], 2)
        self.assertEqual(q1_report["jumlah_respons_valid"], 0)
        self.assertTrue(cleaned["Q1"].isna().all())

    def test_quality_summary_counts_invalid_cells_once(self):
        columns = [f"Q{number}" for number in range(1, 16)]
        data = pd.DataFrame([[4] * 15, [2] * 15], columns=columns, dtype=object)
        data.insert(0, "Respondent_ID", ["R01", "R02"])
        data.loc[1, "Q1"] = ""
        data.loc[1, "Q2"] = "teks"
        data.loc[1, "Q3"] = 2.5
        data.loc[1, "Q4"] = 6

        cleaned, item_report = analysis.clean_likert_responses(data, 1, 5)
        respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.2)
        summary = analysis.build_data_quality_summary(data, cleaned, item_report, respondent_report, 0.2)
        metrics = summary.set_index("metrik")["nilai"]

        self.assertEqual(metrics["jumlah_baris_respons_awal"], 2)
        self.assertEqual(metrics["jumlah_sel_jawaban_diperiksa"], 30)
        self.assertEqual(metrics["jumlah_sel_valid"], 26)
        self.assertEqual(metrics["jumlah_sel_missing_asli"], 1)
        self.assertEqual(metrics["jumlah_sel_nonnumerik"], 1)
        self.assertEqual(metrics["jumlah_sel_desimal_tidak_valid"], 1)
        self.assertEqual(metrics["jumlah_sel_di_luar_rentang"], 1)
        self.assertEqual(metrics["jumlah_sel_tidak_valid_setelah_pembersihan"], 4)
        self.assertAlmostEqual(metrics["persentase_sel_tidak_valid"], 4 / 30 * 100, places=2)
        self.assertEqual(metrics["jumlah_responden_lengkap"], 1)
        self.assertEqual(metrics["jumlah_responden_tidak_lengkap"], 1)
        self.assertEqual(metrics["jumlah_responden_peringatan_missing"], 1)
        self.assertEqual(metrics["jumlah_responden_pola_jawaban_seragam"], 2)
        self.assertEqual(metrics["jumlah_baris_pola_jawaban_duplikat"], 0)
        self.assertEqual(metrics["jumlah_baris_terlibat_id_duplikat"], 0)
        self.assertEqual(metrics["status_kualitas_data"], "Terdapat data hilang atau jawaban tidak valid; Terdapat indikator tanpa variasi; Terdapat pola responden untuk ditinjau manual")

    def test_rejects_decimal_likert_values(self):
        data = pd.DataFrame([[4] * 15], columns=[f"Q{number}" for number in range(1, 16)], dtype=object)
        data.loc[0, "Q1"] = 2.5

        cleaned, report = analysis.clean_likert_responses(data, 1, 5)

        self.assertTrue(pd.isna(cleaned.loc[0, "Q1"]))
        self.assertEqual(report.set_index("kolom").loc["Q1", "jawaban_desimal"], 1)


class ScoringAndExportTests(unittest.TestCase):
    def test_construct_scores_require_all_three_valid_items(self):
        data = pd.DataFrame([[1] * 15, [4] * 15], columns=[f"Q{number}" for number in range(1, 16)])
        data.loc[0, ["Q1", "Q2", "Q3"]] = [1, 2, 3]
        data.loc[1, "Q2"] = float("nan")

        mean_scores, total_scores = analysis.calculate_construct_scores(data)

        self.assertEqual(mean_scores.loc[0, "X1 - System Quality & Features"], 2)
        self.assertEqual(total_scores.loc[0, "X1 - System Quality & Features - total"], 6)
        self.assertTrue(pd.isna(mean_scores.loc[1, "X1 - System Quality & Features"]))
        self.assertTrue(pd.isna(total_scores.loc[1, "X1 - System Quality & Features - total"]))

    def test_clean_export_preserves_ids_rows_and_blank_missing_values(self):
        data = pd.DataFrame([["R01", *([4] * 15)], ["R02", *([4] * 15)]], columns=["Respondent_ID", *[f"Q{number}" for number in range(1, 16)]])
        data.loc[1, "Q3"] = None
        cleaned, _ = analysis.clean_likert_responses(data, 1, 5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = analysis.save_cleaned_indicators(data, cleaned, Path(temporary_directory))
            exported = pd.read_csv(output_file)

        self.assertEqual(len(exported), 2)
        self.assertEqual(exported.columns.tolist(), data.columns.tolist())
        self.assertEqual(exported["Respondent_ID"].tolist(), ["R01", "R02"])
        self.assertTrue(pd.isna(exported.loc[1, "Q3"]))

    def test_constant_constructs_report_unavailable_correlations(self):
        scores = pd.DataFrame({construct: [3.0, 3.0, 3.0] for construct in analysis.CONSTRUCTS})

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Korelasi tidak dapat dihitung"):
                analysis.create_correlation_heatmap(scores, Path(temporary_directory))

    def test_all_missing_indicator_has_no_construct_score(self):
        data = pd.DataFrame([[4] * 15, [5] * 15], columns=[f"Q{number}" for number in range(1, 16)])
        data["Q1"] = ""
        cleaned, report = analysis.clean_likert_responses(data, 1, 5)
        mean_scores, _ = analysis.calculate_construct_scores(cleaned)

        self.assertEqual(report.set_index("kolom").loc["Q1", "jumlah_respons_valid"], 0)
        self.assertTrue(mean_scores["X1 - System Quality & Features"].isna().all())
        self.assertTrue(mean_scores["X2 - Reward & Incentive Structure"].notna().all())

    def test_respondent_quality_flags_missing_duplicates_and_straightlining(self):
        question_columns = [f"Q{number}" for number in range(1, 16)]
        rows = [[4] * 15, [4] * 15, [1] * 15, [4] * 15]
        data = pd.DataFrame(rows, columns=question_columns, dtype=object)
        data.insert(0, "Respondent_ID", ["A", "B", "C", "A"])
        data.loc[1, question_columns[7:]] = ""
        cleaned, _ = analysis.clean_likert_responses(data, 1, 5)

        report = analysis.build_respondent_quality_report(data, cleaned, 0.5)

        self.assertTrue(report.loc[1, "peringatan_missing_tinggi"])
        self.assertEqual(report.loc[1, "persentase_missing_setelah_pembersihan"], 100 * 8 / 15)
        self.assertTrue(report.loc[0, "peringatan_pola_jawaban_seragam"])
        self.assertTrue(report.loc[0, "peringatan_baris_jawaban_duplikat"])
        self.assertTrue(report.loc[0, "peringatan_id_duplikat"])
        self.assertTrue(report.loc[3, "peringatan_id_duplikat"])
        self.assertEqual(len(report), len(data))

    def test_composite_ids_require_all_components_and_share_one_duplicate_rule(self):
        questions = [f"Q{number}" for number in range(1, 16)]
        data = pd.DataFrame(
            [
                ["R1", "A", *([4] * 15)],
                ["R1", "A", *([3] * 15)],
                ["R2", "", *([2] * 15)],
                ["", "", *([1] * 15)],
                ["", "", *("" for _ in range(15))],
            ],
            columns=["Respondent ID", "Participant-ID", *questions],
            dtype=object,
        )
        cleaned, _ = analysis.clean_likert_responses(data, 1, 5)
        quality = analysis.build_respondent_quality_report(data, cleaned, 0.5)
        sample = analysis.build_sample_size_report(data, cleaned, *analysis.calculate_construct_scores(cleaned)[:1], quality)
        sample_metrics = sample.set_index("metrik")["nilai"]

        self.assertTrue(quality.loc[0, "peringatan_id_duplikat"])
        self.assertTrue(quality.loc[1, "peringatan_id_duplikat"])
        self.assertTrue(quality.loc[2, "peringatan_id_tidak_lengkap"])
        self.assertTrue(quality.loc[3, "peringatan_id_kosong"])
        self.assertTrue(quality.loc[4, "baris_kosong_sepenuhnya"])
        self.assertFalse(quality.loc[4, "peringatan_id_kosong"])
        self.assertEqual(sample_metrics["jumlah_responden_dengan_id"], 2)
        self.assertEqual(sample_metrics["jumlah_id_unik"], 1)
        self.assertEqual(sample_metrics["jumlah_id_duplikat"], 2)
        self.assertEqual(sample_metrics["jumlah_responden_id_kosong"], 1)
        self.assertEqual(sample_metrics["jumlah_responden_id_tidak_lengkap"], 1)

    def test_sample_report_counts_ids_and_valid_n_per_construct(self):
        columns = [f"Q{number}" for number in range(1, 16)]
        data = pd.DataFrame([[4] * 15, [2] * 15], columns=columns, dtype=object)
        data.insert(0, "Respondent_ID", ["R01", "R02"])
        data.loc[1, "Q1"] = ""
        data.loc[1, "Q4"] = "teks"
        cleaned, _ = analysis.clean_likert_responses(data, 1, 5)
        means, _ = analysis.calculate_construct_scores(cleaned)
        respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.5)

        sample_report = analysis.build_sample_size_report(data, cleaned, means, respondent_report)
        sample = sample_report[sample_report["metrik"].eq("jumlah_responden_untuk_skor_konstruk")].set_index("keterangan")["nilai"]
        metrics = sample_report.set_index("metrik")["nilai"]

        self.assertEqual(metrics["jumlah_baris_awal"], 2)
        self.assertEqual(metrics["jumlah_responden_dengan_id"], 2)
        self.assertEqual(metrics["jumlah_id_unik"], 2)
        self.assertEqual(metrics["jumlah_id_duplikat"], 0)
        self.assertEqual(metrics["jumlah_responden_lengkap"], 1)
        self.assertEqual(metrics["jumlah_baris_dipertahankan"], 2)
        self.assertEqual(metrics["jumlah_baris_dikeluarkan"], 0)
        self.assertEqual(sample["X1 - System Quality & Features"], 1)
        self.assertEqual(sample["X2 - Reward & Incentive Structure"], 1)
        self.assertEqual(sample["X3 - Drop-Point / Pickup Accessibility"], 2)
        self.assertEqual(sample["M - Environmental Attitude & Engagement"], 2)
        self.assertEqual(sample["Y - Actual Recycling Behavior"], 2)
        pearson_n = sample_report[
            sample_report["keterangan"].eq("Korelasi Pearson deskriptif: X1 - System Quality & Features dengan M - Environmental Attitude & Engagement; pairwise complete.")
        ]["nilai"].iloc[0]
        self.assertEqual(pearson_n, 1)

    def test_sample_report_marks_id_metrics_unavailable_without_id(self):
        data = pd.DataFrame([[4] * 15], columns=[f"Q{number}" for number in range(1, 16)])
        cleaned, _ = analysis.clean_likert_responses(data, 1, 5)
        means, _ = analysis.calculate_construct_scores(cleaned)
        respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.5)

        report = analysis.build_sample_size_report(data, cleaned, means, respondent_report).set_index("metrik")

        self.assertTrue(pd.isna(report.loc["jumlah_responden_dengan_id", "nilai"]))
        self.assertTrue(pd.isna(report.loc["jumlah_id_unik", "nilai"]))
        self.assertTrue(pd.isna(report.loc["jumlah_id_duplikat", "nilai"]))

    def test_descriptive_export_includes_item_frequencies_and_constructs(self):
        data = pd.DataFrame([[1] * 15, [5] * 15], columns=[f"Q{number}" for number in range(1, 16)])
        mean_scores, _ = analysis.calculate_construct_scores(data)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = analysis.save_comprehensive_descriptive_statistics(data, mean_scores, 1, 5, Path(temporary_directory))
            statistics = pd.read_csv(output_file)

        q1_stats = statistics.set_index(["tingkat", "variabel"]).loc[("indikator", "Q1")]
        self.assertEqual(q1_stats["jumlah_respons_valid"], 2)
        self.assertEqual(q1_stats["frekuensi_1"], 1)
        self.assertEqual(q1_stats["frekuensi_5"], 1)
        self.assertIn(("konstruk", "X1 - System Quality & Features"), statistics.set_index(["tingkat", "variabel"]).index)

    def test_construct_descriptive_standard_deviation_is_unavailable_for_one_score(self):
        scores = pd.DataFrame({construct: [4.0] for construct in analysis.CONSTRUCTS})

        with tempfile.TemporaryDirectory() as temporary_directory:
            statistics = analysis.save_descriptive_statistics(scores, Path(temporary_directory))

        self.assertTrue(statistics["std"].isna().all())
        self.assertTrue(statistics["jumlah_respons_valid"].eq(1).all())

    def test_pipeline_keeps_outputs_when_every_answer_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_file = root / "all_missing.csv"
            output_dir = root / "output"
            headers = [f"Q{number}" for number in range(1, 16)]
            csv_file.write_text(",".join(headers) + "\n" + ",".join([""] * 15) + "\n", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(Path(analysis.__file__)), str(csv_file), "--output-dir", str(output_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertTrue((output_dir / "laporan_kualitas_responden.csv").is_file())
            self.assertTrue((output_dir / "laporan_kualitas_data.csv").is_file())
            self.assertTrue((output_dir / "laporan_jumlah_sampel.csv").is_file())
            self.assertTrue((output_dir / "statistik_deskriptif.csv").is_file())
            processing_summary = pd.read_csv(output_dir / "ringkasan_pemrosesan.csv").iloc[0]
            self.assertEqual(processing_summary["baris_awal"], 1)
            self.assertEqual(processing_summary["baris_dipertahankan"], 1)
            self.assertEqual(processing_summary["baris_dikeluarkan"], 0)

    def test_construct_scoring_supports_small_and_large_datasets(self):
        columns = [f"Q{number}" for number in range(1, 16)]
        for response_count in (2, 2000):
            with self.subTest(response_count=response_count):
                data = pd.DataFrame([[4] * 15] * response_count, columns=columns)
                mean_scores, total_scores = analysis.calculate_construct_scores(data)

                self.assertEqual(mean_scores.shape, (response_count, 5))
                self.assertEqual(total_scores.shape, (response_count, 5))


if __name__ == "__main__":
    unittest.main()