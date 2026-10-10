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

    def test_rejects_empty_and_header_only_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "empty.csv"
            csv_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kosong"):
                analysis.read_csv(csv_file)

            csv_file.write_text(",".join(f"Q{number}" for number in range(1, 16)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baris data"):
                analysis.read_csv(csv_file)

    def test_preserves_blank_response_rows_for_missing_audit(self):
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
        self.assertTrue(report.loc[0, "peringatan_missing_tinggi"])

    def test_rejects_duplicate_headers_after_case_normalization(self):
        headers = ["Q1", "q1", *(f"Q{number}" for number in range(2, 16))]
        content = ",".join(headers) + "\n" + ",".join(["1"] * 16) + "\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_file = Path(temporary_directory) / "duplicate.csv"
            csv_file.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplikat"):
                analysis.read_csv(csv_file)

    def test_rejects_missing_required_indicator(self):
        data = pd.DataFrame({f"Q{number}": [4] for number in range(1, 15)})

        with self.assertRaisesRegex(ValueError, "Q15"):
            analysis.clean_likert_responses(data, 1, 5)

    def test_reports_missing_text_and_out_of_range_values(self):
        data = pd.DataFrame([[4] * 15, [4] * 15, [4] * 15], columns=[f"Q{number}" for number in range(1, 16)])
        data["Q1"] = [None, "tidak diketahui", 6]

        cleaned, report = analysis.clean_likert_responses(data, 1, 5)
        q1_report = report.set_index("kolom").loc["Q1"]

        self.assertEqual(q1_report["missing_asli"], 1)
        self.assertEqual(q1_report["jawaban_non_numerik"], 1)
        self.assertEqual(q1_report["di_luar_rentang_likert"], 1)
        self.assertEqual(q1_report["jumlah_respons_valid"], 0)
        self.assertTrue(cleaned["Q1"].isna().all())

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