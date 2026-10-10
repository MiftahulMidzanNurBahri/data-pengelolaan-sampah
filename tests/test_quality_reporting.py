import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

import analisis_sem_pls as analysis


@pytest.fixture
def question_columns():
    return [f"Q{number}" for number in range(1, 16)]


@pytest.fixture
def varied_responses(question_columns):
    rows = [
        [1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4],
        [2, 3, 4, 3, 4, 5, 4, 5, 1, 2, 3, 4, 3, 4, 5],
        [3, 4, 5, 4, 5, 1, 5, 1, 2, 3, 4, 5, 4, 5, 1],
    ]
    return pd.DataFrame(rows, columns=question_columns)


def test_blank_csv_line_is_counted_separately_and_retained(tmp_path, question_columns):
    header = "Respondent_ID," + ",".join(question_columns)
    response = "R01," + ",".join(["4"] * 15)
    csv_file = tmp_path / "blank_line.csv"
    csv_file.write_text(header + "\n\n" + response + "\n", encoding="utf-8")

    data = analysis.read_csv(csv_file)
    cleaned, item_report = analysis.clean_likert_responses(data, 1, 5)
    respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.5)
    quality_summary = analysis.build_data_quality_summary(data, cleaned, item_report, respondent_report, 0.5)
    sample_report = analysis.build_sample_size_report(
        data, cleaned, *analysis.calculate_construct_scores(cleaned)[:1], respondent_report
    )
    quality_metrics = quality_summary.set_index("metrik")["nilai"]
    sample_metrics = sample_report.set_index("metrik")["nilai"]

    assert len(data) == 2
    assert respondent_report["baris_kosong_sepenuhnya"].tolist() == [True, False]
    assert respondent_report["peringatan_missing_tinggi"].tolist() == [False, False]
    assert respondent_report["peringatan_baris_jawaban_duplikat"].tolist() == [False, False]
    assert quality_metrics["jumlah_baris_kosong_sepenuhnya"] == 1
    assert quality_metrics["jumlah_responden_lengkap"] == 1
    assert quality_metrics["jumlah_responden_tidak_lengkap"] == 0
    assert sample_metrics["jumlah_baris_awal"] == 2
    assert sample_metrics["jumlah_baris_dipertahankan"] == 2
    assert sample_metrics["jumlah_baris_dikeluarkan"] == 0


def test_composite_id_requires_all_parts_and_does_not_count_blank_rows(question_columns):
    columns = ["Respondent ID", "Participant-ID", *question_columns]
    data = pd.DataFrame(
        [
            ["R1", "A", *([4] * 15)],
            ["R1", "A", *([3] * 15)],
            ["R2", "", *([2] * 15)],
            ["", "", *([1] * 15)],
            ["", "", *("" for _ in range(15))],
        ],
        columns=columns,
        dtype=object,
    )

    id_columns, has_id, missing_id, partial_id, duplicate_rows, unique_count = analysis.identifier_summary(data)
    quality = analysis.build_respondent_quality_report(
        data, analysis.clean_likert_responses(data, 1, 5)[0], 0.5
    )

    assert id_columns == ["Respondent ID", "Participant-ID"]
    assert has_id.tolist() == [True, True, False, False, False]
    assert missing_id.tolist() == [False, False, False, True, False]
    assert partial_id.tolist() == [False, False, True, False, False]
    assert duplicate_rows.tolist() == [True, True, False, False, False]
    assert unique_count == 1
    assert quality["peringatan_id_duplikat"].tolist() == [True, True, False, False, False]


def test_correlation_pair_count_and_defined_status_match_matrix(varied_responses):
    data = varied_responses.copy()
    data.insert(0, "Respondent_ID", ["R1", "R2", "R3"])
    data[["Q1", "Q2", "Q3"]] = 3
    cleaned, _ = analysis.clean_likert_responses(data, 1, 5)
    mean_scores, _ = analysis.calculate_construct_scores(cleaned)
    respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.5)
    correlation, pairwise_counts = analysis.calculate_correlation_analysis(mean_scores)
    report = analysis.build_sample_size_report(
        data, cleaned, mean_scores, respondent_report, correlation, pairwise_counts
    )
    pair_label = "Korelasi Pearson deskriptif: X1 - System Quality & Features dengan X2 - Reward & Incentive Structure; pairwise complete."
    pair_rows = report[report["keterangan"].eq(pair_label)].set_index("metrik")["nilai"]
    pair_status = report[
        report["metrik"].eq("status_korelasi_pearson")
        & report["keterangan"].str.startswith(pair_label[:-1])
    ]["nilai"].iloc[0]

    assert pair_rows["jumlah_responden_untuk_analisis"] == 3
    assert pair_status == "tidak_terdefinisi"
    assert pd.isna(correlation.loc["X1 - System Quality & Features", "X2 - Reward & Incentive Structure"])


def test_quality_percentages_are_unavailable_when_there_are_no_cells(question_columns):
    data = pd.DataFrame(columns=question_columns)
    cleaned, item_report = analysis.clean_likert_responses(data, 1, 5)
    respondent_report = analysis.build_respondent_quality_report(data, cleaned, 0.5)

    summary = analysis.build_data_quality_summary(data, cleaned, item_report, respondent_report, 0.5).set_index("metrik")

    assert pd.isna(summary.loc["persentase_sel_tidak_valid", "nilai"])
    assert "Tidak ada respons nonkosong" in summary.loc["status_kualitas_data", "nilai"]


def test_cleaning_supports_a_configured_integer_likert_range(question_columns):
    data = pd.DataFrame([[0] * 15, [5] * 15, [10] * 15], columns=question_columns)

    cleaned, report = analysis.clean_likert_responses(data, 0, 10)

    assert cleaned.notna().to_numpy().all()
    assert report["jumlah_respons_valid"].eq(3).all()
    assert report["di_luar_rentang_likert"].eq(0).all()


def test_read_csv_accepts_exactly_150_respondents(tmp_path, question_columns):
    csv_file = tmp_path / "exactly_150.csv"
    rows = [",".join(question_columns)] + [",".join(["4"] * 15) for _ in range(150)]
    csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    data = analysis.read_csv(csv_file)

    assert len(data) == 150
    assert analysis.nonempty_response_rows(data).sum() == 150


def test_read_csv_rejects_respondent_151_without_truncating(tmp_path, question_columns):
    csv_file = tmp_path / "over_limit.csv"
    rows = [",".join(question_columns)] + [",".join(["4"] * 15) for _ in range(151)]
    csv_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="batas maksimum 150.*151.*tidak dipotong"):
        analysis.read_csv(csv_file)


@pytest.mark.parametrize("plot_name", ["boxplot", "heatmap"])
def test_figure_closes_when_image_save_fails(tmp_path, monkeypatch, varied_responses, plot_name):
    mean_scores, _ = analysis.calculate_construct_scores(varied_responses)
    closed_figures = []
    close_figure = analysis.plt.close

    def track_close(figure):
        closed_figures.append(figure)
        close_figure(figure)

    def fail_save(*args, **kwargs):
        raise OSError("image path not writable")

    monkeypatch.setattr(analysis.plt, "close", track_close)
    monkeypatch.setattr(Figure, "savefig", fail_save)
    plot_function = analysis.create_boxplot if plot_name == "boxplot" else analysis.create_correlation_heatmap

    with pytest.raises(OSError, match="not writable"):
        plot_function(mean_scores, tmp_path)

    assert len(closed_figures) == 1


def test_cli_keeps_csv_outputs_when_boxplot_save_fails(tmp_path, monkeypatch, capfd, varied_responses):
    csv_file = tmp_path / "responses.csv"
    output_dir = tmp_path / "new_output"
    data = varied_responses.copy()
    data.insert(0, "Respondent_ID", ["R1", "R2", "R3"])
    data.to_csv(csv_file, index=False)
    monkeypatch.setattr(sys, "argv", ["analisis_sem_pls.py", str(csv_file), "--output-dir", str(output_dir)])

    def fail_boxplot(*args, **kwargs):
        raise OSError("cannot write boxplot")

    monkeypatch.setattr(analysis, "create_boxplot", fail_boxplot)
    analysis.main()
    stdout = capfd.readouterr().out

    assert (output_dir / "laporan_kualitas_data.csv").is_file()
    assert (output_dir / "laporan_jumlah_sampel.csv").is_file()
    assert (output_dir / "data_indikator_bersih.csv").is_file()
    assert (output_dir / "heatmap_korelasi_konstruk.png").is_file()
    assert not (output_dir / "boxplot_skor_konstruk.png").exists()
    assert "Boxplot: cannot write boxplot" in stdout


def test_cli_reports_mixed_synthetic_data_without_excluding_rows(tmp_path, question_columns):
    csv_file = tmp_path / "mixed_synthetic.csv"
    output_dir = tmp_path / "analysis_output"
    header = ["Respondent_ID", *question_columns]
    first = ["R01", *[str((number % 5) + 1) for number in range(15)]]
    second = ["R01", *[str(((number + 1) % 5) + 1) for number in range(15)]]
    third = ["R02", "", "teks", "2.5", "6", *(["4"] * 11)]
    csv_file.write_text(
        ",".join(header) + "\n" + ",".join(first) + "\n" + ",".join(second) + "\n" + ",".join(third) + "\n\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(Path(analysis.__file__)), str(csv_file), "--output-dir", str(output_dir), "--missing-warning-threshold", "0.2"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    quality = pd.read_csv(output_dir / "laporan_kualitas_data.csv").set_index("metrik")["nilai"]
    quality_numbers = pd.to_numeric(quality, errors="coerce")
    sample = pd.read_csv(output_dir / "laporan_jumlah_sampel.csv").set_index("metrik")
    sample_numbers = pd.to_numeric(sample["nilai"], errors="coerce")
    assert quality_numbers["jumlah_baris_respons_awal"] == 4
    assert quality_numbers["jumlah_baris_kosong_sepenuhnya"] == 1
    assert quality_numbers["jumlah_baris_respons_non_kosong"] == 3
    assert quality_numbers["jumlah_sel_jawaban_diperiksa"] == 60
    assert quality_numbers["jumlah_sel_valid"] == 41
    assert quality_numbers["jumlah_sel_tidak_valid_setelah_pembersihan"] == 19
    assert quality_numbers["jumlah_sel_missing_asli"] == 16
    assert quality_numbers["jumlah_sel_nonnumerik"] == 1
    assert quality_numbers["jumlah_sel_desimal_tidak_valid"] == 1
    assert quality_numbers["jumlah_sel_di_luar_rentang"] == 1
    assert quality_numbers["persentase_sel_tidak_valid"] == pytest.approx(31.67)
    assert sample_numbers["jumlah_responden_dengan_id"] == 3
    assert sample_numbers["jumlah_id_unik"] == 2
    assert sample_numbers["jumlah_id_duplikat"] == 2
    assert sample_numbers["jumlah_responden_lengkap"] == 2
    assert sample_numbers["jumlah_responden_tidak_lengkap"] == 1
    assert sample_numbers["jumlah_baris_dipertahankan"] == 4
    assert sample_numbers["jumlah_baris_dikeluarkan"] == 0
    assert (output_dir / "laporan_kualitas_responden.csv").is_file()
    assert (output_dir / "boxplot_skor_konstruk.png").is_file()
    assert (output_dir / "heatmap_korelasi_konstruk.png").is_file()


@pytest.mark.parametrize(
    ("minimum", "maximum", "threshold", "message"),
    [(5, 1, 0.5, "likert-min"), (1, 5, 1.1, "missing-warning-threshold")],
)
def test_cli_rejects_invalid_analysis_configuration(tmp_path, monkeypatch, minimum, maximum, threshold, message):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analisis_sem_pls.py", str(tmp_path / "not_read.csv"), "--likert-min", str(minimum), "--likert-max", str(maximum), "--missing-warning-threshold", str(threshold)],
    )

    with pytest.raises(SystemExit, match=message):
        analysis.main()