"""Unit tests for the CLI module."""

import subprocess
import sys

import pytest
from click.testing import CliRunner

from clusx import __copyright__, __version__
from clusx.cli import cli


def test_version_option():
    """Test that the --version option prints the correct version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"clusx {__version__}" in result.output
    assert __copyright__ in result.output
    assert "This is free software" in result.output
    assert "warranty" in result.output


@pytest.mark.skipif(
    sys.platform == "win32", reason="Command execution differs on Windows"
)
def test_version_option_module():
    """Test that the --version option works when running as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "clusx", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"clusx {__version__}" in result.stdout
    assert __copyright__ in result.stdout
    assert "This is free software" in result.stdout
    assert "warranty" in result.stdout


def test_cluster_command_with_small_text_file(basic_text_file, tmp_path, monkeypatch):
    """Test the cluster command with a text file."""
    from unittest.mock import patch

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ns = "clusx.clustering"

    with (
        patch(f"{ns}.DirichletProcess.fit_predict", return_value=[0, 1, 2]),
        patch(f"{ns}.PitmanYorProcess.fit_predict", return_value=[0, 1, 2]),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "cluster",
                "--input",
                str(basic_text_file),
                "--output",
                "test_output.csv",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Warning: Dataset is very small (< 10 texts)." in result.output
        assert "metrics and visualizations may not be available" in result.output


def test_cluster_command_with_small_csv_file(basic_qa_csv, tmp_path, monkeypatch):
    """Test the cluster command with a CSV file."""
    from unittest.mock import patch

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    ns = "clusx.clustering"

    with (
        patch(f"{ns}.DirichletProcess.fit_predict", return_value=[0, 1]),
        patch(f"{ns}.PitmanYorProcess.fit_predict", return_value=[0, 1]),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "cluster",
                "--input",
                str(basic_qa_csv),
                "--column",
                "question",
                "--output",
                "test_output.csv",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
    assert "Warning: Dataset is very small (< 10 texts)." in result.output
    assert "metrics and visualizations may not be available" in result.output


def test_evaluate_command_show_plot_option(basic_text_file, tmp_path, monkeypatch):
    """Test the evaluate command with --show-plot option."""
    from unittest.mock import patch

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create mock cluster files
    dp_clusters = tmp_path / "dp_clusters.csv"
    pyp_clusters = tmp_path / "pyp_clusters.csv"

    # Write minimal content to mock files
    dp_clusters.write_text("Text,Cluster_DP\ntext1,0\ntext2,1\n")
    pyp_clusters.write_text("Text,Cluster_PYP\ntext1,0\ntext2,1\n")

    # Mock dependencies
    with (
        patch("clusx.clustering.utils.load_data", return_value=["text1", "text2"]),
        patch(
            "clusx.clustering.utils.load_cluster_assignments",
            return_value=([0, 1], {"alpha": 0.5, "sigma": 0.0, "kappa": 0.3}),
        ),
        patch(
            "clusx.clustering.utils.get_embeddings",
            return_value=[[0.1, 0.2], [0.3, 0.4]],
        ),
        patch("clusx.evaluation.ClusterEvaluator.generate_report", return_value={}),
        patch("clusx.evaluation.save_evaluation_report"),
        patch("clusx.visualization.visualize_evaluation_dashboard") as mock_viz,
    ):
        runner = CliRunner()

        # Test with default --no-show-plot (implicit)
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--input",
                str(basic_text_file),
                "--dp-clusters",
                str(dp_clusters),
                "--pyp-clusters",
                str(pyp_clusters),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        mock_viz.assert_called_with(
            {"Dirichlet": {}, "Pitman-Yor": {}}, str(output_dir), show_plot=False
        )

        # Reset mock
        mock_viz.reset_mock()

        # Test with explicit --show-plot
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--input",
                str(basic_text_file),
                "--dp-clusters",
                str(dp_clusters),
                "--pyp-clusters",
                str(pyp_clusters),
                "--output-dir",
                str(output_dir),
                "--show-plot",
            ],
        )
        assert result.exit_code == 0
        mock_viz.assert_called_with(
            {"Dirichlet": {}, "Pitman-Yor": {}}, str(output_dir), show_plot=True
        )
