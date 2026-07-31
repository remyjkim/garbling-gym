# ABOUTME: Tests that `gg run` surfaces the theory benchmark values in its summary

from click.testing import CliRunner

from garbling_gym.cli.__main__ import main


class TestRunSummaryShowsBenchmarks:
    def test_run_output_mentions_qcav_and_cav(self, tmp_path):
        # Run a short game; the summary should mention the benchmark values.
        res = CliRunner().invoke(
            main,
            ["run", "--rounds", "4", "--no-save"],
            catch_exceptions=False,
        )
        assert res.exit_code == 0
        out = res.output.lower()
        # The benchmark ladder is surfaced in the quick summary.
        assert "qcav" in out
        assert "cav" in out
