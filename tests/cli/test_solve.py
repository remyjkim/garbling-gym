# ABOUTME: Tests for the `gg solve` CLI command
# ABOUTME: It prints the benchmark bundle for a config WITHOUT running a game.

import pytest
from click.testing import CliRunner

from garbling_gym.cli.__main__ import main


class TestGgSolve:
    def test_solve_default_config(self):
        res = CliRunner().invoke(main, ["solve"])
        assert res.exit_code == 0
        # The benchmark ladder fields appear in the output.
        assert "cav" in res.output.lower()
        assert "qcav" in res.output.lower()
        assert "babbling" in res.output.lower()

    def test_solve_shows_value_of_commitment(self):
        res = CliRunner().invoke(main, ["solve"])
        assert res.exit_code == 0
        assert "commitment" in res.output.lower()

    def test_solve_with_custom_prior(self):
        res = CliRunner().invoke(
            main, ["solve", "--prior-low", "0.5", "--prior-med", "0.3", "--prior-high", "0.2"]
        )
        assert res.exit_code == 0
        assert "cav" in res.output.lower()

    def test_solve_with_chi_grid(self):
        res = CliRunner().invoke(main, ["solve", "--chi-grid", "0,0.5,1"])
        assert res.exit_code == 0
        # The weak-institution curve is rendered.
        assert "chi" in res.output.lower() or "weak" in res.output.lower()

    def test_solve_invalid_prior_rejected(self):
        # Priors not summing to 1 must be rejected by GameConfig validation.
        res = CliRunner().invoke(
            main, ["solve", "--prior-low", "0.9", "--prior-med", "0.9", "--prior-high", "0.9"]
        )
        assert res.exit_code != 0
