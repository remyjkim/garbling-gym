# ABOUTME: Tests that GameResults carries benchmark + realized-channel fields
# ABOUTME: and that they serialize correctly via the storage layer.

import json

import numpy as np

from garbling_gym.core.config import GameConfig
from garbling_gym.core.game import Game
from garbling_gym.core.agents import agent_factory
from garbling_gym.core.results import GameResults
from garbling_gym.core.storage import ResultsStore


def _run_short_game():
    config = GameConfig(num_rounds=6)
    sender = agent_factory.create_sender()
    receiver = agent_factory.create_receiver(config=config)
    game = Game(config, sender=sender, receiver=receiver)
    return game.play_game(verbose=False), config


class TestResultsHasBenchmarkFields:
    def test_results_carries_benchmarks_and_realized(self):
        results, _ = _run_short_game()
        assert results.benchmarks is not None
        assert results.realized is not None
        # benchmarks contains the ladder
        assert "cav" in results.benchmarks
        assert "qcav" in results.benchmarks
        # realized contains the empirical channel estimate
        assert "channel" in results.realized
        assert "mutual_information" in results.realized

    def test_realized_channel_shape(self):
        results, _ = _run_short_game()
        channel = np.array(results.realized["channel"])
        assert channel.shape == (3, 3)
        assert np.allclose(channel.sum(axis=1), 1.0, atol=1e-6)


class TestResultsSerializable:
    def test_results_round_trip_through_storage(self, tmp_path):
        results, config = _run_short_game()
        store = ResultsStore(base_path=tmp_path)
        run_id = store.generate_run_id("bench_test")
        store.save_run(run_id=run_id, config=config, results=results, name="bench_test")
        loaded = store.load_run(run_id)
        # The benchmarks and realized fields survive serialization.
        assert "benchmarks" in loaded["results"]
        assert "realized" in loaded["results"]
        assert "cav" in loaded["results"]["benchmarks"]

    def test_results_dict_is_json(self):
        results, _ = _run_short_game()
        store = ResultsStore()
        d = store._results_to_dict(results)
        # Must not raise.
        json.dumps(d)


class TestBenchmarksNeverBreakRun:
    """If the solver errors (e.g. a degenerate config), the run must still
    complete with benchmarks=None rather than crashing."""

    def test_default_game_completes_with_benchmarks(self):
        # The normal path: benchmarks are computed and attached.
        results, _ = _run_short_game()
        assert results.benchmarks is not None
        assert results.benchmarks.get("cav") is not None
