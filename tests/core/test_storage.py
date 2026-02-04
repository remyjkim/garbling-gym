"""
Unit tests for results storage system.

Tests database indexing, file storage, and retrieval.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from garbling_gym.core.storage import ResultsStore
from garbling_gym.core.database import ResultsDatabase
from garbling_gym.core.config import GameConfig
from garbling_gym.core.results import GameResults
from garbling_gym.core.types import AssetQuality


class TestResultsDatabase:
    """Test the ResultsDatabase class"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = ResultsDatabase(db_path)
            yield db
            db.close()

    def test_database_initialization(self, temp_db):
        """Test that database initializes with correct schema"""
        # Should not raise
        assert temp_db.db_path.exists()

    def test_insert_and_get_run(self, temp_db):
        """Test inserting and retrieving a run"""
        run_data = {
            'id': '2026-02-03_120000_test',
            'name': 'test_run',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': 10.5,
            'num_rounds': 20,
            'sender_total': 100.0,
            'receiver_total': 50.0,
            'buy_rate': 0.6,
            'avg_informativeness': 0.75,
            'receiver_regret': 25.0,
            'use_llm': False,
            'llm_model': None,
            'git_commit': 'abc123',
            'tags': ['test', 'baseline'],
            'path': '/path/to/run'
        }

        temp_db.insert_run(run_data)

        # Retrieve
        retrieved = temp_db.get_run('2026-02-03_120000_test')
        assert retrieved is not None
        assert retrieved['name'] == 'test_run'
        assert retrieved['num_rounds'] == 20
        assert retrieved['sender_total'] == 100.0

    def test_list_runs_empty(self, temp_db):
        """Test listing runs when database is empty"""
        runs = temp_db.list_runs()
        assert runs == []

    def test_list_runs_with_data(self, temp_db):
        """Test listing runs with multiple entries"""
        # Insert multiple runs
        for i in range(5):
            run_data = {
                'id': f'2026-02-03_12000{i}_test',
                'name': f'test_run_{i}',
                'timestamp': f'2026-02-03T12:00:0{i}',
                'num_rounds': 20,
                'sender_total': float(i * 10),
                'receiver_total': float(i * 5),
                'buy_rate': 0.5,
                'avg_informativeness': 0.5,
                'receiver_regret': 10.0,
                'use_llm': False,
                'path': f'/path/to/run{i}'
            }
            temp_db.insert_run(run_data)

        # List all
        runs = temp_db.list_runs()
        assert len(runs) == 5

        # List with limit
        runs = temp_db.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_with_filters(self, temp_db):
        """Test filtering runs"""
        # Insert runs with different values
        for i in range(5):
            run_data = {
                'id': f'run_{i}',
                'name': f'test_{i}',
                'timestamp': f'2026-02-03T12:00:0{i}',
                'num_rounds': 20,
                'sender_total': float(i * 20),  # 0, 20, 40, 60, 80
                'receiver_total': 0.0,
                'buy_rate': 0.5,
                'avg_informativeness': 0.5,
                'receiver_regret': 0.0,
                'use_llm': False,
                'path': f'/path{i}'
            }
            temp_db.insert_run(run_data)

        # Filter by sender_total
        runs = temp_db.list_runs(filters={'sender_total_min': 40})
        assert len(runs) == 3  # 40, 60, 80

        # Filter by name pattern
        runs = temp_db.list_runs(filters={'name_like': 'test_2'})
        assert len(runs) == 1
        assert runs[0]['name'] == 'test_2'

    def test_count_runs(self, temp_db):
        """Test counting runs"""
        assert temp_db.count_runs() == 0

        # Insert some runs
        for i in range(10):
            run_data = {
                'id': f'run_{i}',
                'timestamp': datetime.now().isoformat(),
                'num_rounds': 20,
                'sender_total': 0.0,
                'receiver_total': 0.0,
                'buy_rate': 0.5,
                'avg_informativeness': 0.5,
                'receiver_regret': 0.0,
                'use_llm': False,
                'path': f'/path{i}'
            }
            temp_db.insert_run(run_data)

        assert temp_db.count_runs() == 10

    def test_delete_run(self, temp_db):
        """Test deleting a run"""
        run_data = {
            'id': 'test_run',
            'timestamp': datetime.now().isoformat(),
            'num_rounds': 20,
            'sender_total': 0.0,
            'receiver_total': 0.0,
            'buy_rate': 0.5,
            'avg_informativeness': 0.5,
            'receiver_regret': 0.0,
            'use_llm': False,
            'path': '/path'
        }
        temp_db.insert_run(run_data)

        # Delete
        assert temp_db.delete_run('test_run') is True
        assert temp_db.get_run('test_run') is None

        # Delete non-existent
        assert temp_db.delete_run('nonexistent') is False


class TestResultsStore:
    """Test the ResultsStore class"""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary results store"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(Path(tmpdir))
            yield store
            store.db.close()

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration"""
        return GameConfig(num_rounds=10, use_llm=False)

    @pytest.fixture
    def sample_results(self):
        """Create sample results"""
        return GameResults(
            total_rounds=10,
            sender_total=80.0,
            receiver_total=40.0,
            strategies_used={'full_revelation': 10},
            buy_rate=0.7,
            quality_stats={'LOW': {'count': 3, 'bought': 2, 'sender_total': 20, 'receiver_total': -30}},
            avg_informativeness=0.85,
            receiver_regret=15.0,
            perfect_info_benchmark=55.0,
            history=[
                {
                    'round': 1,
                    'quality': 'HIGH',
                    'strategy': 'full_revelation',
                    'garbling_info': 1.0,
                    'signal': 'GOOD',
                    'action': 'BUY',
                    'sender_payoff': 10.0,
                    'receiver_payoff': 20.0
                }
            ]
        )

    def test_generate_run_id_with_name(self, temp_store):
        """Test generating run ID with name"""
        run_id = temp_store.generate_run_id("my_experiment")

        # Should have format: YYYY-MM-DD_HHMMSS_name
        # Name is appended after timestamp
        assert "my_experiment" in run_id
        assert run_id.endswith("my_experiment")

    def test_generate_run_id_sanitizes_name(self, temp_store):
        """Test that unsafe characters are sanitized"""
        run_id = temp_store.generate_run_id("test/with:special*chars")

        # Should contain no special characters
        assert "/" not in run_id
        assert ":" not in run_id
        assert "*" not in run_id

    def test_save_and_load_run(self, temp_store, sample_config, sample_results):
        """Test saving and loading a complete run"""
        run_id = temp_store.generate_run_id("test_save")

        # Save
        run_path = temp_store.save_run(
            run_id=run_id,
            config=sample_config,
            results=sample_results,
            duration=5.5,
            name="Test Experiment",
            tags=["test", "baseline"]
        )

        # Verify directory structure
        assert run_path.exists()
        assert (run_path / "config.yaml").exists()
        assert (run_path / "results.json").exists()
        assert (run_path / "metadata.json").exists()
        assert (run_path / "artifacts").exists()

        # Load
        loaded = temp_store.load_run(run_id)
        assert loaded['config']['num_rounds'] == 10
        assert loaded['results']['summary']['sender_total'] == 80.0
        assert loaded['metadata']['name'] == "Test Experiment"

    def test_list_runs_empty(self, temp_store):
        """Test listing runs when store is empty"""
        runs = temp_store.list_runs()
        assert runs == []

    def test_list_runs_with_multiple_saves(self, temp_store, sample_config, sample_results):
        """Test listing after saving multiple runs"""
        # Save multiple runs
        for i in range(3):
            run_id = temp_store.generate_run_id(f"test_{i}")
            temp_store.save_run(
                run_id=run_id,
                config=sample_config,
                results=sample_results,
                name=f"Test {i}"
            )

        # List
        runs = temp_store.list_runs()
        assert len(runs) == 3

    def test_delete_run(self, temp_store, sample_config, sample_results):
        """Test deleting a run"""
        run_id = temp_store.generate_run_id("test_delete")

        # Save
        temp_store.save_run(
            run_id=run_id,
            config=sample_config,
            results=sample_results
        )

        # Verify exists
        assert temp_store.get_run_path(run_id).exists()

        # Delete
        assert temp_store.delete_run(run_id) is True

        # Verify deleted
        assert not temp_store.get_run_path(run_id).exists()

        # Try to delete again
        assert temp_store.delete_run(run_id) is False

    def test_load_nonexistent_run_raises(self, temp_store):
        """Test that loading non-existent run raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            temp_store.load_run("nonexistent_run")
