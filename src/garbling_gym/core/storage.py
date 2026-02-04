# ABOUTME: Results storage system for experiment runs
# ABOUTME: Handles saving/loading results with metadata and SQLite indexing

import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from .results import GameResults
from .config import GameConfig
from .database import ResultsDatabase


class ResultsStore:
    """
    Manages storage and retrieval of experiment results.

    Stores each run in a structured directory with:
    - config.yaml: Normalized configuration
    - results.json: Complete game history and summary
    - metadata.json: Run metadata (timestamp, duration, git hash, etc.)
    - artifacts/: Generated visualizations
    """

    def __init__(self, base_path: Path = None):
        """
        Initialize results store.

        Args:
            base_path: Base directory for storing results (defaults to ./run_results)
        """
        if base_path is None:
            base_path = Path.cwd() / "run_results"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        db_path = self.base_path / ".index.db"
        self.db = ResultsDatabase(db_path)

    def generate_run_id(self, name: Optional[str] = None) -> str:
        """
        Generate a unique run ID with timestamp and name.

        Format: YYYY-MM-DD_HHMMSS_name

        Args:
            name: Optional experiment name

        Returns:
            Unique run identifier
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if name:
            # Sanitize name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            return f"{timestamp}_{safe_name}"
        return timestamp

    def get_run_path(self, run_id: str) -> Path:
        """Get the directory path for a run"""
        return self.base_path / run_id

    def save_run(
        self,
        run_id: str,
        config: GameConfig,
        results: GameResults,
        duration: Optional[float] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None
    ) -> Path:
        """
        Save a complete run with config, results, and metadata.

        Args:
            run_id: Unique run identifier
            config: Game configuration
            results: Game results
            duration: Run duration in seconds
            name: Optional experiment name
            tags: Optional list of tags

        Returns:
            Path to the run directory
        """
        run_path = self.get_run_path(run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        # Create artifacts directory
        artifacts_path = run_path / "artifacts"
        artifacts_path.mkdir(exist_ok=True)

        # Save config as YAML
        config_data = self._config_to_dict(config)
        with open(run_path / "config.yaml", 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        # Save results as JSON
        results_data = self._results_to_dict(results)
        with open(run_path / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)

        # Get git commit hash if available
        git_commit = self._get_git_commit()

        # Save metadata
        metadata = {
            'run_id': run_id,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'git_commit': git_commit,
            'tags': tags or [],
        }
        with open(run_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Index in database
        db_entry = {
            'id': run_id,
            'name': name,
            'timestamp': metadata['timestamp'],
            'duration_seconds': duration,
            'num_rounds': results.total_rounds,
            'sender_total': results.sender_total,
            'receiver_total': results.receiver_total,
            'buy_rate': results.buy_rate,
            'avg_informativeness': results.avg_informativeness,
            'receiver_regret': results.receiver_regret,
            'use_llm': config.use_llm,
            'llm_model': config.llm_model if config.use_llm else None,
            'git_commit': git_commit,
            'tags': tags or [],
            'path': str(run_path)
        }
        self.db.insert_run(db_entry)

        return run_path

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load a complete run's data.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with 'config', 'results', and 'metadata' keys

        Raises:
            FileNotFoundError: If run doesn't exist
        """
        run_path = self.get_run_path(run_id)
        if not run_path.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found at {run_path}")

        # Load config
        with open(run_path / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Load results
        with open(run_path / "results.json", 'r') as f:
            results = json.load(f)

        # Load metadata
        with open(run_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return {
            'config': config,
            'results': results,
            'metadata': metadata,
            'path': run_path
        }

    def list_runs(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        filters: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        List runs with optional filtering and sorting.

        Args:
            limit: Maximum number of runs to return
            sort_by: Column to sort by
            filters: Dictionary of filter conditions

        Returns:
            List of run metadata dictionaries
        """
        return self.db.list_runs(
            limit=limit,
            sort_by=sort_by,
            sort_order="DESC",
            filters=filters
        )

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run and its files.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted successfully
        """
        run_path = self.get_run_path(run_id)

        # Delete from database
        self.db.delete_run(run_id)

        # Delete directory if it exists
        if run_path.exists():
            import shutil
            shutil.rmtree(run_path)
            return True

        return False

    def _config_to_dict(self, config: GameConfig) -> Dict[str, Any]:
        """Convert GameConfig to dictionary for YAML serialization"""
        from .types import AssetQuality

        return {
            'num_rounds': config.num_rounds,
            'use_llm': config.use_llm,
            'llm_model': config.llm_model,
            'prior': {
                quality.name: prob
                for quality, prob in config.prior.items()
            },
            # Note: payoffs structure is complex, using defaults for now
        }

    def _results_to_dict(self, results: GameResults) -> Dict[str, Any]:
        """Convert GameResults to dictionary for JSON serialization"""
        return {
            'summary': {
                'total_rounds': results.total_rounds,
                'sender_total': results.sender_total,
                'receiver_total': results.receiver_total,
                'buy_rate': results.buy_rate,
                'avg_informativeness': results.avg_informativeness,
                'receiver_regret': results.receiver_regret,
                'perfect_info_benchmark': results.perfect_info_benchmark,
                'strategies_used': results.strategies_used,
                'quality_stats': results.quality_stats,
            },
            'history': results.history
        }

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if in a git repository"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
