# ABOUTME: SQLite database for indexing experiment results
# ABOUTME: Provides fast queries and filtering across all runs

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import threading


class ResultsDatabase:
    """
    SQLite database for indexing experiment results.

    Enables fast queries across all runs for listing, filtering, and comparison.
    """

    def __init__(self, db_path: Path):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_database(self):
        """Create database schema if it doesn't exist"""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    timestamp TEXT NOT NULL,
                    duration_seconds REAL,
                    num_rounds INTEGER NOT NULL,
                    sender_total REAL NOT NULL,
                    receiver_total REAL NOT NULL,
                    buy_rate REAL NOT NULL,
                    avg_informativeness REAL NOT NULL,
                    receiver_regret REAL NOT NULL,
                    use_llm INTEGER NOT NULL,
                    llm_model TEXT,
                    git_commit TEXT,
                    tags TEXT,
                    path TEXT NOT NULL
                )
            """)

            # Create indexes for common queries
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_name ON runs(name)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sender_total ON runs(sender_total)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_receiver_total ON runs(receiver_total)
            """)

    def insert_run(self, run_data: Dict[str, Any]) -> None:
        """
        Insert a new run into the database.

        Args:
            run_data: Dictionary containing run metadata
        """
        # Convert tags list to comma-separated string
        tags = run_data.get('tags', [])
        if isinstance(tags, list):
            tags = ','.join(tags)

        with self._conn:
            self._conn.execute("""
                INSERT INTO runs (
                    id, name, timestamp, duration_seconds, num_rounds,
                    sender_total, receiver_total, buy_rate, avg_informativeness,
                    receiver_regret, use_llm, llm_model, git_commit, tags, path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_data['id'],
                run_data.get('name'),
                run_data['timestamp'],
                run_data.get('duration_seconds'),
                run_data['num_rounds'],
                run_data['sender_total'],
                run_data['receiver_total'],
                run_data['buy_rate'],
                run_data['avg_informativeness'],
                run_data['receiver_regret'],
                int(run_data.get('use_llm', False)),
                run_data.get('llm_model'),
                run_data.get('git_commit'),
                tags,
                run_data['path']
            ))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run data dictionary or None if not found
        """
        cursor = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def list_runs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        sort_by: str = "timestamp",
        sort_order: str = "DESC",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering and sorting.

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            sort_by: Column to sort by
            sort_order: Sort order (ASC or DESC)
            filters: Dictionary of filter conditions

        Returns:
            List of run dictionaries
        """
        # Build query
        query = "SELECT * FROM runs"
        params = []

        # Add filters
        if filters:
            conditions = []
            for key, value in filters.items():
                if key.endswith('_min'):
                    field = key[:-4]
                    conditions.append(f"{field} >= ?")
                    params.append(value)
                elif key.endswith('_max'):
                    field = key[:-4]
                    conditions.append(f"{field} <= ?")
                    params.append(value)
                elif key == 'name_like':
                    conditions.append("name LIKE ?")
                    params.append(f"%{value}%")
                elif key == 'tags_contain':
                    conditions.append("tags LIKE ?")
                    params.append(f"%{value}%")
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        # Add sorting
        query += f" ORDER BY {sort_by} {sort_order}"

        # Add pagination
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def count_runs(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count total runs matching filters.

        Args:
            filters: Dictionary of filter conditions

        Returns:
            Count of matching runs
        """
        query = "SELECT COUNT(*) FROM runs"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                if key.endswith('_min'):
                    field = key[:-4]
                    conditions.append(f"{field} >= ?")
                    params.append(value)
                elif key.endswith('_max'):
                    field = key[:-4]
                    conditions.append(f"{field} <= ?")
                    params.append(value)
                elif key == 'name_like':
                    conditions.append("name LIKE ?")
                    params.append(f"%{value}%")
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run from the database.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM runs WHERE id = ?",
                (run_id,)
            )
            return cursor.rowcount > 0

    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')
