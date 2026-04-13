# CLI-Based Simulation Gym Architecture

## Overview

This document captures the complete architecture, patterns, and implementation details for building professional CLI-based simulation frameworks with interactive Marimo dashboards. Based on learnings from **garbling-gym** (cryptographic circuit simulation) and **ads-market-sim** (programmatic advertising marketplace), this guide provides a proven blueprint for similar projects.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Project Structure](#project-structure)
3. [CLI Design Patterns](#cli-design-patterns)
4. [Storage System](#storage-system)
5. [Configuration System](#configuration-system)
6. [Marimo Integration](#marimo-integration)
7. [Export Systems](#export-systems)
8. [Testing Strategy](#testing-strategy)
9. [Migration Path](#migration-path)
10. [Common Pitfalls](#common-pitfalls)

---

## Core Architecture

### Design Philosophy

**Single Responsibility**: Each component has one job
- **CLI**: User interaction and command orchestration
- **Storage**: File-based persistence and indexing
- **Core**: Domain logic and simulation execution
- **Export**: Result transformation and presentation
- **Web**: Interactive visualization and exploration

**Progressive Enhancement**: Build in layers
1. Core simulation engine (standalone)
2. CLI commands (basic functionality)
3. Storage and indexing (persistence)
4. Configuration system (flexibility)
5. Export systems (sharing)
6. Interactive dashboards (exploration)

### High-Level Flow

```
User Input (CLI)
    ↓
Command Handler
    ↓
Configuration Loader ←→ Config Files (YAML/Python)
    ↓
Simulation Execution ←→ Core Engine
    ↓
Results Processing
    ↓
Storage System → Files + SQLite Index
    ↓
Display/Export ←→ Rich Terminal / HTML / Marimo
```

---

## Project Structure

### Standard Layout

```
project/
├── src/
│   └── project_name/
│       ├── core/                    # Domain logic
│       │   ├── __init__.py
│       │   ├── config.py           # Configuration classes
│       │   ├── types.py            # Shared types/enums
│       │   └── ...                 # Core domain modules
│       │
│       ├── cli/                     # Command-line interface
│       │   ├── __init__.py
│       │   ├── __main__.py         # CLI entry point
│       │   ├── config.py           # CLI-specific config
│       │   └── commands/           # Command implementations
│       │       ├── __init__.py
│       │       ├── run.py          # Single run
│       │       ├── list.py         # Browse history
│       │       ├── experiment.py   # Batch runs
│       │       ├── compare.py      # Multi-run comparison
│       │       ├── visualize.py    # Launch notebook for run
│       │       ├── serve.py        # Launch dashboard
│       │       └── export.py       # Export to HTML/CSV
│       │
│       ├── storage/                 # Persistence layer
│       │   ├── __init__.py
│       │   ├── store.py            # File-based storage
│       │   └── database.py         # SQLite indexing
│       │
│       ├── exports/                 # Export systems
│       │   ├── __init__.py
│       │   ├── html.py             # HTML reports
│       │   └── csv.py              # CSV export
│       │
│       ├── web/                     # Interactive dashboards
│       │   ├── __init__.py
│       │   └── notebooks/
│       │       ├── dashboard.py    # Main dashboard
│       │       ├── run_details.py  # Single run viewer
│       │       └── comparison.py   # Multi-run comparison
│       │
│       └── models/                  # Domain models (legacy compat)
│           └── ...
│
├── tests/                           # Test suite
│   ├── core/
│   ├── cli/
│   ├── storage/
│   └── ...
│
├── experiments/                     # Example configs
│   ├── yaml/
│   │   ├── quick_test.yaml
│   │   ├── baseline.yaml
│   │   └── ...
│   └── python/
│       └── ...
│
├── run_results/                     # Simulation outputs
│   ├── .index.db                   # SQLite index
│   ├── 20260203_120000_run_name/
│   │   ├── config.yaml
│   │   ├── results.json
│   │   ├── metadata.json
│   │   └── artifacts/
│   └── ...
│
├── pyproject.toml                   # Package configuration
├── README.md
└── .gitignore
```

### Key Directory Responsibilities

**`core/`**: Pure domain logic, no I/O
- Configuration data classes
- Simulation engine
- Shared types and utilities
- No dependencies on CLI/storage

**`cli/`**: User interaction only
- Argument parsing (Click)
- Progress display (Rich)
- Command orchestration
- Thin layer over core logic

**`storage/`**: Persistence abstraction
- File operations
- Database queries
- No knowledge of CLI or domain logic
- Generic, reusable

**`web/`**: Interactive visualization
- Marimo notebooks
- Plotly/Altair charts
- Reads from storage, never writes
- Optional dependency

---

## CLI Design Patterns

### Command Architecture

**Use Click for Command Groups**

```python
# cli/__main__.py
import click
from pathlib import Path
from dotenv import load_dotenv

# Load .env if exists
env_path = Path.cwd() / ".env"
if env_path.exists():
    load_dotenv(env_path)

@click.group()
@click.version_option()
def cli():
    """Project Name - Description of what it does.

    This CLI provides tools for running simulations, managing
    experiments, and analyzing results.
    """
    pass

# Register commands
from .commands import run, list_runs, experiment, visualize, compare, export, serve

cli.add_command(run.run)
cli.add_command(list_runs.list_runs)
cli.add_command(experiment.experiment)
cli.add_command(visualize.visualize)
cli.add_command(compare.compare)
cli.add_command(export.export)
cli.add_command(serve.serve)

if __name__ == '__main__':
    cli()
```

### Command Pattern: Run

**Single simulation execution with all bells and whistles**

```python
# cli/commands/run.py
import click
from pathlib import Path
from datetime import datetime
from typing import Optional
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from project_name.core.config import SimConfig
from project_name.storage import ResultsStore, RunDatabase

console = Console()

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Config file (YAML/Python)')
@click.option('--param1', type=int, help='Simulation parameter 1')
@click.option('--param2', type=int, help='Simulation parameter 2')
@click.option('--name', '-n', help='Run name')
@click.option('--tags', multiple=True, help='Tags (can be specified multiple times)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
@click.option('--no-save', is_flag=True, help='Do not save results')
def run(config, param1, param2, name, tags, quiet, no_save):
    """Run a single simulation.

    Examples:

        \b
        # Quick test
        project run --param1 100 --name "quick_test"

        \b
        # From config file
        project run --config experiments/yaml/baseline.yaml

        \b
        # Override config values
        project run --config baseline.yaml --param1 200
    """
    # 1. Load base config (from file or defaults)
    if config:
        sim_config = _load_config_file(Path(config))
    else:
        sim_config = SimConfig()

    # 2. Apply CLI overrides
    if param1 is not None:
        sim_config.param1 = param1
    if param2 is not None:
        sim_config.param2 = param2
    if name:
        sim_config.name = name
    if tags:
        sim_config.tags.extend(tags)

    # 3. Generate unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = sim_config.name or "unnamed"
    run_id = f"{timestamp}_{run_name}".replace(" ", "_")

    # 4. Display config (unless quiet)
    if not quiet:
        _display_config(sim_config, run_id)

    # 5. Run simulation with progress bar
    start_time = time.time()

    if not quiet:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("[cyan]Running simulation...", total=None)
            results = run_simulation(sim_config)
            progress.update(task, completed=True)
    else:
        results = run_simulation(sim_config)

    duration = time.time() - start_time

    # 6. Display results (unless quiet)
    if not quiet:
        _display_results(results)

    # 7. Save results
    if not no_save:
        _save_results(run_id, sim_config, results, duration, list(tags))
        if not quiet:
            console.print(f"\n[green]✓[/green] Results saved to run_results/{run_id}")
            console.print(f"[dim]Run ID: {run_id}[/dim]")

    if not quiet:
        console.print(f"\n[green]Simulation completed in {duration:.2f}s[/green]")
```

**Key Patterns**:
1. **Config → Override → Execute**: Load base, apply CLI args, run
2. **Rich Output**: Use Rich Console for beautiful terminal output
3. **Unique IDs**: Timestamp + name for sortable, readable run IDs
4. **Quiet Mode**: Always support `--quiet` for scripting
5. **No-Save Mode**: Allow dry runs with `--no-save`

### Command Pattern: List

**Browse simulation history with filtering**

```python
# cli/commands/list.py
import click
from rich.console import Console
from rich.table import Table

from project_name.storage import RunDatabase

console = Console()

@click.command(name="list-runs")
@click.option('--limit', '-n', type=int, default=20, help='Number of runs to show')
@click.option('--sort', '-s', default='timestamp',
              help='Sort by field (timestamp, duration, metric1)')
@click.option('--tag', multiple=True, help='Filter by tag')
@click.option('--name', help='Filter by name (substring match)')
@click.option('--min-metric', type=float, help='Minimum metric value')
@click.option('--since', help='Show runs since date (YYYY-MM-DD)')
@click.option('--all', '-a', 'show_all', is_flag=True, help='Show all runs')
def list_runs(limit, sort, tag, name, min_metric, since, show_all):
    """List past simulation runs.

    Examples:

        \b
        # Show 20 most recent
        project list-runs

        \b
        # Filter by tag
        project list-runs --tag baseline --all

        \b
        # Sort by metric
        project list-runs --sort metric1 --limit 10
    """
    db = RunDatabase()

    # Build filters
    filters = {}
    if tag:
        filters["tags"] = list(tag)
    if name:
        filters["name"] = name
    if min_metric is not None:
        filters["min_metric"] = min_metric
    if since:
        filters["since"] = since

    # Query database
    query_limit = None if show_all else limit
    runs = db.list_runs(
        limit=query_limit,
        sort_by=sort,
        sort_order="desc",
        filters=filters if filters else None
    )

    if not runs:
        console.print("[yellow]No runs found matching filters[/yellow]")
        return

    # Display as rich table
    _display_runs_table(runs, sort)

    # Show summary
    console.print(f"\n[dim]Showing {len(runs)} run(s)[/dim]")
```

**Key Patterns**:
1. **Flexible Filtering**: Multiple independent filters that combine
2. **Rich Tables**: Beautiful, readable output with Rich
3. **Sorting Support**: Sort by any indexed column
4. **Summary Stats**: Always show what's being displayed

### Command Pattern: Experiment

**Batch execution from config file**

```python
# cli/commands/experiment.py
import click
from pathlib import Path
import yaml
from rich.console import Console
from rich.progress import Progress, TaskProgressColumn

from project_name.core.config import SimConfig
from project_name.cli.commands.run import run_simulation, _save_results

console = Console()

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--parallel', is_flag=True, help='Run experiments in parallel')
@click.option('--tags', multiple=True, help='Additional tags for all runs')
def experiment(config_file, parallel, tags):
    """Run batch experiments from config file.

    Config file format (YAML):

        \b
        experiments:
          - name: baseline
            param1: 100
            param2: 50
            tags: [baseline, test]

          - name: variant_1
            param1: 200
            param2: 50
            tags: [variant, test]

    Examples:

        \b
        # Run experiment batch
        project experiment experiments/batch_comparison.yaml

        \b
        # Add tags to all runs
        project experiment batch.yaml --tags production
    """
    # Load experiment config
    with open(config_file) as f:
        config = yaml.safe_load(f)

    experiments = config['experiments']
    console.print(f"[cyan]Running {len(experiments)} experiments[/cyan]\n")

    results_summary = []

    with Progress(*Progress.get_default_columns(), TaskProgressColumn()) as progress:
        task = progress.add_task("[cyan]Experiments", total=len(experiments))

        for exp in experiments:
            # Create config
            exp_config = SimConfig(**exp)
            if tags:
                exp_config.tags.extend(tags)

            # Generate run ID
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp_config.name}"

            # Run simulation
            progress.console.print(f"  Running: [yellow]{exp_config.name}[/yellow]")
            start_time = time.time()
            results = run_simulation(exp_config)
            duration = time.time() - start_time

            # Save results
            _save_results(run_id, exp_config, results, duration, exp_config.tags)

            # Track summary
            results_summary.append({
                'name': exp_config.name,
                'run_id': run_id,
                'duration': duration,
                'key_metric': results['summary']['key_metric']
            })

            progress.update(task, advance=1)

    # Display summary table
    _display_summary_table(results_summary)

    console.print(f"\n[green]✓ All experiments completed[/green]")
```

**Key Patterns**:
1. **YAML-based**: Easy to edit, version control friendly
2. **Progress Tracking**: Show which experiment is running
3. **Summary Table**: Show all results at the end
4. **Batch Tagging**: Add tags to entire batch

### Command Pattern: Visualize

**Launch interactive Marimo notebook**

```python
# cli/commands/visualize.py
import click
import subprocess
import os
from pathlib import Path

from rich.console import Console

console = Console()

@click.command()
@click.argument('run_id')
@click.option('--port', '-p', type=int, default=None, help='Port number')
def visualize(run_id, port):
    """Launch interactive visualization for a run.

    Opens a Marimo notebook with the specified run pre-loaded.

    Example:

        \b
        project visualize 20260203_120000_baseline
    """
    # Check if marimo is installed
    try:
        import marimo
    except ImportError:
        console.print("[red]Error: marimo not installed[/red]")
        console.print("Install with: uv pip install marimo")
        return

    # Get notebook path
    notebook_path = Path(__file__).parent.parent.parent / "web" / "notebooks" / "run_details.py"

    if not notebook_path.exists():
        console.print(f"[red]Error: Notebook not found at {notebook_path}[/red]")
        return

    # Set environment variable for run ID
    env = os.environ.copy()
    env['SIMULATION_RUN_ID'] = run_id

    # Build command
    cmd = ['marimo', 'edit', str(notebook_path)]
    if port:
        cmd.extend(['--port', str(port)])

    console.print(f"[cyan]Launching visualization for run: {run_id}[/cyan]")
    console.print(f"[dim]Opening notebook: {notebook_path.name}[/dim]\n")

    # Launch marimo
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        console.print("\n[yellow]Visualization closed[/yellow]")
```

**Key Patterns**:
1. **Environment Variables**: Pass run ID to notebook
2. **Optional Dependencies**: Check if marimo installed
3. **Clean Launch**: Use subprocess to launch marimo server

### Command Pattern: Serve

**Launch dashboard server**

```python
# cli/commands/serve.py
import click
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

@click.command()
@click.option('--port', '-p', type=int, default=8000, help='Port number')
@click.option('--host', default='localhost', help='Host address')
def serve(port, host):
    """Launch Marimo dashboard server.

    Opens the main dashboard for browsing all runs.

    Example:

        \b
        project serve --port 8080
    """
    try:
        import marimo
    except ImportError:
        console.print("[red]Error: marimo not installed[/red]")
        console.print("Install with: uv pip install marimo")
        return

    notebook_path = Path(__file__).parent.parent.parent / "web" / "notebooks" / "dashboard.py"

    if not notebook_path.exists():
        console.print(f"[red]Error: Dashboard not found at {notebook_path}[/red]")
        return

    console.print(f"[cyan]Starting dashboard server on {host}:{port}[/cyan]")
    console.print(f"[dim]Dashboard: {notebook_path.name}[/dim]\n")

    cmd = ['marimo', 'edit', str(notebook_path), '--host', host, '--port', str(port)]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard server stopped[/yellow]")
```

---

## Storage System

### File-Based Storage (ResultsStore)

**Structure**: One directory per run

```
run_results/
├── 20260203_120000_baseline/
│   ├── config.yaml          # Human-readable configuration
│   ├── results.json         # Complete results data
│   ├── metadata.json        # Run metadata (timestamp, duration, tags)
│   └── artifacts/           # Additional outputs (plots, exports)
│       ├── plot1.png
│       └── export.html
└── .index.db                # SQLite index for fast queries
```

**Implementation**:

```python
# storage/store.py
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from datetime import datetime

class ResultsStore:
    """File-based storage for simulation results."""

    def __init__(self, base_dir: Path = Path("run_results")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        duration: float,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Path:
        """Save a simulation run."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config as YAML (human-readable)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Save results as JSON
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save metadata
        metadata = {
            "run_id": run_id,
            "name": name,
            "tags": tags or [],
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create artifacts directory
        (run_dir / "artifacts").mkdir(exist_ok=True)

        return run_dir

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load a simulation run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_id} not found")

        with open(run_dir / "metadata.json") as f:
            metadata = json.load(f)

        with open(run_dir / "results.json") as f:
            results = json.load(f)

        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        return {
            "metadata": metadata,
            "results": results,
            "config": config,
            "run_dir": run_dir
        }

    def list_run_ids(self) -> List[str]:
        """List all run IDs, sorted by modification time."""
        if not self.base_dir.exists():
            return []

        run_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [d.name for d in run_dirs]
```

**Key Design Decisions**:
1. **YAML for config**: Human-readable, easy to diff in git
2. **JSON for results**: Structured, widely supported
3. **Separate metadata**: Fast access to run info without loading full results
4. **Artifacts directory**: Keep all run-related files together

### SQLite Index (RunDatabase)

**Purpose**: Fast filtering and sorting without loading files

```python
# storage/database.py
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class RunDatabase:
    """SQLite database for indexing simulation runs."""

    def __init__(self, db_path: Path = Path("run_results/.index.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Main runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT,
                    timestamp TEXT,
                    duration REAL,
                    config_param1 INTEGER,
                    config_param2 INTEGER,
                    result_metric1 REAL,
                    result_metric2 REAL
                )
            """)

            # Tags table (many-to-many)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    run_id TEXT,
                    tag TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_run_id ON tags(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert_run(self, metadata: Dict[str, Any], results: Dict[str, Any]):
        """Insert a run into the database."""
        with self._get_connection() as conn:
            # Extract data
            config = metadata["config"]
            summary = results.get("summary", {})

            # Insert run
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, name, timestamp, duration,
                    config_param1, config_param2,
                    result_metric1, result_metric2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata["run_id"],
                metadata.get("name"),
                metadata["timestamp"],
                metadata["duration"],
                config.get("param1"),
                config.get("param2"),
                summary.get("metric1"),
                summary.get("metric2")
            ))

            # Delete old tags
            conn.execute("DELETE FROM tags WHERE run_id = ?", (metadata["run_id"],))

            # Insert tags
            for tag in metadata.get("tags", []):
                conn.execute("INSERT INTO tags (run_id, tag) VALUES (?, ?)",
                           (metadata["run_id"], tag))

            conn.commit()

    def list_runs(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List runs with filtering and sorting."""
        filters = filters or {}

        # Build WHERE clauses
        where_clauses = []
        params = []

        if "name" in filters:
            where_clauses.append("runs.name LIKE ?")
            params.append(f"%{filters['name']}%")

        if "min_metric1" in filters:
            where_clauses.append("result_metric1 >= ?")
            params.append(filters["min_metric1"])

        if "tags" in filters and filters["tags"]:
            tag_placeholders = ",".join("?" * len(filters["tags"]))
            where_clauses.append(f"run_id IN (SELECT run_id FROM tags WHERE tag IN ({tag_placeholders}))")
            params.extend(filters["tags"])

        # Build query
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Validate sort_by (prevent SQL injection)
        allowed_sorts = ["run_id", "name", "timestamp", "duration",
                        "result_metric1", "result_metric2"]
        if sort_by not in allowed_sorts:
            sort_by = "timestamp"

        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        order_sql = f"ORDER BY {sort_by} {sort_direction}"

        limit_sql = f"LIMIT {limit}" if limit else ""

        query = f"SELECT * FROM runs {where_sql} {order_sql} {limit_sql}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            runs = [dict(row) for row in cursor.fetchall()]

            # Fetch tags for each run
            for run in runs:
                cursor = conn.execute(
                    "SELECT tag FROM tags WHERE run_id = ?",
                    (run["run_id"],)
                )
                run["tags"] = [row["tag"] for row in cursor.fetchall()]

        return runs
```

**Key Design Decisions**:
1. **Denormalized metrics**: Store key metrics in runs table for fast queries
2. **Many-to-many tags**: Flexible tagging system
3. **Indexes**: On timestamp and tags for fast filtering
4. **Row factory**: Return dicts, not tuples
5. **SQL injection protection**: Whitelist sort fields

---

## Configuration System

### Two-Level Configuration

**1. SimConfig (Core)**: Domain-specific settings

```python
# core/config.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SimConfig:
    """Core simulation configuration."""
    # Simulation parameters
    param1: int = 100
    param2: int = 50
    param3: float = 0.5
    seed: Optional[int] = 42

    # Output settings
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        assert self.param1 > 0, "param1 must be positive"
        assert 0 <= self.param3 <= 1, "param3 must be in [0, 1]"
```

**2. CLI Config (Optional)**: Additional CLI settings

```python
# cli/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CLIConfig:
    """CLI-specific configuration."""
    results_dir: Path = Path("run_results")
    default_limit: int = 20
    auto_open_browser: bool = True

    @classmethod
    def load_from_env(cls):
        """Load from environment variables."""
        import os
        return cls(
            results_dir=Path(os.getenv("RESULTS_DIR", "run_results")),
            default_limit=int(os.getenv("DEFAULT_LIMIT", "20")),
            auto_open_browser=os.getenv("AUTO_OPEN_BROWSER", "true").lower() == "true"
        )
```

### Configuration File Support

**YAML Configuration**:

```yaml
# experiments/yaml/baseline.yaml
name: baseline_experiment
param1: 100
param2: 50
param3: 0.8
seed: 42
tags:
  - baseline
  - production
```

**Python Configuration** (for complex cases):

```python
# experiments/python/advanced.py
from project_name.core.config import SimConfig

config = SimConfig(
    name="advanced_experiment",
    param1=200,
    param2=100,
    param3=0.9,
    seed=123,
    tags=["advanced", "test"]
)

# Can include logic
if some_condition:
    config.param1 *= 2
```

**Config Loader**:

```python
# cli/config.py (additional methods)
def load_config_file(path: Path) -> SimConfig:
    """Load configuration from YAML or Python file."""
    if path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return SimConfig(**data)

    elif path.suffix == '.py':
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}")
```

---

## Marimo Integration

### Dashboard Architecture

**Three-Tier Dashboard System**:

1. **Main Dashboard** (`dashboard.py`): Browse all runs
2. **Run Details** (`run_details.py`): Deep dive into single run
3. **Comparison** (`comparison.py`): Multi-run comparison

### Dashboard Pattern: Main Dashboard

```python
# web/notebooks/dashboard.py
import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from pathlib import Path
    from project_name.storage import ResultsStore, RunDatabase
    return mo, pd, go, Path, ResultsStore, RunDatabase

@app.cell
def __(ResultsStore, RunDatabase):
    """Load all runs from storage"""
    store = ResultsStore()
    db = RunDatabase()

    # Get all runs from database
    all_runs = db.list_runs(limit=None)

    return store, db, all_runs

@app.cell
def __(mo, all_runs):
    """Create filters"""
    # Extract unique tags
    all_tags = sorted(list(set(tag for run in all_runs for tag in run.get('tags', []))))

    # Create UI elements
    tag_filter = mo.ui.multiselect(
        options=all_tags,
        label="Filter by tags",
        value=[]
    )

    name_filter = mo.ui.text(
        label="Filter by name",
        placeholder="Enter name substring..."
    )

    _filters_output = mo.vstack([
        mo.md("## Filters"),
        tag_filter,
        name_filter
    ])

    _filters_output
    return tag_filter, name_filter, all_tags

@app.cell
def __(all_runs, tag_filter, name_filter):
    """Apply filters to runs"""
    filtered_runs = all_runs

    # Filter by tags
    if tag_filter.value:
        filtered_runs = [
            run for run in filtered_runs
            if any(tag in run.get('tags', []) for tag in tag_filter.value)
        ]

    # Filter by name
    if name_filter.value:
        filtered_runs = [
            run for run in filtered_runs
            if name_filter.value.lower() in (run.get('name') or '').lower()
        ]

    return filtered_runs,

@app.cell
def __(mo, pd, filtered_runs):
    """Display runs table with selection"""
    # Convert to DataFrame
    df_runs = pd.DataFrame(filtered_runs)

    # Format for display
    df_display = df_runs[['run_id', 'name', 'timestamp', 'duration',
                           'result_metric1', 'result_metric2']].copy()
    df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    df_display['duration'] = df_display['duration'].round(2).astype(str) + 's'

    # Create interactive table
    runs_table = mo.ui.table(
        df_display,
        selection='multi',
        page_size=20,
        label="Select runs for comparison"
    )

    runs_table
    return runs_table, df_runs, df_display

@app.cell
def __(mo, runs_table, df_runs):
    """Comparison chart (only if 2+ runs selected)"""
    if runs_table is not None and len(runs_table.value) >= 2:
        # Get selected run IDs
        selected_ids = runs_table.value['run_id'].tolist()
        selected_runs = df_runs[df_runs['run_id'].isin(selected_ids)]

        # Create comparison chart
        import plotly.graph_objects as go

        fig = go.Figure()

        for _, run in selected_runs.iterrows():
            fig.add_trace(go.Bar(
                name=run['name'] or run['run_id'][:20],
                x=['Metric 1', 'Metric 2'],
                y=[run['result_metric1'], run['result_metric2']]
            ))

        fig.update_layout(
            title="Selected Runs Comparison",
            barmode='group',
            height=400
        )

        _chart_output = mo.vstack([
            mo.md("## Comparison"),
            mo.ui.plotly(fig)
        ])

        _chart_output
    return

@app.cell
def __(mo, filtered_runs):
    """Summary statistics"""
    if filtered_runs:
        avg_duration = sum(r['duration'] for r in filtered_runs) / len(filtered_runs)
        avg_metric1 = sum(r['result_metric1'] for r in filtered_runs) / len(filtered_runs)

        _stats_output = mo.md(f"""
        ## Summary Statistics

        - **Total runs**: {len(filtered_runs)}
        - **Avg duration**: {avg_duration:.2f}s
        - **Avg metric1**: {avg_metric1:.4f}
        """)

        _stats_output
    return

if __name__ == "__main__":
    app.run()
```

**Key Patterns**:
1. **Cell output**: Always assign to `_output` and display as last statement
2. **Table selection**: Use `table.value['column'].tolist()` to extract data
3. **Conditional rendering**: Check `len(table.value) >= 2` before rendering
4. **Private variables**: Use `_` prefix for local variables
5. **Reactivity**: Cells auto-rerun when dependencies change

### Run Details Pattern

```python
# web/notebooks/run_details.py
import marimo
import os

app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from project_name.storage import ResultsStore
    return mo, pd, go, ResultsStore

@app.cell
def __(mo, ResultsStore):
    """Load run from environment variable or prompt user"""
    import os

    # Check if run ID passed via environment
    run_id_from_env = os.getenv('SIMULATION_RUN_ID')

    if run_id_from_env:
        run_id_input = mo.md(f"**Run ID**: `{run_id_from_env}`")
    else:
        run_id_input = mo.ui.text(
            label="Enter Run ID",
            placeholder="20260203_120000_baseline"
        )

    run_id_input
    return run_id_from_env, run_id_input

@app.cell
def __(mo, ResultsStore, run_id_from_env, run_id_input):
    """Load run data"""
    run_id = run_id_from_env or run_id_input.value

    if not run_id:
        _no_id_output = mo.md("_Enter a run ID above to load data_")
        _no_id_output
        return

    try:
        store = ResultsStore()
        run_data = store.load_run(run_id)

        _loaded_output = mo.md(f"✓ Loaded run: **{run_id}**")
        _loaded_output

    except FileNotFoundError:
        _error_output = mo.md(f"❌ Run not found: {run_id}")
        _error_output
        return

    return run_data, run_id

@app.cell
def __(mo, run_data):
    """Display run metadata"""
    metadata = run_data['metadata']

    _metadata_output = mo.md(f"""
    ## Run Metadata

    - **Name**: {metadata.get('name', 'N/A')}
    - **Timestamp**: {metadata['timestamp']}
    - **Duration**: {metadata['duration']:.2f}s
    - **Tags**: {', '.join(metadata.get('tags', []))}
    """)

    _metadata_output
    return metadata,

@app.cell
def __(mo, pd, go, run_data):
    """Visualize results"""
    results = run_data['results']

    # Create time series plot (example)
    df = pd.DataFrame(results.get('time_series', []))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['metric1'],
        mode='lines',
        name='Metric 1'
    ))

    fig.update_layout(
        title="Metric Over Time",
        xaxis_title="Time",
        yaxis_title="Metric Value"
    )

    _chart_output = mo.ui.plotly(fig)
    _chart_output
    return
```

---

## Export Systems

### HTML Export Pattern

**Self-contained HTML with embedded charts**

```python
# exports/html.py
from pathlib import Path
from typing import Dict, Any
import base64
from io import BytesIO

import matplotlib.pyplot as plt

def export_to_html(run_data: Dict[str, Any], output_path: Path):
    """Export run to self-contained HTML report."""
    metadata = run_data['metadata']
    results = run_data['results']
    config = run_data['config']

    # Generate charts as base64-encoded images
    charts = _generate_charts(results)

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Run Report: {metadata.get('name', metadata['run_id'])}</title>
        <style>
            {_get_css()}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Simulation Run Report</h1>

            <section class="metadata">
                <h2>Metadata</h2>
                <table>
                    <tr><th>Run ID</th><td>{metadata['run_id']}</td></tr>
                    <tr><th>Name</th><td>{metadata.get('name', 'N/A')}</td></tr>
                    <tr><th>Timestamp</th><td>{metadata['timestamp']}</td></tr>
                    <tr><th>Duration</th><td>{metadata['duration']:.2f}s</td></tr>
                    <tr><th>Tags</th><td>{', '.join(metadata.get('tags', []))}</td></tr>
                </table>
            </section>

            <section class="config">
                <h2>Configuration</h2>
                <pre>{_format_config(config)}</pre>
            </section>

            <section class="results">
                <h2>Results</h2>
                {_format_results_summary(results)}
            </section>

            <section class="charts">
                <h2>Visualizations</h2>
                {_embed_charts(charts)}
            </section>
        </div>
    </body>
    </html>
    """

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)

def _generate_charts(results: Dict[str, Any]) -> Dict[str, str]:
    """Generate charts as base64-encoded PNG images."""
    charts = {}

    # Example: Time series chart
    fig, ax = plt.subplots(figsize=(10, 6))
    time_series = results.get('time_series', [])
    if time_series:
        times = [p['time'] for p in time_series]
        values = [p['metric1'] for p in time_series]
        ax.plot(times, values)
        ax.set_xlabel('Time')
        ax.set_ylabel('Metric 1')
        ax.set_title('Metric 1 Over Time')

        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        charts['time_series'] = f"data:image/png;base64,{img_base64}"
        plt.close()

    return charts

def _embed_charts(charts: Dict[str, str]) -> str:
    """Embed charts as <img> tags."""
    html_parts = []
    for name, data_uri in charts.items():
        html_parts.append(f'<div class="chart">')
        html_parts.append(f'  <h3>{name.replace("_", " ").title()}</h3>')
        html_parts.append(f'  <img src="{data_uri}" alt="{name}">')
        html_parts.append(f'</div>')
    return '\n'.join(html_parts)

def _get_css() -> str:
    """Return CSS styles."""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        section {
            margin: 30px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #ecf0f1;
            font-weight: 600;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .chart {
            margin: 20px 0;
        }
        .chart img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    """
```

**Key Design Decisions**:
1. **Self-contained**: All assets embedded (CSS, images)
2. **Base64 encoding**: Charts embedded as data URIs
3. **Responsive**: Works on any screen size
4. **Professional styling**: Clean, modern design

### CSV Export Pattern

```python
# exports/csv.py
from pathlib import Path
from typing import Dict, Any
import csv

def export_to_csv(run_data: Dict[str, Any], output_dir: Path):
    """Export run to directory of CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_data['results']

    # Export summary
    _export_summary(run_data, output_dir / "summary.csv")

    # Export time series
    if 'time_series' in results:
        _export_time_series(results['time_series'], output_dir / "time_series.csv")

    # Export detailed results
    for key, data in results.items():
        if isinstance(data, list) and data and isinstance(data[0], dict):
            _export_table(data, output_dir / f"{key}.csv")

def _export_summary(run_data: Dict[str, Any], path: Path):
    """Export summary metrics."""
    metadata = run_data['metadata']
    summary = run_data['results'].get('summary', {})

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Run ID', metadata['run_id']])
        writer.writerow(['Name', metadata.get('name', '')])
        writer.writerow(['Timestamp', metadata['timestamp']])
        writer.writerow(['Duration', f"{metadata['duration']:.2f}"])

        for key, value in summary.items():
            writer.writerow([key, value])

def _export_time_series(data: list, path: Path):
    """Export time series data."""
    if not data:
        return

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def _export_table(data: list, path: Path):
    """Export tabular data."""
    if not data:
        return

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
```

---

## Testing Strategy

### Test Structure

```
tests/
├── core/
│   ├── test_config.py          # Configuration validation
│   └── test_simulation.py      # Core simulation logic
├── storage/
│   ├── test_store.py           # File storage
│   └── test_database.py        # SQLite operations
├── cli/
│   ├── test_run.py             # Run command
│   ├── test_list.py            # List command
│   └── test_config.py          # Config loading
├── exports/
│   ├── test_html.py            # HTML export
│   └── test_csv.py             # CSV export
└── integration/
    └── test_full_workflow.py   # End-to-end tests
```

### Testing Patterns

**Storage Tests**:

```python
# tests/storage/test_store.py
import pytest
from pathlib import Path
import tempfile

from project_name.storage import ResultsStore

@pytest.fixture
def temp_store():
    """Create temporary results store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ResultsStore(Path(tmpdir))

def test_save_and_load_run(temp_store):
    """Test saving and loading a run."""
    config = {"param1": 100, "param2": 50}
    results = {"summary": {"metric1": 1.5}}

    # Save
    run_dir = temp_store.save_run(
        run_id="test_run",
        config=config,
        results=results,
        duration=10.5,
        name="Test Run",
        tags=["test"]
    )

    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "metadata.json").exists()

    # Load
    loaded = temp_store.load_run("test_run")
    assert loaded["config"] == config
    assert loaded["results"] == results
    assert loaded["metadata"]["name"] == "Test Run"

def test_list_run_ids(temp_store):
    """Test listing run IDs."""
    # Save multiple runs
    for i in range(3):
        temp_store.save_run(
            run_id=f"run_{i}",
            config={},
            results={},
            duration=1.0
        )

    run_ids = temp_store.list_run_ids()
    assert len(run_ids) == 3
    assert "run_0" in run_ids
```

**CLI Tests**:

```python
# tests/cli/test_run.py
from click.testing import CliRunner
from project_name.cli.__main__ import cli

def test_run_command_basic():
    """Test basic run command."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['run', '--param1', '100', '--name', 'test', '--no-save'])
        assert result.exit_code == 0
        assert 'Simulation completed' in result.output

def test_run_command_with_config(tmp_path):
    """Test run command with config file."""
    # Create config file
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("""
    name: test_config
    param1: 200
    param2: 100
    tags: [test, config]
    """)

    runner = CliRunner()
    result = runner.invoke(cli, ['run', '--config', str(config_path), '--no-save'])
    assert result.exit_code == 0
```

**Integration Tests**:

```python
# tests/integration/test_full_workflow.py
def test_full_workflow(tmp_path):
    """Test complete workflow: run -> save -> list -> load."""
    from project_name.core.config import SimConfig
    from project_name.storage import ResultsStore, RunDatabase

    store = ResultsStore(tmp_path / "results")
    db = RunDatabase(tmp_path / "results" / ".index.db")

    # Run simulation
    config = SimConfig(param1=100, name="integration_test", tags=["test"])
    results = run_simulation(config)

    # Save results
    run_id = "20260203_120000_integration_test"
    metadata = {
        "run_id": run_id,
        "name": config.name,
        "tags": config.tags,
        "duration": 5.0,
        "timestamp": "2026-02-03T12:00:00",
        "config": {"param1": config.param1}
    }

    store.save_run(run_id, metadata["config"], results, 5.0, config.name, config.tags)
    db.insert_run(metadata, results)

    # List runs
    runs = db.list_runs(filters={"tags": ["test"]})
    assert len(runs) == 1
    assert runs[0]["name"] == "integration_test"

    # Load run
    loaded = store.load_run(run_id)
    assert loaded["metadata"]["name"] == "integration_test"
```

---

## Migration Path

### From Monolithic Script to CLI Gym

**Phase 1: Foundation (Critical)**
- Fix any framework compatibility issues (e.g., Mesa 3.x migration)
- Create core module with types and configuration
- Ensure all tests pass
- Add python-dotenv support

**Phase 2: CLI Infrastructure**
- Create CLI package structure
- Add CLI dependencies (Click, Rich, PyYAML)
- Create storage system (ResultsStore)
- Create database indexing (RunDatabase)
- Implement `run` command (port existing script)
- Implement `list-runs` command

**Phase 3: Configuration & Experiments**
- Create YAML config loader
- Create example configs
- Update `run` command for config files
- Implement `experiment` command for batch runs

**Phase 4: Visualization & Export**
- Create HTML export system
- Create CSV export system
- Implement `export` command
- Add Marimo dependency (optional)
- Create dashboard notebooks
- Implement `visualize` and `serve` commands

**Phase 5: Advanced Features**
- Implement `compare` command
- Add advanced filtering to `list-runs`
- Create comparison notebook

**Phase 6: Documentation & Testing**
- Write comprehensive README
- Expand test coverage (target >70%)
- Add docstrings
- Create usage examples

**Phase 7: Git & Cleanup**
- Create .gitignore
- Initialize git repository
- Add LICENSE
- Create incremental commits
- Remove old files

### Time Estimates

- **Small project** (~1,000 lines): 8-12 hours
- **Medium project** (~5,000 lines): 20-30 hours
- **Large project** (~10,000+ lines): 40-60 hours

---

## Common Pitfalls

### 1. CLI Design

**❌ Wrong**: Inconsistent naming
```bash
project run-simulation
project listRuns
project exp
```

**✅ Correct**: Consistent, predictable naming
```bash
project run
project list-runs
project experiment
```

### 2. Storage Design

**❌ Wrong**: Storing everything in SQLite
- Makes database large
- Hard to inspect results
- Backup/export difficult

**✅ Correct**: Hybrid approach
- Full results in files (JSON)
- Metadata in SQLite for queries
- Best of both worlds

### 3. Configuration

**❌ Wrong**: CLI args only
```bash
project run --p1 100 --p2 50 --p3 0.8 --p4 True --p5 10 --p6 ...
# Unwieldy, error-prone
```

**✅ Correct**: Config files with CLI overrides
```bash
project run --config baseline.yaml --p1 200
# Readable, maintainable, reproducible
```

### 4. Marimo Integration

**❌ Wrong**: Tight coupling to Marimo
```python
# core/simulation.py
import marimo as mo  # NO!
```

**✅ Correct**: Optional dependency
```python
# web/notebooks/dashboard.py
import marimo as mo  # YES - only in web/
```

### 5. Error Handling

**❌ Wrong**: Silent failures
```python
try:
    results = run_simulation(config)
except Exception:
    pass  # Silently fails
```

**✅ Correct**: Informative errors
```python
try:
    results = run_simulation(config)
except ValidationError as e:
    console.print(f"[red]Configuration error: {e}[/red]")
    raise click.Abort()
except Exception as e:
    console.print(f"[red]Simulation failed: {e}[/red]")
    # Log full traceback
    logger.exception("Simulation error")
    raise
```

### 6. Progress Display

**❌ Wrong**: Print statements
```python
for i in range(1000):
    print(f"Step {i}/1000")  # Clutters output
```

**✅ Correct**: Rich progress bars
```python
with Progress() as progress:
    task = progress.add_task("Running...", total=1000)
    for i in range(1000):
        # Do work
        progress.update(task, advance=1)
```

### 7. Database Schema

**❌ Wrong**: Storing nested data
```sql
CREATE TABLE runs (
    ...
    results TEXT  -- Storing JSON blob
)
```

**✅ Correct**: Denormalized metrics
```sql
CREATE TABLE runs (
    ...
    result_metric1 REAL,
    result_metric2 REAL,
    ...
)
-- Query by metrics directly
```

### 8. Testing

**❌ Wrong**: Testing CLI output strings
```python
assert "Simulation completed" in result.output
# Brittle, breaks with formatting changes
```

**✅ Correct**: Testing behavior
```python
assert result.exit_code == 0
assert store.load_run(run_id) is not None
assert len(db.list_runs()) == 1
```

---

## Summary Checklist

### Project Structure
- [ ] Core module with domain logic (no I/O)
- [ ] CLI module with commands
- [ ] Storage module (files + database)
- [ ] Exports module (HTML, CSV)
- [ ] Web module with Marimo notebooks
- [ ] Tests for all modules

### CLI Commands
- [ ] `run` - Single simulation
- [ ] `list-runs` - Browse history
- [ ] `experiment` - Batch execution
- [ ] `visualize` - Run details notebook
- [ ] `serve` - Dashboard server
- [ ] `compare` - Multi-run comparison
- [ ] `export` - HTML/CSV export

### Storage System
- [ ] File-based storage (YAML config, JSON results)
- [ ] SQLite index for fast queries
- [ ] Unique run IDs (timestamp + name)
- [ ] Tag system for organization
- [ ] Metadata extraction

### Configuration
- [ ] YAML config support
- [ ] Python config support
- [ ] CLI overrides
- [ ] Validation in `__post_init__`
- [ ] Example configs in repository

### Marimo Dashboards
- [ ] Main dashboard (browse all runs)
- [ ] Run details (deep dive)
- [ ] Comparison (multi-run)
- [ ] Cell output pattern (assign + display)
- [ ] Table selections handled correctly

### Export Systems
- [ ] HTML (self-contained, embedded charts)
- [ ] CSV (multiple files)
- [ ] Charts as base64 images
- [ ] Professional styling

### Testing
- [ ] Core logic tests
- [ ] Storage tests
- [ ] CLI tests (click.testing.CliRunner)
- [ ] Integration tests
- [ ] Coverage >70%

### Documentation
- [ ] Comprehensive README
- [ ] Command examples
- [ ] Config examples
- [ ] Architecture documentation
- [ ] Docstrings on public APIs

---

## References

### Implementation Examples

- **garbling-gym**: Cryptographic circuit simulation
  - Location: `/Users/pureicis/dev/garbling-sims`
  - Features: Full CLI, Marimo dashboards, HTML export
  - Command: `gg`

- **ads-market-sim**: Programmatic advertising marketplace
  - Location: `/Users/pureicis/dev/ads-sims`
  - Features: CLI, storage system, Rich output
  - Command: `adsim`

### Tools & Frameworks

- **Click**: CLI framework - https://click.palletsprojects.com
- **Rich**: Terminal formatting - https://rich.readthedocs.io
- **Marimo**: Reactive notebooks - https://marimo.io
- **PyYAML**: YAML parsing - https://pyyaml.org
- **uv**: Fast package manager - https://github.com/astral-sh/uv

---

**Last Updated**: 2026-02-04
**Version**: 1.0
**Authors**: Based on garbling-gym and ads-market-sim implementations
