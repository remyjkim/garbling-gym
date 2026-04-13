# Garbling Gym: Comprehensive Refactoring & CLI Implementation

## Overview
Transform the existing garbling economics simulation into a modular, extensible experiment framework with CLI interface and interactive visualization.

## Design Document
See brainstorming session 2026-02-03 for full architecture and design decisions.

## Key Decisions
- **Structure**: Monorepo style under `src/garbling_gym/`
- **CLI**: `gg` entry point with comprehensive subcommands (run, experiment, visualize, compare, list, replay, export, serve)
- **Config**: Dual system - YAML (simple) and Python (advanced with custom strategies)
- **Storage**: Structured directories under `run_results/YYYY-MM-DD_HHMMSS_name/` with JSON + SQLite index
- **Visualization**: Marimo for interactive dashboards + static exports (PNG, HTML, CSV)
- **Package Manager**: uv for all dependency management

---

## Phase 1: Project Setup & Core Refactoring

### Task 1.1: Initialize uv Project Structure
**Goal**: Set up modern Python project with uv

- [ ] Create `pyproject.toml` with project metadata and dependencies
- [ ] Define `[project.scripts]` entry point: `gg = "garbling_gym.cli.__main__:main"`
- [ ] Add core dependencies: numpy, click, rich, pyyaml, pandas, matplotlib, seaborn, plotly, marimo
- [ ] Add optional dependencies: `llm` extra for openai
- [ ] Add dev dependencies: pytest, pytest-cov, black, ruff
- [ ] Create `src/garbling_gym/` directory structure
- [ ] Run `uv sync` to initialize environment
- [ ] Test that `uv run gg --help` works (even if just placeholder)

**Acceptance**: `uv sync` succeeds, project structure is created, placeholder CLI responds

---

### Task 1.2: Create Core Module Structure
**Goal**: Establish directory hierarchy for monorepo organization

Create these directories under `src/garbling_gym/`:
- [ ] `core/` - Game engine and domain logic
- [ ] `core/agents/` - Agent implementations
- [ ] `core/strategies/` - Garbling strategies
- [ ] `cli/` - CLI implementation
- [ ] `cli/commands/` - Individual command modules
- [ ] `web/` - Marimo dashboards
- [ ] `visualization/` - Static visualization utilities
- [ ] `experiments/yaml/` - Example YAML configs
- [ ] `experiments/python/` - Example Python configs

Add `__init__.py` files to make them proper packages.

**Acceptance**: All directories exist with `__init__.py`, can import `garbling_gym.core`

---

### Task 1.3: Extract Core Game Logic
**Goal**: Refactor `garbling_game.py` into modular components

**Files to create**:
- [ ] `core/types.py` - Enums (AssetQuality, Signal, Action) and base types
- [ ] `core/config.py` - GameConfig dataclass
- [ ] `core/game.py` - Game class and orchestration logic
- [ ] `core/results.py` - RoundResult, GameResults dataclasses
- [ ] `core/payoffs.py` - PayoffStructure (extracted from config)

**Refactoring steps**:
- [ ] Move enums (AssetQuality, Signal, Action) to `core/types.py`
- [ ] Move GameConfig to `core/config.py`
- [ ] Extract GarblingEconomicsGame class to `core/game.py`
- [ ] Create result classes in `core/results.py`
- [ ] Add type hints throughout
- [ ] Preserve all existing functionality

**Testing**:
- [ ] Write `tests/core/test_game.py` to verify game runs identically to old version
- [ ] Test with same random seed produces same results

**Acceptance**: Game can be instantiated and run from new modules, tests pass

---

### Task 1.4: Extract Garbling Strategies
**Goal**: Create pluggable strategy system

**Files to create**:
- [ ] `core/strategies/base.py` - GarblingStrategy base class
- [ ] `core/strategies/builtin.py` - All existing strategies (full_revelation, pool_low_medium, etc.)
- [ ] `core/strategies/registry.py` - StrategyRegistry for custom strategies

**Implementation**:
- [ ] Move GarblingMatrix to `base.py` as GarblingStrategy
- [ ] Move GARBLING_STRATEGIES dict to `builtin.py`
- [ ] Create StrategyRegistry class with register/get methods
- [ ] Pre-register all builtin strategies
- [ ] Update game.py to use registry

**Testing**:
- [ ] Write `tests/core/test_strategies.py`
- [ ] Test all builtin strategies are valid (rows sum to 1)
- [ ] Test custom strategy registration
- [ ] Test informativeness_score calculation

**Acceptance**: Strategies can be retrieved from registry, custom strategies can be registered

---

### Task 1.5: Extract Agent System
**Goal**: Create extensible agent architecture

**Files to create**:
- [ ] `core/agents/base.py` - Agent abstract base class
- [ ] `core/agents/sender.py` - SenderAgent (extracted from LLMAgent/SenderAgent)
- [ ] `core/agents/receiver.py` - ReceiverAgent (extracted from LLMAgent/ReceiverAgent)
- [ ] `core/agents/llm.py` - LLMAgent base class with OpenAI integration
- [ ] `core/agents/registry.py` - AgentFactory for creating agents

**Implementation**:
- [ ] Define Agent ABC with abstract decide() method
- [ ] Refactor SenderAgent to inherit from Agent
- [ ] Refactor ReceiverAgent to inherit from Agent
- [ ] Keep LLM and heuristic fallback logic intact
- [ ] Create factory pattern for agent instantiation

**Testing**:
- [ ] Write `tests/core/test_agents.py`
- [ ] Test heuristic agents (no API key required)
- [ ] Test agent decision-making consistency
- [ ] Mock LLM responses and test parsing

**Acceptance**: Agents can be instantiated via factory, both LLM and heuristic modes work

---

## Phase 2: CLI Implementation

### Task 2.1: Create CLI Entry Point & Framework
**Goal**: Set up Click-based CLI with subcommand structure

**Files to create**:
- [ ] `cli/__main__.py` - Main entry point with `main()` function
- [ ] `cli/utils.py` - Shared CLI utilities (error handling, output formatting)

**Implementation**:
- [ ] Create Click group for `gg` command
- [ ] Add version flag
- [ ] Add global options (--verbose, --quiet)
- [ ] Set up rich console for pretty output
- [ ] Add error handling wrapper

**Testing**:
- [ ] Test `gg --help` shows usage
- [ ] Test `gg --version` shows version
- [ ] Test invalid command shows helpful error

**Acceptance**: `uv run gg --help` shows well-formatted help text

---

### Task 2.2: Implement Results Storage System
**Goal**: Create structured result storage with SQLite indexing

**Files to create**:
- [ ] `core/storage.py` - ResultsStore class for saving/loading results
- [ ] `core/database.py` - SQLite index management

**Implementation**:
- [ ] Create `run_results/` directory structure
- [ ] Implement directory naming: `YYYY-MM-DD_HHMMSS_name/`
- [ ] Save config.yaml (normalized)
- [ ] Save results.json (full game history + summary)
- [ ] Save metadata.json (timestamp, duration, git hash, tags)
- [ ] Create `artifacts/` subdirectory
- [ ] SQLite schema: runs table with indexed columns
- [ ] Insert/query operations for index

**Testing**:
- [ ] Test directory creation with timestamp
- [ ] Test saving/loading round-trip
- [ ] Test SQLite queries (filter by date, metrics)
- [ ] Test concurrent writes (locking)

**Acceptance**: Results can be saved and retrieved, SQLite index is queryable

---

### Task 2.3: Implement Config Loading System
**Goal**: Support YAML and Python configuration files

**Files to create**:
- [ ] `cli/config.py` - ConfigLoader class

**Implementation**:
- [ ] YAML loader with schema validation (pydantic or dataclasses)
- [ ] Python loader (importlib, extract 'config' variable)
- [ ] Convert both formats to GameConfig
- [ ] Support custom strategies in Python configs
- [ ] Default config fallback

**Create example configs**:
- [ ] `experiments/yaml/baseline.yaml` - Simple YAML example
- [ ] `experiments/python/custom_strategy.py` - Python with custom strategy

**Testing**:
- [ ] Write `tests/cli/test_config.py`
- [ ] Test YAML loading
- [ ] Test Python loading with custom strategy
- [ ] Test validation errors
- [ ] Test missing file handling

**Acceptance**: Both YAML and Python configs load correctly into GameConfig

---

### Task 2.4: Implement `gg run` Command
**Goal**: Single game runner (replaces `python garbling_game.py`)

**Files to create**:
- [ ] `cli/commands/run.py`

**Implementation**:
- [ ] Click command with options: --rounds, --config, --name, --llm/--no-llm
- [ ] Load config (from file or CLI args)
- [ ] Generate run ID and create directory
- [ ] Instantiate game with config
- [ ] Run game with progress indicator (rich.progress)
- [ ] Save results using ResultsStore
- [ ] Display ASCII summary using visualization.ascii
- [ ] Print run ID and path for reference

**Testing**:
- [ ] Test basic run without config
- [ ] Test run with YAML config
- [ ] Test run with CLI overrides
- [ ] Test results are saved correctly

**Acceptance**: `gg run` executes game and saves results to timestamped directory

---

### Task 2.5: Implement `gg list` Command
**Goal**: List and filter past experiments

**Files to create**:
- [ ] `cli/commands/list.py`

**Implementation**:
- [ ] Click command with options: --sort, --filter, --limit
- [ ] Query SQLite index
- [ ] Support sorting by date, name, duration, metrics
- [ ] Support filtering by tags, date range, metric thresholds
- [ ] Display as rich table with columns: ID, Name, Date, Rounds, Sender, Receiver, Info
- [ ] Add --detail flag for verbose output

**Testing**:
- [ ] Test listing with no runs
- [ ] Test sorting by various fields
- [ ] Test filtering by metrics
- [ ] Test output formatting

**Acceptance**: `gg list` shows formatted table of past experiments with sorting/filtering

---

### Task 2.6: Implement `gg experiment` Command
**Goal**: Batch experiment runner

**Files to create**:
- [ ] `cli/commands/experiment.py`

**Implementation**:
- [ ] Click command taking config file (YAML or Python)
- [ ] Config format supports list of experiment configs
- [ ] Options: --parallel N (number of concurrent runs)
- [ ] Run experiments with progress bar (overall + per-experiment)
- [ ] Generate summary comparison after batch completes
- [ ] Save batch metadata (which runs belong to this batch)

**Example batch config**:
```yaml
experiments:
  - name: baseline-noisy
    rounds: 50
    llm: false
  - name: baseline-llm
    rounds: 50
    llm: true
```

**Testing**:
- [ ] Test single experiment in batch
- [ ] Test multiple experiments run sequentially
- [ ] Test parallel execution (--parallel 2)
- [ ] Test batch summary output

**Acceptance**: `gg experiment batch.yaml` runs multiple configs and generates summary

---

### Task 2.7: Implement `gg visualize` Command
**Goal**: Generate visualizations from saved results

**Files to create**:
- [ ] `cli/commands/visualize.py`

**Implementation**:
- [ ] Click command taking run ID or path
- [ ] Options: --format ascii|html|png, --export path
- [ ] Load results from storage
- [ ] Generate requested visualization format
- [ ] For ASCII: print to stdout using visualization.ascii
- [ ] For html/png: save to artifacts/ or --export path
- [ ] Generate multiple chart types: strategy dist, payoff timeline, info analysis

**Testing**:
- [ ] Test ASCII output to stdout
- [ ] Test HTML export to file
- [ ] Test PNG export to file
- [ ] Test invalid run ID handling

**Acceptance**: `gg visualize <run-id>` generates and displays/exports visualizations

---

### Task 2.8: Implement `gg compare` Command
**Goal**: Side-by-side comparison of multiple runs

**Files to create**:
- [ ] `cli/commands/compare.py`

**Implementation**:
- [ ] Click command taking multiple run IDs
- [ ] Options: --metric payoffs|strategies|info
- [ ] Load all results
- [ ] Generate comparison table with key metrics
- [ ] Show side-by-side charts (ASCII or exported)
- [ ] Highlight differences and outliers
- [ ] Statistical tests (if applicable)

**Testing**:
- [ ] Test comparing 2 runs
- [ ] Test comparing 3+ runs
- [ ] Test different metric focuses
- [ ] Test handling missing runs

**Acceptance**: `gg compare <id1> <id2>` shows detailed comparison

---

### Task 2.9: Implement `gg replay` Command
**Goal**: Re-run experiment from saved config

**Files to create**:
- [ ] `cli/commands/replay.py`

**Implementation**:
- [ ] Click command taking run ID
- [ ] Options: --override key=value (e.g., rounds=100)
- [ ] Load original config from run
- [ ] Apply overrides
- [ ] Run new experiment with same base config
- [ ] Link in metadata (replayed_from: original_id)

**Testing**:
- [ ] Test replay without overrides
- [ ] Test replay with overrides
- [ ] Test metadata linking

**Acceptance**: `gg replay <run-id>` re-runs with same config, creating new result

---

### Task 2.10: Implement `gg export` Command
**Goal**: Export results to various formats

**Files to create**:
- [ ] `cli/commands/export.py`

**Implementation**:
- [ ] Click command taking run ID
- [ ] Options: --format csv|json|html, --output path
- [ ] CSV: flatten history into pandas DataFrame
- [ ] JSON: dump results.json (or custom format)
- [ ] HTML: generate standalone report with embedded charts
- [ ] Default output to artifacts/ if no --output specified

**Testing**:
- [ ] Test CSV export (verify columns)
- [ ] Test JSON export (verify structure)
- [ ] Test HTML export (verify standalone)
- [ ] Test custom output path

**Acceptance**: `gg export <run-id> --format csv` generates usable CSV file

---

## Phase 3: Visualization Layer

### Task 3.1: Enhanced ASCII Visualizations
**Goal**: Rich terminal output for CLI commands

**Files to create/update**:
- [ ] `visualization/ascii.py` - Port and enhance visualize_game.py

**Implementation**:
- [ ] Use rich library for color and formatting
- [ ] Strategy distribution bar chart
- [ ] Payoff timeline table (every N rounds)
- [ ] Summary statistics cards
- [ ] Comparison tables for multiple runs
- [ ] Progress bars for long operations
- [ ] Sparklines for inline trends

**Testing**:
- [ ] Visual inspection of output
- [ ] Test with various result sizes
- [ ] Test terminal width handling

**Acceptance**: ASCII visualizations are clear, colorful, and informative

---

### Task 3.2: Static Export System
**Goal**: Generate PNG and HTML artifacts

**Files to create**:
- [ ] `visualization/exports.py` - Static export utilities

**Implementation**:
- [ ] matplotlib/seaborn for PNG charts:
  - Strategy distribution (bar/pie)
  - Payoff trajectories (line)
  - Information quality scatter
  - Signal-quality heatmap
- [ ] plotly for interactive HTML:
  - Same charts but interactive
  - Hover tooltips with details
- [ ] Pandas for CSV export (flatten game history)
- [ ] Auto-save to artifacts/ during run

**Testing**:
- [ ] Test each chart type generates correctly
- [ ] Test HTML interactivity
- [ ] Test file saving and loading

**Acceptance**: All chart types can be generated and saved as PNG/HTML

---

## Phase 4: Marimo Dashboards

### Task 4.1: Create Main Dashboard App
**Goal**: Overview + drill-down dashboard

**Files to create**:
- [ ] `web/dashboard.py` - Main marimo app

**Implementation**:
- [ ] Import marimo and create app
- [ ] Top section: Summary statistics
  - Query SQLite for aggregate metrics
  - Recent activity timeline
- [ ] Middle section: Interactive filters
  - Date range picker (mo.ui.date_range)
  - Strategy multi-select (mo.ui.multiselect)
  - Metric sliders (mo.ui.slider)
  - Text search (mo.ui.text)
- [ ] Bottom section: Results table
  - Load filtered results from SQLite
  - Display as mo.ui.table (sortable)
  - Row click → navigate to run viewer
  - Multi-select + compare button
- [ ] Interactive charts:
  - Scatter: informativeness vs regret (plotly)
  - Timeline: payoffs over experiments
  - Histogram: strategy usage
- [ ] Use reactive cells for filter → query → display pipeline
- [ ] Use mo.cache for expensive queries

**Testing**:
- [ ] Test with empty database
- [ ] Test with many runs
- [ ] Test filtering updates charts
- [ ] Test navigation to run viewer

**Acceptance**: `marimo run web/dashboard.py` shows functional overview dashboard

---

### Task 4.2: Create Run Viewer App
**Goal**: Detailed single-run analysis

**Files to create**:
- [ ] `web/run_viewer.py` - Single run marimo app

**Implementation**:
- [ ] Accept run_id as parameter (mo.cli_args or query param)
- [ ] Load results from storage
- [ ] Config display (collapsible mo.accordion)
- [ ] Summary stats (mo.stat cards)
- [ ] Round-by-round timeline (plotly with hover)
- [ ] Strategy distribution (pie chart)
- [ ] Payoff trajectories (line chart)
- [ ] Signal-quality confusion matrix (heatmap)
- [ ] Bayesian updating analysis (probability curves)
- [ ] Export buttons (trigger download)
- [ ] Back to dashboard link

**Testing**:
- [ ] Test with various run IDs
- [ ] Test chart interactivity
- [ ] Test export buttons
- [ ] Test invalid run ID handling

**Acceptance**: Run viewer shows comprehensive analysis of single experiment

---

### Task 4.3: Create Live Viewer App
**Goal**: Real-time experiment monitoring

**Files to create**:
- [ ] `web/live.py` - Live monitoring marimo app

**Implementation**:
- [ ] Monitor run_results/ for latest in-progress run
- [ ] Display current config
- [ ] Progress bar (rounds completed)
- [ ] Live updating charts (auto-refresh every N seconds):
  - Cumulative payoffs
  - Running averages
  - Recent rounds table
- [ ] Use mo.state for tracking updates
- [ ] Pause/resume controls (if experiment supports signals)
- [ ] Refresh interval slider

**Testing**:
- [ ] Test monitoring active run
- [ ] Test auto-refresh behavior
- [ ] Test handling run completion

**Acceptance**: Live viewer updates in real-time as experiment runs

---

### Task 4.4: Implement `gg serve` Command
**Goal**: Launch marimo dashboard server

**Files to create**:
- [ ] `cli/commands/serve.py`

**Implementation**:
- [ ] Click command with optional subcommand: dashboard|run|live
- [ ] Options: --port 8080
- [ ] Default: launch dashboard.py
- [ ] `gg serve run <id>`: launch run_viewer.py with run_id
- [ ] `gg serve live`: launch live.py
- [ ] Use subprocess to run `marimo run <app> --port <port>`
- [ ] Pass parameters via CLI args or environment
- [ ] Print URL and keep process running

**Testing**:
- [ ] Test default dashboard launch
- [ ] Test specific run viewer
- [ ] Test live viewer
- [ ] Test port specification

**Acceptance**: `gg serve` launches marimo dashboard accessible in browser

---

## Phase 5: Integration & Polish

### Task 5.1: Integration Testing
**Goal**: End-to-end workflow tests

**Files to create**:
- [ ] `tests/integration/test_end_to_end.py`

**Test scenarios**:
- [ ] Run game → list → visualize → export workflow
- [ ] Batch experiment → compare workflow
- [ ] Run → replay with override workflow
- [ ] Config loading → game execution → results storage
- [ ] SQLite index consistency after many runs

**Acceptance**: All workflows complete successfully

---

### Task 5.2: Documentation
**Goal**: Comprehensive user documentation

**Files to create/update**:
- [ ] `README.md` - Update with new CLI usage
- [ ] `docs/quickstart.md` - Getting started guide
- [ ] `docs/configuration.md` - Config file reference
- [ ] `docs/extending.md` - How to add custom agents/strategies
- [ ] `docs/api.md` - Python API reference

**Content**:
- [ ] Installation instructions (uv sync)
- [ ] CLI command examples
- [ ] Config file examples (YAML and Python)
- [ ] Dashboard screenshots
- [ ] Extensibility guide

**Acceptance**: New user can follow docs to install and run experiments

---

### Task 5.3: Example Experiments
**Goal**: Provide useful experiment templates

**Files to create**:
- [ ] `experiments/yaml/baseline.yaml` - Standard game
- [ ] `experiments/yaml/high_rounds.yaml` - Long game
- [ ] `experiments/yaml/skewed_prior.yaml` - Unbalanced priors
- [ ] `experiments/python/custom_strategy.py` - Custom garbling matrix
- [ ] `experiments/python/custom_payoffs.py` - Different payoff structure
- [ ] `experiments/python/batch_comparison.py` - Batch of experiments

**Acceptance**: Example configs run successfully and demonstrate features

---

### Task 5.4: Cleanup & Migration
**Goal**: Remove old code, update repository

**Tasks**:
- [ ] Archive old `garbling_game.py` (move to `archive/`)
- [ ] Archive old `visualize_game.py` (move to `archive/`)
- [ ] Update `.gitignore`:
  - Add `run_results/` (but keep `.gitkeep`)
  - Add `__marimo__/cache/`
  - Add `.venv/`
- [ ] Update README.md with new usage
- [ ] Add `run_results/.gitkeep` and `run_results/.index.db` placeholder
- [ ] Ensure all tests pass with new structure

**Acceptance**: Old code is archived, repository is clean, new CLI is primary interface

---

### Task 5.5: Performance Optimization
**Goal**: Ensure system scales to many experiments

**Optimizations**:
- [ ] Add SQLite indexes on common query columns
- [ ] Use mo.cache in marimo apps for expensive computations
- [ ] Lazy loading in dashboard (pagination for large result sets)
- [ ] Parallel batch execution with proper concurrency limits
- [ ] Profile and optimize hot paths in game loop

**Testing**:
- [ ] Benchmark with 100+ experiments
- [ ] Test dashboard with large dataset
- [ ] Monitor memory usage during batch runs

**Acceptance**: System handles 100+ experiments efficiently

---

## Phase 6: Future Enhancements (Post-MVP)

### Task 6.1: Web-Based Configuration Editor
- [ ] Marimo app for creating/editing configs
- [ ] Form-based UI for YAML configs
- [ ] Visual strategy matrix editor

### Task 6.2: Advanced Analysis Features
- [ ] Statistical significance testing in comparisons
- [ ] Automated experiment optimization (hyperparameter search)
- [ ] Correlation analysis across experiments
- [ ] Export to academic paper format (LaTeX tables)

### Task 6.3: Collaboration Features
- [ ] Export/import experiment bundles
- [ ] Shared experiment repository
- [ ] Annotation/note system for runs
- [ ] Git integration for config versioning

---

## Success Criteria

**MVP is complete when**:
1. All Phase 1-5 tasks are checked off
2. `gg --help` shows all commands
3. `gg run` successfully executes game and saves results
4. `gg list` shows past experiments
5. `gg serve` launches functional marimo dashboard
6. Tests pass with >80% coverage
7. Documentation allows new user to get started in <10 minutes
8. Example experiments run successfully

**Definition of Done per task**:
- [ ] Code written and committed
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Manually verified functionality
- [ ] Code reviewed (if applicable)
- [ ] No regressions in existing functionality
