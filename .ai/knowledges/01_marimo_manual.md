## marimo manual (core components, workflows, best practices, tips)

### 1) What marimo is (and why it’s different)

**marimo** is a *reactive* Python notebook that’s stored as **plain `.py` (or `.md`)**, designed to keep **code, outputs, and runtime state consistent** by automatically re-running (or marking stale) downstream cells when inputs change. It’s positioned as a batteries-included replacement for combinations of Jupyter + widgets + app frameworks, and it also supports **running notebooks as scripts and as web apps**. ([Marimo][1])

Key implications:

* **No hidden state**: deleting/altering code updates program state coherently, reducing “out-of-order execution” bugs common in traditional notebooks. ([Marimo][2])
* **Deterministic execution order** is based on variable dependencies (a dataflow/DAG), not cell position. ([Marimo][3])
* Notebooks are **Git-friendly** because they’re code-first files, not JSON. ([Marimo][4])

---

### 2) Installation and “recommended” extras

Install base:

* `pip install marimo` or `conda install -c conda-forge marimo` or `uv add marimo` ([Marimo][5])

Optional “recommended” dependencies enable SQL cells, some chart/data features, AI/code formatting integrations, etc.:

* `pip install "marimo[recommended]"` (or `uv add "marimo[recommended]"`) ([Marimo][6])

For SQL specifically:

* `pip install "marimo[sql]"` (or `uv add "marimo[sql]"`) ([Marimo][7])

---

### 3) The marimo file model (what’s actually in a notebook)

#### 3.1 `.py` notebook format: an `App` with reactive “cells”

A marimo notebook is a Python module defining an `App` and a set of cells. Cells are represented as functions whose **signature encodes dependencies** and whose **return values encode definitions** (variables exported to the dataflow graph). The **last expression** is the *visual output*. marimo can manage these signatures/returns for you in the editor. ([Marimo][8])

A representative pattern:

```python
import marimo as mo

app = mo.App()

@app.cell
def _():
    x = 1
    return x

@app.cell
def _(x):
    y = x + 1
    y  # visual output
    return y

if __name__ == "__main__":
    app.run()
```

#### 3.2 `.md` notebook format (prose-heavy workflows)

marimo also supports notebooks as Markdown files; you can export `.py -> .md` and (in some workflows) use Quarto/MkDocs publishing. ([Marimo][8])

---

### 4) Reactive execution: the core mental model

#### 4.1 Dependency-driven re-runs

When you run a cell, marimo updates all dependent cells that reference its defined variables—like a spreadsheet recalculation model. ([Marimo][3])

#### 4.2 Automatic vs lazy (stale) execution for expensive work

You can configure runtime behavior:

* **Automatic**: downstream cells re-run immediately.
* **Lazy**: downstream cells are marked **stale** instead of running automatically—useful for long-running/side-effectful notebooks. ([Marimo][9])

#### 4.3 Reactivity constraints (things to avoid)

Because marimo builds a dependency graph via code analysis + structured notebook format, some patterns are intentionally constrained. Example: `import *` is disallowed because it obscures global names and conflicts with the format/execution model. ([Marimo][10])

---

### 5) Core user workflows (CLI and editor)

marimo is typically driven via the `marimo` CLI:

* **Create/edit**: `marimo edit notebook.py` ([Marimo][11])
* **Run as an app** (read-only UI): `marimo run notebook.py` ([Marimo][12])
* **Run as a script**: `python notebook.py` (useful for side effects like writing files) ([Marimo][13])
* **Convert from Jupyter**: `marimo convert notebook.ipynb -o notebook.py` ([Marimo][11])
* **Export** to other formats (HTML, WASM HTML, ipynb, md, etc.): `marimo export ...` ([Marimo][14])
* **Static check/lint** before running: `marimo check notebook.py` ([Marimo][12])
* **Tutorials**: `marimo tutorial intro` (and others like `fileformat`) ([Marimo][6])

Quality-of-life flags:

* `--watch` to reflect changes from external editors; `--sandbox` for per-notebook isolated env (uv/PEP 723); `--trusted/--untrusted` for remote notebooks (untrusted uses Docker). ([Marimo][15])

---

### 6) Interactive UI (widgets) without callback spaghetti

marimo provides UI elements in `marimo.ui`. Interacting with a UI element **re-runs cells that reference it**, as long as it’s bound in a way marimo can track. ([Marimo][16])

Typical pattern:

```python
import marimo as mo

slider = mo.ui.slider(0, 100, value=50)
slider  # show widget

# downstream cells reference slider.value
```

Key rules/tips:

* **Bind widgets to stable, global variables** so marimo can synchronize and trigger reactivity. ([Marimo][16])
* If you need a *dynamic set* of widgets (created at runtime), use composite helpers like `mo.ui.dictionary` / `mo.ui.array` so the whole collection stays reactive. ([Marimo][17])
* For “run on demand” compute (instead of re-running on every tweak), use `mo.ui.run_button` patterns and/or lazy runtime. (See recipes/examples.) ([Marimo][18])

---

### 7) State management: ordinary variables vs `mo.state`

In many cases you can just use normal Python variables—marimo keeps state coherent by re-running the right cells. When you need **mutable reactive state that persists across reruns of the defining cell**, use `mo.state` (advanced topic). ([Marimo][19])

Example from troubleshooting guidance (state for persistence):

```python
get_value, set_value = mo.state(initial_value)
element = mo.ui.slider(0, 10, value=get_value(), on_change=set_value)
```

([Marimo][20])

Caution: `mo.state` is powerful but easier to misuse (it can reintroduce “hidden state” style problems if overused). The docs explicitly frame it as advanced. ([Marimo][19])

---

### 8) Caching and performance (critical for real data science)

marimo includes caching primitives that integrate with reactive execution:

* `@mo.cache` (in-memory, unlimited)
* `@mo.lru_cache` (bounded memory)
* `@mo.persistent_cache` (disk-backed; survives restart)
* All support sync + async; persistent caches default to `__marimo__/cache/` and should typically be gitignored. ([Marimo][21])

Key behaviors:

* Cache invalidation accounts for dependencies, including **UI element and `mo.state` changes** (so UI interactions can correctly refresh cached computations when needed). ([Marimo][21])
* Persistent caching can be used as a decorator **or** a context manager (useful for caching whole blocks). ([Marimo][21])

Best-practice pattern:

* Put expensive operations behind cached functions, then keep downstream visualization/interaction cells fast. ([Marimo][22])

---

### 9) Working with data: dataframes, SQL, plotting

#### 9.1 Interactive dataframe exploration

Displaying a dataframe (as the last expression of a cell) enables fast paging/filtering/sorting in the UI. ([Marimo][23])

#### 9.2 SQL cells (DuckDB-powered) + Python variables

marimo supports SQL cells as syntactic sugar that compiles to calls like `mo.sql(...)`, and SQL can depend on Python values (e.g., f-strings) with results returned as a Python dataframe. DuckDB is the default engine for SQL execution. ([Marimo][7])

#### 9.3 Reactive plots

marimo can connect selections from Altair/Plotly back into Python via helpers like `mo.ui.altair_chart` / `mo.ui.plotly`. Plotly selection support is limited to specific chart types; Altair transformations may require `vegafusion` for transformed data. ([Marimo][24])

---

### 10) Apps, layouts, and publishing

#### 10.1 Run as an app

`marimo run notebook.py` serves the notebook as a web app (code hidden/uneditable by default renderer; layouts available). ([Marimo][12])

#### 10.2 Layout options

* **Grid layout**: drag-and-drop arrangement for app UI.
* **Slides layout**: notebook as a slideshow (order tied to cell order; less customizable). ([Marimo][12])
  Also available: programmatic layouts like stacks, tabs, sidebar, etc. via the layouts API. ([Marimo][25])

#### 10.3 WASM/HTML export (share interactive notebooks without Python)

`marimo export html-wasm ...` produces a WASM-powered HTML artifact that runs in the browser; it must be served over HTTP (not `file://`), and not all Python packages will work in WASM. ([Marimo][14])

Publishing workflows:

* Quarto / MkDocs extensions can be used with marimo’s markdown format for docs/sites. ([Marimo][26])

---

### 11) Package management and reproducibility

marimo supports `pip`, `uv`, `poetry`, `pixi`, `rye` and can prompt to install missing imports from the editor; installing a module triggers dependent cells to re-run (or become stale). ([Marimo][27])

#### 11.1 Sandboxed notebooks (PEP 723 + uv)

If you create/edit with `--sandbox`, marimo tracks dependencies in **inline script metadata (PEP 723)**, and re-creates an isolated environment for that notebook. This is especially good for shareable, self-contained notebooks and library examples. ([Marimo][28])

#### 11.2 Notebooks inside a larger project

You can also manage notebooks as part of a normal project environment (`pyproject.toml`), and marimo documents patterns for “notebooks in existing projects”. ([Marimo][29])

---

### 12) Testing and reuse (treat notebooks as software)

#### 12.1 Reuse functions/classes from notebooks

You can import top-level functions/classes from marimo notebooks with normal Python imports, as long as they meet simple criteria (single def/class cell, limited dependencies). ([Marimo][30])

#### 12.2 pytest and doctest

Because notebooks are Python programs, you can run pytest directly; marimo supports conventions for testing notebook cells (e.g., `test_`-prefixed cell names). ([Marimo][31])
There’s also a cell execution API that supports running named cells and retrieving outputs/definitions (useful for tests and composition). ([Marimo][32])

---

### 13) Security model (important if you open random notebooks)

marimo’s security stance is: **no user code executes without explicit user action**. Notebooks opened in `marimo edit` are **statically loaded** (parsed, not executed) and content is sanitized until you run code; `marimo run` is treated more like hosting a trusted website/app, and auth can be configured. There are also CLI concepts around trusted/untrusted remote execution. ([Marimo][33])

---

## Best practices (opinionated, high-leverage)

### A) Write notebooks like small, testable programs

* Keep **global variables minimal**; prefer functions for intermediate values or prefix cell-local temps with `_` to avoid name collisions. ([Marimo][34])
* Treat the notebook as an orchestrator/DAG; push heavy logic into modules and use module autoreloading during development. ([Marimo][35])

### B) Design for reactivity (avoid “mutation traps”)

* Prefer **pure-ish transforms** (new objects) over mutating shared objects across cells; mutation can lead to surprising non-reactive behavior because dependencies are on *names/values*, not “deep object mutation”. (This is a common troubleshooting theme; use dataflow tools to verify connections.) ([Marimo][11])
* If you truly need persistence across reruns, use `mo.state` deliberately—and sparingly. ([Marimo][19])

### C) Control expensive execution explicitly

* Use **lazy runtime**, **run buttons**, **disable expensive cells**, and **`mo.cache`/`mo.persistent_cache`** to keep iteration tight. ([Marimo][9])

### D) Make reproducibility a default, not an afterthought

* For shareable artifacts, prefer **`--sandbox`** notebooks with inline deps (PEP 723) so someone can run it without chasing env files. ([Marimo][29])
* Add `__marimo__/cache/` to `.gitignore` if using persistent cache. ([Marimo][21])

### E) Prefer parameterization patterns that match the execution mode

* For scripts/apps, use `argparse` (or similar) and pass args after `--` when using `marimo edit/run/export`; `mo.cli_args()` exists but is intentionally minimal. ([Marimo][36])
* For web apps, use `mo.query_params` to sync state into the URL for shareable/bookmarkable app states. ([Marimo][37])

---

## Practical tips & “gotchas” checklist

* **If a downstream cell isn’t updating**, open the dependency graph/minimap/variables panel and confirm the cell actually *references* the upstream symbol (dataflow tools). ([Marimo][11])
* **If cells run too often**, switch to lazy mode, gate with a run button, and cache the expensive function(s). ([Marimo][9])
* **If exporting WASM HTML**, remember: serve over HTTP and expect package limitations. ([Marimo][14])
* **Avoid `import *`** and other patterns that make name dependencies implicit. ([Marimo][10])
* **Sandbox feature requires uv** (today), so standardize on uv if you want inline deps. ([Marimo][29])
* **Security**: treat untrusted notebooks like any arbitrary code—review before running; marimo reduces *load-time* risk but not the risk of executing malicious code you choose to run. ([Marimo][33])


[1]: https://docs.marimo.io/?utm_source=chatgpt.com "marimo"
[2]: https://docs.marimo.io/getting_started/key_concepts/?utm_source=chatgpt.com "Key Concepts"
[3]: https://docs.marimo.io/guides/reactivity/?utm_source=chatgpt.com "Running cells"
[4]: https://docs.marimo.io/guides/coming_from/jupyter/?utm_source=chatgpt.com "Migrate from Jupyter"
[5]: https://docs.marimo.io/getting_started/installation/?utm_source=chatgpt.com "Installation"
[6]: https://docs.marimo.io/getting_started/quickstart/?utm_source=chatgpt.com "Quickstart"
[7]: https://docs.marimo.io/guides/working_with_data/sql/?utm_source=chatgpt.com "SQL"
[8]: https://docs.marimo.io/guides/editor_features/watching/?utm_source=chatgpt.com "Using your own editor"
[9]: https://docs.marimo.io/guides/configuration/runtime_configuration/?utm_source=chatgpt.com "Runtime configuration"
[10]: https://docs.marimo.io/guides/understanding_errors/import_star/?utm_source=chatgpt.com "Import star"
[11]: https://docs.marimo.io/llms.txt?utm_source=chatgpt.com "https://docs.marimo.io/llms.txt"
[12]: https://docs.marimo.io/guides/apps/?utm_source=chatgpt.com "Run notebooks as apps"
[13]: https://docs.marimo.io/guides/scripts/?utm_source=chatgpt.com "Run notebooks as scripts"
[14]: https://docs.marimo.io/guides/exporting/?utm_source=chatgpt.com "Exporting to HTML and other formats"
[15]: https://docs.marimo.io/cli/?utm_source=chatgpt.com "Commands - marimo"
[16]: https://docs.marimo.io/guides/interactivity/?utm_source=chatgpt.com "Interactive elements"
[17]: https://docs.marimo.io/api/inputs/dictionary/?utm_source=chatgpt.com "Dictionary"
[18]: https://docs.marimo.io/recipes/?utm_source=chatgpt.com "Recipes"
[19]: https://docs.marimo.io/api/state/?utm_source=chatgpt.com "State"
[20]: https://docs.marimo.io/guides/troubleshooting/?utm_source=chatgpt.com "Troubleshooting"
[21]: https://docs.marimo.io/api/caching/?utm_source=chatgpt.com "Caching"
[22]: https://docs.marimo.io/guides/expensive_notebooks/?utm_source=chatgpt.com "Expensive notebooks"
[23]: https://docs.marimo.io/guides/working_with_data/dataframes/?utm_source=chatgpt.com "DataFrames"
[24]: https://docs.marimo.io/guides/working_with_data/plotting/?utm_source=chatgpt.com "Plotting"
[25]: https://docs.marimo.io/api/layouts/?utm_source=chatgpt.com "Layouts"
[26]: https://docs.marimo.io/guides/publishing/quarto/?utm_source=chatgpt.com "Quarto"
[27]: https://docs.marimo.io/guides/editor_features/package_management/?utm_source=chatgpt.com "Package management"
[28]: https://docs.marimo.io/guides/package_management/importing_packages/?utm_source=chatgpt.com "Importing packages"
[29]: https://docs.marimo.io/guides/package_management/notebooks_in_projects/?utm_source=chatgpt.com "Notebooks in existing projects"
[30]: https://docs.marimo.io/guides/reusing_functions/?utm_source=chatgpt.com "Reuse functions and classes"
[31]: https://docs.marimo.io/guides/testing/pytest/?utm_source=chatgpt.com "pytest"
[32]: https://docs.marimo.io/api/cell/?utm_source=chatgpt.com "Cell"
[33]: https://docs.marimo.io/security/?utm_source=chatgpt.com "Security"
[34]: https://docs.marimo.io/guides/best_practices/?utm_source=chatgpt.com "Best practices"
[35]: https://docs.marimo.io/guides/editor_features/module_autoreloading/?utm_source=chatgpt.com "Module autoreloading"
[36]: https://docs.marimo.io/api/cli_args/?utm_source=chatgpt.com "Command Line Arguments"
[37]: https://docs.marimo.io/api/query_params/?utm_source=chatgpt.com "Query Parameters"
