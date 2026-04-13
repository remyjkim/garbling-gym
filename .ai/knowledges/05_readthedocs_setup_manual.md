Below is an *updated* (current RTD + Sphinx + **uv**) setup manual that assumes:

* Your docs live in `docs/`
* You want **RTD builds to install deps via `uv.lock`** (fast + reproducible)
* You’re using Sphinx (autodoc API docs, plus optional CLI docs)

I’m basing the RTD YAML and job semantics on RTD’s current “Build process customization” docs (which now include an explicit `uv` example). ([Read the Docs Documentation][1])

---

## 1) Structure your repo for Sphinx + RTD

Typical layout:

```
your-project/
├─ pyproject.toml
├─ uv.lock
├─ src/your_pkg/...
└─ docs/
   ├─ conf.py
   ├─ index.rst
   └─ ...
```

RTD will run your build from a checkout of the repo, and expects Sphinx config at the path you declare (e.g., `docs/conf.py`). ([Read the Docs Documentation][2])

---

## 2) Put doc dependencies in `pyproject.toml` (uv-friendly)

You have two good options:

### Option A (recommended): a dedicated `docs` **dependency group**

uv reads dev/dependency groups from `[dependency-groups]` (PEP 735). ([docs.astral.sh][3])

Example:

```toml
[dependency-groups]
docs = [
  "sphinx>=8",
  "furo>=2024.0.0",
  "sphinx-autodoc-typehints",
  "sphinx-click",                  # for CLI docs (optional)
  "myst-parser",                   # if you want Markdown (optional)
]
```

Then lock:

```bash
uv lock
```

Commit `uv.lock`.

### Option B: `[project.optional-dependencies]` (extras)

This is nice if you also want `pip install .[docs]` to work outside uv. uv can sync extras with `--extra docs` / `--all-extras`. ([docs.astral.sh][3])

---

## 3) Make local Sphinx builds match RTD (using uv)

From repo root:

```bash
uv sync --group docs
uv run sphinx-build -T -W --keep-going -b html docs docs/_build/html
```

Notes:

* `-W --keep-going` helps you keep docs warning-free (highly recommended).
* `uv run` will “lock and sync” automatically unless you use `--locked/--frozen`. ([docs.astral.sh][3])

---

## 4) Add `.readthedocs.yaml` (the important part)

RTD config is **v2**. You should set:

* OS image + tool versions (for reproducibility) ([Read the Docs Documentation][4])
* Sphinx configuration path
* Custom build jobs to use **uv** (and respect `uv.lock`) ([Read the Docs Documentation][1])

### Canonical `uv`-based RTD config (updated)

Create `.readthedocs.yaml` at repo root:

```yaml
version: 2

sphinx:
  configuration: docs/conf.py

build:
  os: ubuntu-24.04
  tools:
    python: "3.13"
  jobs:
    # Install uv (RTD docs show asdf-based install)
    pre_create_environment:
      - asdf plugin add uv
      - asdf install uv latest
      - asdf global uv latest

    # Create the RTD virtualenv using uv at RTD's expected location
    create_environment:
      - uv venv "${READTHEDOCS_VIRTUALENV_PATH}"

    # Sync deps from uv.lock into that environment
    install:
      - UV_PROJECT_ENVIRONMENT="${READTHEDOCS_VIRTUALENV_PATH}" uv sync --frozen --group docs
```

This is essentially RTD’s own example for uv projects: it creates the venv at `${READTHEDOCS_VIRTUALENV_PATH}` and then runs `uv sync` with `UV_PROJECT_ENVIRONMENT` pointing at that venv. ([Read the Docs Documentation][1])

**Why `UV_PROJECT_ENVIRONMENT`?**
uv’s default project environment is `.venv`, but you can override it with `UV_PROJECT_ENVIRONMENT`. ([docs.astral.sh][5])

**Why `--frozen`?**
`uv sync` will otherwise “re-lock before syncing”; `--locked`/`--frozen` stop that behavior. ([docs.astral.sh][6])

* Use `--locked` if you want the build to **fail** when `uv.lock` is out of date (CI-style strictness).
* Use `--frozen` if you want it to just use the lockfile without checking freshness.

### Important RTD nuance: each job command runs in a fresh shell

RTD explicitly notes that **each command is executed in a new shell process**, so environment mutations won’t persist between commands. That’s why the config sets `UV_PROJECT_ENVIRONMENT=...` inline on the same command that calls `uv sync`. ([Read the Docs Documentation][1])

---

## 5) Sphinx configuration that works well on RTD

In `docs/conf.py`, the minimum RTD-friendly bits:

```python
import os
import sys
from datetime import datetime

# If using src/ layout:
sys.path.insert(0, os.path.abspath("../src"))

project = "your-project"
author = "Your Team"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "furo"  # or "sphinx_rtd_theme", etc.
```

### If you want CLI docs

If your tool is Click/Typer, add `sphinx-click` or equivalent and document commands. (That’s why I suggested including `sphinx-click` in the `docs` group.)

---

## 6) Turn on RTD builds and versioning

In the RTD UI, after importing your repo from GitHub:

* Enable builds for:

  * Default branch (often `main`)
  * Tags (so releases get versioned docs)
* Configure “pull request builds” if you want preview docs per PR.

RTD’s build process overview is worth skimming once so you know where `create_environment/install/build` fit. ([Read the Docs Documentation][7])

---

## 7) Troubleshooting checklist (common Sphinx+RTD failures)

### A) `autodoc` can’t import your package

Cause: your package isn’t installed (or C extensions/system deps missing).

Fix:

* Confirm your `uv sync` installs the project itself (default behavior is to install the project editable during sync). ([docs.astral.sh][3])
* If you rely on system libraries, add `build.apt_packages` (RTD supports installing apt packages as part of build config). ([Read the Docs Documentation][1])

### B) “Your documentation did not generate an index.html…”

Cause: Sphinx build didn’t output to the expected place, or build failed before output.

Fix:

* Ensure `docs/index.rst` exists and is in your toctree.
* If you override `build.jobs.build` later, ensure output goes to `$READTHEDOCS_OUTPUT/html/` (RTD hosts that directory). ([Read the Docs Documentation][1])

### C) Builds “randomly” change

Fix: pin OS + Python versions in `.readthedocs.yaml` and rely on lockfiles / pinned deps. RTD recommends this explicitly for reproducibility. ([Read the Docs Documentation][4])

---

## 8) A stricter, CI-grade variant (recommended once stable)

Swap `--frozen` for `--locked` so RTD fails if `pyproject.toml` changed but `uv.lock` wasn’t updated:

```yaml
install:
  - UV_PROJECT_ENVIRONMENT="${READTHEDOCS_VIRTUALENV_PATH}" uv sync --locked --group docs
```

`uv sync` documentation: it “re-locks before syncing unless `--locked` or `--frozen` is provided.” ([docs.astral.sh][6])

---

If you paste your current `pyproject.toml` (docs deps section) and your `docs/conf.py`, I can give you a “drop-in” `.readthedocs.yaml` tailored to:

* src vs flat layout
* whether you want `--locked` vs `--frozen`
* whether the docs build needs your optional extras (e.g., `--extra multiagent`) in addition to `--group docs`

[1]: https://docs.readthedocs.com/platform/stable/build-customization.html "Build process customization — Read the Docs user documentation"
[2]: https://docs.readthedocs.com/platform/stable/config-file/v2.html?utm_source=chatgpt.com "Configuration file reference - readthedocs.yaml"
[3]: https://docs.astral.sh/uv/concepts/projects/sync/ "Locking and syncing | uv"
[4]: https://docs.readthedocs.com/platform/latest/guides/reproducible-builds.html?utm_source=chatgpt.com "How to create reproducible builds"
[5]: https://docs.astral.sh/uv/concepts/projects/config/?utm_source=chatgpt.com "Configuring projects | uv - Astral Docs"
[6]: https://docs.astral.sh/uv/reference/cli/ "Commands | uv"
[7]: https://docs.readthedocs.com/platform/latest/builds.html?utm_source=chatgpt.com "Build process overview"
