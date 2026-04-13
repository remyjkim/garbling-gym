# Marimo Dashboard Best Practices

## Overview

This document captures critical learnings from implementing and debugging the Garbling Gym dashboard using Marimo. These patterns and pitfalls apply to any Marimo notebook implementation.

## Critical: Cell Output Pattern

**THE MOST IMPORTANT RULE:** In Marimo, simply writing a variable name inside an if/else block does NOT output it. You must assign to a variable and write that variable as the final statement.

### ❌ WRONG - Cell Won't Display Anything

```python
@app.cell
def __(mo, data):
    if len(data) > 0:
        _output = mo.md("Has data")
        _output  # This is inside the if block
    else:
        mo.md("No data")  # This is inside the else block
    return
```

### ✅ CORRECT - Cell Will Display

```python
@app.cell
def __(mo, data):
    if len(data) > 0:
        _final_output = mo.md("Has data")
    else:
        _final_output = mo.md("No data")

    _final_output  # This is OUTSIDE and will actually display
    return
```

### Why This Matters

- Python evaluates `_output` but doesn't do anything with it
- Marimo only displays the **last expression** in a cell
- If the last expression is `return`, nothing displays
- If the last expression is inside a conditional, it may not execute

## Table Selections in Marimo

### Understanding `table.value`

When using `mo.ui.table()` with `selection='multi'`:

```python
table = mo.ui.table(df, selection='multi')
```

**Key Facts:**
- `table.value` is a **pandas DataFrame**, NOT a list
- It contains the **actual row data** of selected rows
- To get specific column: `table.value['column_name'].tolist()`
- Check if empty: `len(table.value) > 0` NOT `if table.value:`

### ❌ WRONG - Causes Errors

```python
# Treating it as a list
selected = [row[0] for row in table.value]  # KeyError!

# Boolean check
if table.value:  # ValueError: DataFrame truth is ambiguous
```

### ✅ CORRECT

```python
# Check if selections exist
if table is not None and len(table.value) > 0:
    # Extract column as list
    selected_ids = table.value['id'].tolist()

    # Use the IDs
    selected_items = [item for item in all_items if item['id'] in selected_ids]
```

## Conditional Rendering Pattern

For cells that should only display when data exists:

```python
@app.cell
def __(mo, selected_items):
    """Chart that only appears when 2+ items selected"""
    if len(selected_items) >= 2:
        # Build the chart
        chart = create_chart(selected_items)

        _output = mo.vstack([
            mo.md("### My Chart"),
            mo.ui.plotly(chart)
        ])
    else:
        _output = None  # Or mo.md("Select 2+ items")

    # Critical: Output even if None
    if _output:
        _output
    return
```

**Alternative (cleaner):**

```python
@app.cell
def __(mo, selected_items):
    if len(selected_items) >= 2:
        _output = create_chart_output(selected_items)
        _output  # Display
    return  # Display nothing if condition false
```

## Variable Naming Conventions

Marimo has special handling for variable names:

- **Private variables** (starting with `_`): Local to the cell, not shared
- **Public variables**: Shared across cells, create dependencies

### Best Practice

```python
@app.cell
def __(mo, data):
    """Process data and output"""
    # Private - won't pollute namespace
    _processed = process(data)
    _chart = create_chart(_processed)
    _output = mo.vstack([_chart])

    # Public - other cells can use this
    final_result = compute_result(_processed)

    _output
    return final_result  # Only return what other cells need
```

## Debugging Patterns

### Add Debug Output

```python
@app.cell
def __(mo, table):
    _debug_info = [
        f"table is None: {table is None}",
        f"table.value type: {type(table.value) if table else 'N/A'}",
        f"table.value len: {len(table.value) if table else 0}"
    ]

    mo.md("**Debug:**\n\n" + "\n\n".join(_debug_info))
    return
```

### Server Logs

When using `gg serve`, check server logs for Python errors:
```bash
# The dashboard runs in background
# Check logs with: look at stderr output
```

Common errors in logs:
- `KeyError`: Wrong column name or treating DataFrame as list
- `ValueError: truth value ambiguous`: Using DataFrame in boolean context
- `TypeError: list indices must be integers`: Wrong indexing approach

## Cell Dependencies & Reactivity

Marimo automatically tracks dependencies:

```python
# Cell 1: Creates data
@app.cell
def __():
    selected_runs = []
    return selected_runs

# Cell 2: Uses data - will auto-rerun when selected_runs changes
@app.cell
def __(selected_runs):
    if len(selected_runs) >= 2:
        # This will automatically update when selected_runs changes
        create_chart(selected_runs)
    return
```

**Important:** Cells only re-execute if their **inputs** change. If a cell depends on `selected_runs`, it won't re-execute unless `selected_runs` is modified.

## Common Pitfalls

### 1. Forgetting to Output

```python
# ❌ Chart is created but never displayed
if condition:
    _chart = mo.ui.plotly(fig)
return
```

### 2. Incorrect Table Value Access

```python
# ❌ Trying to iterate like a list
for row in table.value:  # Works but...
    id = row[0]  # KeyError! row is a Series, not a list
```

### 3. Boolean Checks on DataFrames

```python
# ❌ Ambiguous truth value
if table.value:
    do_something()

# ✅ Explicit check
if len(table.value) > 0:
    do_something()
```

### 4. Modifying DataFrames for Display

```python
# ❌ Modifying source data
df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
# Now filtered_runs has modified timestamps!

# ✅ Copy first
df_display = df_runs.copy()
df_display['timestamp'] = pd.to_datetime(df_display['timestamp'])
```

## Testing Strategy

1. **Start simple**: Create cells that just display static content
2. **Add interactivity**: Add UI elements (table, dropdown, etc.)
3. **Add debug output**: Show what data you're getting
4. **Build incrementally**: Add one dependent cell at a time
5. **Check server logs**: Watch for Python errors
6. **Verify reactivity**: Change inputs, ensure cells update

## File Organization

```python
# Good cell organization:

# 1. Imports
@app.cell
def __():
    import marimo as mo
    import plotly.graph_objects as go
    return mo, go

# 2. Data loading
@app.cell
def __(ResultsStore):
    store = ResultsStore()
    runs = store.list_runs()
    return store, runs

# 3. Filters/inputs
@app.cell
def __(mo, runs):
    filter_widget = mo.ui.dropdown(...)
    return filter_widget

# 4. Data processing
@app.cell
def __(runs, filter_widget):
    filtered = apply_filter(runs, filter_widget.value)
    return filtered

# 5. Display/output
@app.cell
def __(mo, filtered):
    _output = create_display(filtered)
    _output
    return
```

## When to Use Marimo vs Static HTML

**Use Marimo when:**
- Need interactivity (filters, selections, sliders)
- Want live updates as data changes
- Exploring data iteratively

**Use Static HTML when:**
- Final reports/presentations
- Sharing with non-technical users
- Archiving results
- No need for interactivity

## Summary Checklist

- [ ] Cell outputs: Assign to variable, output as last statement
- [ ] Table selections: Use `table.value['column'].tolist()`
- [ ] Boolean checks: Use `len(df) > 0`, not `if df:`
- [ ] Private variables: Prefix with `_` to keep local
- [ ] Debug output: Add temporarily to understand data flow
- [ ] Test incrementally: Build one cell at a time
- [ ] Check logs: Monitor server stderr for errors
- [ ] Conditional display: Handle both branches, output outside condition
