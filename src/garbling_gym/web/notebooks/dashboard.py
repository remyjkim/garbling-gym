# ABOUTME: Marimo dashboard for browsing and comparing runs
# ABOUTME: Interactive table with filtering, sorting, and visualization

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import sys
    from pathlib import Path

    # Add garbling_gym to path
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from garbling_gym.core.storage import ResultsStore
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    from datetime import datetime

    return mo, ResultsStore, go, px, pd, datetime, Path


@app.cell
def __(mo):
    """Header"""
    mo.md("""
    # 🎮 Garbling Gym - Experiment Dashboard

    Browse, filter, and analyze your garbling economics experiments.
    """)
    return


@app.cell
def __(mo, ResultsStore):
    """Load all runs"""
    store = ResultsStore()
    runs = store.list_runs(limit=200, sort_by='timestamp')

    if not runs:
        mo.callout(
            "⚠️ No runs found. Run your first experiment: `gg run`",
            kind="warn"
        )

    return store, runs


@app.cell
def __(mo, runs):
    """Filters"""
    if runs:
        # Get unique tags
        all_tags = set()
        for _run in runs:
            tags = _run.get('tags', [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(',') if t.strip()]
            all_tags.update(tags)

        tag_filter = mo.ui.multiselect(
            options=sorted(list(all_tags)) if all_tags else [],
            label="Filter by Tags:",
        )

        llm_filter = mo.ui.dropdown(
            options={"All": None, "LLM Only": 1, "No LLM": 0},
            value="All",
            label="LLM Usage:"
        )

        _output = mo.vstack([
            mo.md("## 🔍 Filters"),
            mo.hstack([tag_filter, llm_filter], widths="equal", gap=2)
        ])
    else:
        tag_filter = None
        llm_filter = None
        all_tags = set()
        _output = mo.md("_No runs to filter_")

    _output
    return tag_filter, llm_filter, all_tags


@app.cell
def __(runs, tag_filter, llm_filter, pd):
    """Filter runs"""
    if runs:
        filtered_runs = runs

        # Apply tag filter
        if tag_filter and tag_filter.value:
            filtered_runs = [
                r for r in filtered_runs
                if any(tag in r.get('tags', []) for tag in tag_filter.value)
            ]

        # Apply LLM filter
        if llm_filter and llm_filter.value is not None:
            filtered_runs = [
                r for r in filtered_runs
                if r.get('use_llm') == llm_filter.value
            ]

        # Convert to DataFrame
        df_runs = pd.DataFrame(filtered_runs)

        # Format columns
        if not df_runs.empty:
            df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            df_runs['sender_total'] = df_runs['sender_total'].apply(lambda x: f"{x:+.0f}")
            df_runs['receiver_total'] = df_runs['receiver_total'].apply(lambda x: f"{x:+.0f}")
            df_runs['buy_rate'] = df_runs['buy_rate'].apply(lambda x: f"{x:.1%}")
            df_runs['avg_informativeness'] = df_runs['avg_informativeness'].apply(lambda x: f"{x:.2f}")

    else:
        filtered_runs = []
        df_runs = pd.DataFrame()

    return filtered_runs, df_runs


@app.cell
def __(mo, df_runs, filtered_runs):
    """Display runs table"""
    if not df_runs.empty:
        # Select columns to display
        display_cols = ['id', 'name', 'timestamp', 'num_rounds', 'sender_total',
                        'receiver_total', 'buy_rate', 'avg_informativeness']

        table = mo.ui.table(
            df_runs[display_cols],
            selection='multi',
            pagination=True,
            page_size=20
        )

        _table_output = mo.vstack([
            mo.md(f"## 📊 Experiment Runs ({len(filtered_runs)} runs)"),
            table
        ])
    else:
        table = None
        display_cols = []
        _table_output = mo.md("_No runs match the filters_")

    _table_output
    return table, display_cols


@app.cell
def __(mo, table, filtered_runs):
    """Selected runs for comparison"""
    if table is not None and len(table.value) > 0:
        # table.value is a DataFrame of selected rows
        # Extract the 'id' column from selected rows
        selected_ids = table.value['id'].tolist()

        # Match selected IDs against filtered_runs
        selected_runs = [run for run in filtered_runs if run['id'] in selected_ids]

        _selected_output = mo.md(f"""
        ## 🔬 Selected Runs ({len(selected_runs)})

        Select 2-5 runs from the table above to compare them.
        """)
    else:
        selected_ids = []
        selected_runs = []
        _selected_output = None

    if _selected_output:
        _selected_output

    return selected_ids, selected_runs


@app.cell
def __(mo, selected_runs, go):
    """Comparison chart"""
    if len(selected_runs) >= 2:
        _fig = go.Figure()

        for _run in selected_runs[:5]:  # Limit to 5 runs
            _run_label = f"{_run.get('name', 'unnamed')[:20]}"

            _fig.add_trace(go.Scatter(
                x=[_run['sender_total']],
                y=[_run['receiver_total']],
                mode='markers+text',
                name=_run_label,
                text=[_run_label],
                textposition="top center",
                marker=dict(size=15)
            ))

        _fig.update_layout(
            xaxis_title="Sender Total",
            yaxis_title="Receiver Total",
            hovermode='closest',
            height=500
        )

        _output_chart = mo.vstack([
            mo.md("### Sender vs Receiver Totals"),
            mo.ui.plotly(_fig)
        ])
        _output_chart
    return


@app.cell
def __(mo, selected_runs, go):
    """Buy rate comparison"""
    if len(selected_runs) >= 2:
        _names = [r.get('name', 'unnamed')[:20] for r in selected_runs[:5]]
        _buy_rates = [r['buy_rate'] * 100 for r in selected_runs[:5]]

        _fig = go.Figure(data=[go.Bar(
            x=_names,
            y=_buy_rates,
            marker_color='#667eea'
        )])

        _fig.update_layout(
            xaxis_title="Run",
            yaxis_title="Buy Rate (%)",
            height=400
        )

        _output_buy = mo.vstack([
            mo.md("### Buy Rate Comparison"),
            mo.ui.plotly(_fig)
        ])
        _output_buy
    return


@app.cell
def __(mo, selected_runs, go):
    """Informativeness comparison"""
    if len(selected_runs) >= 2:
        _names = [r.get('name', 'unnamed')[:20] for r in selected_runs[:5]]
        _info_scores = [r['avg_informativeness'] for r in selected_runs[:5]]

        _fig = go.Figure(data=[go.Bar(
            x=_names,
            y=_info_scores,
            marker_color='#4facfe'
        )])

        _fig.update_layout(
            xaxis_title="Run",
            yaxis_title="Avg Informativeness",
            height=400,
            yaxis=dict(range=[0, 1])
        )

        _output_info = mo.vstack([
            mo.md("### Informativeness Comparison"),
            mo.ui.plotly(_fig)
        ])
        _output_info
    return


@app.cell
def __(mo, runs, go):
    """Aggregate statistics"""
    if runs:
        mo.md("## 📈 Aggregate Statistics")

        total_rounds = sum(r['num_rounds'] for r in runs)
        avg_sender = sum(r['sender_total'] for r in runs) / len(runs)
        avg_receiver = sum(r['receiver_total'] for r in runs) / len(runs)
        avg_buy_rate = sum(r['buy_rate'] for r in runs) / len(runs)

        stats = mo.hstack([
            mo.stat(
                label="Total Experiments",
                value=str(len(runs)),
                bordered=True
            ),
            mo.stat(
                label="Total Rounds Played",
                value=str(total_rounds),
                bordered=True
            ),
            mo.stat(
                label="Avg Sender Payoff",
                value=f"{avg_sender:+.1f}",
                bordered=True
            ),
            mo.stat(
                label="Avg Receiver Payoff",
                value=f"{avg_receiver:+.1f}",
                bordered=True
            ),
            mo.stat(
                label="Avg Buy Rate",
                value=f"{avg_buy_rate:.1%}",
                bordered=True
            ),
        ])

        stats
    return total_rounds, avg_sender, avg_receiver, avg_buy_rate, stats


@app.cell
def __(mo, runs, pd, px):
    """Timeline of experiments"""
    if runs:
        mo.md("### Experiment Timeline")

        df_timeline = pd.DataFrame(runs)
        df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])

        # Group by date
        df_timeline['date'] = df_timeline['timestamp'].dt.date
        daily_counts = df_timeline.groupby('date').size().reset_index(name='count')

        _fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title='Experiments per Day',
            markers=True
        )

        _fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Experiments",
            height=400
        )

        mo.ui.plotly(_fig)
    return df_timeline, daily_counts


@app.cell
def __(mo):
    """Footer"""
    mo.md("""
    ---
    **Garbling Gym Dashboard**

    Commands:
    - `gg run` - Run new experiment
    - `gg experiment <file>` - Batch experiments
    - `gg visualize <run-id>` - Detailed analysis
    - `gg compare <id1> <id2>` - Compare runs
    """)
    return


if __name__ == "__main__":
    app.run()
