# ABOUTME: Marimo notebook for interactive run visualization
# ABOUTME: Displays detailed analysis and charts for a single game run

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

    return mo, ResultsStore, go, px, pd, Path


@app.cell
def __(mo):
    """Run selector"""
    mo.md("# 🎮 Garbling Economics - Run Visualization")
    return


@app.cell
def __(mo, ResultsStore):
    """Load available runs"""
    store = ResultsStore()
    runs = store.list_runs(limit=50, sort_by='timestamp')

    if not runs:
        mo.md("⚠️ No runs found. Run your first experiment: `gg run`")
        run_options = []
    else:
        run_options = [(f"{r['id']} ({r.get('name', 'unnamed')})", r['id']) for r in runs]

    return store, runs, run_options


@app.cell
def __(mo, run_options):
    """Run selector dropdown"""
    if run_options:
        run_selector = mo.ui.dropdown(
            options=dict(run_options),
            value=run_options[0][1] if run_options else None,
            label="Select Run:",
            full_width=True
        )
        mo.md(f"## Select Run\n{run_selector}")
    else:
        run_selector = None
        mo.md("No runs available")

    return run_selector,


@app.cell
def __(mo, run_selector, store):
    """Load run data"""
    if run_selector is None or run_selector.value is None:
        run_data = None
        results = None
        metadata = None
        history = None
    else:
        run_data = store.load_run(run_selector.value)

        # Handle nested structure
        if 'summary' in run_data['results']:
            results = run_data['results']['summary']
            history = run_data['results'].get('history', [])
        else:
            results = run_data['results']
            history = results.get('history', [])

        metadata = run_data.get('metadata', {})

    return run_data, results, metadata, history


@app.cell
def __(mo, results, metadata):
    """Display summary metrics"""
    if results is None:
        mo.md("Please select a run")
    else:
        mo.md(f"""
        ## 📊 Summary Statistics

        **Run:** {metadata.get('name', 'Unnamed')}
        **Timestamp:** {metadata.get('timestamp', 'Unknown')}
        **Duration:** {metadata.get('duration', 0):.2f}s
        """)
    return


@app.cell
def __(mo, results):
    """Metrics cards"""
    if results:
        cards = mo.hstack([
            mo.stat(
                label="Total Rounds",
                value=str(results['total_rounds']),
                bordered=True
            ),
            mo.stat(
                label="Sender Total",
                value=f"{results['sender_total']:+.0f}",
                bordered=True,
                caption="Sender payoff"
            ),
            mo.stat(
                label="Receiver Total",
                value=f"{results['receiver_total']:+.0f}",
                bordered=True,
                caption="Receiver payoff"
            ),
            mo.stat(
                label="Buy Rate",
                value=f"{results['buy_rate']:.1%}",
                bordered=True
            ),
            mo.stat(
                label="Informativeness",
                value=f"{results['avg_informativeness']:.2f}",
                bordered=True,
                caption="Avg information score"
            ),
            mo.stat(
                label="Receiver Regret",
                value=f"{results['receiver_regret']:+.0f}",
                bordered=True,
                caption="Loss from garbling"
            ),
        ])
        cards
    return cards,


@app.cell
def __(mo, results, history, go):
    """Cumulative payoffs chart"""
    if results and history:
        mo.md("## 📈 Cumulative Payoffs Over Time")

        rounds = [h['round'] for h in history]
        sender_cumulative = []
        receiver_cumulative = []

        s_total = 0
        r_total = 0
        for h in history:
            s_total += h['sender_payoff']
            r_total += h['receiver_payoff']
            sender_cumulative.append(s_total)
            receiver_cumulative.append(r_total)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=sender_cumulative,
            mode='lines+markers',
            name='Sender',
            line=dict(color='#f5576c', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=rounds,
            y=receiver_cumulative,
            mode='lines+markers',
            name='Receiver',
            line=dict(color='#00f2fe', width=3)
        ))

        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Cumulative Payoff",
            hovermode='x unified',
            height=400
        )

        mo.ui.plotly(fig)
    return rounds, sender_cumulative, receiver_cumulative, fig


@app.cell
def __(mo, results, go):
    """Strategy distribution chart"""
    if results:
        strategies = results.get('strategies_used', {})
        if strategies:
            mo.md("## 🎯 Strategy Distribution")

            fig = go.Figure(data=[go.Bar(
                x=list(strategies.keys()),
                y=list(strategies.values()),
                marker_color='#667eea'
            )])

            fig.update_layout(
                xaxis_title="Strategy",
                yaxis_title="Count",
                height=400
            )

            mo.ui.plotly(fig)
    return strategies, fig


@app.cell
def __(mo, results, go):
    """Quality distribution chart"""
    if results:
        quality_stats = results.get('quality_stats', {})
        if quality_stats:
            mo.md("## 📊 Quality Distribution & Buy Rates")

            qualities = ['HIGH', 'MEDIUM', 'LOW']
            counts = [quality_stats.get(q, {}).get('count', 0) for q in qualities]
            buy_rates = [quality_stats.get(q, {}).get('buy_rate', 0) * 100 for q in qualities]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Count',
                x=qualities,
                y=counts,
                marker_color='#4facfe'
            ))
            fig.add_trace(go.Bar(
                name='Buy Rate (%)',
                x=qualities,
                y=buy_rates,
                marker_color='#f093fb'
            ))

            fig.update_layout(
                barmode='group',
                xaxis_title="Quality",
                yaxis_title="Value",
                height=400
            )

            mo.ui.plotly(fig)
    return quality_stats, qualities, counts, buy_rates, fig


@app.cell
def __(mo, history, pd):
    """Round-by-round data table"""
    if history:
        mo.md("## 📋 Round-by-Round Details")

        df = pd.DataFrame(history)

        # Select relevant columns
        columns = ['round', 'quality', 'strategy', 'signal', 'action',
                   'sender_payoff', 'receiver_payoff', 'garbling_info']
        df = df[columns]

        # Format numeric columns
        df['sender_payoff'] = df['sender_payoff'].apply(lambda x: f"{x:+.0f}")
        df['receiver_payoff'] = df['receiver_payoff'].apply(lambda x: f"{x:+.0f}")
        df['garbling_info'] = df['garbling_info'].apply(lambda x: f"{x:.2f}")

        mo.ui.table(df, selection=None)
    return df, columns


@app.cell
def __(mo):
    """Footer"""
    mo.md("""
    ---
    **Garbling Gym** - Interactive experiment visualization
    Use `gg serve` to launch this dashboard
    """)
    return


if __name__ == "__main__":
    app.run()
