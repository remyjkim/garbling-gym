# ABOUTME: Compare command for comparing two game runs
# ABOUTME: Shows side-by-side comparison of metrics, strategies, and outcomes

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from ...core.storage import ResultsStore


console = Console()


@click.command()
@click.argument('run_id_1')
@click.argument('run_id_2')
@click.option(
    '--detail', '-d',
    is_flag=True,
    help='Show detailed comparison including round-by-round'
)
def compare(run_id_1, run_id_2, detail):
    """
    Compare two game runs side-by-side.

    Shows differences in strategies, payoffs, and outcomes.

    Examples:
      gg compare run1 run2           # Basic comparison
      gg compare run1 run2 --detail  # Detailed comparison
    """
    store = ResultsStore()

    # Load both runs
    try:
        run1_data = store.load_run(run_id_1)
        # Handle nested structure - results may have 'summary' key
        if 'summary' in run1_data['results']:
            run1_results = run1_data['results']['summary']
            run1_results['history'] = run1_data['results'].get('history', [])
        else:
            run1_results = run1_data['results']
        run1_metadata = run1_data['metadata']
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Run '{run_id_1}' not found")
        console.print("\nAvailable runs: [bold]gg list[/bold]")
        raise click.Abort()

    try:
        run2_data = store.load_run(run_id_2)
        # Handle nested structure - results may have 'summary' key
        if 'summary' in run2_data['results']:
            run2_results = run2_data['results']['summary']
            run2_results['history'] = run2_data['results'].get('history', [])
        else:
            run2_results = run2_data['results']
        run2_metadata = run2_data['metadata']
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Run '{run_id_2}' not found")
        console.print("\nAvailable runs: [bold]gg list[/bold]")
        raise click.Abort()

    # Display header
    console.print(f"\n[bold cyan]Comparing Runs[/bold cyan]")
    console.print(f"[dim]Run 1:[/dim] {run_id_1}")
    console.print(f"[dim]Run 2:[/dim] {run_id_2}\n")

    # Create summary comparison table
    _show_summary_comparison(run1_results, run2_results, run1_metadata, run2_metadata)

    # Strategy comparison
    _show_strategy_comparison(run1_results, run2_results)

    # Quality stats comparison
    _show_quality_comparison(run1_results, run2_results)

    if detail:
        _show_detailed_comparison(run1_results, run2_results)


def _show_summary_comparison(results1, results2, meta1, meta2):
    """Show high-level metric comparison."""
    table = Table(title="Summary Comparison", show_header=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Run 1", justify="right", style="green")
    table.add_column("Run 2", justify="right", style="blue")
    table.add_column("Difference", justify="right", style="yellow")

    # Calculate differences
    sender_diff = results2['sender_total'] - results1['sender_total']
    receiver_diff = results2['receiver_total'] - results1['receiver_total']
    buy_diff = results2['buy_rate'] - results1['buy_rate']
    info_diff = results2['avg_informativeness'] - results1['avg_informativeness']
    regret_diff = results2['receiver_regret'] - results1['receiver_regret']

    # Add rows
    table.add_row(
        "Rounds",
        str(results1['total_rounds']),
        str(results2['total_rounds']),
        ""
    )
    table.add_row(
        "Sender Total",
        f"{results1['sender_total']:+.0f}",
        f"{results2['sender_total']:+.0f}",
        f"{sender_diff:+.0f}"
    )
    table.add_row(
        "Receiver Total",
        f"{results1['receiver_total']:+.0f}",
        f"{results2['receiver_total']:+.0f}",
        f"{receiver_diff:+.0f}"
    )
    table.add_row(
        "Buy Rate",
        f"{results1['buy_rate']:.1%}",
        f"{results2['buy_rate']:.1%}",
        f"{buy_diff:+.1%}"
    )
    table.add_row(
        "Informativeness",
        f"{results1['avg_informativeness']:.2f}",
        f"{results2['avg_informativeness']:.2f}",
        f"{info_diff:+.2f}"
    )
    table.add_row(
        "Receiver Regret",
        f"{results1['receiver_regret']:+.0f}",
        f"{results2['receiver_regret']:+.0f}",
        f"{regret_diff:+.0f}"
    )

    # Add metadata
    table.add_section()
    table.add_row(
        "Duration",
        f"{meta1.get('duration', 0):.1f}s",
        f"{meta2.get('duration', 0):.1f}s",
        ""
    )

    console.print(table)
    console.print()


def _show_strategy_comparison(results1, results2):
    """Show strategy usage comparison."""
    strats1 = results1.get('strategies_used', {})
    strats2 = results2.get('strategies_used', {})

    # Get all unique strategies
    all_strats = set(strats1.keys()) | set(strats2.keys())

    if not all_strats:
        return

    table = Table(title="Strategy Usage Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Run 1", justify="right", style="green")
    table.add_column("Run 2", justify="right", style="blue")
    table.add_column("Difference", justify="right", style="yellow")

    for strat in sorted(all_strats):
        count1 = strats1.get(strat, 0)
        count2 = strats2.get(strat, 0)
        diff = count2 - count1

        # Calculate percentages
        pct1 = count1 / results1['total_rounds'] * 100 if results1['total_rounds'] > 0 else 0
        pct2 = count2 / results2['total_rounds'] * 100 if results2['total_rounds'] > 0 else 0

        table.add_row(
            strat,
            f"{count1} ({pct1:.0f}%)",
            f"{count2} ({pct2:.0f}%)",
            f"{diff:+d}"
        )

    console.print(table)
    console.print()


def _show_quality_comparison(results1, results2):
    """Show quality distribution comparison."""
    stats1 = results1.get('quality_stats', {})
    stats2 = results2.get('quality_stats', {})

    if not stats1 and not stats2:
        return

    table = Table(title="Quality Distribution Comparison")
    table.add_column("Quality", style="cyan")
    table.add_column("Run 1 Count", justify="right", style="green")
    table.add_column("Run 2 Count", justify="right", style="blue")
    table.add_column("Run 1 Buy Rate", justify="right", style="green")
    table.add_column("Run 2 Buy Rate", justify="right", style="blue")

    for quality in ['LOW', 'MEDIUM', 'HIGH']:
        s1 = stats1.get(quality, {})
        s2 = stats2.get(quality, {})

        count1 = s1.get('count', 0)
        count2 = s2.get('count', 0)
        buy1 = s1.get('buy_rate', 0)
        buy2 = s2.get('buy_rate', 0)

        table.add_row(
            quality,
            str(count1),
            str(count2),
            f"{buy1:.1%}",
            f"{buy2:.1%}"
        )

    console.print(table)
    console.print()


def _show_detailed_comparison(results1, results2):
    """Show round-by-round detailed comparison."""
    console.print("[bold]Round-by-Round Comparison[/bold]")

    history1 = results1.get('history', [])
    history2 = results2.get('history', [])

    max_rounds = max(len(history1), len(history2))

    table = Table()
    table.add_column("Rnd", style="dim")
    table.add_column("Run 1: Q→S→A", style="green")
    table.add_column("Run 1: Payoffs", justify="right", style="green")
    table.add_column("Run 2: Q→S→A", style="blue")
    table.add_column("Run 2: Payoffs", justify="right", style="blue")

    for i in range(min(max_rounds, 20)):  # Show first 20 rounds
        h1 = history1[i] if i < len(history1) else {}
        h2 = history2[i] if i < len(history2) else {}

        r1_flow = f"{h1.get('quality', '-')}→{h1.get('signal', '-')}→{h1.get('action', '-')}" if h1 else "-"
        r1_payoffs = f"S:{h1.get('sender_payoff', 0):+.0f} R:{h1.get('receiver_payoff', 0):+.0f}" if h1 else "-"

        r2_flow = f"{h2.get('quality', '-')}→{h2.get('signal', '-')}→{h2.get('action', '-')}" if h2 else "-"
        r2_payoffs = f"S:{h2.get('sender_payoff', 0):+.0f} R:{h2.get('receiver_payoff', 0):+.0f}" if h2 else "-"

        table.add_row(
            str(i + 1),
            r1_flow,
            r1_payoffs,
            r2_flow,
            r2_payoffs
        )

    if max_rounds > 20:
        table.add_row("...", "...", "...", "...", "...")

    console.print(table)
