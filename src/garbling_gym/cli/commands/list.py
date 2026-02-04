# ABOUTME: List command for browsing past experiments
# ABOUTME: Provides filtering, sorting, and formatted display of runs

import click
from rich.console import Console
from rich.table import Table
from datetime import datetime

from ...core.storage import ResultsStore


console = Console()


@click.command()
@click.option(
    '--limit', '-l',
    type=int,
    default=20,
    help='Maximum number of runs to display'
)
@click.option(
    '--sort',
    type=click.Choice(['timestamp', 'name', 'sender_total', 'receiver_total', 'rounds']),
    default='timestamp',
    help='Sort by field'
)
@click.option(
    '--filter-name',
    type=str,
    help='Filter by name pattern'
)
@click.option(
    '--min-sender',
    type=float,
    help='Minimum sender total'
)
@click.option(
    '--max-sender',
    type=float,
    help='Maximum sender total'
)
@click.option(
    '--llm/--no-llm',
    default=None,
    help='Filter by LLM usage'
)
def list_runs(limit, sort, filter_name, min_sender, max_sender, llm):
    """
    List past experiment runs.

    Examples:
      gg list                       # Show recent runs
      gg list --limit 10            # Show last 10 runs
      gg list --sort sender_total   # Sort by sender payoff
      gg list --filter-name baseline # Filter by name
      gg list --min-sender 100      # Filter by minimum sender total
    """
    store = ResultsStore()

    # Build filters
    filters = {}
    if filter_name:
        filters['name_like'] = filter_name
    if min_sender is not None:
        filters['sender_total_min'] = min_sender
    if max_sender is not None:
        filters['sender_total_max'] = max_sender
    if llm is not None:
        filters['use_llm'] = int(llm)

    # Map sort field to database column
    sort_field = {
        'timestamp': 'timestamp',
        'name': 'name',
        'sender_total': 'sender_total',
        'receiver_total': 'receiver_total',
        'rounds': 'num_rounds'
    }.get(sort, 'timestamp')

    # Get runs
    runs = store.list_runs(limit=limit, sort_by=sort_field, filters=filters)

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        console.print("\nRun your first experiment: [bold]gg run[/bold]")
        return

    # Create table
    table = Table(title=f"Experiment Runs (showing {len(runs)})")

    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Date", style="dim")
    table.add_column("Rounds", justify="right")
    table.add_column("Sender", justify="right", style="green")
    table.add_column("Receiver", justify="right", style="blue")
    table.add_column("Buy Rate", justify="right")
    table.add_column("Info", justify="right")

    for run in runs:
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(run['timestamp'])
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = run['timestamp'][:16]

        # Format values
        sender_str = f"{run['sender_total']:+.0f}"
        receiver_str = f"{run['receiver_total']:+.0f}"
        buy_rate_str = f"{run['buy_rate']:.1%}"
        info_str = f"{run['avg_informativeness']:.2f}"

        table.add_row(
            run['id'][:20],  # Truncate long IDs
            run.get('name', '')[:20] or '-',
            date_str,
            str(run['num_rounds']),
            sender_str,
            receiver_str,
            buy_rate_str,
            info_str
        )

    console.print(table)

    # Show commands
    console.print(f"\n[dim]View details:[/dim] [bold]gg visualize <run-id>[/bold]")
    console.print(f"[dim]Compare runs:[/dim] [bold]gg compare <id1> <id2>[/bold]")


# Alias for convenience
@click.command(hidden=True)
@click.pass_context
def ls(ctx):
    """Alias for 'list' command"""
    ctx.forward(list_runs)
