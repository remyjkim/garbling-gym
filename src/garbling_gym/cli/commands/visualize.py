# ABOUTME: Visualize command for displaying game analysis
# ABOUTME: Shows detailed ASCII visualizations and statistics for completed runs

import click
from rich.console import Console
from pathlib import Path

from ...core.storage import ResultsStore
from ...visualization.ascii import visualize_results, analyze_bayesian_updating


console = Console()


@click.command()
@click.argument('run_id', required=False)
@click.option(
    '--section', '-s',
    type=click.Choice(['summary', 'bayesian', 'all']),
    default='all',
    help='Which sections to display'
)
@click.option(
    '--export', '-e',
    type=click.Path(),
    help='Export visualization to file'
)
def visualize(run_id, section, export):
    """
    Visualize results from a completed run.

    If no run_id is provided, visualizes the most recent run.

    Examples:
      gg visualize                          # Show latest run
      gg visualize 2026-02-03_120000_test   # Show specific run
      gg visualize --section summary        # Show only summary section
      gg visualize --export report.txt      # Export to file
    """
    store = ResultsStore()

    # Get run_id if not provided
    if not run_id:
        recent_runs = store.list_runs(limit=1, sort_by='timestamp')
        if not recent_runs:
            console.print("[yellow]No runs found. Run your first experiment:[/yellow] [bold]gg run[/bold]")
            raise click.Abort()
        run_id = recent_runs[0]['id']
        console.print(f"[dim]Visualizing most recent run: {run_id}[/dim]\n")

    # Load run results
    try:
        run_data = store.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Run '{run_id}' not found")
        console.print("\nAvailable runs: [bold]gg list[/bold]")
        raise click.Abort()

    # Extract results dict - handle nested structure
    if 'summary' in run_data['results']:
        # Flatten the nested structure for visualization
        results_dict = run_data['results']['summary'].copy()
        results_dict['history'] = run_data['results'].get('history', [])
    else:
        results_dict = run_data['results']

    # Generate visualizations
    output = []

    if section in ['summary', 'all']:
        output.append(visualize_results(results_dict))

    if section in ['bayesian', 'all']:
        output.append(analyze_bayesian_updating(results_dict['history']))

    visualization = '\n'.join(output)

    # Display or export
    if export:
        export_path = Path(export)
        with open(export_path, 'w') as f:
            f.write(visualization)
        console.print(f"[green]✓[/green] Exported visualization to: {export_path}")
    else:
        # Print directly without Rich formatting to preserve ASCII art
        print(visualization)

    # Show related commands
    if not export:
        console.print(f"\n[dim]Compare with another run:[/dim] [bold]gg compare {run_id} <other-run-id>[/bold]")
