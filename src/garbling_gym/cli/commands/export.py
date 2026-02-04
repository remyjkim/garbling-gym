# ABOUTME: Export command for generating static reports
# ABOUTME: Supports HTML and CSV export formats

import click
from rich.console import Console
from pathlib import Path

from ...core.storage import ResultsStore
from ...exports import HTMLExporter, CSVExporter


console = Console()


@click.command()
@click.argument('run_id')
@click.option(
    '--format', '-f',
    type=click.Choice(['html', 'csv', 'all']),
    default='html',
    help='Export format'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output path (default: run_results/{run_id}/exports/)'
)
def export(run_id, format, output):
    """
    Export run results to static files.

    Supports HTML reports and CSV data exports for analysis.

    Examples:
      gg export 2026-02-03_120000_test              # Export as HTML
      gg export run_id --format csv                  # Export as CSV
      gg export run_id --format all                  # Export all formats
      gg export run_id --output /path/to/report.html # Custom output path
    """
    store = ResultsStore()

    # Load run data
    try:
        run_data = store.load_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Run '{run_id}' not found")
        console.print("\nAvailable runs: [bold]gg list[/bold]")
        raise click.Abort()

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        # Default to exports directory within run directory
        run_path = store.get_run_path(run_id)
        output_path = run_path / "exports"
        output_path.mkdir(exist_ok=True)

    console.print(f"[cyan]Exporting run:[/cyan] {run_id}")

    exported_files = []

    # Export HTML
    if format in ['html', 'all']:
        console.print("[dim]Generating HTML report...[/dim]")
        html_exporter = HTMLExporter()

        if format == 'all':
            html_path = output_path / f"{run_id}.html"
        else:
            html_path = output_path if output_path.suffix == '.html' else output_path / "report.html"

        html_file = html_exporter.export(run_data, html_path)
        exported_files.append(html_file)
        console.print(f"[green]✓[/green] HTML report: {html_file}")

    # Export CSV
    if format in ['csv', 'all']:
        console.print("[dim]Generating CSV files...[/dim]")
        csv_exporter = CSVExporter()

        if format == 'all':
            csv_dir = output_path / "csv"
            csv_dir.mkdir(exist_ok=True)
            csv_path = csv_dir / run_id
        else:
            csv_path = output_path

        csv_dir = csv_exporter.export(run_data, csv_path)
        exported_files.append(csv_dir)

        # List generated CSV files
        csv_files = list(csv_dir.glob("*.csv"))
        for csv_file in csv_files:
            console.print(f"[green]✓[/green] CSV file: {csv_file}")

    console.print(f"\n[bold green]Export complete![/bold green]")

    if format == 'html' or format == 'all':
        console.print(f"\n[dim]Open in browser:[/dim] [bold]open {exported_files[0]}[/bold]")
