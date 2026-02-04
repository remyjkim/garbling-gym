# ABOUTME: Serve command for launching interactive dashboards
# ABOUTME: Starts Marimo web server with visualizations and dashboards

import click
from rich.console import Console
from pathlib import Path
import subprocess
import sys

console = Console()


@click.command()
@click.option(
    '--mode', '-m',
    type=click.Choice(['dashboard', 'run', 'edit']),
    default='dashboard',
    help='Dashboard mode to serve'
)
@click.option(
    '--run-id', '-r',
    type=str,
    help='Specific run ID to visualize (for run mode)'
)
@click.option(
    '--port', '-p',
    type=int,
    default=2718,
    help='Port to run server on'
)
@click.option(
    '--host',
    type=str,
    default='localhost',
    help='Host to bind to'
)
@click.option(
    '--no-browser',
    is_flag=True,
    help='Do not open browser automatically'
)
def serve(mode, run_id, port, host, no_browser):
    """
    Launch interactive web dashboards.

    Modes:
      - dashboard: Browse and compare all runs (default)
      - run: Visualize specific run in detail
      - edit: Open notebook in Marimo editor

    Examples:
      gg serve                              # Launch dashboard
      gg serve --mode run --run-id <id>     # Visualize specific run
      gg serve --port 8080                  # Custom port
      gg serve --mode edit                  # Edit notebook
    """

    # Check if marimo is installed
    try:
        import marimo
    except ImportError:
        console.print("[red]Error:[/red] marimo is not installed")
        console.print("\nInstall with: [bold]uv pip install marimo[/bold]")
        raise click.Abort()

    # Determine which notebook to serve
    notebooks_dir = Path(__file__).parent.parent.parent / "web" / "notebooks"

    if mode == 'dashboard':
        notebook_path = notebooks_dir / "dashboard.py"
        console.print(f"[cyan]Launching dashboard on {host}:{port}[/cyan]")
    elif mode == 'run':
        notebook_path = notebooks_dir / "run_visualization.py"
        if run_id:
            console.print(f"[cyan]Visualizing run: {run_id}[/cyan]")
        else:
            console.print("[cyan]Launching run visualization[/cyan]")
    else:  # edit mode
        notebook_path = notebooks_dir / "dashboard.py"
        console.print("[cyan]Opening notebook editor[/cyan]")

    if not notebook_path.exists():
        console.print(f"[red]Error:[/red] Notebook not found: {notebook_path}")
        raise click.Abort()

    # Build marimo command
    cmd = [sys.executable, "-m", "marimo"]

    if mode == 'edit':
        cmd.append("edit")
    else:
        cmd.append("run")
        cmd.extend(["--port", str(port)])
        cmd.extend(["--host", host])

        if no_browser:
            cmd.append("--no-browser")

    cmd.append(str(notebook_path))

    # Display info
    console.print("\n[bold green]Server starting...[/bold green]")
    console.print(f"[dim]Notebook:[/dim] {notebook_path.name}")

    if mode != 'edit':
        console.print(f"\n[bold]Open in browser:[/bold] http://{host}:{port}")

    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    # Run marimo server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Error running marimo:[/red] {e}")
        raise click.Abort()
