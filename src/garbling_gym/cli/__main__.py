# ABOUTME: Main CLI entry point for the garbling-gym command
# ABOUTME: Implements the 'gg' command with subcommand structure

import click
from rich.console import Console
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path.cwd() / ".env"
if env_path.exists():
    load_dotenv(env_path)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="garbling-gym")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def main(ctx, verbose, quiet):
    """
    Garbling Gym - An extensible experiment framework for garbling economics.

    Run simulations, analyze results, and explore information economics concepts.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["console"] = console


# Register commands
from .commands.run import run
from .commands.list import list_runs
from .commands.experiment import experiment
from .commands.visualize import visualize
from .commands.compare import compare
from .commands.export import export
from .commands.serve import serve

main.add_command(run)
main.add_command(list_runs)
main.add_command(experiment)
main.add_command(visualize)
main.add_command(compare)
main.add_command(export)
main.add_command(serve)


if __name__ == "__main__":
    main()
