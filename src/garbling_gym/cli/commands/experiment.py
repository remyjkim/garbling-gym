# ABOUTME: Experiment command for running batch experiments
# ABOUTME: Supports parameter sweeps and comparative analysis

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from pathlib import Path
import yaml
from typing import List, Dict, Any
from datetime import datetime

from ...core.game import Game
from ...core.config import GameConfig
from ...core.storage import ResultsStore
from ...core.agents import agent_factory


console = Console()


@click.command()
@click.argument('experiment_file', type=click.Path(exists=True))
@click.option(
    '--parallel', '-p',
    is_flag=True,
    help='Run experiments in parallel (experimental)'
)
@click.option(
    '--tag', '-t',
    multiple=True,
    help='Add tags to all runs in this experiment'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be run without executing'
)
def experiment(experiment_file, parallel, tag, dry_run):
    """
    Run batch experiments from configuration file.

    The experiment file should be a YAML file with this structure:

    \b
    experiments:
      - name: baseline
        rounds: 20
        llm: false

      - name: llm_sender
        rounds: 20
        llm: true
        llm_roles: [sender]

    Examples:
      gg experiment experiments/parameter_sweep.yaml
      gg experiment experiments/ablation.yaml --tag ablation-v1
      gg experiment experiments/large.yaml --parallel
    """
    # Load experiment configuration
    exp_path = Path(experiment_file)

    try:
        with open(exp_path, 'r') as f:
            exp_config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading experiment file:[/red] {e}")
        raise click.Abort()

    if 'experiments' not in exp_config:
        console.print("[red]Error:[/red] Experiment file must have 'experiments' key")
        raise click.Abort()

    experiments = exp_config['experiments']

    if not experiments:
        console.print("[yellow]No experiments found in file.[/yellow]")
        return

    console.print(f"\n[bold cyan]Loaded {len(experiments)} experiments from {exp_path.name}[/bold cyan]")

    # Dry run mode
    if dry_run:
        _show_dry_run(experiments)
        return

    # Run experiments
    if parallel:
        console.print("[yellow]Warning: Parallel execution is experimental and may not work with LLM agents[/yellow]")
        results = _run_parallel(experiments, list(tag))
    else:
        results = _run_sequential(experiments, list(tag))

    # Display summary
    _show_summary(results)


def _show_dry_run(experiments: List[Dict[str, Any]]):
    """Show what would be run without executing."""
    table = Table(title="Experiment Plan (Dry Run)")

    table.add_column("Name", style="cyan")
    table.add_column("Rounds", justify="right")
    table.add_column("LLM", justify="center")
    table.add_column("LLM Roles")
    table.add_column("Other Config")

    for exp in experiments:
        name = exp.get('name', 'unnamed')
        rounds = str(exp.get('rounds', 20))
        llm = "✓" if exp.get('llm', False) else "✗"
        llm_roles = ', '.join(exp.get('llm_roles', [])) if exp.get('llm') else '-'

        # Show other config keys
        other_keys = [k for k in exp.keys() if k not in ['name', 'rounds', 'llm', 'llm_roles']]
        other = ', '.join(other_keys) if other_keys else '-'

        table.add_row(name, rounds, llm, llm_roles, other)

    console.print(table)
    console.print(f"\n[dim]Total: {len(experiments)} experiments[/dim]")


def _run_sequential(experiments: List[Dict[str, Any]], tags: List[str]) -> List[Dict[str, Any]]:
    """Run experiments sequentially with progress tracking."""
    results = []
    store = ResultsStore()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        overall_task = progress.add_task(
            f"[cyan]Running {len(experiments)} experiments...",
            total=len(experiments)
        )

        for i, exp_config in enumerate(experiments, 1):
            name = exp_config.get('name', f'experiment_{i}')

            # Create GameConfig from experiment config
            config = _create_game_config(exp_config)

            # Update progress
            progress.update(overall_task, description=f"[cyan]Running: {name} ({i}/{len(experiments)})")

            # Run the game
            start_time = datetime.now()

            try:
                game_results = _run_single_experiment(config, exp_config)
                duration = (datetime.now() - start_time).total_seconds()

                # Add experiment-level tags
                all_tags = tags + exp_config.get('tags', [])

                # Generate run ID and save results
                run_id = store.generate_run_id(name)
                run_path = store.save_run(
                    run_id=run_id,
                    config=config,
                    results=game_results,
                    duration=duration,
                    name=name,
                    tags=all_tags
                )

                results.append({
                    'name': name,
                    'run_id': run_id,
                    'sender_total': game_results.sender_total,
                    'receiver_total': game_results.receiver_total,
                    'buy_rate': game_results.buy_rate,
                    'avg_informativeness': game_results.avg_informativeness,
                    'duration': duration,
                    'status': 'success'
                })

            except Exception as e:
                console.print(f"\n[red]Error in experiment '{name}':[/red] {e}")
                results.append({
                    'name': name,
                    'run_id': None,
                    'status': 'error',
                    'error': str(e)
                })

            progress.advance(overall_task)

    return results


def _run_parallel(experiments: List[Dict[str, Any]], tags: List[str]) -> List[Dict[str, Any]]:
    """Run experiments in parallel (experimental)."""
    console.print("[yellow]Parallel execution not yet implemented. Falling back to sequential.[/yellow]")
    return _run_sequential(experiments, tags)


def _create_game_config(exp_config: Dict[str, Any]) -> GameConfig:
    """Create GameConfig from experiment configuration."""
    # Extract known GameConfig fields
    config_kwargs = {}

    if 'rounds' in exp_config:
        config_kwargs['num_rounds'] = exp_config['rounds']
    if 'prior' in exp_config:
        config_kwargs['prior'] = exp_config['prior']
    if 'payoffs' in exp_config:
        config_kwargs['payoffs'] = exp_config['payoffs']
    if 'llm' in exp_config:
        config_kwargs['use_llm'] = exp_config['llm']
    if 'llm_model' in exp_config:
        config_kwargs['llm_model'] = exp_config['llm_model']

    return GameConfig(**config_kwargs)


def _run_single_experiment(config: GameConfig, exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    # Create agents using the factory
    sender = agent_factory.create_sender(model=config.llm_model)
    receiver = agent_factory.create_receiver(model=config.llm_model)

    # Run the game
    game = Game(config, sender=sender, receiver=receiver)
    results = game.play_game(verbose=False)

    return results


def _show_summary(results: List[Dict[str, Any]]):
    """Display summary table of experiment results."""
    console.print("\n")

    # Count successes and failures
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes

    if successes > 0:
        table = Table(title=f"Experiment Results ({successes} successful, {failures} failed)")

        table.add_column("Name", style="cyan")
        table.add_column("Run ID", style="dim", no_wrap=True)
        table.add_column("Sender", justify="right", style="green")
        table.add_column("Receiver", justify="right", style="blue")
        table.add_column("Buy Rate", justify="right")
        table.add_column("Info", justify="right")
        table.add_column("Duration", justify="right", style="dim")

        for result in results:
            if result['status'] == 'success':
                table.add_row(
                    result['name'],
                    result['run_id'][:20],
                    f"{result['sender_total']:+.0f}",
                    f"{result['receiver_total']:+.0f}",
                    f"{result['buy_rate']:.1%}",
                    f"{result['avg_informativeness']:.2f}",
                    f"{result['duration']:.1f}s"
                )

        console.print(table)

    if failures > 0:
        console.print(f"\n[red]{failures} experiments failed[/red]")
        for result in results:
            if result['status'] == 'error':
                console.print(f"  [red]✗[/red] {result['name']}: {result.get('error', 'Unknown error')}")

    if successes > 0:
        console.print(f"\n[dim]View details:[/dim] [bold]gg visualize <run-id>[/bold]")
        console.print(f"[dim]Compare runs:[/dim] [bold]gg compare <id1> <id2>[/bold]")
