# ABOUTME: Run command implementation for executing single games
# ABOUTME: Handles game execution with results storage and progress display

import time
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.config import GameConfig
from ...core.game import Game
from ...core.agents import agent_factory
from ...core.storage import ResultsStore
from ..config import ConfigLoader, merge_config_overrides


console = Console()


@click.command()
@click.option(
    '--rounds', '-r',
    type=int,
    default=None,
    help='Number of rounds to play'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to config file (YAML or Python)'
)
@click.option(
    '--name', '-n',
    type=str,
    help='Experiment name'
)
@click.option(
    '--llm/--no-llm',
    default=None,
    help='Use LLM agents (requires OPENAI_API_KEY)'
)
@click.option(
    '--llm-model',
    type=str,
    default=None,
    help='LLM model to use (e.g., gpt-4o-mini)'
)
@click.option(
    '--verbose/--quiet', '-v/-q',
    default=True,
    help='Show detailed output'
)
@click.option(
    '--save/--no-save',
    default=True,
    help='Save results to run_results/'
)
@click.pass_context
def run(ctx, rounds, config, name, llm, llm_model, verbose, save):
    """
    Run a single garbling economics game.

    Examples:
      gg run                          # Run with defaults
      gg run --rounds 50              # Override rounds
      gg run --config experiment.yaml # Load from config
      gg run --llm --llm-model gpt-4o # Use specific LLM
    """
    start_time = time.time()

    # Load or create config
    if config:
        console.print(f"[cyan]Loading config from {config}[/cyan]")
        game_config = ConfigLoader.load(config)
    else:
        game_config = GameConfig()

    # Apply CLI overrides
    overrides = {}
    if rounds is not None:
        overrides['num_rounds'] = rounds
    if llm is not None:
        overrides['use_llm'] = llm
    if llm_model is not None:
        overrides['llm_model'] = llm_model

    if overrides:
        game_config = merge_config_overrides(game_config, overrides)

    # Display config summary
    console.print("\n[bold]Game Configuration[/bold]")
    console.print(f"  Rounds: {game_config.num_rounds}")
    console.print(f"  LLM: {game_config.use_llm}")
    if game_config.use_llm:
        console.print(f"  Model: {game_config.llm_model}")
    console.print()

    # Create agents
    sender = agent_factory.create_sender(model=game_config.llm_model)
    receiver = agent_factory.create_receiver(model=game_config.llm_model)

    # Create and run game
    game = Game(game_config, sender=sender, receiver=receiver)

    console.print("[bold green]Starting game...[/bold green]\n")

    results = game.play_game(verbose=verbose)

    duration = time.time() - start_time

    # Save results if requested
    if save:
        console.print("\n[cyan]Saving results...[/cyan]")

        store = ResultsStore()
        run_id = store.generate_run_id(name)

        run_path = store.save_run(
            run_id=run_id,
            config=game_config,
            results=results,
            duration=duration,
            name=name
        )

        console.print(f"[green]✓[/green] Results saved to: {run_path}")
        console.print(f"[green]✓[/green] Run ID: [bold]{run_id}[/bold]")

        # Show quick summary
        console.print("\n[bold]Quick Summary[/bold]")
        console.print(f"  Sender Total:   {results.sender_total:+.0f}")
        console.print(f"  Receiver Total: {results.receiver_total:+.0f}")
        console.print(f"  Buy Rate:       {results.buy_rate:.1%}")
        console.print(f"  Informativeness: {results.avg_informativeness:.2f}")
        console.print(f"  Duration:       {duration:.1f}s")

        console.print(f"\nView details: [bold]gg visualize {run_id}[/bold]")

    return results
