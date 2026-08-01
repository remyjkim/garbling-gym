# ABOUTME: `gg solve` — print the theory benchmark bundle for a config
# ABOUTME: Computes benchmarks WITHOUT running a game (Proposal 08).

import numpy as np
import click
from rich.console import Console
from rich.table import Table

from ...core.config import GameConfig
from ...core.types import AssetQuality
from ...core.theory.benchmarks import compute_benchmarks
from ..config import ConfigLoader


console = Console()


def _parse_chi_grid(value: str) -> np.ndarray:
    """Parse a comma-separated chi grid string like '0,0.5,1' into an array."""
    return np.array([float(x) for x in value.split(",") if x.strip() != ""], dtype=float)


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to config file (YAML) for prior/payoffs.'
)
@click.option('--prior-low', type=float, default=None, help='Prior P(LOW)')
@click.option('--prior-med', type=float, default=None, help='Prior P(MEDIUM)')
@click.option('--prior-high', type=float, default=None, help='Prior P(HIGH)')
@click.option(
    '--chi-grid', type=str, default=None,
    help='Comma-separated credibility values for the weak-institution curve '
         '(e.g. "0,0.25,0.5,0.75,1"). Default: a 0..1 grid in steps of 0.1.'
)
@click.option(
    '--grid-n', type=int, default=30,
    help='Simplex grid resolution (higher = more accurate, slower). Default 30.'
)
def solve(config, prior_low, prior_med, prior_high, chi_grid, grid_n):
    """Print the theory benchmark bundle for a game configuration.

    Computes the Kamenica-Gentzkow (cav), Lipnowski-Ravid (qcav), and LRS
    weak-institution values for the configured prior and payoffs, without
    running a game. This is the analytical ground truth that experiments
    measure against.

    Examples:
      gg solve                                   # default config
      gg solve --prior-low 0.5 --prior-high 0.2  # custom prior
      gg solve --chi-grid 0,0.25,0.5,0.75,1      # fine weak-institution curve
    """
    # Build the config.
    if config:
        game_config = ConfigLoader.load(config)
    else:
        game_config = GameConfig()

    # Apply prior overrides.
    overrides = {}
    if any(p is not None for p in (prior_low, prior_med, prior_high)):
        # Prior defaults to the current config values, then overrides.
        cur = game_config.prior
        low = prior_low if prior_low is not None else cur[AssetQuality.LOW]
        med = prior_med if prior_med is not None else cur[AssetQuality.MEDIUM]
        high = prior_high if prior_high is not None else cur[AssetQuality.HIGH]
        overrides['prior'] = {
            AssetQuality.LOW: low,
            AssetQuality.MEDIUM: med,
            AssetQuality.HIGH: high,
        }

    if overrides:
        from ..config import merge_config_overrides
        game_config = merge_config_overrides(game_config, overrides)

    chi = _parse_chi_grid(chi_grid) if chi_grid else None

    console.print("\n[bold cyan]Computing theory benchmarks...[/bold cyan]")
    bundle = compute_benchmarks(game_config, chi_grid=chi, grid_n=grid_n)

    # --- Summary table ---
    table = Table(title="Benchmark Bundle", show_header=True)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Sender value", justify="right", style="green")
    table.add_column("Meaning", style="dim")

    table.add_row("babbling", f"{bundle.babbling:.3f}", "no information (posterior = prior)")
    table.add_row("qcav", f"{bundle.qcav:.3f}", "cheap-talk value (Lipnowski-Ravid)")
    table.add_row("cav", f"{bundle.cav:.3f}", "commitment value (Kamenica-Gentzkow)")
    table.add_row(
        "[bold]value of commitment[/bold]",
        f"[bold green]{bundle.value_of_commitment:+.3f}[/bold green]",
        "cav - qcav (the gap Proposals 09-11 measure)",
    )
    console.print(table)

    # --- Receiver bounds ---
    rcv = Table(title="Receiver bounds", show_header=False)
    rcv.add_column(style="dim")
    rcv.add_column(justify="right")
    rcv.add_row("floor (no-info)", f"{bundle.floor:.3f}")
    rcv.add_row("ceiling (perfect info)", f"{bundle.ceiling:.3f}")
    console.print(rcv)

    # --- Weak-institution curve ---
    curve = Table(title="Weak-institution curve v*(chi)", show_header=True)
    curve.add_column("chi", justify="right", style="magenta")
    curve.add_column("v*(chi)", justify="right", style="yellow")
    for chi_val in sorted(bundle.weak_inst):
        curve.add_row(f"{chi_val:.2f}", f"{bundle.weak_inst[chi_val]:.3f}")
    console.print(curve)

    console.print(
        f"\n[dim]Prior: LOW={game_config.prior[AssetQuality.LOW]:.2f} "
        f"MEDIUM={game_config.prior[AssetQuality.MEDIUM]:.2f} "
        f"HIGH={game_config.prior[AssetQuality.HIGH]:.2f} | "
        f"grid_n={grid_n}[/dim]"
    )
