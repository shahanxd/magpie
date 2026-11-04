"""``magpie`` command-line entry point."""

from __future__ import annotations

import json

import typer

from magpie import __version__
from magpie.costs import TINY_AYA_FIRE, CostModel

app = typer.Typer(add_completion=False, help="Magpie: robustness priced as attacker cost.")


@app.command()
def version() -> None:
    """Print the Magpie version."""
    typer.echo(__version__)


@app.command()
def cost(
    prompt_tokens: int = typer.Option(400, help="prompt tokens incl. the chat preamble"),
    output_tokens: int = typer.Option(256, help="generated tokens per sample"),
    n_samples: int = typer.Option(1, help="independent samples (best-of-N)"),
) -> None:
    """Price a generation workload on the Tiny Aya Fire cost model."""
    cm = CostModel(TINY_AYA_FIRE)
    c = cm.generation(prompt_tokens, output_tokens, n_samples)
    out = {
        "params_total": TINY_AYA_FIRE.params_total,
        "params_non_embedding": TINY_AYA_FIRE.params_non_embedding,
        **c.to_dict(),
        "log10_flops": round(c.log10_flops, 3),
    }
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
