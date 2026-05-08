"""
training/callbacks.py
======================
Training callbacks for MedQLoRA:
  - WandbCallback: logs loss, val_loss, lr to W&B
  - LocalLogCallback: always writes training log to JSONL (fallback when no W&B)
  - CheckpointCallback: tracks checkpoint cadence (mlx_lm saves the actual files)

Callbacks are called by train.py at each logging step.
"""

import json
import os
import time
from pathlib import Path

from rich.console import Console
from rich import print as rprint

console = Console()


# ── W&B callback ──────────────────────────────────────────────────────────────
class WandbCallback:
    """
    Logs training metrics to Weights & Biases.
    Gracefully degrades if WANDB_API_KEY is not set.
    """

    def __init__(self, project: str, run_name: str, config: dict, entity: str | None = None):
        self.enabled = False
        self.run = None

        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            rprint(
                "[yellow]⚠ WANDB_API_KEY not set — W&B logging disabled.[/yellow]\n"
                "  Set it in .env or export WANDB_API_KEY=... to enable."
            )
            return

        try:
            import wandb  # type: ignore

            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                reinit=True,
            )
            self.enabled = True
            console.log(f"[green]W&B run started[/green]: {wandb.run.url}")
        except Exception as e:
            rprint(f"[yellow]⚠ W&B init failed ({e}) — falling back to local logging.[/yellow]")

    def log(self, metrics: dict, step: int) -> None:
        if self.enabled and self.run:
            try:
                import wandb  # type: ignore
                wandb.log(metrics, step=step)
            except Exception as e:
                rprint(f"[dim yellow]W&B log error at step {step}: {e}[/dim yellow]")

    def log_text(self, key: str, text: str, step: int) -> None:
        """Log a text sample (e.g. model generation) to W&B."""
        if self.enabled and self.run:
            try:
                import wandb  # type: ignore
                wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        if self.enabled and self.run:
            try:
                import wandb  # type: ignore
                wandb.finish()
                console.log("[green]W&B run finished[/green]")
            except Exception:
                pass


# ── Local log callback ────────────────────────────────────────────────────────
class LocalLogCallback:
    """
    Always writes training metrics to a local JSONL file.
    Serves as the fallback when W&B is not configured,
    and as a supplementary log when W&B is active.
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.log_file, "a", encoding="utf-8")
        console.log(f"[dim]Local training log → {self.log_file}[/dim]")

    def log(self, metrics: dict, step: int) -> None:
        record = {"step": step, "ts": time.time(), **metrics}
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── Checkpoint callback ────────────────────────────────────────────────────────
class CheckpointCallback:
    """
    Tracks the checkpoint cadence. The actual adapter files are written by
    mlx_lm.lora itself (we run it as a subprocess and don't hold the model
    object), so this class only answers "is this step a checkpoint step?".
    """

    def __init__(self, output_dir: Path, save_every: int):
        self.output_dir = output_dir
        self.save_every = save_every
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.save_every == 0
