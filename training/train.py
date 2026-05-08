"""
training/train.py
==================
Main QLoRA training script for MedQLoRA using Apple MLX.

This script wraps mlx_lm.lora (Apple's native LoRA trainer) with:
  - Config loading from config.yaml
  - W&B + local logging via callbacks.py
  - Pre/post validation loss tracking
  - Checkpoint saving every N steps
  - Sample generation for qualitative monitoring
  - Guard: verifies baseline scores exist before training starts

Usage:
  python training/train.py                    # full training run
  python training/train.py --test-run         # 1 step (verify setup)
  python training/train.py --resume           # resume from latest checkpoint
  python training/train.py --epochs 1         # override config epochs

How MLX-LM LoRA works:
  mlx_lm provides a built-in LoRA trainer via `mlx_lm.lora`.
  We call it programmatically with our config and hook in callbacks.
  The trainer handles gradient accumulation, cosine LR, adapter saving.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from training.callbacks import (
    WandbCallback,
    LocalLogCallback,
)

console = Console()
load_dotenv(ROOT / ".env")


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Pre-flight checks ─────────────────────────────────────────────────────────
def preflight_checks(cfg: dict, data_dir: Path | None = None) -> None:
    """Verify all prerequisites before starting an expensive training run."""
    console.rule("[bold yellow]Pre-flight Checks[/bold yellow]")

    errors = []
    warnings = []

    # 1. Baseline scores (warning, not error). Without a "before" number the
    #    final compare.py report has nothing to subtract from — but training
    #    itself still works, and you can always run eval_qa.py --which base
    #    afterwards (the base model is independent of any adapters).
    baseline_file = ROOT / cfg["evaluation"]["baseline_scores_file"]
    if not baseline_file.exists():
        warnings.append(
            f"Baseline scores not found: {baseline_file}\n"
            "  → For a complete before/after report, run "
            "[cyan]python evaluation/eval_qa.py --which base[/cyan] before training.\n"
            "  Training will still succeed — you can also run baseline eval after."
        )
    else:
        with open(baseline_file) as f:
            baseline = json.load(f)
        m = baseline["metrics"]
        # New QA eval format uses char_similarity_mean + keyword_recall_mean.
        # Old PubMedQA format used accuracy. Display whichever fields exist.
        if "char_similarity_mean" in m:
            console.log(
                f"[green]✓[/green] Baseline scores found — "
                f"char_sim {m['char_similarity_mean']:.3f}, "
                f"keyword_recall {m['keyword_recall_mean']:.3f}"
            )
        elif "accuracy" in m:
            console.log(
                f"[green]✓[/green] Baseline scores found — accuracy {m['accuracy']:.1%}"
            )
        else:
            console.log("[yellow]⚠[/yellow] Baseline scores file present but no recognized metrics")

    # 2. Training data must exist
    if data_dir is None:
        train_file = ROOT / cfg["data"]["train_file"]
        val_file = ROOT / cfg["data"]["val_file"]
    else:
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "valid.jsonl"
    if not train_file.exists() or not val_file.exists():
        errors.append(
            f"Training data not found: {train_file}\n"
            "  → Run: python data/prepare_dataset.py"
        )
    else:
        n_train = sum(1 for _ in open(train_file))
        n_val = sum(1 for _ in open(val_file))
        console.log(
            f"[green]✓[/green] Training data: {n_train:,} train / {n_val:,} val examples"
        )

    # 3. HF token for model download
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        warnings.append(
            "HF_TOKEN not set — may fail to download gated models like Llama 3.1\n"
            "  → Run: huggingface-cli login  OR  set HF_TOKEN in .env"
        )
    else:
        console.log("[green]✓[/green] HF_TOKEN is set")

    # 4. W&B token
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        warnings.append("WANDB_API_KEY not set — training will log locally only")
    else:
        console.log("[green]✓[/green] WANDB_API_KEY is set")

    # 5. MLX available
    try:
        import mlx.core as mx
        import mlx_lm
        console.log(f"[green]✓[/green] mlx-lm {mlx_lm.__version__} available")
    except ImportError as e:
        errors.append(f"mlx-lm not installed: {e}\n  → pip install mlx-lm")

    # Print warnings
    for w in warnings:
        rprint(f"[yellow]⚠[/yellow] {w}")

    # Print errors and exit if any
    if errors:
        console.print()
        for e in errors:
            rprint(f"[red]✗[/red] {e}")
        rprint("\n[red]Fix the above errors before training.[/red]")
        sys.exit(1)

    console.log("[bold green]All pre-flight checks passed ✓[/bold green]")


# ── Build MLX-LM training args ─────────────────────────────────────────────────
def build_mlx_lm_args(
    cfg: dict,
    output_dir: Path,
    data_dir: Path,
    test_run: bool = False,
) -> list[str]:
    """
    Build the argument list for mlx_lm lora training.

    Invocation: `python -m mlx_lm lora ...`
    (The old `python -m mlx_lm.lora` form is deprecated as of mlx-lm 0.20+)

    Key flags:
      --num-layers N    : number of transformer layers to apply LoRA (from top);
                          use -1 for all layers (slowest, best quality)
      --max-seq-length  : sequences longer than this are silently truncated.
                          Set to 2048 in config; drop to 1024 to halve RAM usage.
    """
    lora_layers = cfg["lora"].get("lora_layers", 16)
    # -1 means "all layers" — mlx_lm interprets this correctly
    lora_layers_arg = str(lora_layers) if lora_layers != -1 else str(lora_layers)
    lora_rank = cfg["lora"].get("r", 16)
    lora_dropout = cfg["lora"].get("dropout", 0.0)

    # Path to the LoRA hyperparameter config (rank, dropout, scale).
    # mlx_lm does not expose these as CLI flags; they must go via --config.
    lora_cfg_path = Path(__file__).parent / "lora_config.yaml"

    args = [
        sys.executable, "-m", "mlx_lm", "lora",   # ← fixed: not mlx_lm.lora
        "--train",
        "-c", str(lora_cfg_path),                  # LoRA rank=16, dropout=0.05, scale=1.0
        "--model", cfg["model"]["name"],
        "--data", str(data_dir),
        "--batch-size", str(cfg["training"]["batch_size"]),
        "--learning-rate", str(cfg["training"]["learning_rate"]),
        "--num-layers", lora_layers_arg,
        "--iters", str(50 if test_run else _compute_iters(cfg)),
        "--val-batches", str(cfg["training"].get("val_batches", 25)),
        # Print train loss every 50 iters (~2 min cadence at ~2.4 s/iter);
        # validation pass runs every val_every iters (~500, ~20 min).
        "--steps-per-report", "50",
        "--steps-per-eval", str(cfg["training"]["val_every"]),
        "--grad-accumulation-steps", str(cfg["training"]["gradient_accumulation"]),
        "--save-every", str(cfg["training"]["save_every"]),
        "--adapter-path", str(output_dir),
        "--max-seq-length", str(cfg["model"]["max_seq_length"]),
        "--grad-checkpoint",  # gradient checkpointing saves ~30% memory
        # KEY FIX: only compute loss on assistant tokens, not the question.
        # Without this, ~75% of gradient signal is wasted predicting the
        # prompt back — the model never properly memorises answers.
        "--mask-prompt",
    ]

    return args




def _compute_iters(cfg: dict, train_file: Path | None = None) -> int:
    """
    Number of optimizer steps to train for, computed from epochs + dataset size.

    mlx_lm.lora's `--iters` counts *optimizer steps*. Each step processes
    `batch_size * gradient_accumulation` examples (one effective batch).
    So:
        iters = epochs * (n_examples / (batch_size * gradient_accumulation))

    NOTE: a previous version ignored gradient_accumulation and trained
    `gradient_accumulation×` more steps than intended. With small datasets
    (TechMojo's 99 examples, 30 epochs) the bug would be catastrophic — 240
    effective epochs instead of 30 — so this formula is load-bearing.
    """
    if train_file is None:
        train_file = ROOT / cfg["data"]["train_file"]
    if train_file.exists():
        n_examples = sum(1 for _ in open(train_file))
    else:
        n_examples = 100  # safe fallback

    batch_size = cfg["training"]["batch_size"]
    grad_accum = cfg["training"].get("gradient_accumulation", 1)
    epochs = cfg["training"]["epochs"]
    effective_batch = max(1, batch_size * grad_accum)
    steps_per_epoch = max(1, n_examples // effective_batch)
    return steps_per_epoch * epochs


def _latest_resume_checkpoint(output_dir: Path) -> tuple[Path | None, int]:
    """
    Return latest step checkpoint and its step number from adapters dir.

    Expected checkpoint format:
      0002500_adapters.safetensors
    """
    candidates = sorted(output_dir.glob("*_adapters.safetensors"))
    if not candidates:
        return None, 0

    latest = candidates[-1]
    step = 0
    try:
        step = int(latest.name.split("_", 1)[0])
    except (ValueError, IndexError):
        step = 0

    return latest, step


def _rewrite_iter_prefix(line: str, resume_step: int) -> tuple[str, int | None]:
    """
    Rewrite `Iter N:` to global iteration when resuming from checkpoint step.

    Example:
      resume_step=2500, incoming "Iter 1: ..." -> "Iter 2501 (session 1): ..."
    """
    if resume_step <= 0:
        return line, None

    m = re.search(r"Iter\s+(\d+):", line)
    if not m:
        return line, None

    session_step = int(m.group(1))
    global_step = resume_step + session_step
    rewritten = line.replace(
        f"Iter {session_step}:",
        f"Iter {global_step} (session {session_step}):",
        1,
    )
    return rewritten, global_step


# ── Print training config ─────────────────────────────────────────────────────
def print_training_config(cfg: dict, iters: int) -> None:
    table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Model", cfg["model"]["name"])
    table.add_row("LoRA Rank (r)", str(cfg["lora"]["r"]))
    table.add_row("LoRA Alpha", str(cfg["lora"]["alpha"]))
    table.add_row("Target Modules", ", ".join(cfg["lora"]["target_modules"]))
    table.add_row("Batch Size", str(cfg["training"]["batch_size"]))
    table.add_row("Gradient Accumulation", str(cfg["training"]["gradient_accumulation"]))
    table.add_row("Effective Batch Size", str(
        cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation"]
    ))
    table.add_row("Learning Rate", str(cfg["training"]["learning_rate"]))
    table.add_row("LR Scheduler", cfg["training"]["lr_scheduler"])
    table.add_row("Warmup Steps", str(cfg["training"]["warmup_steps"]))
    table.add_row("Epochs", str(cfg["training"]["epochs"]))
    table.add_row("Est. Total Steps", f"~{iters:,}")
    table.add_row("Checkpoint Every", f"{cfg['training']['save_every']} steps")
    table.add_row("Framework", "Apple MLX")
    table.add_row("Dtype", cfg["hardware"]["dtype"])

    console.print(table)


# ── Main training entry point ─────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Train MedQLoRA with Apple MLX")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run 50 steps only — validates setup without full training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in adapters/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (not recommended)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Dataset directory containing train.jsonl/valid.jsonl/test.jsonl (default: data)",
    )
    parser.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Resume from a specific checkpoint file (overrides --resume latest lookup)",
    )
    args = parser.parse_args()

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    console.rule("[bold blue]MedQLoRA Training[/bold blue]")
    rprint(
        f"[bold]Model:[/bold] {cfg['model']['name']}\n"
        f"[bold]Framework:[/bold] Apple MLX\n"
        f"[bold]Hardware:[/bold] Apple Silicon (Mac Mini 24GB)\n"
    )

    data_dir = ROOT / args.data_dir
    if not args.skip_preflight:
        preflight_checks(cfg, data_dir=data_dir)

    output_dir = ROOT / cfg["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    est_iters = _compute_iters(cfg, train_file=data_dir / "train.jsonl")

    resume_file = None
    resume_step = 0
    train_iters = 50 if args.test_run else est_iters
    if args.resume_file and not args.test_run:
        resume_file = Path(args.resume_file)
        if not resume_file.is_absolute():
            resume_file = ROOT / resume_file
        if not resume_file.exists():
            rprint(f"[red]--resume-file not found: {resume_file}[/red]")
            sys.exit(1)
        try:
            resume_step = int(resume_file.name.split("_", 1)[0])
        except (ValueError, IndexError):
            resume_step = 0
        default_train_file = ROOT / cfg["data"]["train_file"]
        same_data_regime = (data_dir / "train.jsonl").resolve() == default_train_file.resolve()
        if same_data_regime:
            train_iters = max(est_iters - resume_step, 1)
        else:
            # For a new dataset regime (e.g., class-balanced stage), run full
            # iteration budget computed from that dataset instead of subtracting
            # old checkpoint step counters.
            train_iters = est_iters
        rprint(
            f"[cyan]Resuming from explicit file:[/cyan] {resume_file}\n"
            f"[cyan]Checkpoint step:[/cyan] {resume_step:,}\n"
            f"[cyan]Remaining steps:[/cyan] {train_iters:,}"
        )
    elif args.resume and not args.test_run:
        resume_file, resume_step = _latest_resume_checkpoint(output_dir)
        if resume_file is None:
            rprint(
                f"[red]--resume requested but no checkpoint found in {output_dir}[/red]\n"
                f"[dim]Expected files like: 0002500_adapters.safetensors[/dim]"
            )
            sys.exit(1)

        default_train_file = ROOT / cfg["data"]["train_file"]
        same_data_regime = (data_dir / "train.jsonl").resolve() == default_train_file.resolve()
        # Train only remaining steps when resuming the same dataset regime.
        train_iters = max(est_iters - resume_step, 1) if same_data_regime else est_iters
        rprint(
            f"[cyan]Resuming from:[/cyan] {resume_file}\n"
            f"[cyan]Checkpoint step:[/cyan] {resume_step:,}\n"
            f"[cyan]Remaining steps:[/cyan] {train_iters:,}"
        )

    print_training_config(cfg, train_iters)

    if args.test_run:
        rprint("\n[yellow]⚡ TEST RUN MODE — 50 steps only[/yellow]")

    # ── Init callbacks ────────────────────────────────────────────────────────
    run_name = f"medqlora-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    wandb_cb = WandbCallback(
        project=cfg["logging"]["wandb_project"],
        run_name=run_name,
        config=cfg,
        entity=cfg["logging"].get("wandb_entity"),
    )

    local_log = LocalLogCallback(
        ROOT / cfg["logging"]["local_log_file"]
    )

    # Note: CheckpointCallback is no longer wired in — mlx_lm.lora handles all
    # checkpoint writes internally via --save-every. Its dir is `output_dir`.

    # ── Build and run mlx_lm.lora ─────────────────────────────────────────────
    mlx_args = build_mlx_lm_args(
        cfg,
        output_dir,
        data_dir=data_dir,
        test_run=args.test_run,
    )
    if not args.test_run:
        # Override iteration budget when resuming from a non-zero step.
        if train_iters != est_iters:
            iters_idx = mlx_args.index("--iters") + 1
            mlx_args[iters_idx] = str(train_iters)
        if resume_file is not None:
            mlx_args.extend(["--resume-adapter-file", str(resume_file)])

    console.rule("[bold green]Starting Training[/bold green]")
    rprint(f"[dim]Command: {' '.join(mlx_args)}[/dim]\n")

    start_time = time.time()

    # Log initial entry
    local_log.log({"event": "training_start", "run_name": run_name}, step=0)
    wandb_cb.log({"event": "start"}, step=0)

    # Stream child output and rewrite Iter prefixes on resume so logs show
    # global progress (instead of restarting from Iter 1 every session).
    step = 0
    process = None
    try:
        process = subprocess.Popen(
            mlx_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line, maybe_global_step = _rewrite_iter_prefix(raw_line, resume_step)
            if maybe_global_step is not None:
                step = maybe_global_step
            print(line, end="", flush=True)

        process.wait()
        if process.returncode != 0:
            rprint(f"[red]mlx_lm.lora exited with code {process.returncode}[/red]")

    except KeyboardInterrupt:
        rprint("\n[yellow]Training interrupted by user.[/yellow]")
        if process is not None:
            process.terminate()

    except Exception as e:
        rprint(f"[red]Training error: {e}[/red]")
        raise

    finally:
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        local_log.log(
            {"event": "training_end", "elapsed_s": elapsed, "steps": step},
            step=step,
        )
        wandb_cb.log({"elapsed_s": elapsed}, step=step)
        wandb_cb.finish()
        local_log.close()

    # ── Post-training summary ─────────────────────────────────────────────────
    console.rule("[bold green]Training Complete[/bold green]")
    console.print(
        Panel(
            f"[bold]Elapsed time:[/bold] {elapsed_str}\n"
            f"[bold]Adapter weights saved to:[/bold] {output_dir}\n\n"
            f"[bold]Next steps:[/bold]\n"
            f"  1. [cyan]python evaluation/eval_finetuned.py[/cyan]\n"
            f"  2. [cyan]python evaluation/compare.py[/cyan]\n"
            f"  3. [cyan]python app.py[/cyan]",
            title="✅ MedQLoRA Training Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
