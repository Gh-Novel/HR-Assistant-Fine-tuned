"""
data/prepare_dataset.py
========================
Prepare the TechMojo HR fine-tune dataset.

Pipeline:
  1. Load raw TechMojo Q&A from data/techmojo/raw.jsonl (already downloaded
     from PranavTM/LeavePolicy on HuggingFace via curl).
  2. Convert each {"conversations": [...]} record into mlx-lm's expected
     {"messages": [system, user, assistant]} chat format, prepending the
     system prompt from config.yaml.
  3. Shuffle deterministically, split into train + held-out eval.
  4. Write data/train.jsonl, data/val.jsonl, data/eval.jsonl.
     (val.jsonl is a copy of eval.jsonl — mlx_lm.lora needs `valid.jsonl`
     during training and we don't have enough data to also carve a separate
     val set. We symlink valid.jsonl → val.jsonl and test.jsonl → eval.jsonl
     for the trainer.)
  5. Print a summary so the user can sanity-check before training starts.

Usage:
  python data/prepare_dataset.py
  python data/prepare_dataset.py --dry-run    # don't write files
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_raw(raw_path: Path) -> tuple[list[dict], Path]:
    """
    Prefer the augmented file (each answer is a 2-3 sentence explanation
    followed by **TechMojo policy:** + verbatim policy line) if it exists
    and is complete (=117 records). Otherwise fall back to the raw terse
    answers. Returns (records, source_path) so the caller can log which
    source was used.
    """
    augmented = raw_path.parent / "raw_augmented.jsonl"
    if augmented.exists():
        recs = []
        with open(augmented) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        if len(recs) >= 100:  # tolerate a few skipped examples
            return recs, augmented
        rprint(
            f"[yellow]⚠ {augmented} only has {len(recs)} records — "
            "falling back to raw (run augment_explanations.py to complete it)[/yellow]"
        )

    if not raw_path.exists():
        rprint(
            f"[red]✗ Raw dataset not found: {raw_path}[/red]\n"
            f"Download it first:\n"
            f"  [cyan]curl -L https://huggingface.co/datasets/PranavTM/LeavePolicy/"
            f"resolve/main/techmojo_leave_policy_finetune.jsonl "
            f"-o {raw_path}[/cyan]"
        )
        sys.exit(1)

    records: list[dict] = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records, raw_path


def to_mlx_lm_record(raw: dict, system_prompt: str) -> dict | None:
    """
    Convert a {conversations: [user, assistant]} record into the
    {messages: [system, user, assistant]} chat format mlx-lm trains on.
    Returns None if the record is malformed.
    """
    convs = raw.get("conversations") or []
    user = next((c["content"] for c in convs if c["role"] == "user"), None)
    assistant = next((c["content"] for c in convs if c["role"] == "assistant"), None)
    if not user or not assistant:
        return None

    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        # Keep raw fields too — eval_qa.py reads `question` / `answer` for
        # the held-out scoring loop instead of re-extracting from messages.
        "question": user.strip(),
        "answer": assistant.strip(),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_mlx_symlinks(data_dir: Path) -> None:
    """
    mlx_lm.lora expects {train,valid,test}.jsonl in the data directory.
    We write train.jsonl directly; symlink valid→val and test→eval so the
    trainer can find them without extra files on disk.
    """
    pairs = [("valid.jsonl", "val.jsonl"), ("test.jsonl", "eval.jsonl")]
    for link_name, target in pairs:
        link = data_dir / link_name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)


def print_summary(train: list[dict], evald: list[dict]) -> None:
    table = Table(title="TechMojo HR Dataset", header_style="bold cyan")
    table.add_column("Split")
    table.add_column("Examples", justify="right")
    table.add_row("train", str(len(train)))
    table.add_row("eval (held out)", str(len(evald)))
    table.add_row("[bold]total[/bold]", f"[bold]{len(train) + len(evald)}[/bold]")
    console.print(table)

    if train:
        sample = train[0]
        console.print(
            Panel(
                f"[bold yellow]Q:[/bold yellow] {sample['question']}\n\n"
                f"[bold green]A:[/bold green] {sample['answer']}",
                title="Sample train example",
                border_style="green",
            )
        )
    if evald:
        sample = evald[0]
        console.print(
            Panel(
                f"[bold yellow]Q:[/bold yellow] {sample['question']}\n\n"
                f"[bold green]A:[/bold green] {sample['answer']}",
                title="Sample held-out eval example",
                border_style="cyan",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TechMojo HR fine-tune data")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    cfg = load_config()
    raw_path = ROOT / cfg["data"]["raw_file"]
    train_path = ROOT / cfg["data"]["train_file"]
    val_path = ROOT / cfg["data"]["val_file"]
    eval_path = ROOT / cfg["data"]["eval_file"]
    eval_frac = float(cfg["data"]["eval_fraction"])
    seed = int(cfg["data"]["seed"])
    system_prompt = cfg["model"]["system_prompt"]

    console.rule("[bold blue]Loading TechMojo dataset[/bold blue]")
    raw, source_path = load_raw(raw_path)
    is_augmented = source_path.name == "raw_augmented.jsonl"
    flavor = "AUGMENTED (explanation + policy)" if is_augmented else "RAW (terse)"
    console.log(f"Loaded {len(raw)} {flavor} records from {source_path}")

    console.rule("[bold blue]Converting to mlx-lm chat format[/bold blue]")
    records = [r for r in (to_mlx_lm_record(r, system_prompt) for r in raw) if r]
    dropped = len(raw) - len(records)
    if dropped:
        console.log(f"[yellow]Dropped {dropped} malformed records[/yellow]")
    console.log(f"Kept {len(records)} valid records")

    console.rule("[bold blue]Splitting train / eval[/bold blue]")
    rng = random.Random(seed)
    rng.shuffle(records)
    n_eval = max(1, int(round(len(records) * eval_frac)))
    evald = records[:n_eval]
    train = records[n_eval:]
    console.log(f"train={len(train)}  eval={len(evald)}  (eval_frac={eval_frac:.0%})")

    print_summary(train, evald)

    if args.dry_run:
        rprint("\n[yellow]⚡ DRY RUN — no files written[/yellow]")
        return

    console.rule("[bold blue]Writing splits[/bold blue]")
    write_jsonl(train_path, train)
    write_jsonl(eval_path, evald)
    # mlx_lm expects valid.jsonl during training; we mirror eval into val.
    write_jsonl(val_path, evald)
    ensure_mlx_symlinks(train_path.parent)
    console.log(f"[green]✓[/green] {train_path}")
    console.log(f"[green]✓[/green] {val_path}  (= eval.jsonl, in-training loss signal)")
    console.log(f"[green]✓[/green] {eval_path}")
    console.log(f"[green]✓[/green] symlinks: data/valid.jsonl → val.jsonl, data/test.jsonl → eval.jsonl")

    rprint(
        "\n[bold]Next steps:[/bold]\n"
        "  1. [cyan]python evaluation/eval_qa.py --which base[/cyan]   "
        "— score base model on held-out eval (the 'before')\n"
        "  2. [cyan]python training/train.py --test-run[/cyan]         "
        "— 50-iter smoke test (loss must decrease)\n"
        "  3. [cyan]python training/train.py[/cyan]                    "
        "— full training\n"
        "  4. [cyan]python evaluation/eval_qa.py --which finetuned[/cyan]\n"
        "  5. [cyan]python evaluation/compare.py[/cyan]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
