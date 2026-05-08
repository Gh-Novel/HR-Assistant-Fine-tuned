"""
export/push_to_hub.py
======================
Upload MedQLoRA LoRA adapter weights to HuggingFace Hub.

Only uploads the adapter weights (~150MB), not the full 16GB model.
The adapter is shared as a standalone model card that references
the base model — users download it with mlx-lm and the base model.

Prerequisites:
  - Set HF_TOKEN in .env (or run: huggingface-cli login)
  - Training must be complete (adapters/ directory must exist)
  - eval_finetuned.py must have been run (for model card metrics)

Usage:
  python export/push_to_hub.py                         # push to default repo
  python export/push_to_hub.py --repo myuser/medqlora  # custom repo name
  python export/push_to_hub.py --dry-run               # preview only
"""

import json
import os
import sys
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

ROOT = Path(__file__).parent.parent
console = Console()
load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_metrics() -> dict | None:
    """
    Load baseline + finetuned scores for the model card.

    Prefers the logit-based MC results (cleaner methodology) and falls back to
    generation-based results if logit scores aren't available yet.
    """
    cfg = load_config()
    eval_dir = ROOT / "evaluation"
    logit_base = eval_dir / "baseline_scores_logit.json"
    logit_ft = eval_dir / "finetuned_scores_logit.json"
    gen_base = ROOT / cfg["evaluation"]["baseline_scores_file"]
    gen_ft = ROOT / cfg["evaluation"]["finetuned_scores_file"]

    if logit_base.exists() and logit_ft.exists():
        baseline_file, finetuned_file, method = logit_base, logit_ft, "logit-based MC"
    elif gen_base.exists() and gen_ft.exists():
        baseline_file, finetuned_file, method = gen_base, gen_ft, "generation + regex"
    else:
        return None

    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(finetuned_file) as f:
        finetuned = json.load(f)

    bm = baseline["metrics"]
    fm = finetuned["metrics"]

    return {
        "method": method,
        "baseline_accuracy": bm["accuracy"],
        "finetuned_accuracy": fm["accuracy"],
        "baseline_f1": bm["macro_f1"],
        "finetuned_f1": fm["macro_f1"],
        "per_class": {
            cls: {
                "base_f1": bm["per_class"][cls]["f1"],
                "ft_f1": fm["per_class"][cls]["f1"],
            }
            for cls in ["yes", "no", "maybe"]
        },
    }


def generate_model_card(
    repo_id: str,
    cfg: dict,
    metrics: dict | None,
) -> str:
    """Generate a HuggingFace model card (README.md) for the adapter."""
    base_model = cfg["model"]["name"]
    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]

    metrics_section = ""
    if metrics:
        b_acc = metrics["baseline_accuracy"]
        f_acc = metrics["finetuned_accuracy"]
        diff = f_acc - b_acc
        b_f1 = metrics["baseline_f1"]
        f_f1 = metrics["finetuned_f1"]
        method = metrics.get("method", "logit-based MC")
        sign = "+" if diff >= 0 else ""

        per_class_rows = []
        per_class_notes = {
            "yes": "Already strong; fine-tuning roughly neutral here.",
            "no": "Largest real gain — model commits to 'no' more confidently.",
            "maybe": "Regression: training data has zero uncertain labels, so fine-tuning suppresses 'maybe'.",
        }
        for cls in ["yes", "no", "maybe"]:
            pc = metrics["per_class"][cls]
            d = pc["ft_f1"] - pc["base_f1"]
            s = "+" if d >= 0 else ""
            per_class_rows.append(
                f"| {cls.capitalize()} | {pc['base_f1']:.3f} | {pc['ft_f1']:.3f} | {s}{d:.3f} | {per_class_notes[cls]} |"
            )

        metrics_section = f"""
## Evaluation Results

Evaluated on **PubMedQA** (`pqa_labeled`, yes/no/maybe classification, {method}):

| Metric | Base Model | MedQLoRA | Δ |
|--------|-----------|----------|---|
| Accuracy | {b_acc:.1%} | **{f_acc:.1%}** | {sign}{diff * 100:.1f}pp |
| Macro F1 | {b_f1:.3f} | {f_f1:.3f} | {'+' if f_f1 >= b_f1 else ''}{f_f1 - b_f1:.3f} |

### Per-class F1 (with what fine-tuning actually did)

| Class | Base | MedQLoRA | Δ | Note |
|---|---|---|---|---|
{chr(10).join(per_class_rows)}

The headline accuracy gain comes almost entirely from improved 'no' classification.
The 'maybe' regression is a documented limitation of training on definitive medical Q&A
data that contains no uncertain labels.
"""

    return f"""---
base_model: {base_model}
library_name: mlx
tags:
  - medical
  - llama
  - lora
  - qlora
  - mlx
  - fine-tuned
  - pubmedqa
  - medical-qa
datasets:
  - medalpaca/medical_meadow_medqa
  - medalpaca/medical_meadow_wikidoc
license: llama3
---

# MedQLoRA — Medical Q&A Fine-tuned Llama 3.1 8B

LoRA adapter weights for Llama 3.1 8B fine-tuned on medical Q&A datasets using QLoRA via Apple MLX.

**Adapter size:** ~150MB (vs 16GB for the full model)
**Base model:** [`{base_model}`](https://huggingface.co/{base_model})

{metrics_section}

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `{base_model}` |
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_alpha} |
| Trainable Parameters | ~0.8% of total |
| Epochs | {epochs} |
| Learning Rate | {lr} |
| Framework | Apple MLX |
| Hardware | Mac Mini, Apple Silicon 24GB |

## Training Data

- **medalpaca/medical_meadow_medqa** — 10K medical Q&A pairs (NIH-curated)
- **medalpaca/medical_meadow_wikidoc** — 10K clinical explanation pairs

## Usage

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "{base_model}",
    adapter_path="{repo_id}"  # HuggingFace repo ID
)

messages = [
    {{"role": "system", "content": "You are an expert medical AI assistant."}},
    {{"role": "user", "content": "What are the symptoms of Type 2 diabetes?"}}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

## Disclaimer

This model is for research purposes only. It is not a substitute for professional medical advice.
Always consult a qualified healthcare provider for medical decisions.

## License

Llama 3.1 is licensed under the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).
"""


def push_to_hub(
    adapter_path: Path,
    repo_id: str,
    cfg: dict,
    metrics: dict | None,
    dry_run: bool = False,
) -> None:
    """Upload adapter weights and model card to HuggingFace Hub."""
    if not adapter_path.exists():
        rprint(
            f"[red]✗ Adapter directory not found: {adapter_path}[/red]\n"
            "Run training first: [cyan]python training/train.py[/cyan]"
        )
        sys.exit(1)

    # List files to upload
    adapter_files = list(adapter_path.glob("*.safetensors")) + list(adapter_path.glob("*.json"))
    if not adapter_files:
        rprint(f"[red]✗ No adapter files found in {adapter_path}[/red]")
        sys.exit(1)

    console.log(f"\n[bold]Repository:[/bold] {repo_id}")
    console.log(f"[bold]Files to upload:[/bold]")
    total_size = 0
    for f in adapter_files:
        size_mb = f.stat().st_size / 1_048_576
        total_size += size_mb
        console.log(f"  {f.name} ({size_mb:.1f} MB)")
    console.log(f"\n[bold]Total upload size:[/bold] {total_size:.1f} MB\n")

    if dry_run:
        # Render the model card so the user can preview what will go up
        model_card = generate_model_card(repo_id, cfg, metrics)
        preview_path = adapter_path / "_README_preview.md"
        preview_path.write_text(model_card)
        rprint(
            "[yellow]⚡ DRY RUN — no files uploaded[/yellow]\n"
            f"Model card preview written to: [cyan]{preview_path}[/cyan]"
        )
        return

    from huggingface_hub import HfApi, create_repo  # type: ignore

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        rprint(
            "[red]✗ HF_TOKEN not set.[/red]\n"
            "Set it in .env or run: [cyan]huggingface-cli login[/cyan]"
        )
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    console.log(f"Creating repo: {repo_id}")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    # Generate and upload model card
    model_card = generate_model_card(repo_id, cfg, metrics)
    model_card_path = adapter_path / "_README.md"
    model_card_path.write_text(model_card)

    api.upload_file(
        path_or_fileobj=str(model_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )
    console.log("[green]✓[/green] Model card uploaded")

    # Upload adapter files
    for f in adapter_files:
        console.log(f"  Uploading {f.name}…")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            token=token,
        )
        console.log(f"  [green]✓[/green] {f.name}")

    # Clean up temp file
    model_card_path.unlink()

    console.print(
        Panel(
            f"[bold green]Adapter weights uploaded successfully![/bold green]\n\n"
            f"View at: [cyan]https://huggingface.co/{repo_id}[/cyan]\n\n"
            f"Users can load with:\n"
            f"  [dim]from mlx_lm import load\n"
            f'  model, tokenizer = load("{cfg["model"]["name"]}", adapter_path="{repo_id}")[/dim]',
            title="✅ HuggingFace Hub Upload Complete",
            border_style="green",
        )
    )


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Push MedQLoRA adapter weights to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=cfg["export"]["hub_model_id"],
        help="HuggingFace repo ID (e.g. username/medqlora-llama-3.1-8b)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=cfg["inference"]["adapter_path"],
        help="Local adapter directory to upload (default: best-checkpoint dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview upload without pushing",
    )
    args = parser.parse_args()

    if "your-username" in args.repo or "your-actual-username" in args.repo:
        rprint(
            "[red]✗ Please set your actual HuggingFace username in config.yaml:[/red]\n"
            "  export.hub_model_id: '<your-username>/medqlora-llama-3.1-8b'\n"
            "OR use: [cyan]python export/push_to_hub.py --repo <username>/medqlora[/cyan]"
        )
        sys.exit(1)

    metrics = load_metrics()
    if not metrics:
        rprint("[yellow]⚠ Evaluation scores not found — model card will not include benchmark table.[/yellow]")

    push_to_hub(
        adapter_path=ROOT / args.adapter,
        repo_id=args.repo,
        cfg=cfg,
        metrics=metrics,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
