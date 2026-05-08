"""
evaluation/eval_qa.py
======================
Free-form QA evaluation for the TechMojo HR fine-tune.

Why a new evaluator (vs the PubMedQA classification one): TechMojo answers are
free-text (sentences, processes, day counts), not yes/no/maybe labels. We need
two complementary metrics:

  1. char_similarity — difflib.SequenceMatcher ratio between model output and
     ground truth. Measures *how close* the wording is; an aligned model gets
     ~0.6+, the base model rambling gets ~0.1-0.3.

  2. keyword_recall — fraction of TechMojo-specific tokens (numbers, proper
     nouns: 'Freshteams', 'ADP', '20 days', '5 days', etc.) from the ground
     truth that appear in the model's response. This is the factual-recall
     score and is the most discriminative metric for memorization-style
     fine-tuning.

Usage:
  python evaluation/eval_qa.py --which base       # score base Llama 3.1 8B
  python evaluation/eval_qa.py --which finetuned  # score with adapters_techmojo
  python evaluation/eval_qa.py --which both       # both, back to back
"""

import argparse
import difflib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

ROOT = Path(__file__).parent.parent
console = Console()


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Eval data loader ──────────────────────────────────────────────────────────
def load_eval(eval_file: Path) -> list[dict]:
    if not eval_file.exists():
        rprint(
            f"[red]✗ Eval file not found: {eval_file}[/red]\n"
            "Run: [cyan]python data/prepare_dataset.py[/cyan]"
        )
        sys.exit(1)
    records = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Keyword extraction (TechMojo-specific facts) ──────────────────────────────
# Capture anything that looks like a hard fact: numbers + units, proper nouns,
# and a fixed list of TechMojo-internal tool/policy names.
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|hours?|%)\b", re.I)
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z]+){2,}\b|\b[A-Z]{2,}\b")
_TECHMOJO_TERMS = {
    "techmojo", "freshteams", "adp", "form 16", "comp off", "compensatory",
    "casual leave", "earned leave", "sick leave", "sabbatical",
    "maternity", "paternity", "flexi", "client team manager",
    "hr", "manager", "probation", "referral bonus",
}


def extract_keywords(text: str) -> set[str]:
    """Pull tokens we treat as 'facts': numbers+units, proper nouns, internal terms."""
    text_lower = text.lower()
    keywords: set[str] = set()

    for m in _NUMBER_PATTERN.findall(text):
        keywords.add(m.lower().strip())
    for m in _PROPER_NOUN.findall(text):
        keywords.add(m.lower().strip())
    for term in _TECHMOJO_TERMS:
        if term in text_lower:
            keywords.add(term)

    return {k for k in keywords if len(k) > 1}


# ── Scoring ───────────────────────────────────────────────────────────────────
def char_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio. 0=no overlap, 1=identical."""
    return difflib.SequenceMatcher(a=a.strip().lower(), b=b.strip().lower()).ratio()


def keyword_recall(prediction: str, truth: str) -> tuple[float, list[str], list[str]]:
    """
    Fraction of truth keywords also present in prediction.

    Returns (recall, hits, misses). If truth has no extractable keywords
    (rare — short generic answer), returns 1.0 (vacuously true).
    """
    truth_keys = extract_keywords(truth)
    if not truth_keys:
        return 1.0, [], []
    pred_lower = prediction.lower()
    hits = sorted(k for k in truth_keys if k in pred_lower)
    misses = sorted(truth_keys - set(hits))
    return len(hits) / len(truth_keys), hits, misses


# ── Eval loop ─────────────────────────────────────────────────────────────────
def run_eval(
    model_name: str,
    adapter_path: Path | None,
    records: list[dict],
    system_prompt: str,
    output_file: Path,
    label: str,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    """
    Note on system prompts: this function is called separately for base and
    fine-tuned, with *different* system prompts each time. The base model
    receives a generic "helpful assistant" prompt with no mention of TechMojo;
    the fine-tuned model receives the full TechMojo HR persona it was trained
    on. This is the fair OOD comparison — both models see the same user
    question, but only the fine-tune has been told what TechMojo is.
    """
    console.rule(f"[bold blue]{label}[/bold blue]")
    console.log(f"Base:    [cyan]{model_name}[/cyan]")
    if adapter_path is not None:
        console.log(f"Adapter: [cyan]{adapter_path}[/cyan]")

    t0 = time.time()
    if adapter_path is not None:
        model, tokenizer = load(model_name, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_name)
    console.log(f"[green]Loaded in {time.time() - t0:.1f}s[/green]")

    sampler = make_sampler(temp=temperature)

    sims: list[float] = []
    recalls: list[float] = []
    detail: list[dict] = []
    latencies: list[float] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Eval-QA: {label}", total=len(records))
        for record in records:
            question = record["question"]
            truth = record["answer"]

            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            t = time.time()
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                sampler=sampler,
                verbose=False,
            )
            latencies.append(time.time() - t)
            response = response.strip()

            sim = char_similarity(response, truth)
            recall, hits, misses = keyword_recall(response, truth)
            sims.append(sim)
            recalls.append(recall)

            detail.append({
                "question": question,
                "truth": truth,
                "response": response,
                "char_similarity": round(sim, 3),
                "keyword_recall": round(recall, 3),
                "keyword_hits": hits,
                "keyword_misses": misses,
            })
            progress.advance(task)

    metrics = {
        "char_similarity_mean": round(float(np.mean(sims)), 4) if sims else 0.0,
        "keyword_recall_mean": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "char_similarity_std": round(float(np.std(sims)), 4) if sims else 0.0,
        "keyword_recall_std": round(float(np.std(recalls)), 4) if recalls else 0.0,
        "n_examples": len(records),
        "avg_latency_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
    }

    result = {
        "model": model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "eval_type": "qa_freeform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "samples": detail[:20],   # keep first 20 for quick inspection
        "all_responses": detail,  # full set for compare.py
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    console.log(f"[bold green]✅ Saved → {output_file}[/bold green]")
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_metrics(metrics: dict, label: str) -> None:
    panel = Panel(
        f"[bold]Char-similarity (mean):[/bold] {metrics['char_similarity_mean']:.3f}  "
        f"(σ={metrics['char_similarity_std']:.3f})\n"
        f"[bold]Keyword recall (mean):[/bold]  "
        f"{metrics['keyword_recall_mean']:.3f}  (σ={metrics['keyword_recall_std']:.3f})\n"
        f"[bold]N:[/bold] {metrics['n_examples']}  "
        f"[bold]Avg latency:[/bold] {metrics['avg_latency_s']:.2f}s",
        title=f"[bold]{label}[/bold]",
        border_style="green",
    )
    console.print(panel)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Free-form QA eval for TechMojo HR")
    parser.add_argument("--which", choices=["base", "finetuned", "both"], default="both")
    parser.add_argument("--sample", type=int, default=None, help="Eval only N examples")
    parser.add_argument("--adapter", type=str, default=cfg["inference"]["adapter_path"])
    parser.add_argument("--model", type=str, default=cfg["model"]["name"])
    args = parser.parse_args()

    eval_file = ROOT / cfg["data"]["eval_file"]
    records = load_eval(eval_file)
    if args.sample:
        records = records[: args.sample]
    console.log(f"Held-out eval set: {len(records)} examples")

    finetuned_prompt = cfg["model"]["system_prompt"].strip()
    base_prompt = cfg["model"].get(
        "base_system_prompt",
        "You are a helpful AI assistant. Answer the user's question accurately and honestly. "
        "If the user asks about a specific company, organization, internal tool, or "
        "proprietary policy that you do not have verified information about, say so clearly "
        "and avoid inventing details.",
    ).strip()
    base_out = ROOT / cfg["evaluation"]["baseline_scores_file"]
    ft_out = ROOT / cfg["evaluation"]["finetuned_scores_file"]
    temperature = cfg["evaluation"]["temperature"]
    max_new_tokens = cfg["evaluation"]["max_new_tokens"]

    if args.which in ("base", "both"):
        m = run_eval(
            args.model, None, records, base_prompt, base_out,
            "Base Llama 3.1 8B (generic prompt, no TechMojo mention)",
            temperature, max_new_tokens,
        )
        print_metrics(m, "Base Model (generic prompt)")

    if args.which in ("finetuned", "both"):
        adapter = ROOT / args.adapter
        if not adapter.exists():
            rprint(f"[red]Adapter dir not found: {adapter}[/red]")
            sys.exit(1)
        m = run_eval(
            args.model, adapter, records, finetuned_prompt, ft_out,
            f"Fine-tuned ({args.adapter}) with TechMojo HR system prompt",
            temperature, max_new_tokens,
        )
        print_metrics(m, "Fine-tuned Model")

    if args.which == "both" and base_out.exists() and ft_out.exists():
        with open(base_out) as f:
            base = json.load(f)["metrics"]
        with open(ft_out) as f:
            ft = json.load(f)["metrics"]

        table = Table(title="Δ Improvement", header_style="bold magenta")
        table.add_column("Metric", style="bold")
        table.add_column("Base", justify="right")
        table.add_column("Fine-tuned", justify="right")
        table.add_column("Δ", justify="right")
        sims = (base["char_similarity_mean"], ft["char_similarity_mean"])
        rec = (base["keyword_recall_mean"], ft["keyword_recall_mean"])
        table.add_row(
            "Char-similarity",
            f"{sims[0]:.3f}",
            f"[green]{sims[1]:.3f}[/green]",
            f"[green]+{sims[1] - sims[0]:.3f}[/green]",
        )
        table.add_row(
            "Keyword recall",
            f"{rec[0]:.3f}",
            f"[green]{rec[1]:.3f}[/green]",
            f"[green]+{rec[1] - rec[0]:.3f}[/green]",
        )
        console.print(table)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
