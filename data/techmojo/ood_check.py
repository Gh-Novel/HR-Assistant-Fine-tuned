"""
data/techmojo/ood_check.py
==========================
Quick OOD check: ask the base Llama 3.1 8B 6 specific TechMojo handbook questions
and print its answers alongside ground truth. If the base model is genuinely
out-of-distribution on this data, we'll see hallucination/generic-HR-boilerplate
answers vs the specific TechMojo facts.
"""

import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent.parent.parent
console = Console()

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
N_PROBES = 6


def main() -> None:
    raw_path = ROOT / "data/techmojo/raw.jsonl"
    with open(raw_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    random.seed(7)
    probes = random.sample(examples, N_PROBES)

    console.rule(f"[bold blue]Loading base model[/bold blue]")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    console.log(f"Loaded in {time.time() - t0:.1f}s")

    sampler = make_sampler(temp=0.2)

    for i, ex in enumerate(probes, 1):
        convs = ex["conversations"]
        question = next(c["content"] for c in convs if c["role"] == "user")
        truth = next(c["content"] for c in convs if c["role"] == "assistant")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an HR assistant at TechMojo. Answer specifically and concisely. "
                    "If you don't know a TechMojo-specific policy, say so explicitly."
                ),
            },
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=120,
            sampler=sampler,
            verbose=False,
        )
        latency = time.time() - t0

        console.print(
            Panel(
                f"[bold yellow]Q ({i}/{N_PROBES}):[/bold yellow] {question}\n\n"
                f"[bold green]Ground truth (TechMojo):[/bold green] {truth}\n\n"
                f"[bold red]Base Llama 3.1 8B response[/bold red] ({latency:.1f}s):\n{response.strip()}",
                border_style="cyan",
            )
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(1)
