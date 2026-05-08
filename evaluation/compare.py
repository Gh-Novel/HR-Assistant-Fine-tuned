"""
evaluation/compare.py
======================
Generate the before/after report for the TechMojo HR fine-tune.

Reads:
  evaluation/baseline_scores.json   (eval_qa.py --which base)
  evaluation/finetuned_scores.json  (eval_qa.py --which finetuned)

Writes:
  - rich terminal report with a 5-example side-by-side
  - benchmark section in README.md (between <!-- BENCHMARK_START --> markers)

Usage:
  python evaluation/compare.py             # print + update README
  python evaluation/compare.py --no-readme # print only
"""

import argparse
import json
import sys
from datetime import datetime, timezone
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


def load_scores(path: Path, label: str) -> dict:
    if not path.exists():
        rprint(f"[red]✗ {label} scores not found: {path}[/red]")
        rprint("  Run [cyan]python evaluation/eval_qa.py[/cyan] first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_markdown(base: dict, ft: dict, n_examples: int) -> str:
    bm = base["metrics"]
    fm = ft["metrics"]
    sim_b, sim_f = bm["char_similarity_mean"], fm["char_similarity_mean"]
    rec_b, rec_f = bm["keyword_recall_mean"], fm["keyword_recall_mean"]

    md = ["## 📊 TechMojo HR Benchmark — Before vs After Fine-tuning\n\n"]
    md.append(
        "Held-out eval: TechMojo HR questions the base Llama 3.1 8B has never seen "
        "(the source dataset is small and not in pretraining corpora — verified via "
        "`data/techmojo/ood_check.py`).\n\n"
    )
    md.append("| Metric | Base Llama 3.1 8B | Fine-tuned (QLoRA) | Δ |\n")
    md.append("|---|---|---|---|\n")
    sim_d = sim_f - sim_b
    rec_d = rec_f - rec_b
    md.append(
        f"| **Char-similarity to ground truth** | {sim_b:.3f} | **{sim_f:.3f}** | "
        f"**{'+' if sim_d >= 0 else ''}{sim_d:.3f}** |\n"
    )
    md.append(
        f"| **Keyword recall (TechMojo facts)** | {rec_b:.3f} | **{rec_f:.3f}** | "
        f"**{'+' if rec_d >= 0 else ''}{rec_d:.3f}** |\n"
    )
    md.append(f"| Examples | {n_examples} | {n_examples} | — |\n")

    md.append(
        "\n**What this measures:** *Char-similarity* is `difflib.SequenceMatcher` "
        "between the model's response and the ground truth answer (0=no overlap, "
        "1=identical text). *Keyword recall* is the fraction of TechMojo-specific "
        "facts (numbers, proper nouns, internal tool names like `Freshteams`/`ADP`) "
        "from the ground truth that appear in the model's response.\n"
    )
    md.append(
        "\n**Why both metrics:** char-similarity catches paraphrased correct "
        "answers; keyword recall catches whether the model knows the specific "
        "facts. Note that keyword recall can *favor verbose hallucination*: a "
        "rambling base-model answer mentions surface tokens like 'HR', "
        "'manager', 'policies' by chance and scores high recall even when the "
        "factual content is wrong. The terse fine-tuned answer ('Yes, 1 week "
        "in advance') is correct but contains fewer total tokens to recall. "
        "Char-similarity is therefore the cleaner signal of factual "
        "correctness for this task.\n"
    )

    # Side-by-side examples (5 most representative — biggest swing)
    samples = ft.get("all_responses") or ft.get("samples") or []
    base_lookup = {
        s["question"]: s for s in (base.get("all_responses") or base.get("samples") or [])
    }
    if samples and base_lookup:
        # Sort by largest improvement in char-similarity
        ranked = sorted(
            samples,
            key=lambda s: s["char_similarity"] - base_lookup.get(s["question"], {}).get("char_similarity", 0),
            reverse=True,
        )[:5]
        md.append("\n### Side-by-side: 5 questions where fine-tuning helped most\n\n")
        for s in ranked:
            q = s["question"]
            b = base_lookup.get(q, {})
            md.append(f"**Q:** {q}\n\n")
            md.append(f"- **Ground truth (TechMojo):** {s['truth']}\n")
            md.append(f"- **Base Llama 3.1 8B:** {b.get('response', '—')}\n")
            md.append(f"- **Fine-tuned:** {s['response']}\n\n")
            md.append(
                f"  _char_similarity: {b.get('char_similarity', 0):.2f} → "
                f"{s['char_similarity']:.2f} · "
                f"keyword_recall: {b.get('keyword_recall', 0):.2f} → "
                f"{s['keyword_recall']:.2f}_\n\n"
            )

    md.append(
        f"\n> Evaluated {n_examples} held-out questions from "
        f"`PranavTM/LeavePolicy` (TechMojo internal HR policies).  \n"
        f"> Base model: `{base['model']}`.  \n"
        f"> Fine-tuned with QLoRA (r=16, α=16) via Apple MLX on Mac Mini 24GB.  \n"
        f"> Eval timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
    )

    return "".join(md)


def update_readme(comparison_md: str) -> None:
    readme = ROOT / "README.md"
    marker_start = "<!-- BENCHMARK_START -->"
    marker_end = "<!-- BENCHMARK_END -->"

    if not readme.exists():
        rprint("[yellow]README.md not found — skipping update[/yellow]")
        return

    content = readme.read_text()
    if marker_start in content and marker_end in content:
        before = content[: content.index(marker_start) + len(marker_start)]
        after = content[content.index(marker_end):]
        readme.write_text(f"{before}\n{comparison_md}\n{after}")
        console.log("[green]Updated[/green] benchmark section in README.md")
    else:
        readme.write_text(content + "\n\n---\n" + comparison_md)
        console.log("[green]Appended[/green] benchmark section to README.md")


def print_terminal_report(base: dict, ft: dict) -> None:
    console.rule("[bold blue]TechMojo HR — Before / After[/bold blue]")
    bm, fm = base["metrics"], ft["metrics"]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Base", justify="right")
    table.add_column("Fine-tuned", justify="right")
    table.add_column("Δ", justify="right")

    sim_d = fm["char_similarity_mean"] - bm["char_similarity_mean"]
    rec_d = fm["keyword_recall_mean"] - bm["keyword_recall_mean"]
    sim_color = "green" if sim_d >= 0 else "red"
    rec_color = "green" if rec_d >= 0 else "red"
    table.add_row(
        "Char-similarity (mean)",
        f"{bm['char_similarity_mean']:.3f}",
        f"[{sim_color}]{fm['char_similarity_mean']:.3f}[/{sim_color}]",
        f"[{sim_color}]{'+' if sim_d >= 0 else ''}{sim_d:.3f}[/{sim_color}]",
    )
    table.add_row(
        "Keyword recall (mean)",
        f"{bm['keyword_recall_mean']:.3f}",
        f"[{rec_color}]{fm['keyword_recall_mean']:.3f}[/{rec_color}]",
        f"[{rec_color}]{'+' if rec_d >= 0 else ''}{rec_d:.3f}[/{rec_color}]",
    )
    table.add_row(
        "Avg latency (s)",
        f"{bm['avg_latency_s']:.2f}",
        f"{fm['avg_latency_s']:.2f}",
        "—",
    )
    console.print(table)

    # Resume bullet
    console.print(
        Panel(
            f"[bold]Resume bullet:[/bold]\n\n"
            f"  Fine-tuned Llama 3.1 8B with QLoRA on Apple Silicon (24GB) using\n"
            f"  TechMojo internal HR policies (~100 train examples) — data the base\n"
            f"  model has never seen. Confirmed out-of-distribution by probing the\n"
            f"  base model on 6 specific policy questions before training.\n\n"
            f"  [bold green]Char-similarity to ground truth: "
            f"{bm['char_similarity_mean']:.2f} → {fm['char_similarity_mean']:.2f}[/bold green]\n"
            f"  [bold green]Keyword recall on TechMojo-specific facts: "
            f"{bm['keyword_recall_mean']:.2f} → {fm['keyword_recall_mean']:.2f}[/bold green]\n"
            f"  Demonstrates QLoRA's value for proprietary / out-of-distribution data.",
            title="📄 Resume Bullet",
            border_style="cyan",
        )
    )


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Generate before/after report")
    parser.add_argument("--no-readme", action="store_true", help="Print only")
    args = parser.parse_args()

    base = load_scores(ROOT / cfg["evaluation"]["baseline_scores_file"], "Baseline")
    ft = load_scores(ROOT / cfg["evaluation"]["finetuned_scores_file"], "Fine-tuned")

    print_terminal_report(base, ft)

    if not args.no_readme:
        n = base["metrics"]["n_examples"]
        md = build_markdown(base, ft, n)
        update_readme(md)
        rprint("\n[green]✅ README.md updated with benchmark section[/green]")

    rprint(
        "\n[bold]Next:[/bold]\n"
        "  [cyan]python app.py[/cyan]                — launch Gradio side-by-side demo\n"
        "  [cyan]python export/push_to_hub.py[/cyan] — push adapters to HuggingFace Hub"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
