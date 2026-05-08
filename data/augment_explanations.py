"""
data/augment_explanations.py
=============================
Use the base Llama 3.1 8B to generate a 2-3 sentence elaborated explanation
for every TechMojo Q&A pair, then bundle it with the original terse policy
line into a single training-ready response.

Why: the source PranavTM/LeavePolicy answers read like one-line policy
extracts. A model trained verbatim on those will produce one-line answers
forever — useful for accuracy, but not user-friendly. By rewriting each
training response as

    {explanation paragraph}

    **TechMojo policy:** {original terse answer}

the model learns to *both* explain in plain language *and* cite the exact
policy rule. The terse fact stays anchored — there's no risk of the
explanation drifting and replacing the actual rule.

Output:
    data/techmojo/raw_augmented.jsonl

Format (compatible with prepare_dataset.py — same `conversations` shape):
    {"conversations": [
        {"role": "user",      "content": "<original question>"},
        {"role": "assistant", "content": "<explanation>\n\n**TechMojo policy:** <original answer>"}
    ]}

Usage:
    python data/augment_explanations.py
    python data/augment_explanations.py --limit 5     # smoke test
    python data/augment_explanations.py --resume      # skip already-done examples
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).parent.parent
console = Console()

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
RAW_PATH = ROOT / "data/techmojo/raw.jsonl"
OUT_PATH = ROOT / "data/techmojo/raw_augmented.jsonl"

REWRITER_SYSTEM = (
    "You are an expert HR communication writer. Your job is to take a single "
    "policy Q&A pair and write a friendly, professional 2-3 sentence "
    "explanation that an employee would actually find helpful. Your output "
    "must:\n"
    "  - explain the *context* and *why* of the policy in plain English\n"
    "  - read like a human HR partner, not a legal document\n"
    "  - NOT contradict the given policy answer\n"
    "  - NOT add facts that aren't implied by the answer\n"
    "  - NOT include the literal policy line itself (that gets appended separately)\n"
    "  - be 2 to 3 complete sentences, no bullet points, no headers"
)


def build_rewriter_prompt(tokenizer, question: str, terse_answer: str) -> str:
    user_content = (
        "Here is one TechMojo HR Q&A pair. Write the 2-3 sentence "
        "elaborated explanation as instructed.\n\n"
        f"Question: {question}\n\n"
        f"Policy answer: {terse_answer}\n\n"
        "Now write the explanation:"
    )
    messages = [
        {"role": "system", "content": REWRITER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def assemble_response(explanation: str, terse_answer: str) -> str:
    expl = explanation.strip()
    # Defensively strip the model's tendency to add the policy line itself
    for marker in ("\n\n**TechMojo policy:**", "**TechMojo policy:**", "TechMojo policy:"):
        if marker in expl:
            expl = expl.split(marker, 1)[0].strip()
    return f"{expl}\n\n**TechMojo policy:** {terse_answer.strip()}"


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                user_msg = next(
                    (c["content"] for c in rec.get("conversations", []) if c["role"] == "user"),
                    None,
                )
                if user_msg:
                    seen.add(user_msg.strip())
            except json.JSONDecodeError:
                continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment TechMojo answers with explanations")
    parser.add_argument("--limit", type=int, default=None, help="Augment only N examples")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already present in raw_augmented.jsonl",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    if not RAW_PATH.exists():
        console.print(f"[red]✗ Source file missing: {RAW_PATH}[/red]")
        sys.exit(1)

    raw_records: list[dict] = []
    with open(RAW_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    console.log(f"Loaded {len(raw_records)} source records from {RAW_PATH}")

    seen = already_done(OUT_PATH) if args.resume else set()
    if seen:
        console.log(f"Resuming — {len(seen)} already done, will skip them")

    # Fresh start unless resuming
    mode = "a" if args.resume else "w"

    console.log(f"Loading [cyan]{MODEL}[/cyan] (rewriter)…")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    console.log(f"Loaded in {time.time() - t0:.1f}s")

    sampler = make_sampler(temp=args.temperature)

    # Filter & limit
    todo: list[dict] = []
    for rec in raw_records:
        convs = rec.get("conversations", [])
        user = next((c["content"] for c in convs if c["role"] == "user"), None)
        assistant = next((c["content"] for c in convs if c["role"] == "assistant"), None)
        if not user or not assistant:
            continue
        if user.strip() in seen:
            continue
        todo.append({"question": user.strip(), "answer": assistant.strip()})
    if args.limit:
        todo = todo[: args.limit]
    console.log(f"Will augment {len(todo)} records")

    written = 0
    with open(OUT_PATH, mode) as f, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Augmenting", total=len(todo))
        for ex in todo:
            prompt = build_rewriter_prompt(tokenizer, ex["question"], ex["answer"])
            try:
                explanation = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            except Exception as e:
                console.log(f"[red]Error on '{ex['question'][:60]}…': {e}[/red]")
                progress.advance(task)
                continue

            full_response = assemble_response(explanation, ex["answer"])
            out_record = {
                "conversations": [
                    {"role": "user", "content": ex["question"]},
                    {"role": "assistant", "content": full_response},
                ]
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            progress.advance(task)

    console.log(f"[bold green]✅ Wrote {written} augmented records → {OUT_PATH}[/bold green]")
    if written:
        # Show one sample so the user can sanity-check style
        with open(OUT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                convs = rec["conversations"]
                console.print("\n[bold yellow]Sample augmented record:[/bold yellow]")
                console.print(f"[bold]Q:[/bold] {convs[0]['content']}")
                console.print(f"[bold]A:[/bold]\n{convs[1]['content']}")
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(1)
