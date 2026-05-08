"""
inference/chat.py
==================
Interactive CLI for the fine-tuned TechMojo HR assistant.

Features:
  - Loads base model + LoRA adapters from adapters/ directory
  - Rich terminal UI with color-coded output
  - Streaming token generation
  - Conversation history (multi-turn)
  - Response timing

Usage:
  python inference/chat.py                              # fine-tuned model
  python inference/chat.py --base-only                  # base model only
  python inference/chat.py --adapter adapters/          # custom adapter path
  python inference/chat.py --compare                    # side-by-side comparison
"""

import sys
import time
import argparse
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import print as rprint

ROOT = Path(__file__).parent.parent
console = Console()

def _load_system_prompt() -> str:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)["model"]["system_prompt"].strip()


SYSTEM_PROMPT = _load_system_prompt()

WELCOME_MESSAGE = """
# 🏢 TechMojo HR Assistant

Llama 3.1 8B fine-tuned with QLoRA on TechMojo internal HR policies.

**Commands:**
  - Type your HR question and press Enter
  - `/compare` — toggle side-by-side comparison mode
  - `/clear`   — clear conversation history
  - `/quit`    — exit

**Disclaimer:** Demo only. Trained on the public `PranavTM/LeavePolicy` dataset.
"""


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, adapter_path: Path | None = None):
    """Load model and tokenizer, optionally with LoRA adapters."""
    try:
        from mlx_lm import load
    except ImportError:
        rprint("[red]✗ mlx-lm not installed. Run: pip install mlx-lm[/red]")
        sys.exit(1)

    if adapter_path and adapter_path.exists():
        console.log(
            f"[cyan]Loading[/cyan] {model_name} + adapters from {adapter_path}…"
        )
        model, tokenizer = load(str(model_name), adapter_path=str(adapter_path))
        label = "🏢 TechMojo HR (Fine-tuned)"
    else:
        if adapter_path:
            rprint(
                f"[yellow]⚠ Adapter path not found: {adapter_path}[/yellow]\n"
                "Falling back to base model."
            )
        console.log(f"[cyan]Loading[/cyan] base model {model_name}…")
        model, tokenizer = load(str(model_name))
        label = "🤖 Base Llama 3.1 8B"

    return model, tokenizer, label


def generate_response(
    model,
    tokenizer,
    conversation: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, float]:
    """Generate a response and return (text, latency_seconds)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    t_start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        sampler=make_sampler(temp=temperature, top_p=top_p),
        verbose=False,
    )
    latency = time.time() - t_start
    return response, latency


def chat_loop(
    model,
    tokenizer,
    model_label: str,
    base_model=None,
    base_tokenizer=None,
    base_label: str = "Base Model",
    cfg: dict | None = None,
) -> None:
    """Main interactive chat loop."""
    cfg = cfg or {}
    max_new_tokens = cfg.get("inference", {}).get("max_new_tokens", 512)
    temperature = cfg.get("inference", {}).get("temperature", 0.7)

    console.print(Markdown(WELCOME_MESSAGE))

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    compare_mode = base_model is not None

    if compare_mode:
        base_history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        rprint(f"[bold]Mode:[/bold] Side-by-side comparison — {model_label} vs {base_label}")
    else:
        rprint(f"[bold]Model:[/bold] {model_label}")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            rprint("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.strip().lower() == "/quit":
            rprint("[yellow]Goodbye![/yellow]")
            break
        elif user_input.strip().lower() == "/clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            if compare_mode:
                base_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            rprint("[dim]Conversation cleared.[/dim]")
            continue
        elif user_input.strip().lower() == "/compare":
            if base_model is None:
                rprint("[yellow]Start with --compare flag to enable side-by-side mode.[/yellow]")
            else:
                compare_mode = not compare_mode
                rprint(f"[dim]Compare mode: {'ON' if compare_mode else 'OFF'}[/dim]")
            continue

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Generate fine-tuned response
        console.print(f"\n[bold green]{model_label}[/bold green]")
        with console.status("[dim]Thinking…[/dim]"):
            response, latency = generate_response(
                model,
                tokenizer,
                history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        console.print(
            Panel(
                Markdown(response),
                border_style="green",
                subtitle=f"[dim]{latency:.2f}s[/dim]",
            )
        )
        history.append({"role": "assistant", "content": response})

        # Side-by-side comparison with base model
        if compare_mode and base_model is not None:
            base_history.append({"role": "user", "content": user_input})

            console.print(f"\n[bold yellow]{base_label}[/bold yellow]")
            with console.status("[dim]Generating base model response…[/dim]"):
                base_response, base_latency = generate_response(
                    base_model,
                    base_tokenizer,
                    base_history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            console.print(
                Panel(
                    Markdown(base_response),
                    border_style="yellow",
                    subtitle=f"[dim]{base_latency:.2f}s[/dim]",
                )
            )
            base_history.append({"role": "assistant", "content": base_response})


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="TechMojo HR assistant CLI")
    parser.add_argument(
        "--adapter",
        type=str,
        default=cfg["training"]["output_dir"],
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Use base model without LoRA adapters",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Load both base and fine-tuned models for side-by-side comparison",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg["model"]["name"],
        help="Base model HuggingFace ID",
    )
    args = parser.parse_args()

    adapter_path = None if args.base_only else ROOT / args.adapter

    model, tokenizer, label = load_model(args.model, adapter_path=adapter_path)

    base_model = base_tokenizer = None
    base_label = "Base Llama 3.1 8B"

    if args.compare and not args.base_only:
        rprint("[dim]Loading base model for comparison…[/dim]")
        base_model, base_tokenizer, base_label = load_model(args.model, adapter_path=None)

    chat_loop(
        model=model,
        tokenizer=tokenizer,
        model_label=label,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        base_label=base_label,
        cfg=cfg,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
