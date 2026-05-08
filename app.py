"""
app.py
=======
Gradio demo for TechMojo HR Assistant — side-by-side base vs fine-tuned chat.

Three tabs:
  1. HR Q&A            — Base vs Fine-tuned side-by-side chat
  2. Benchmark Results — Before/after metrics on held-out TechMojo questions
  3. About             — Training pipeline + OOD justification

Usage:
  python app.py                  # local demo (localhost:7860)
  python app.py --share          # public Gradio share link
  python app.py --base-only      # load base model only (no adapters)
"""

import json
import sys
import time
import argparse
from pathlib import Path

import yaml
import gradio as gr

ROOT = Path(__file__).parent


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


cfg = load_config()

# ── MLX availability check ────────────────────────────────────────────────────
# `mlx-lm` only runs on Apple Silicon. On Linux (e.g. the HF Spaces runtime),
# `import mlx.core` fails with `OSError: libmlx.so: cannot open shared object
# file`. We probe at import time so the UI can boot even when the inference
# backend is missing — the chat handler then shows a clear notice instead of
# a 500 traceback.
try:
    import mlx.core  # noqa: F401
    MLX_AVAILABLE = True
    MLX_UNAVAILABLE_REASON = ""
except (ImportError, OSError) as _mlx_err:
    MLX_AVAILABLE = False
    MLX_UNAVAILABLE_REASON = str(_mlx_err) or "MLX backend not available on this host"
    print(f"[app] MLX backend unavailable: {MLX_UNAVAILABLE_REASON}")
    print("[app] Falling back to a non-MLX inference backend (Transformers) if possible.")


# ── Transformers (Linux/Spaces) fallback ───────────────────────────────────────
TORCH_AVAILABLE = False
TORCH_UNAVAILABLE_REASON = ""
try:
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    TORCH_AVAILABLE = True
except Exception as _torch_err:  # noqa: BLE001
    TORCH_AVAILABLE = False
    TORCH_UNAVAILABLE_REASON = str(_torch_err) or "torch/transformers not available"

_torch_models = {}


def _get_torch_models(base_only: bool = False):
    """
    Load models via Transformers for Linux (HF Spaces) runtime.

    Notes:
    - This loads the base model from Hugging Face Hub.
    - Your MLX adapter files are not guaranteed to be directly compatible with
      Transformers/PEFT. If no compatible adapter is available, both panels
      will use the base model.
    """
    if _torch_models:
        return _torch_models

    if not TORCH_AVAILABLE:
        raise RuntimeError(TORCH_UNAVAILABLE_REASON)

    model_name = cfg["model"]["name"]

    # Best-effort: GPU if available, else CPU (8B on CPU may be very slow).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[app] (Transformers) Loading model on {device} …")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        model = model.to(device)

    _torch_models["base"] = (model, tok, device)
    _torch_models["finetuned"] = _torch_models["base"]
    return _torch_models


def _infer_torch(model_key: str, question: str, history: list, max_tokens: int, temperature: float) -> tuple[str, float]:
    import torch

    models = _get_torch_models()
    model, tok, device = models[model_key]

    system_prompt = (
        FINETUNED_SYSTEM_PROMPT if model_key == "finetuned" else BASE_SYSTEM_PROMPT
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    latency = time.time() - t0

    text = tok.decode(out[0], skip_special_tokens=True)
    # Return only newly generated part (best-effort).
    answer = text[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)) :].strip()
    return answer or text.strip(), latency

FINETUNED_SYSTEM_PROMPT = cfg["model"]["system_prompt"].strip()

# Base model gets a deliberately generic prompt — no mention of TechMojo.
# This is the fair OOD comparison: both models see the same user question,
# but only the fine-tuned model has been told who TechMojo is.
BASE_SYSTEM_PROMPT = cfg["model"].get(
    "base_system_prompt",
    "You are a helpful AI assistant. Answer the user's question accurately and honestly. "
    "If the user asks about a specific company, organization, internal tool, or "
    "proprietary policy that you do not have verified information about, say so clearly "
    "and avoid inventing details.",
).strip()

DISCLAIMER = (
    "Demo project — Llama 3.1 8B fine-tuned on the public `PranavTM/LeavePolicy` "
    "dataset (TechMojo internal HR policies). Both panels receive the same user "
    "question; only the fine-tuned panel is told what TechMojo is."
)

# Example questions are taken from the *training* set (not held-out eval), so
# the fine-tuned model has actually been shown the answer and can produce it
# verbatim. The held-out eval questions are reserved for benchmarking, not the
# live demo.
EXAMPLE_QUESTIONS = [
    "Where should employees apply for their leaves?",
    "Will I get my appraisal while on maternity leave?",
    "How do I report a technical issue with my system?",
    "How much paid maternity leave can I take?",
    "Are Flexi leaves included in my annual leave quota?",
    "Can I encash my leftover leaves at the end of the year?",
    "When does the leave year start and end?",
    "Will unused leaves accumulate year after year?",
]

# ── Model loader ──────────────────────────────────────────────────────────────
_models = {}


def get_models(base_only: bool = False):
    """Lazy-load models on first call."""
    if _models:
        return _models

    try:
        from mlx_lm import load
    except ImportError:
        print("ERROR: pip install mlx-lm")
        sys.exit(1)

    model_name = cfg["model"]["name"]
    # Use the curated best-checkpoint dir from inference config, not the
    # active training output dir which holds the latest (not necessarily best) weights.
    adapter_path = ROOT / cfg["inference"]["adapter_path"]

    print(f"Loading base model: {model_name}")
    base_model, base_tok = load(str(model_name))
    _models["base"] = (base_model, base_tok)
    print("  ✓ Base model ready")

    if not base_only and adapter_path.exists():
        print(f"Loading fine-tuned model (adapters: {adapter_path})")
        ft_model, ft_tok = load(str(model_name), adapter_path=str(adapter_path))
        _models["finetuned"] = (ft_model, ft_tok)
        print("  ✓ Fine-tuned model ready")
    else:
        _models["finetuned"] = _models["base"]
        if not base_only:
            print(f"  ⚠ Adapter path {adapter_path} not found — using base model for both")

    return _models


# ── Inference helper ──────────────────────────────────────────────────────────
def infer(model_key: str, question: str, history: list, max_tokens: int, temperature: float) -> tuple[str, float]:
    """
    `history` is a list of OpenAI-style message dicts: {"role": ..., "content": ...}
    (Gradio 6.x Chatbot format.)

    Threading note: this function MUST run on the same thread the model was loaded
    on, otherwise mx.eval raises "There is no Stream(gpu, 1) in current thread."
    The MLX prompt_cache holds stream state in a thread-local registry. To enforce
    that, the Gradio handler is async — Gradio awaits async handlers on the event
    loop thread (main thread = where get_models() was called), bypassing the
    anyio.to_thread worker pool that breaks MLX.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    models = get_models()
    model, tokenizer = models[model_key]

    # Different system prompts per model: the fine-tuned model gets the full
    # TechMojo HR persona it was trained with; the base model gets a generic
    # assistant prompt with NO mention of TechMojo. This is the fair OOD
    # comparison: same user question, only the fine-tune knows the company.
    system_prompt = (
        FINETUNED_SYSTEM_PROMPT if model_key == "finetuned" else BASE_SYSTEM_PROMPT
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
    )
    latency = time.time() - t0
    return response.strip(), latency


# ── Gradio event handlers ─────────────────────────────────────────────────────
async def respond(
    question: str,
    ft_history: list,
    base_history: list,
    max_tokens: int,
    temperature: float,
):
    """
    Handle user question → generate both model responses.

    Async on purpose: Gradio runs sync handlers via anyio.to_thread (worker pool),
    but MLX prompt_cache stream state is thread-local — calling generate() from a
    worker thread when the model was loaded on the main thread raises
    "There is no Stream(gpu, 1) in current thread." Async handlers are awaited on
    the event loop's main thread, matching where get_models() ran at startup.
    The blocking mx.eval inside generate() will pause the event loop during
    inference, which is acceptable for this single-user demo.
    """
    if not question.strip():
        return ft_history, base_history, "", ""

    if not MLX_AVAILABLE:
        # HF Spaces / Linux: try Transformers fallback.
        try:
            ft_answer, ft_latency = _infer_torch("finetuned", question, ft_history, max_tokens, temperature)
            base_answer, base_latency = _infer_torch("base", question, base_history, max_tokens, temperature)
        except Exception as e:  # noqa: BLE001
            notice = (
                "⚠️ **Live inference couldn't start on this Space.**\n\n"
                "This app can run either:\n"
                "- **Apple MLX** (macOS + Apple Silicon), or\n"
                "- **Transformers** (recommended: GPU Space).\n\n"
                f"Error: `{str(e)[:400]}`\n\n"
                "**Fix:** switch the Space hardware to a GPU (T4/A10G), or run locally on a Mac."
            )
            ft_history = ft_history + [{"role": "user", "content": question}, {"role": "assistant", "content": notice}]
            base_history = base_history + [{"role": "user", "content": question}, {"role": "assistant", "content": notice}]
            return ft_history, base_history, "", "No inference backend available"

        ft_history = ft_history + [{"role": "user", "content": question}, {"role": "assistant", "content": ft_answer}]
        base_history = base_history + [{"role": "user", "content": question}, {"role": "assistant", "content": base_answer}]
        status = f"✅ (Transformers) Fine-tuned: {ft_latency:.2f}s | Base: {base_latency:.2f}s"
        return ft_history, base_history, "", status

    # Fine-tuned response
    ft_answer, ft_latency = infer("finetuned", question, ft_history, max_tokens, temperature)

    # Base model response
    base_answer, base_latency = infer("base", question, base_history, max_tokens, temperature)

    ft_history = ft_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ft_answer},
    ]
    base_history = base_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": base_answer},
    ]

    status = (
        f"✅ Fine-tuned: {ft_latency:.2f}s | Base: {base_latency:.2f}s"
    )

    return ft_history, base_history, "", status


def clear_history():
    return [], [], "", ""


def use_example(example: str) -> str:
    return example


# ── Load benchmark results ────────────────────────────────────────────────────
def load_benchmark_html() -> str:
    base_path = ROOT / cfg["evaluation"]["baseline_scores_file"]
    ft_path = ROOT / cfg["evaluation"]["finetuned_scores_file"]

    if not base_path.exists():
        return """
        <div style='padding: 20px; background: #1a1a2e; border-radius: 12px; color: #e0e0e0;'>
            <h3>⏳ Baseline scores not yet available</h3>
            <p>Run: <code>python evaluation/eval_qa.py --which base</code> before training,
            then <code>--which finetuned</code> after training.</p>
        </div>
        """

    with open(base_path) as f:
        base = json.load(f)
    b = base["metrics"]

    if not ft_path.exists():
        return f"""
        <div style='font-family: Inter, sans-serif; padding: 24px; background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); border-radius: 16px; color: #e2e8f0;'>
            <h2 style='color: #818cf8; margin-top: 0;'>📊 Baseline Results (training not done yet)</h2>
            <p><strong>Char-similarity:</strong>
                <span style='color: #fbbf24; font-size: 1.2em;'>{b['char_similarity_mean']:.3f}</span>
                · <strong>Keyword recall:</strong>
                <span style='color: #fbbf24; font-size: 1.2em;'>{b['keyword_recall_mean']:.3f}</span>
            </p>
            <p style='color: #94a3b8;'>Fine-tuned results will appear here after
                <code>python training/train.py</code> +
                <code>python evaluation/eval_qa.py --which finetuned</code>
            </p>
        </div>
        """

    with open(ft_path) as f:
        ft = json.load(f)
    fm = ft["metrics"]
    sim_delta = fm["char_similarity_mean"] - b["char_similarity_mean"]
    rec_delta = fm["keyword_recall_mean"] - b["keyword_recall_mean"]

    return f"""
    <div style='font-family: Inter, sans-serif; padding: 24px; background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); border-radius: 16px; color: #e2e8f0;'>
        <h2 style='color: #818cf8; margin-top: 0;'>📊 TechMojo HR Benchmark</h2>
        <table style='width: 100%; border-collapse: collapse; margin-top: 16px;'>
            <thead>
                <tr style='background: rgba(129, 140, 248, 0.15);'>
                    <th style='padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1);'>Metric</th>
                    <th style='padding: 12px 16px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);'>Base Llama 3.1 8B</th>
                    <th style='padding: 12px 16px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);'>Fine-tuned</th>
                    <th style='padding: 12px 16px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.1);'>Δ</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style='padding: 12px 16px; font-weight: 600;'>Char-similarity to ground truth</td>
                    <td style='padding: 12px 16px; text-align: center;'>{b['char_similarity_mean']:.3f}</td>
                    <td style='padding: 12px 16px; text-align: center; color: #4ade80; font-weight: 700;'>{fm['char_similarity_mean']:.3f}</td>
                    <td style='padding: 12px 16px; text-align: center; color: #4ade80; font-weight: 700;'>+{sim_delta:.3f}</td>
                </tr>
                <tr style='background: rgba(255,255,255,0.03);'>
                    <td style='padding: 12px 16px; font-weight: 600;'>Keyword recall (TechMojo facts)</td>
                    <td style='padding: 12px 16px; text-align: center;'>{b['keyword_recall_mean']:.3f}</td>
                    <td style='padding: 12px 16px; text-align: center; color: #4ade80; font-weight: 700;'>{fm['keyword_recall_mean']:.3f}</td>
                    <td style='padding: 12px 16px; text-align: center; color: #4ade80; font-weight: 700;'>+{rec_delta:.3f}</td>
                </tr>
                <tr>
                    <td style='padding: 12px 16px; font-weight: 600;'>Held-out examples</td>
                    <td style='padding: 12px 16px; text-align: center;'>{b['n_examples']}</td>
                    <td style='padding: 12px 16px; text-align: center;'>{fm['n_examples']}</td>
                    <td style='padding: 12px 16px; text-align: center;'>—</td>
                </tr>
            </tbody>
        </table>
        <p style='margin-top: 16px; font-size: 0.85em; color: #94a3b8;'>
            Source: <code>PranavTM/LeavePolicy</code> · Base: {base['model'].split('/')[-1]}
        </p>
    </div>
    """


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_app(base_only: bool = False) -> gr.Blocks:
    """Build and return the Gradio app."""

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Color tokens (single source of truth) ─────────────────────────── */
    :root {
        --bg-page: #0c0f14;
        --bg-surface: #131820;
        --bg-elevated: #1a212c;
        --border: #232a36;
        --border-strong: #2f3845;
        --text: #e6e9ef;
        --text-muted: #8b94a3;
        --text-subtle: #5e6776;
        --accent: #4f8cf6;
        --accent-soft: rgba(79, 140, 246, 0.10);
        --positive: #3ecf8e;
        --positive-soft: rgba(62, 207, 142, 0.08);
        --warning: #c97a4a;
        --warning-soft: rgba(201, 122, 74, 0.08);
    }

    * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    code, pre, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

    body, .gradio-container {
        background: var(--bg-page) !important;
        color: var(--text) !important;
    }

    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 24px !important; }

    /* ── Header ───────────────────────────────────────────────────────── */
    .header-banner {
        background: var(--bg-surface);
        padding: 28px 32px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
    }
    .header-banner h1 {
        font-size: 1.6em;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .header-banner .subline {
        color: var(--text-muted);
        font-size: 0.92em;
        margin: 0;
    }
    .header-banner .badges {
        margin-top: 14px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .header-banner .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72em;
        font-weight: 500;
        color: var(--text-muted);
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 4px;
    }

    /* ── Model column labels ─────────────────────────────────────────── */
    .model-label-ft, .model-label-base {
        font-size: 0.78em;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 6px 12px;
        border-radius: 4px;
        border: 1px solid var(--border);
        background: var(--bg-surface);
        margin-bottom: 8px;
        display: inline-block;
    }
    .model-label-ft { color: var(--positive); border-color: rgba(62, 207, 142, 0.3); }
    .model-label-base { color: var(--text-muted); }
    .model-label-base::before { content: ""; display: inline-block; width: 6px; height: 6px; background: var(--text-subtle); border-radius: 50%; margin-right: 8px; vertical-align: middle; }
    .model-label-ft::before { content: ""; display: inline-block; width: 6px; height: 6px; background: var(--positive); border-radius: 50%; margin-right: 8px; vertical-align: middle; }

    .disclaimer {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        padding: 10px 14px;
        border-radius: 4px;
        color: var(--text-muted);
        font-size: 0.84em;
        margin-bottom: 18px;
    }

    /* ── Chat bubbles ────────────────────────────────────────────────── */
    .chatbot {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .chatbot .message.user {
        background: var(--accent-soft) !important;
        border: 1px solid rgba(79, 140, 246, 0.2) !important;
        border-radius: 6px !important;
    }
    .chatbot .message.bot {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .submit-btn {
        background: var(--accent) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 9px 20px !important;
        transition: background 0.15s !important;
    }
    .submit-btn:hover { background: #6aa1f8 !important; }

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .tab-nav {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
    }
    .tab-nav button {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.92em !important;
        padding: 10px 18px !important;
    }
    .tab-nav button.selected {
        border-bottom: 2px solid var(--accent) !important;
        color: var(--text) !important;
    }

    /* ── Inputs ──────────────────────────────────────────────────────── */
    input[type=text], textarea {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    """

    with gr.Blocks(title="TechMojo HR Assistant — Fine-tune Demo") as demo:
        demo._launch_kwargs = {"css": css, "theme": gr.themes.Base()}

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <h1>TechMojo HR Assistant</h1>
            <p class="subline">Llama 3.1 8B fine-tuned with QLoRA on TechMojo internal HR policies. Side-by-side comparison: base model (no company knowledge) vs. fine-tuned model.</p>
            <div class="badges">
                <span class="badge">Llama 3.1 8B 4-bit</span>
                <span class="badge">QLoRA r=16</span>
                <span class="badge">Apple MLX</span>
                <span class="badge">99 train · 18 eval</span>
            </div>
        </div>
        """)

        gr.HTML(f'<div class="disclaimer">{DISCLAIMER}</div>')

        if not MLX_AVAILABLE:
            gr.HTML(
                '<div class="disclaimer" style="border-left-color:#c97a4a; '
                'background:rgba(201,122,74,0.06); color:#e6c39a;">'
                '<strong>Running on Linux.</strong> '
                'This Space will try a <strong>Transformers</strong> fallback backend for live inference. '
                'For best performance, use a <strong>GPU Space</strong>. '
                'If loading fails, the app will show an actionable error message.'
                '</div>'
            )

        with gr.Tabs():

            # ── Tab 1: HR Q&A ─────────────────────────────────────────────────
            with gr.Tab("💬 HR Q&A"):
                gr.Markdown("### Ask a TechMojo HR question — compare base vs fine-tuned responses")

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="model-label-ft">Fine-tuned · TechMojo HR</div>')
                        ft_chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            elem_classes=["chatbot"],
                            show_label=False,
                        )

                    with gr.Column():
                        gr.HTML('<div class="model-label-base">Base · Llama 3.1 8B (no fine-tune)</div>')
                        base_chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            elem_classes=["chatbot"],
                            show_label=False,
                        )

                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="e.g. How many leave days do I get per year at TechMojo?",
                        label="Your HR question",
                        scale=4,
                        lines=1,
                        show_label=True,
                    )
                    submit_btn = gr.Button("Ask →", variant="primary", scale=1, elem_classes=["submit-btn"])

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear", scale=1)
                    status_box = gr.Textbox(label="Status", scale=3, interactive=False, show_label=False)

                # Advanced settings
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
                        temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.1, label="Temperature")

                # Example questions
                gr.Markdown("**Quick examples:**")
                with gr.Row():
                    for ex in EXAMPLE_QUESTIONS[:4]:
                        gr.Button(ex[:45] + "…" if len(ex) > 45 else ex, size="sm").click(
                            fn=lambda e=ex: e,
                            outputs=question_box,
                        )

                with gr.Row():
                    for ex in EXAMPLE_QUESTIONS[4:]:
                        gr.Button(ex[:45] + "…" if len(ex) > 45 else ex, size="sm").click(
                            fn=lambda e=ex: e,
                            outputs=question_box,
                        )

                # State (conversation histories)
                ft_state = gr.State([])
                base_state = gr.State([])

                submit_btn.click(
                    fn=respond,
                    inputs=[question_box, ft_state, base_state, max_tokens, temperature],
                    outputs=[ft_chatbot, base_chatbot, question_box, status_box],
                )
                question_box.submit(
                    fn=respond,
                    inputs=[question_box, ft_state, base_state, max_tokens, temperature],
                    outputs=[ft_chatbot, base_chatbot, question_box, status_box],
                )
                clear_btn.click(
                    fn=clear_history,
                    outputs=[ft_chatbot, base_chatbot, question_box, status_box],
                )

            # ── Tab 2: Benchmark Results ───────────────────────────────────────
            with gr.Tab("📊 Benchmark Results"):
                gr.Markdown("## TechMojo HR — Before/After Fine-tuning")
                gr.Markdown(
                    "Held-out eval on TechMojo internal HR questions. "
                    "Source: `PranavTM/LeavePolicy` on HuggingFace — "
                    "verified to be out-of-distribution for the base model "
                    "(see `data/techmojo/ood_check.py`)."
                )
                benchmark_html = gr.HTML(load_benchmark_html())
                gr.Button("🔄 Refresh Results").click(
                    fn=load_benchmark_html,
                    outputs=benchmark_html,
                )

                gr.Markdown("""
### 💾 Model Size Comparison

| Component | Size |
|-----------|------|
| Full Llama 3.1 8B | ~16 GB |
| 4-bit quantized base | ~5 GB |
| **LoRA adapters only** | **~40 MB** |

> Only the adapter weights are trained and saved — a tiny fraction of the full model.
""")

            # ── Tab 3: About ─────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## What is this?

A QLoRA fine-tune of Llama 3.1 8B on **TechMojo's internal HR policies** —
specifically chosen because the base model has *never seen this data*. We
verified this before training by probing the base Llama with 6 specific
TechMojo questions: it either hallucinated plausible-but-wrong details
(invented "Employee Referral Program portal" instead of `Freshteams`) or
admitted ignorance ("I'm not aware of the exact leave policy at TechMojo").

That's the criterion for a fine-tune that *creates capability* rather than
*restyling existing capability*: the base model has to fail the OOD probe
first.

## Pipeline

```
1. Source                 → PranavTM/LeavePolicy (HuggingFace, 117 examples)
2. OOD probe              → data/techmojo/ood_check.py (verify base fails)
3. Dataset prep           → prepare_dataset.py (chat format + train/eval split)
4. Baseline eval          → evaluation/eval_qa.py --which base
5. QLoRA training         → training/train.py (mlx_lm.lora wrapper)
6. After eval + compare   → eval_qa.py --which finetuned + compare.py
7. Demo                   → app.py (this Gradio side-by-side)
8. Publish                → export/push_to_hub.py
```

## Why this works as a portfolio demo

Most fine-tuning portfolio projects pick mainstream targets (medical, code,
chat) where the base model already has the knowledge — the "improvement" is
mostly style transfer, which is unimpressive. By picking proprietary HR data
the base model has never seen, the before/after is genuinely dramatic and
demonstrates QLoRA's actual strength: cheap, parameter-efficient adaptation
to private/niche data.

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | Llama 3.1 8B (Meta), 4-bit quantized |
| Fine-tuning | QLoRA (r=16, α=16) |
| Framework | Apple MLX |
| Training | mlx-lm LoRA trainer |
| Demo | Gradio (side-by-side base vs fine-tuned) |
| Hardware | Mac Mini, Apple M-series, 24 GB |

---
*Built on Apple Silicon · {cfg['model']['name']}*
""")

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch TechMojo HR Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--base-only", action="store_true", help="Load base model only")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    # Don't pre-load models on the main thread — Gradio's queue processes
    # requests on its own event-loop thread, and MLX prompt_cache stream state
    # is thread-local. Lazy-load inside the first inference so models live on
    # the same thread that calls generate().
    print("Skipping eager model load — models will load on first question (~60-90s).")

    demo = build_app(base_only=args.base_only)

    print(f"\n✅ TechMojo HR demo ready at http://localhost:{args.port}")
    if args.share:
        print("Creating public share link…")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        **getattr(demo, "_launch_kwargs", {}),
    )
