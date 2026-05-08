"""
tools/convert_mlx_to_peft.py
=============================
Convert an mlx-lm LoRA adapter to PEFT format so it can be loaded with
transformers + peft on Linux (where mlx-lm cannot run).

MLX layout (input):
  Keys:    model.layers.{i}.{module}.lora_a   shape: (in_features, r)
           model.layers.{i}.{module}.lora_b   shape: (r, out_features)

PEFT layout (output):
  Keys:    base_model.model.{model.layers.{i}.{module}}.lora_A.default.weight
                                                          shape: (r, in_features)
           base_model.model.{model.layers.{i}.{module}}.lora_B.default.weight
                                                          shape: (out_features, r)
  Plus an adapter_config.json with the LoRA hyperparameters.

Both encode the same delta. The difference is the key naming and that PEFT
stores weights transposed relative to MLX (PEFT follows nn.Linear's
(out, in) convention).

Usage:
  python tools/convert_mlx_to_peft.py
  python tools/convert_mlx_to_peft.py --src adapters_techmojo_best \
                                      --dst adapters_techmojo_best_peft
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_np

ROOT = Path(__file__).parent.parent

# Match MLX adapter keys like:
#   model.layers.16.self_attn.q_proj.lora_a
#   model.layers.16.mlp.down_proj.lora_b
MLX_KEY_RE = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_([ab])$"
)


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def mlx_to_peft_key(mlx_key: str) -> str | None:
    """
    'model.layers.16.self_attn.q_proj.lora_a'
        ↓
    'base_model.model.model.layers.16.self_attn.q_proj.lora_A.default.weight'

    The double 'base_model.model.' prefix is intentional — that's what PEFT
    produces when wrapping a HuggingFace `AutoModelForCausalLM` (whose own
    forward accesses `.model` internally).
    """
    m = MLX_KEY_RE.match(mlx_key)
    if not m:
        return None
    inner = m.group(1)
    ab = m.group(2).upper()
    return f"base_model.model.{inner}.lora_{ab}.default.weight"


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Convert MLX LoRA → PEFT format")
    parser.add_argument(
        "--src",
        type=str,
        default=cfg["inference"]["adapter_path"],
        help="MLX adapter dir (must contain adapters.safetensors)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=cfg["inference"].get("adapter_path_linux", "adapters_peft"),
        help="Output dir for PEFT-format adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=cfg["model"].get("hf_name_linux") or cfg["model"]["name"],
        help="base_model_name_or_path for adapter_config.json",
    )
    args = parser.parse_args()

    src_dir = ROOT / args.src
    dst_dir = ROOT / args.dst
    src_file = src_dir / "adapters.safetensors"
    if not src_file.exists():
        print(f"[convert] ✗ MLX adapter not found: {src_file}")
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[convert] src: {src_file}")
    print(f"[convert] dst: {dst_dir}")
    print(f"[convert] target base model (for adapter_config): {args.base_model}")
    print()

    # ── Read + rename + transpose ──────────────────────────────────────────
    peft_tensors: dict = {}
    skipped: list[str] = []

    with safe_open(str(src_file), framework="numpy") as f:
        mlx_keys = list(f.keys())
        for mlx_key in mlx_keys:
            new_key = mlx_to_peft_key(mlx_key)
            if new_key is None:
                skipped.append(mlx_key)
                continue
            arr = f.get_tensor(mlx_key)
            # Transpose: MLX (in,r)/(r,out) → PEFT (r,in)/(out,r)
            peft_tensors[new_key] = arr.T.copy()

    print(f"[convert] converted {len(peft_tensors)}/{len(mlx_keys)} keys")
    if skipped:
        print(f"[convert] skipped {len(skipped)} keys that didn't match pattern:")
        for k in skipped[:5]:
            print(f"           - {k}")

    # ── Write PEFT-format safetensors ──────────────────────────────────────
    out_safetensors = dst_dir / "adapter_model.safetensors"
    save_safetensors_np(peft_tensors, str(out_safetensors))
    print(f"[convert] ✓ wrote {out_safetensors} ({out_safetensors.stat().st_size / 1_048_576:.1f} MB)")

    # ── Write PEFT adapter_config.json ─────────────────────────────────────
    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": args.base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": cfg["lora"]["alpha"],
        "lora_dropout": cfg["lora"]["dropout"],
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": cfg["lora"]["r"],
        "rank_pattern": {},
        "revision": None,
        "target_modules": cfg["lora"]["target_modules"],
        "task_type": "CAUSAL_LM",
    }
    out_config = dst_dir / "adapter_config.json"
    with open(out_config, "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"[convert] ✓ wrote {out_config}")

    print()
    print(f"[convert] Done. Load on Linux/Spaces with:")
    print(f"           from peft import PeftModel")
    print(f"           model = PeftModel.from_pretrained(base_model, '{dst_dir.name}')")


if __name__ == "__main__":
    main()
