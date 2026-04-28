#!/usr/bin/env python3
"""Load data and run SFT via ``sft_experiment`` (``python sft_train.py`` from this dir or repo root)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SFT_DIR.parent.parent
CHAPTER5 = REPO_ROOT / "chapter5"
# Default checkpoint for this assignment (offline-friendly when files are present).
PREFERRED_SFT_MODEL = CHAPTER5 / "models" / "Qwen2.5-Math-1.5B"

if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

import torch

from sft_experiment import SFTConfig, sft_experiment
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path, max_examples: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def rows_to_sft_examples(rows: list[dict]) -> list[dict]:
    """MATH-style JSONL: ``problem`` + ``answer``. Also accepts ``question`` + ``answer``."""
    examples: list[dict] = []
    for r in rows:
        problem = r.get("problem") if "problem" in r else r.get("question")
        answer = r.get("answer")
        if problem is None or answer is None:
            continue
        prompt = (
            "You are a helpful assistant. Solve the problem and give a concise final answer.\n"
            f"User: {problem}\nAssistant:"
        )
        examples.append({"prompt": prompt, "output": f" {answer}"})
    return examples


def default_train_path() -> Path:
    for candidate in (
        CHAPTER5 / "MATH" / "train.jsonl",
        CHAPTER5 / "MATH" / "validation.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No default train file found. Pass --train-file explicitly "
        f"(looked under {CHAPTER5 / 'MATH'})."
    )


def default_local_model_dir() -> Path | None:
    """Prefer Qwen2.5-Math-1.5B, else first HuggingFace-style dir under ``chapter5/models``."""
    if PREFERRED_SFT_MODEL.is_dir() and (PREFERRED_SFT_MODEL / "config.json").is_file():
        return PREFERRED_SFT_MODEL
    root = CHAPTER5 / "models"
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child
    return None


def resolve_model_and_local_only(
    model_arg: str, local_files_only_flag: bool
) -> tuple[str, bool]:
    """
    Pick model id/path and whether to load from disk only (no HuggingFace Hub).
    """
    hub_offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    trans_offline = os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")

    model_id = model_arg.strip()
    if not model_id:
        found = default_local_model_dir()
        if found is None:
            raise SystemExit(
                "No --model given and no usable local checkpoint.\n"
                f"Expected first: {PREFERRED_SFT_MODEL} (with config.json), "
                f"or any HF snapshot under {CHAPTER5 / 'models'}.\n"
                "Or pass explicitly, e.g.:\n"
                f"  --model {PREFERRED_SFT_MODEL}\n"
                "With internet, a Hub id also works, e.g. --model gpt2"
            )
        model_id = str(found)

    path = Path(model_id).expanduser()
    looks_local = path.is_dir() and (path / "config.json").is_file()
    local_files_only = (
        local_files_only_flag or looks_local or hub_offline or trans_offline
    )
    return model_id, local_files_only


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training (cs336_bacis/sft)")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="HF Hub id or local checkpoint dir (with config.json). "
        f"Default (if empty): {PREFERRED_SFT_MODEL} when present, else first under chapter5/models/.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact HuggingFace Hub (use with local snapshot or offline cache).",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="",
        help="JSONL with problem+answer (or question+answer). Empty = auto under chapter5/MATH.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=256,
        help="Cap number of JSONL rows loaded (default 256). Use -1 for all.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Per-optimizer-step batch size (default 1 to reduce VRAM).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max prompt+output tokens per example (-1 = no cap).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable bf16/fp16 autocast (more VRAM, fp32 on GPU if model loads in fp32).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster, much more VRAM).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SFT_DIR / "outputs"),
        help="Directory for loss plot and optional metrics JSON.",
    )
    args = parser.parse_args()

    model_id, local_files_only = resolve_model_and_local_only(
        args.model, args.local_files_only
    )

    train_path = Path(args.train_file) if args.train_file else default_train_path()
    max_ex = None if args.max_examples < 0 else args.max_examples
    raw_rows = load_jsonl(train_path, max_ex)
    train_examples = rows_to_sft_examples(raw_rows)
    if not train_examples:
        raise SystemExit(
            f"No training examples parsed from {train_path}. "
            "Expected each line to include 'problem' (or 'question') and 'answer'."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_plot = str(out_dir / "sft_loss.png")

    load_kw: dict = {"trust_remote_code": True, "local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    weight_dtype = None
    if str(args.device).startswith("cuda") and not args.no_amp:
        weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kw = dict(load_kw)
    if weight_dtype is not None:
        model_kw["dtype"] = weight_dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)
    except TypeError:
        # Older transformers: ``torch_dtype`` instead of ``dtype``.
        if "dtype" in model_kw:
            model_kw["torch_dtype"] = model_kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)

    max_seq = None if args.max_seq_length < 0 else args.max_seq_length
    cfg = SFTConfig(
        train_device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        loss_plot_path=loss_plot,
        max_seq_length=max_seq,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_amp=not args.no_amp,
    )

    print(f"Training on {len(train_examples)} examples from {train_path}")
    print(
        f"Model: {model_id}, local_files_only={local_files_only}, device: {args.device}, "
        f"amp={cfg.use_amp}, gc={cfg.gradient_checkpointing}, max_seq_length={cfg.max_seq_length}"
    )
    result = sft_experiment(model, tokenizer, train_examples, cfg)

    metrics_path = out_dir / "sft_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "final_loss": result["final_loss"],
                "num_optimizer_updates": result["num_optimizer_updates"],
                "loss_plot_path": result.get("loss_plot_path"),
                "train_history": result["train_history"],
            },
            f,
            indent=2,
        )

    print(f"Final loss: {result['final_loss']:.6f}")
    print(f"Optimizer updates: {result['num_optimizer_updates']}")
    if result.get("loss_plot_path"):
        print(f"Loss plot: {result['loss_plot_path']}")
    elif loss_plot:
        print("Loss plot skipped (matplotlib not installed). pip install matplotlib")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
