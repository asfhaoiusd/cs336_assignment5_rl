
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

SFT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SFT_DIR.parent.parent
CHAPTER5 = REPO_ROOT / "chapter5"
PREFERRED_SFT_MODEL = CHAPTER5 / "models" / "Qwen2.5-Math-1.5B"


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


#解析本地模型路径
def default_local_model_dir() -> Path | None:
    if PREFERRED_SFT_MODEL.is_dir() and (PREFERRED_SFT_MODEL / "config.json").is_file():
        return PREFERRED_SFT_MODEL
    root = CHAPTER5 / "models"
    if not root.is_dir():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").is_file():
            return child
    return None


def resolve_qwen_model() -> str:
    """Resolve local Qwen checkpoint only (no Hub fallback)."""
    found = default_local_model_dir()
    if found is None:
        raise SystemExit(
            "No local Qwen checkpoint found.\n"
            f"Expected: {PREFERRED_SFT_MODEL} (with config.json), "
            f"or another snapshot under {CHAPTER5 / 'models'}."
        )
    return str(found)

#将数据集转换为文本
def rows_to_texts(rows: list[dict]) -> list[str]:
    texts: list[str] = []
    for r in rows:
        problem = r.get("problem") if "problem" in r else r.get("question")
        answer = r.get("answer")
        if problem is None or answer is None:
            continue
        texts.append(
            "You are a helpful assistant. Solve the problem and give a concise final answer.\n"
            f"User: {problem}\nAssistant: {answer}"
        )
    return texts

#记录损失历史
class LossHistoryCallback(TrainerCallback):
    """Track loss in-memory for optional JSON export."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if "loss" in logs:
            self.history.append(
                {
                    "step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "loss": float(logs["loss"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT for local Qwen with TRL")
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "pretrain_model"),
        help="Directory to save SFT model/tokenizer (default: pretrain_model).",
    )
    #解析参数
    args = parser.parse_args()

    #解析模型路径
    model_id = resolve_qwen_model()
    #解析训练文件路径
    train_path = Path(args.train_file) if args.train_file else default_train_path()
    max_ex = None if args.max_examples < 0 else args.max_examples
    rows = load_jsonl(train_path, max_ex)
    texts = rows_to_texts(rows)
    if not texts:
        raise SystemExit(
            f"No training examples parsed from {train_path}. "
            "Expected each line to include 'problem' (or 'question') and 'answer'."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    load_kw = {"trust_remote_code": True, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kw = dict(load_kw)
    if torch.cuda.is_available() and not args.no_amp:
        model_kw["dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kw)
    model.config.use_cache = False if not args.no_gradient_checkpointing else True
    
    #创建训练数据集，huggface的dataset库还是很好用的，能直接将数据集转换为Dataset对象
    train_dataset = Dataset.from_dict({"text": texts})
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.no_amp
    use_fp16 = torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()) and not args.no_amp

    config = SFTConfig(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_length=(None if args.max_seq_length < 0 else args.max_seq_length),
        packing=True,
        dataset_text_field="text",
        seed=args.seed,
        report_to=[],
    )

    history = LossHistoryCallback()
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[history],
    )

    print(f"Training on {len(train_dataset)} examples from {train_path}")
    print(f"Model: {model_id} (local-only)")
    result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "final_train_loss": float(result.training_loss),
        "global_step": int(trainer.state.global_step),
        "train_samples": len(train_dataset),
        "log_history": history.history,
    }
    metrics_path = out_dir / "sft_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Done. Final train loss: {metrics['final_train_loss']:.6f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
