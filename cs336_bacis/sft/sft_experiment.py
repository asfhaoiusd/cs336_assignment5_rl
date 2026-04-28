from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from get_response_log_probs import get_response_log_probs
from sft_microbatch_train_step import sft_microbatch_train_step
from tokenize_prompt_and_output import tokenize_prompt_and_output


@dataclass
class SFTConfig:
    train_device: str = "cuda"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    num_epochs: int = 1
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    normalize_constant: float = 1.0
    seed: int = 42
    # If set, save a loss curve (optimizer step vs loss) to this path after training.
    loss_plot_path: str | None = None
    # Cap prompt+output token count (None = no truncation). Reduces VRAM for long MATH prompts.
    max_seq_length: int | None = 2048
    gradient_checkpointing: bool = True
    # CUDA only: mixed precision forward/backward (bf16 if supported, else fp16).
    use_amp: bool = True


def _save_loss_plot(train_history: list[dict[str, Any]], path: str) -> bool:
    """Save loss curve; return False if matplotlib is not installed."""
    if not train_history:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [h["optimizer_update"] for h in train_history]
    losses = [h["loss"] for h in train_history]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color="tab:blue", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _unpack_example(ex: dict[str, Any]) -> tuple[str, str]:
    if "prompt" in ex and "output" in ex:
        return str(ex["prompt"]), str(ex["output"])
    if "question" in ex and "answer" in ex:
        return str(ex["question"]), str(ex["answer"])
    raise KeyError(
        "Expected keys ('prompt','output') or ('question','answer'). "
        f"Got: {list(ex.keys())}"
    )


def sft_experiment(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_examples: list[dict[str, Any]],
    cfg: SFTConfig,
) -> dict[str, Any]:
    if cfg.train_batch_size % cfg.gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps "
            f"(got {cfg.train_batch_size} and {cfg.gradient_accumulation_steps})."
        )

    device = torch.device(cfg.train_device)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    policy.train()
    policy.to(device)

    if cfg.gradient_checkpointing and hasattr(policy, "gradient_checkpointing_enable"):
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable()

    optimizer = AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    n = len(train_examples)
    train_history: list[dict[str, Any]] = []
    global_update = 0

    for epoch in range(1, cfg.num_epochs + 1):
        order = list(range(n))
        random.Random(cfg.seed + epoch).shuffle(order)

        limit = n - (n % cfg.train_batch_size)
        for batch_start in range(0, limit, cfg.train_batch_size):
            batch_indices = order[batch_start : batch_start + cfg.train_batch_size]
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[float] = []

            for micro_start in range(0, cfg.train_batch_size, micro_train_batch_size):
                micro_indices = batch_indices[micro_start : micro_start + micro_train_batch_size]
                prompt_strs = []
                output_strs = []
                for idx in micro_indices:
                    p, o = _unpack_example(train_examples[idx])
                    prompt_strs.append(p)
                    output_strs.append(o)

                batch = tokenize_prompt_and_output(
                    prompt_strs,
                    output_strs,
                    tokenizer,
                    max_seq_length=cfg.max_seq_length,
                )
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                response_mask = batch["response_mask"].to(device)

                use_amp = cfg.use_amp and device.type == "cuda"
                amp_ctx = nullcontext()
                if use_amp:
                    amp_dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)

                with amp_ctx:
                    out = get_response_log_probs(
                        model=policy,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=False,
                    )
                    policy_log_probs = out["log_probs"]

                    scaled_loss, _ = sft_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                        normalize_constant=cfg.normalize_constant,
                    )
                step_losses.append(scaled_loss.item() * cfg.gradient_accumulation_steps)

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm).item()
            )
            optimizer.step()
            global_update += 1

            avg_loss = float(sum(step_losses) / len(step_losses))
            train_history.append(
                {
                    "epoch": epoch,
                    "optimizer_update": global_update,
                    "loss": avg_loss,
                    "grad_norm": grad_norm,
                }
            )

    loss_plot_path: str | None = None
    if cfg.loss_plot_path and _save_loss_plot(train_history, cfg.loss_plot_path):
        loss_plot_path = cfg.loss_plot_path

    return {
        "train_history": train_history,
        "final_loss": train_history[-1]["loss"] if train_history else float("nan"),
        "num_optimizer_updates": global_update,
        "loss_plot_path": loss_plot_path,
    }
