#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAPTER5 = REPO_ROOT / "chapter5"
SFT_MODEL_DIR = REPO_ROOT / "pretrain_model"
RL_MODEL_DIR = REPO_ROOT / "rl_model"

import sys

if str(CHAPTER5) not in sys.path:
    sys.path.insert(0, str(CHAPTER5))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> "
    "</think> and answer is enclosed within <answer> </answer> tags.\n"
    "User: {question}\nAssistant: <think>"
)


@dataclass
class RolloutItem:
    prompt: str
    response: str
    ground_truth: str
    reward: float
    advantage: float


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
    for candidate in (CHAPTER5 / "MATH" / "train.jsonl", CHAPTER5 / "MATH" / "validation.jsonl"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No train file found under chapter5/MATH.")


def rows_to_examples(rows: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for r in rows:
        q = r.get("problem", r.get("question"))
        a = r.get("answer")
        if q is None or a is None:
            continue
        out.append((str(q), str(a)))
    return out


def pad_and_tensorize(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in seqs)
    ids, mask = [], []
    for s in seqs:
        pad_n = max_len - len(s)
        ids.append(s + [pad_id] * pad_n)
        mask.append([1] * len(s) + [0] * pad_n)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # labels may contain -100 ignore index; clamp to valid ids before gather
    # and zero-out ignored positions after gather.
    safe_labels = labels.clamp_min(0)
    logp = torch.gather(
        F.log_softmax(logits, dim=-1), dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return torch.where(labels == -100, torch.zeros_like(logp), logp)


def make_batch_tensors(
    tokenizer,
    rollouts: list[RolloutItem],
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    resp_mask_list: list[list[int]] = []

    for it in rollouts:
        prompt_ids = tokenizer(it.prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(it.prompt + it.response, add_special_tokens=False)["input_ids"]
        if len(full_ids) < 2:
            continue
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
        x = full_ids[:-1]
        y = full_ids[1:]
        prompt_pred_start = max(0, min(len(prompt_ids) - 1, len(y)))
        response_mask = [0] * prompt_pred_start + [1] * (len(y) - prompt_pred_start)
        input_ids_list.append(x)
        labels_list.append(y)
        resp_mask_list.append(response_mask)

    input_ids, _ = pad_and_tensorize(input_ids_list, pad_id)
    labels, _ = pad_and_tensorize(labels_list, -100)
    response_mask, _ = pad_and_tensorize(resp_mask_list, 0)
    return input_ids.to(device), labels.to(device), response_mask.to(device)


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def evaluate_avg_reward(
    model,
    tokenizer,
    examples: list[tuple[str, str]],
    max_eval_examples: int,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> float:
    subset = examples[:max_eval_examples]
    rewards: list[float] = []
    for q, gt in subset:
        prompt = PROMPT_TEMPLATE.format(question=q)
        batch_prompt = [prompt] * group_size
        tok = tokenizer(batch_prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **tok,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        plen = tok["input_ids"].shape[1]
        for i in range(group_size):
            resp = tokenizer.decode(out[i, plen:], skip_special_tokens=True)
            rewards.append(float(r1_zero_reward_fn(resp, gt)["reward"]))
    return float(sum(rewards) / max(1, len(rewards)))


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training from SFT checkpoint")
    parser.add_argument("--sft-model-dir", type=str, default=str(SFT_MODEL_DIR))
    parser.add_argument("--output-model-dir", type=str, default=str(RL_MODEL_DIR))
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-eval-examples", type=int, default=32)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print training metrics every N optimization steps.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    train_path = Path(args.train_file) if args.train_file else default_train_path()
    rows = load_jsonl(train_path, None if args.max_examples < 0 else args.max_examples)
    examples = rows_to_examples(rows)
    if not examples:
        raise SystemExit("No valid training examples found.")

    out_model_dir = Path(args.output_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = out_model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else None
    )
    tokenizer_load_kw = {"trust_remote_code": True, "local_files_only": True}
    model_load_kw = {"trust_remote_code": True, "local_files_only": True}
    if dtype is not None:
        # Keep dtype only on model loading. Passing torch dtype to tokenizer
        # can leak into tokenizer config and break JSON serialization on save.
        model_load_kw["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_dir, fix_mistral_regex=True, **tokenizer_load_kw
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_dir, **model_load_kw).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_dir, **model_load_kw).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    history: list[dict] = []
    global_step = 0
    total_steps_per_epoch = (len(examples) + args.prompt_batch_size - 1) // args.prompt_batch_size
    total_steps = total_steps_per_epoch * args.epochs
    print(
        f"Starting GRPO training | examples={len(examples)} epochs={args.epochs} "
        f"steps/epoch={total_steps_per_epoch} total_steps={total_steps}",
        flush=True,
    )
    for ep in range(1, args.epochs + 1):
        order = list(range(len(examples)))
        random.shuffle(order)
        epoch_pbar = tqdm(
            range(0, len(order), args.prompt_batch_size),
            desc=f"Epoch {ep}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
        )
        for start in epoch_pbar:
            idxs = order[start : start + args.prompt_batch_size]
            batch = [examples[i] for i in idxs]

            rollouts: list[RolloutItem] = []
            policy.eval()
            rollout_pbar = tqdm(
                batch,
                desc="Generating rollouts",
                unit="prompt",
                leave=False,
                dynamic_ncols=True,
            )
            for q, gt in rollout_pbar:
                prompt = PROMPT_TEMPLATE.format(question=q)
                batch_prompt = [prompt] * args.group_size
                tok = tokenizer(batch_prompt, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = policy.generate(
                        **tok,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                plen = tok["input_ids"].shape[1]
                responses = [tokenizer.decode(out[i, plen:], skip_special_tokens=True) for i in range(args.group_size)]
                rewards = torch.tensor([float(r1_zero_reward_fn(r, gt)["reward"]) for r in responses], dtype=torch.float32)
                adv = rewards - rewards.mean()
                adv = adv / (rewards.std(unbiased=False) + 1e-6)
                for i in range(args.group_size):
                    rollouts.append(RolloutItem(prompt=prompt, response=responses[i], ground_truth=gt, reward=float(rewards[i]), advantage=float(adv[i])))
                rollout_pbar.set_postfix(avg_reward=f"{float(rewards.mean()):.4f}")
            rollout_pbar.close()

            policy.train()
            input_ids, labels, response_mask = make_batch_tensors(tokenizer, rollouts, args.max_seq_length, device)
            if input_ids.shape[0] == 0:
                continue

            outputs = policy(input_ids=input_ids, return_dict=True)
            new_logp = token_log_probs(outputs.logits, labels)
            with torch.no_grad():
                old_logp = new_logp.detach()
                ref_logits = ref_model(input_ids=input_ids, return_dict=True).logits
                ref_logp = token_log_probs(ref_logits, labels)

            adv_tensor = torch.tensor([r.advantage for r in rollouts], dtype=torch.float32, device=device).unsqueeze(1)
            adv_tensor = adv_tensor.expand_as(new_logp)
            ratio = torch.exp(new_logp - old_logp)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
            pg_loss_tok = -torch.minimum(ratio * adv_tensor, clipped_ratio * adv_tensor)
            pg_loss = masked_mean(pg_loss_tok, response_mask)

            kl_tok = (new_logp - ref_logp)
            kl_loss = masked_mean(kl_tok, response_mask)
            loss = pg_loss + args.kl_coef * kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm))
            optimizer.step()
            global_step += 1

            mean_reward = float(sum(r.reward for r in rollouts) / max(1, len(rollouts)))
            history.append(
                {
                    "epoch": ep,
                    "step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "pg_loss": float(pg_loss.detach().cpu()),
                    "kl_loss": float(kl_loss.detach().cpu()),
                    "mean_reward": mean_reward,
                    "grad_norm": grad_norm,
                }
            )
            epoch_pbar.set_postfix(
                step=global_step,
                loss=f"{history[-1]['loss']:.4f}",
                reward=f"{mean_reward:.4f}",
            )
            if args.log_every > 0 and (global_step % args.log_every == 0):
                print(
                    f"[ep {ep} step {global_step}] "
                    f"loss={history[-1]['loss']:.4f} "
                    f"pg={history[-1]['pg_loss']:.4f} "
                    f"kl={history[-1]['kl_loss']:.4f} "
                    f"reward={mean_reward:.4f} "
                    f"grad_norm={grad_norm:.4f}",
                    flush=True,
                )
        epoch_pbar.close()

    policy.save_pretrained(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)

    # Curves
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    loss_plot = output_dir / "grpo_loss_curve.png"
    reward_plot = output_dir / "grpo_reward_curve.png"
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, color="tab:red", linewidth=1.2)
    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title("GRPO Training Loss (from SFT init)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rewards, color="tab:green", linewidth=1.2)
    plt.xlabel("Optimization step")
    plt.ylabel("Mean rollout reward")
    plt.title("GRPO Training Reward (from SFT init)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(reward_plot, dpi=150)
    plt.close()

    # Quick comparison: SFT vs GRPO
    sft_for_eval = AutoModelForCausalLM.from_pretrained(args.sft_model_dir, **model_load_kw).to(device).eval()
    grpo_for_eval = AutoModelForCausalLM.from_pretrained(str(out_model_dir), **model_load_kw).to(device).eval()
    sft_reward = evaluate_avg_reward(
        sft_for_eval,
        tokenizer,
        examples,
        args.quick_eval_examples,
        args.group_size,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )
    grpo_reward = evaluate_avg_reward(
        grpo_for_eval,
        tokenizer,
        examples,
        args.quick_eval_examples,
        args.group_size,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )
    compare = {
        "evaluation_type": "quick",
        "note": "Quick comparison only; use larger eval set for final conclusion.",
        "sft_model": str(Path(args.sft_model_dir).resolve()),
        "grpo_model": str(out_model_dir.resolve()),
        "quick_eval_examples": args.quick_eval_examples,
        "group_size": args.group_size,
        "sft_avg_reward": sft_reward,
        "grpo_avg_reward": grpo_reward,
        "reward_improvement_abs": grpo_reward - sft_reward,
    }
    compare_plot = output_dir / "grpo_vs_sft_reward_comparison.png"
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["SFT", "GRPO"], [sft_reward, grpo_reward], color=["tab:blue", "tab:orange"])
    plt.ylabel("Average reward")
    plt.title("SFT vs GRPO Quick Evaluation")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(compare_plot, dpi=150)
    plt.close()

    compare_path = output_dir / "grpo_vs_sft_quick_comparison.json"
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "num_steps": global_step,
        "train_examples": len(examples),
        "history": history,
        "loss_plot": str(loss_plot),
        "reward_plot": str(reward_plot),
        "comparison_plot": str(compare_plot),
        "comparison_json": str(compare_path),
        "output_model_dir": str(out_model_dir),
    }
    metrics_path = output_dir / "grpo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved GRPO model to: {out_model_dir}")
    print(f"Loss curve: {loss_plot}")
    print(f"Reward curve: {reward_plot}")
    print(f"SFT vs GRPO quick comparison: {compare_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
