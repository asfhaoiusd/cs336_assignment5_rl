import torch

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> torch.Tensor:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clip(ratio, min = 1.0 - cliprange, max = 1.0 + cliprange)
    loss = -torch.mininum(ratio * advantages, clipped_ratio * advantages)
    # 计算被clip 掉的次数占总次数的比例
    was_clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    clip_fraction = was_clipped.float().mean()
    return loss, {"clip_fraction": clip_fraction}     