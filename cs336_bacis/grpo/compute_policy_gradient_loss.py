import torch
from .compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from .compute_grpo_clip_loss import compute_grpo_clip_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> torch.Tensor:
    if loss_type == "no_baseline":
       loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
       metadata = {}
    if loss_type == "reinforce_with_baseline":
       loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
       metadata = {}
    if loss_type == "grpo_clip":
       loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
       raise ValueError(f"Invalid loss type: {loss_type}")
    return loss, metadata