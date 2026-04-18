import torch
from typing import Callable

def compute_group_normalized_reward(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算每个 rollout 的 reward，并按组进行归一化。每 group_size 个连续的 rollouts 属于同一组，归一化时会使用该组内的 reward 进行中心化（减均值）和可选的标准差归一化。
    """
    # zip的作用是将两个列表对应位置的元素打包成一个元组，然后返回一个包含这些元组的列表。
    raw_rewards = torch.Tensor(
        [reward_fn(r, g)["reward"] for r, g in zip(rollout_responses, repeated_ground_truths)]
    )
    r = raw_rewards.reshape(-1, group_size)
    centered = r - r.mean(dim = -1, keepdim = True)

    if normalize_by_std:
        normalized = centered / (r.std(dim = -1, keepdim = True) + advantage_eps)
    else:
        normalized = centered

    return normalized.flatten(), raw_rewards, {}