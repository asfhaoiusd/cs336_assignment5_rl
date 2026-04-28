import torch

def masked_normalize(
    tensor: torch.tensor,
    mask: torch.tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant