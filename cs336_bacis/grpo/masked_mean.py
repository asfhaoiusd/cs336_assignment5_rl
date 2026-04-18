import torch

def masked_mean(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.
    """
    masked_sum = (tensor * mask).sum(dim = dim)
    masked_count = mask.sum(dim = dim)
    return masked_sum / masked_count    