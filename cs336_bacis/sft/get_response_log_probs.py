import torch
import torch.nn.functional as F
from compute_entropy import compute_entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    outputs = model(input_ids = input_ids, return_dict = True)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)
    probs = F.log_softmax(logits, dim=-1)
    labels = labels.unsqueeze(-1) 
    #gather函数是用来根据索引从张量中提取值的，这里索引是labels，所以是根据labels的值从probs中提取值。
    log_probs = torch.gather(probs, dim=-1, index=labels) 
    log_probs = log_probs.squeeze(-1) 
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}