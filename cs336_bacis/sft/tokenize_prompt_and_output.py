import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def _truncate_prompt_output_ids(
    prompt_ids: list[int],
    output_ids: list[int],
    max_total_tokens: int,
) -> tuple[list[int], list[int]]:
    """If prompt+output exceeds ``max_total_tokens``, keep all output tokens and trim prompt from the left."""
    if len(prompt_ids) + len(output_ids) <= max_total_tokens:
        return prompt_ids, output_ids
    out = output_ids
    if len(out) >= max_total_tokens:
        return [], out[-max_total_tokens:]
    budget_prompt = max_total_tokens - len(out)
    return prompt_ids[-budget_prompt:], out


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int | None = None,
) -> dict[str, Tensor]:
    full_sequence_ids: list[list[int]] = []
    padding_length = 0
    prompt_length_index: list[int] = []
    output_length_index: list[int] = []

    for i in range(len(prompt_strs)):
        prompt_ids_i = tokenizer.encode(prompt_strs[i])
        output_ids_i = tokenizer.encode(output_strs[i])
        if max_seq_length is not None:
            prompt_ids_i, output_ids_i = _truncate_prompt_output_ids(
                prompt_ids_i, output_ids_i, max_seq_length
            )
        full_sequence_ids_i = prompt_ids_i + output_ids_i
        full_sequence_ids.append(full_sequence_ids_i)

        prompt_length_index.append(len(prompt_ids_i))
        output_length_index.append(len(output_ids_i) + prompt_length_index[i])
        padding_length = max(padding_length, len(full_sequence_ids_i))

    batch_size = len(full_sequence_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    full = torch.full((batch_size, padding_length), pad_id, dtype=torch.long)
    for i, ids in enumerate(full_sequence_ids):
        full[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    full_sequence_ids_tensor = full

    raw_mask = torch.zeros((batch_size, padding_length), dtype=torch.bool)
    for i in range(len(prompt_strs)):
        raw_mask[i, : prompt_length_index[i]] = False
        raw_mask[i, prompt_length_index[i] : output_length_index[i]] = True
        raw_mask[i, output_length_index[i] :] = False

    return {
        "input_ids": full_sequence_ids_tensor[:, :-1],
        "labels": full_sequence_ids_tensor[:, 1:],
        "response_mask": raw_mask[:, 1:],
    }
