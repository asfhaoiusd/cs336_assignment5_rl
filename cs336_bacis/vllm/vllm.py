import torch
from vllm import LLM, SamplingParams

def run_vllm(
    model_path: str,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[str]:
    llm = LLM(model=model_path)
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]