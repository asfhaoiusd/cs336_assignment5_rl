import json
import sys
from pathlib import Path
from typing import Callable, List

from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAPTER5 = REPO_ROOT / "chapter5"
sys.path.insert(0, str(CHAPTER5))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    output_file: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for i, output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]

        scores = reward_fn(generated_text, ground_truth)

        result_entry = {
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "scores": scores,
        }
        results.append(result_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    MODEL_PATH = str((CHAPTER5 / "models" / "Qwen2.5-Math-1.5B").resolve())

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    validation_file = CHAPTER5 / "MATH" / "validation.jsonl"
    with open(validation_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    prompts = [PROMPTS_TEMPLATE.format(question=item["problem"]) for item in data]
    ground_truths = [item["answer"] for item in data]

    llm = LLM(model=MODEL_PATH)
    try:
        evaluate_vllm(
            llm,
            r1_zero_reward_fn,
            prompts,
            sampling_params,
            ground_truths,
            "results.jsonl",
        )
    finally:
        # 避免进程退出时 EngineCore 子进程先结束，被监控线程误报为 “died unexpectedly”
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
