# Assignment5-RL: SFT + GRPO + vLLM Eval

这个项目用于在 MATH 数据上完成一个简化的 RL 对齐流程：

1. SFT 微调（监督学习）
2. GRPO 训练（基于奖励函数优化）
3. vLLM / 脚本评测与结果可视化

> 备注：目录名 `cs336_bacis` 是历史拼写，实际就是本项目核心代码目录。

## 项目结构

- `cs336_bacis/sft/`：SFT 训练代码（如 `sft_train.py`）
- `cs336_bacis/grpo/`：GRPO 训练代码（`train_grpo.py`）
- `cs336_bacis/vllm/`：vLLM 评测脚本（`run_evaluate_llm.py`）
- `chapter5/cs336_alignment/`：奖励函数与判题逻辑（`r1_zero_reward_fn`）
- `chapter5/MATH/`：训练/验证数据（`train.jsonl`、`validation.jsonl`）
- `pretrain_model/`：SFT 初始化模型或预训练模型目录
- `rl_model/`：GRPO 训练输出模型与图表
- `results.jsonl`：评测结果输出文件

## 环境要求

- Python `>=3.12`
- 建议使用 GPU（NVIDIA + CUDA）
- 关键依赖：`torch`、`transformers`、`vllm`、`matplotlib`

如果你已在当前项目 `.venv` 中安装依赖，可直接运行下面命令。

## 快速运行

在项目根目录执行：

```bash
python /root/autodl-fs/assignment5-rl/cs336_bacis/grpo/train_grpo.py
```

训练默认会：

- 从 `pretrain_model/` 读取 SFT 模型
- 使用 `chapter5/MATH/` 数据
- 输出到 `rl_model/`
- 生成以下文件：
  - `grpo_loss_curve.png`
  - `grpo_reward_curve.png`
  - `grpo_vs_sft_reward_comparison.png`
  - `grpo_vs_sft_quick_comparison.json`
  - `grpo_metrics.json`

## 常用参数（GRPO）

```bash
python /root/autodl-fs/assignment5-rl/cs336_bacis/grpo/train_grpo.py \
  --max-examples 128 \
  --epochs 1 \
  --prompt-batch-size 2 \
  --group-size 2 \
  --max-new-tokens 96 \
  --log-every 10
```

参数说明（重点）：

- `--prompt-batch-size`：每个 step 采样多少个题目
- `--group-size`：每个题目采样多少个回答（rollouts）
- `--max-new-tokens`：每条回答最大生成长度
- `--log-every`：每多少 step 打印一次详细日志

每个优化 step 的样本规模约为：

`prompt_batch_size * group_size`

## 为什么 SFT 对比 GRPO 奖励低很多

这是这个任务里很常见的现象，核心原因是“训练目标不一样”：

1. SFT 优化的是“模仿数据分布”的 token 似然，不直接优化奖励函数。
2. GRPO 直接用 `r1_zero_reward_fn` 的反馈更新策略，目标就是提高该奖励。
3. 奖励函数较严格（答案格式、等价性、标签规范），SFT 采样输出容易拿 0 分。
4. 评测若使用采样解码（`temperature/top_p`），SFT 波动更大，更容易偏离标准答案。
5. 你的 quick eval 样本量有限时，均值看起来会更“接近 0”。

所以图里“GRPO reward 明显高于 SFT reward”通常不是异常，而是目标函数对齐后的预期结果。

## 训练慢时的优化建议

优先按下面顺序调：

1. 降低 `--max-new-tokens`（例如 256 -> 96）
2. 降低 `--group-size`（例如 4 -> 2）
3. 调小 `--max-examples` 做快速迭代
4. 增大 `--log-every` 降低日志刷新开销

## 排查建议

- 如果看到 `loss` 几乎总是 `0.0000`：
  - 先看 `reward` 是否几乎全 0（常见）
  - 你打印精度是 4 位，小量值会显示成 0
- 如果报 `dtype is not JSON serializable`：
  - 已在当前代码修复（`dtype` 仅传给模型，不传 tokenizer）

## License

见根目录 `LICENSE`。
