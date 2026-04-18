# CS336 笔记与作业（个人仓库）

Stanford **CS336: Language Modeling / Building LLMs** 相关学习记录与代码练习。课程主页可参考 [Stanford CS336 (Spring 2025)](https://stanford-cs336.github.io/spring2025/)。

本仓库用于个人备份与分享；若你克隆后运行对齐 / 推理相关脚本，请自行准备模型权重与课程配套依赖。

## 仓库里有什么

| 路径 | 说明 |
|------|------||
| `cs336_bacis/` | 本地辅助代码（含 `grpo/`、`vllm/` 等模块；目录名为历史拼写） |
| `cs336_spring2025_assignment5_*.pdf` | 课程作业 PDF（若你本地有则与仓库一起保留） |

说明：若你从本仓库的 **zip 源码包** 解压得到副本，可能不包含上游完整仓库中的全部章节目录；以你磁盘上的实际结构为准。

## 环境要求

- **Python**：`>= 3.12`（见 `.python-version` 与 `pyproject.toml`）
- **包管理**：推荐使用 [uv](https://github.com/astral-sh/uv)
- **GPU**：运行 `chapter5/hw3/evaluate_llm.py` 等脚本一般需要 **NVIDIA GPU** 与匹配的 **CUDA** 驱动；具体以 [vLLM](https://github.com/vllm-project/vllm) 官方要求为准

## 快速开始（根项目依赖）

根目录 `pyproject.toml` 目前仅声明了轻量依赖（如 `requests`），用于最小可运行示例：

```bash
cd cs336_note_and_hw
uv sync
uv run python main.py
```

## 第 5 章脚本（额外依赖）

例如 `chapter5/hw3/evaluate_llm.py` 会：

- 使用 **`vllm`** 加载模型并批量生成
- 从 **`cs336_alignment`** 导入评分函数（通常来自 Stanford CS336 **Assignment 5 alignment** 官方/课程代码包，需单独安装或把包放到 `PYTHONPATH` 可发现的路径）

脚本默认期望：

- 本地模型目录：`chapter5/models/Qwen2.5-Math-1.5B`（可在源码中改为 Hugging Face Hub 上的模型 ID）
- 数据文件：`chapter5/MATH/validation.jsonl`

请根据你的机器与课程环境，自行安装 **PyTorch（CUDA 版）**、**vLLM**、**transformers** 等，并保证 `cs336_alignment` 可被导入。

## 许可证

本仓库采用 **CC BY-NC-SA 4.0**（署名-非商业性-相同方式共享）。详见根目录 [LICENSE](LICENSE)。

## 致谢

课程内容版权归 Stanford CS336 课程方；本仓库中的笔记与练习代码为个人学习整理。
