# Project Overview: vLLM-Ascend Hidden States Extraction & Length Predictor

## 1. 项目目标 (Project Goal)
本项目旨在基于华为昇腾（Ascend NPU）环境下的 `vllm-ascend` 框架，实现大语言模型（LLM）推理过程中中间/隐层状态（Hidden States）的提取，并基于提取到的隐藏层特征训练/开发一个**长度预测器（Length Predictor）**，用于预测模型生成序列的长度。

---

## 2. 环境与技术栈 (Tech Stack & Environment)
- **硬件平台**: 华为昇腾 Ascend NPU (8卡910B4平台，单卡32GB显存)
- **推理框架**: `vllm-ascend` (适配昇腾的 vLLM 分支/插件)
- **深度学习框架**: PyTorch (Ascend NPU 支持版本 / `torch_npu`)
- **开发语言**: Python 3.10+
- **模型**: Qwen/Qwen3.5-27B

---

## 3. 核心任务清单 (Tasks & Workflows)

### 任务一：vLLM-Ascend 隐藏层状态提取 (Hidden States Extraction)
- **目标**: 修改或扩展 `vllm-ascend` 的 Model Runner / Forward 过程，使模型在推理输入 Prompt（Prefill 阶段）或生成 Token（Decode 阶段）时，能够导出特定层或最后一层的 Hidden States。
- **关键细节**:
 阅读参考资料中的相关内容，确认提取隐藏层状态的方法

### 任务二：数据集构建与特征处理 (Data & Feature Processing)
- **目标**: 准备用于训练长度预测器的数据集。
- **关键细节**:
  1. 收集代表性的 Prompt 输入及其对应的真实生成长度（Ground Truth Target Sequence Length）。
  2. 使用任务一开发的工具，提取这些 Prompt 在 Prefill 结束时的 Hidden States 作为输入特征 $X$。
  3. 将对应生成的 Token 长度作为目标标签 $y$。
  4. 以任务一和参考资料中的内容为准

### 任务三：长度预测器开发与训练 (Length Predictor Development & Training)
- **目标**: 设计并训练一个轻量级的回归/分类模型（Length Predictor），根据输入的隐藏层向量直接预测输出长度。
- **关键细节**:
  1. **架构选择**: 可先从简单 MLP (Multi-Layer Perceptron)、Linear Regressor 或轻量 Transformer Block 开始验证。
  2. **输入处理**: 可采用 Pooling（如 Mean/Last Token Pooling）将 `[seq_len, hidden_size]` 转化为固定维度的特征向量。
  3. **损失函数**: MSE Loss / L1 Loss（若作为回归任务）或 Cross Entropy / Focal Loss（若按长度区间分类）。
  4. **评估指标**: MAE (Mean Absolute Error), RMSE, $R^2$ Score, Kendall_tau 等。
- **交付物**:
  相关代码请在 `src/` 下进行开发

---

## 4. 代码规范与操作准则 (Coding Standards & Rules for Codex)
1. **昇腾适配注意事项**:
   - 涉及到 Tensor 操作时，确保设备类型正确（使用 `torch_npu` 或匹配 vLLM 内部的 device 管理）。
   - 避免在极高频的 Loop 内频繁进行 `tensor.cpu()` 或 `.numpy()` 操作，以防止 NPU/CPU 同步造成性能急剧下降。
2. **验证与测试**:
   - 编写任何新功能后，需确保有对应的单元测试或 Minimal Reproducible Example (MRE) 脚本，位于 `tests/` 目录下。

---

## 5. 参考资料
- [https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/feature_guide/speculative_decoding.html#_7]
- [https://github.com/vllm-project/vllm/pull/49811]
- [https://vllm.com.cn/blog/2026-03-30-extract-hidden-states]
- [https://docs.vllm.com.cn/en/latest/features/speculative_decoding/extract_hidden_states/]