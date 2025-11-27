# Qwen3-14B 直接加载测试脚本使用说明

## 概述

`test_qwen3_direct.py` 是一个直接使用 MindFormers Qwen3 模块的测试脚本，用于加载 Qwen3-14B 模型权重和分词器，并测试 `generate` 和 `infer` 功能。

**重要说明**：如果您已经安装了 mindformers，Qwen3 支持应该已经包含在内，只是可能缺少官方文档说明。您可以直接导入使用：
```python
from mindformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
```

## 快速验证

在使用主脚本之前，建议先运行验证脚本确认 Qwen3 支持：

```bash
python ShieldLM/verify_qwen3_support.py
```

如果验证通过，您就可以直接使用 Qwen3 模型了！

## 与 test_qwen3_mindformers.py 的区别

| 特性 | test_qwen3_direct.py | test_qwen3_mindformers.py |
|------|---------------------|--------------------------|
| 模型加载方式 | 直接导入 Qwen3ForCausalLM | 使用 AutoModel |
| 权重加载 | 显式加载 safetensors/ckpt | 依赖 AutoModel 自动加载 |
| 权重转换 | 支持 HuggingFace -> MindFormers | 期望已转换的权重 |
| 调试信息 | 详细的权重加载日志 | 一般日志 |
| 适用场景 | 调试权重加载问题 | 常规使用 |

## 前置要求

1. **MindFormers 源码**：确保 `mindformers` 文件夹包含 Qwen3 模块
   ```bash
   ls mindformers/mindformers/models/qwen3/
   # 应该看到：__init__.py, configuration_qwen3.py, modeling_qwen3.py 等
   ```

2. **模型权重**：准备 Qwen3-14B 的权重文件
   - 支持 **safetensors** 格式（HuggingFace 格式）
   - 支持 **ckpt** 格式（MindSpore 格式）

3. **分词器**：tokenizer.json 和相关配置文件

4. **配置文件**：MindFormers YAML 配置文件

## 目录结构示例

```
your_model_directory/
├── config.json                    # 模型配置
├── tokenizer.json                 # 分词器
├── tokenizer_config.json          # 分词器配置
├── model-00001-of-00008.safetensors  # 权重文件（safetensors）
├── model-00002-of-00008.safetensors
├── ...
└── model.safetensors.index.json   # 权重索引（可选）

your_config_directory/
└── predict_qwen3_14b.yaml         # MindFormers 配置文件
```

## 使用方法

### 基本用法

```bash
python ShieldLM/test_qwen3_direct.py \
    --config /path/to/predict_qwen3_14b.yaml \
    --model_dir /path/to/model_weights \
    --tokenizer_path /path/to/tokenizer \
    --test_mode both \
    --prompt "你好，请介绍一下你自己。"
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--config` | 是 | MindFormers YAML 配置文件路径 | - |
| `--model_dir` | 推荐 | 模型权重目录（包含 .safetensors 或 .ckpt） | None |
| `--tokenizer_path` | 可选 | 分词器目录，默认使用 model_dir | None |
| `--test_mode` | 否 | 测试模式：generate/infer/both | both |
| `--prompt` | 否 | 测试提示词 | "你好，请介绍一下你自己。" |
| `--max_new_tokens` | 否 | 最大生成 token 数 | 50 |
| `--use_past` | 否 | 是否使用增量推理（KV cache） | False |
| `--batch_size` | 否 | 批次大小 | 1 |

### 示例 1：仅测试 generate

```bash
python ShieldLM/test_qwen3_direct.py \
    --config configs/predict_qwen3_14b.yaml \
    --model_dir /data/models/Qwen3-14B \
    --test_mode generate \
    --max_new_tokens 100
```

### 示例 2：仅测试 infer

```bash
python ShieldLM/test_qwen3_direct.py \
    --config configs/predict_qwen3_14b.yaml \
    --model_dir /data/models/Qwen3-14B \
    --test_mode infer
```

### 示例 3：使用 KV cache 加速推理

```bash
python ShieldLM/test_qwen3_direct.py \
    --config configs/predict_qwen3_14b.yaml \
    --model_dir /data/models/Qwen3-14B \
    --use_past
```

## 关键功能

### 1. 自动权重格式检测

脚本会自动检测并加载以下格式的权重：
- **safetensors**：支持 HuggingFace 格式，自动转换权重名称
- **ckpt**：支持 MindSpore 原生格式

### 2. 权重名称自动转换

如果检测到 HuggingFace 格式的权重（如 `model.embed_tokens`），会自动调用 `model.convert_weight_dict()` 进行转换：

```
HuggingFace 格式              ->  MindFormers 格式
model.embed_tokens.           ->  embedding.word_embeddings.
.self_attn.q_proj.            ->  .self_attention.linear_q.
.self_attn.k_proj.            ->  .self_attention.linear_k.
.self_attn.v_proj.            ->  .self_attention.linear_v.
.self_attn.o_proj.            ->  .self_attention.linear_proj.
.mlp.gate_proj.               ->  .mlp.gating.
.mlp.down_proj.               ->  .mlp.linear_fc2.
.mlp.up_proj.                 ->  .mlp.hidden.
model.norm.                   ->  decoder.final_layernorm.
lm_head.                      ->  output_layer.
```

### 3. 权重加载验证

脚本会验证加载的权重：
- 检查是否有全零参数
- 检查是否有 NaN/Inf
- 显示关键层的统计信息

### 4. 模型架构检测

自动检测模型是 Legacy 架构还是 MCore 架构，并使用相应的 infer 接口：
- **Legacy 模型**：使用 `model.infer()`
- **MCore 模型**：使用 `model.infer_mcore()`

## 输出说明

### 成功的输出示例

```
============================================================
Qwen3-14B 直接加载测试
============================================================
配置文件: configs/predict_qwen3_14b.yaml
模型目录: /data/models/Qwen3-14B
测试模式: both
============================================================

============================================================
加载Qwen3模型
============================================================
✓ 设置 RUN_MODE=predict
...
✓ 模型实例创建成功
  - 模型类型: InferenceQwen3ForCausalLM

从目录加载权重: /data/models/Qwen3-14B
找到 8 个 safetensors 文件
  加载: model-00001-of-00008.safetensors
  ...
  检测到 HuggingFace 权重格式，执行名称转换...
  ✓ 权重名称转换完成
  ✓ 所有权重加载成功
✓ safetensors 权重加载完成
...

============================================================
测试 generate 接口
============================================================
...
✓ generate调用成功

------------------------------------------------------------
生成结果:
------------------------------------------------------------
你好！我是通义千问，由阿里云开发的AI助手...
------------------------------------------------------------
```

## 常见问题排查

### 1. 找不到 qwen3 模块

**错误**：`ImportError: cannot import name 'Qwen3ForCausalLM'`

**解决方案**：
```bash
# 确认 qwen3 模块存在
ls mindformers/mindformers/models/qwen3/

# 确认脚本正确添加了 mindformers 路径
# 检查输出中是否有：✓ 添加本地mindformers路径
```

### 2. 权重未加载

**症状**：
```
⚠️ 发现 X 个全零参数（可能未正确加载）
```

**可能原因**：
1. model_dir 路径不正确
2. 权重文件格式不支持
3. 权重名称不匹配

**解决方案**：
```bash
# 检查权重文件
ls -lh /path/to/model_dir/*.safetensors
ls -lh /path/to/model_dir/*.ckpt

# 查看详细的加载日志，检查是否有 "not_loaded" 提示
```

### 3. infer 失败：block_tables/slot_mapping 未准备好

**错误**：`MCore 模型需要 block_tables 和 slot_mapping，但准备失败`

**解决方案**：
1. 检查配置文件中的 `seq_length` 和 `num_blocks` 设置
2. 确保配置文件正确设置了 `use_past=True`（如果需要）
3. 尝试减少输入序列长度

### 4. 内存不足

**错误**：OOM (Out of Memory)

**解决方案**：
1. 减小 `batch_size`
2. 减小 `max_new_tokens`
3. 使用更大内存的设备
4. 检查是否有内存泄漏

## 环境变量

脚本会自动设置以下环境变量：

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `RUN_MODE` | `predict` | 强制使用推理模型而非训练模型 |
| `DEVICE_ID` | 从环境读取或默认 0 | 指定使用的设备 ID |

您可以在运行前手动设置：

```bash
export DEVICE_ID=0
export RUN_MODE=predict
python ShieldLM/test_qwen3_direct.py ...
```

## 与原始测试脚本的对比

如果 `test_qwen3_direct.py` 能成功加载权重，但 `test_qwen3_mindformers.py` 失败，可能的原因：

1. **AutoModel 路径问题**：AutoModel.from_pretrained 可能无法正确找到权重
2. **配置文件问题**：YAML 配置中的 `pretrained_model_dir` 设置不正确
3. **权重格式问题**：AutoModel 期望特定的文件结构

建议：
1. 先使用 `test_qwen3_direct.py` 验证权重能正常加载
2. 然后检查 `test_qwen3_mindformers.py` 中的路径配置
3. 确保 YAML 配置文件正确指向权重目录

## 进一步的改进

如果脚本运行成功，您可以：

1. **保存转换后的权重**：将 HuggingFace 格式转换为 MindSpore 格式保存
2. **修改配置文件**：更新 YAML 配置以正确指向权重路径
3. **集成到原始脚本**：将成功的加载逻辑集成回 `test_qwen3_mindformers.py`

## 技术支持

如果遇到问题，请提供：
1. 完整的错误堆栈
2. 模型目录结构（`ls -lh /path/to/model_dir`）
3. 配置文件内容（特别是 model 部分）
4. MindFormers 版本信息

