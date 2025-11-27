# ⚠️ 配置不匹配问题 - 解决方案

## 问题描述

运行时出现以下错误：

```
RuntimeError: For 'load_param_into_net', embedding.word_embeddings.weight in the argument 'net' 
should have the same shape as embedding.word_embeddings.weight in the argument 'parameter_dict'. 
But got its shape (151936, 4096) in the argument 'net' and shape (151936, 5120) in the argument 'parameter_dict'.
```

## 问题原因

**配置文件与权重文件不匹配**！

| 项目 | 配置文件 (YAML) | 权重文件 | 说明 |
|------|----------------|---------|------|
| `hidden_size` | 4096 | 5120 | ❌ 不匹配 |
| `vocab_size` | 151936 | 151936 | ✅ 匹配 |

您的配置文件设置了 `hidden_size=4096`，但权重文件实际是 `hidden_size=5120`（Qwen3-14B 的正确值）。

## 快速解决方案

### 方案 1: 使用权重目录的 config.json（推荐）⭐

**最简单的方法**：确保权重目录包含 `config.json` 文件。

```bash
# 检查权重目录
ls /path/to/Qwen3-14B/

# 应该看到：
# config.json                      ← 必须有这个文件！
# tokenizer.json
# tokenizer_config.json
# model-00001-of-00008.safetensors
# ...
```

如果有 `config.json`，脚本会**自动**从该文件加载正确的配置，无需手动修改！

```bash
# 运行时脚本会自动使用 config.json
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir /path/to/Qwen3-14B \
    --use_training_conversion
```

**预期输出**：
```
============================================================
检查权重目录的 config.json
============================================================
  ✓ 找到 config.json
  关键配置项:
    - hidden_size: 5120
    - num_hidden_layers: 48
    - num_attention_heads: 40
    - num_key_value_heads: 8
    - vocab_size: 151936
    - intermediate_size: 13824

============================================================
创建Qwen3Config
============================================================
尝试从权重目录加载 config.json: /path/to/Qwen3-14B
✓ 从权重目录的 config.json 加载配置成功
```

### 方案 2: 手动修改 YAML 配置文件

如果权重目录没有 `config.json`，需要手动修改 YAML 配置文件。

#### Qwen3-14B 的正确配置：

```yaml
model:
  model_config:
    type: Qwen3Config
    vocab_size: 151936
    hidden_size: 5120          # ⭐ 改为 5120
    num_hidden_layers: 48       # ⭐ 改为 48
    num_attention_heads: 40     # ⭐ 改为 40
    num_key_value_heads: 8      # ⭐ 改为 8
    intermediate_size: 13824    # ⭐ 改为 13824
    max_position_embeddings: 32768
    rms_norm_eps: 1.0e-6
    rope_theta: 1000000.0
    attention_bias: true
```

## 常见 Qwen3 模型配置

| 模型 | hidden_size | num_layers | num_heads | num_kv_heads | intermediate_size |
|------|-------------|------------|-----------|--------------|-------------------|
| Qwen3-0.5B | 896 | 24 | 14 | 2 | 4864 |
| Qwen3-1.8B | 2048 | 28 | 16 | 4 | 11008 |
| Qwen3-4B | 2560 | 40 | 20 | 4 | 13824 |
| Qwen3-7B | 3584 | 28 | 28 | 4 | 18944 |
| **Qwen3-14B** | **5120** | **48** | **40** | **8** | **13824** |
| Qwen3-32B | 5120 | 64 | 40 | 8 | 27648 |

## 脚本的自动修复功能

更新后的 `test_qwen3_direct.py` 包含以下自动检测和修复功能：

### 1. 自动检测 config.json

```python
# 脚本会自动检查权重目录
if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
    # 优先从 config.json 加载
    qwen3_config = Qwen3Config.from_pretrained(model_dir)
```

### 2. 权重验证

在加载权重前，脚本会验证配置是否匹配：

```
============================================================
验证权重与配置的匹配性
============================================================

从权重检测到的配置:
  - vocab_size: 151936
  - hidden_size: 5120

当前模型配置:
  - vocab_size: 151936
  - hidden_size: 4096

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
检测到配置不匹配问题:
  ❌ hidden_size 不匹配：权重=5120, 配置=4096
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
```

### 3. 自动修复

如果检测到不匹配，脚本会：
1. 显示详细的修复建议
2. 自动使用检测到的配置重新创建模型
3. 继续加载权重

```
尝试使用检测到的配置重新创建模型...
✓ 配置已更新:
  - vocab_size: 151936
  - hidden_size: 5120

重新创建模型实例...
✓ 模型实例重新创建成功
```

## 完整使用流程

### 步骤 1: 准备权重文件

确保权重目录包含以下文件：

```
Qwen3-14B/
├── config.json                      ← ⭐ 最重要！
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── model-00001-of-00008.safetensors
├── model-00002-of-00008.safetensors
├── ...
└── model.safetensors.index.json
```

**如何获取 config.json**：
- 如果从 HuggingFace 下载，应该会自动包含
- 如果缺失，可以从 HuggingFace 模型页面下载
- Qwen3-14B: https://huggingface.co/Qwen/Qwen2.5-14B/blob/main/config.json

### 步骤 2: 运行脚本

```bash
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir /path/to/Qwen3-14B \
    --tokenizer_path /path/to/Qwen3-14B \
    --use_training_conversion \
    --test_mode both
```

### 步骤 3: 验证输出

成功时应该看到：

```
✓ 找到 config.json
✓ 从权重目录的 config.json 加载配置成功
✓ 权重与配置匹配
✓ 所有权重加载成功
```

## 如果仍然失败

### 检查清单

1. **确认模型类型**
   ```bash
   # 查看权重文件大小
   du -sh /path/to/Qwen3-14B/*.safetensors
   
   # Qwen3-14B (BF16) 应该约 28GB
   # 如果差异很大，可能下载的是其他模型
   ```

2. **验证 config.json 内容**
   ```bash
   cat /path/to/Qwen3-14B/config.json | grep hidden_size
   # 应该显示: "hidden_size": 5120
   ```

3. **检查 YAML 配置**
   ```bash
   cat your_config.yaml | grep hidden_size
   # 如果没有 config.json，确保 YAML 中是 5120
   ```

### 手动创建 config.json

如果权重目录真的没有 `config.json`，可以手动创建：

```bash
cat > /path/to/Qwen3-14B/config.json << 'EOF'
{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 13824,
  "max_position_embeddings": 32768,
  "max_window_layers": 48,
  "model_type": "qwen2",
  "num_attention_heads": 40,
  "num_hidden_layers": 48,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936,
  "attention_bias": true
}
EOF
```

## 诊断命令

### 快速检查配置

```bash
# 检查权重目录
ls -lh /path/to/Qwen3-14B/

# 查看 config.json（如果存在）
python -c "import json; print(json.load(open('/path/to/Qwen3-14B/config.json'))['hidden_size'])"

# 应该输出: 5120
```

### 验证权重文件

```python
# 快速验证脚本
import os
from safetensors import safe_open

model_dir = "/path/to/Qwen3-14B"
safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]

if safetensors_files:
    first_file = os.path.join(model_dir, safetensors_files[0])
    with safe_open(first_file, framework="np") as f:
        # 查找 embedding 层
        for key in f.keys():
            if 'embed_tokens' in key:
                tensor = f.get_tensor(key)
                print(f"Key: {key}")
                print(f"Shape: {tensor.shape}")
                print(f"Hidden size: {tensor.shape[1]}")
                break
```

## 总结

**最佳实践**：
1. ✅ 确保权重目录有 `config.json`
2. ✅ 使用 `--use_training_conversion` 参数
3. ✅ 让脚本自动检测和修复配置不匹配

**不推荐**：
1. ❌ 手动猜测配置参数
2. ❌ 不检查就运行
3. ❌ 忽略配置不匹配的警告

按照以上步骤，配置不匹配问题应该可以完全解决！🎉

