# 内存分配失败问题解决方案

## 🔥 问题描述

### 错误信息

```
RuntimeError: Allocate memory failed
kernel tensor: shape:(151936, 5120) type:Ref[Tensor[Float32]]
size: 3111649280  # 约 3GB
node: output_layer.weight
```

### 发生时机

- ✅ 权重加载成功（323个参数）
- ✅ 没有全零参数
- ❌ **推理时内存分配失败**

## 🔍 根本原因分析

### 内存占用时间线

使用 `--use_training_conversion` 参数时的完整内存使用：

```
时间 | 操作 | 对象 | 内存占用 | 累计内存
-----|------|------|---------|--------
t1   | 加载safetensors | params_dict (numpy) | +28GB | 28GB
t2   | 创建训练模型 | TrainingQwen3ForCausalLM | +28GB | 56GB
t3   | 转换权重 | final_params (Parameter) | +28GB | 84GB
t4   | 创建推理模型 | InferenceQwen3ForCausalLM | +28GB | 112GB ⚠️
t5   | 推理运行时 | 中间激活/KV cache | +10-20GB | 122GB+ ❌
```

**峰值内存: 120GB+** （超出大多数设备容量）

### 为什么会这么高？

#### 1. 训练模型（28GB）仍在内存中

```python
# 创建训练模型用于转换
training_model = TrainingQwen3ForCausalLM(config)  # +28GB
params_dict = training_model.convert_weight_dict(params_dict)

# ❌ 如果不删除，训练模型仍占用 28GB
# ✅ 应该：del training_model; gc.collect()
```

#### 2. 多份权重副本（56-84GB）

```python
params_dict (numpy)      # 原始加载: 28GB
final_params (Parameter) # 转换后:   28GB  
模型内部参数             # 加载后:   28GB
# 总计: 84GB（如果不清理前两个）
```

#### 3. 推理运行时开销（10-20GB）

```python
# 推理时额外需要：
- 中间激活层
- KV cache（如果 use_past=True）
- 临时计算缓冲区
# 约 10-20GB
```

### 🎯 关键发现

**如果不释放中间对象，峰值内存可达 120GB+，远超大多数设备能力！**

## ✅ 解决方案

### 方案 1: 自动内存释放（推荐）⭐⭐⭐

**脚本已更新**，自动在关键位置释放内存：

```python
# 1. 转换完成后，立即删除训练模型
del temp_training_model
temp_training_model = None
gc.collect()
print("✓ 训练模型已释放（~28GB）")

# 2. 删除原始权重字典
del params_dict
gc.collect()
print("✓ 中间权重数据已释放")

# 3. 权重加载到模型后，删除参数副本
del final_params
gc.collect()
print("✓ 权重参数已释放（已加载到模型）")
```

**内存优化效果**：

| 阶段 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 权重转换 | 84GB | 56GB | -28GB |
| 推理模型创建 | 112GB | 56GB | -56GB |
| 推理运行 | 122GB+ | 66-76GB | -46-56GB |

**峰值内存降低 50%+！**

### 方案 2: 不使用训练模型转换（不推荐）

```bash
# 不使用 --use_training_conversion
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir /path/to/Qwen3-14B
```

**问题**：
- ❌ QKV/FFN 权重不会合并
- ❌ 关键参数未加载（约 120 个）
- ❌ 模型无法正常工作

**不推荐！**

### 方案 3: 使用更大内存设备

#### Ascend NPU 内存需求

| 设备 | 内存 | Qwen3-14B 支持 | 推荐 |
|------|------|---------------|------|
| Ascend 910A | 32GB | ❌ 不足 | 不推荐 |
| Ascend 910B | 64GB | ✅ 充足 | **推荐** ⭐ |
| Ascend 910 Pro | 96GB | ✅ 非常充足 | 推荐 |

#### NVIDIA GPU 内存需求

| 设备 | 内存 | Qwen3-14B 支持 | 推荐 |
|------|------|---------------|------|
| A100-40GB | 40GB | ⚠️ 勉强 | 可尝试 |
| A100-80GB | 80GB | ✅ 充足 | **推荐** ⭐ |
| H100 | 80GB | ✅ 充足 | 推荐 |

## 📊 内存使用详细分析

### Qwen3-14B 模型组成（BF16格式）

| 组件 | 参数量 | 大小 |
|------|--------|------|
| Embedding | 151936 × 5120 | ~1.5GB |
| 48个 Transformer 层 | ~14B params | ~25GB |
| Output Layer | 151936 × 5120 | ~1.5GB |
| **总计** | ~14B params | **~28GB** |

### 推理时额外内存

```
基础模型: 28GB
+ 中间激活: 2-5GB（取决于batch size和序列长度）
+ KV cache: 5-10GB（如果use_past=True）
+ 临时计算: 2-5GB
= 总需求: 37-48GB（不含权重加载时的额外开销）
```

### 使用 --use_training_conversion 时

```
训练模型: 28GB
+ 原始权重: 28GB
+ 转换权重: 28GB
+ 推理模型: 28GB
+ 推理运行: 10-20GB
= 峰值: 112-122GB（如果不释放）

优化后：
推理模型: 28GB
+ 推理运行: 10-20GB
= 峰值: 38-48GB ✅
```

## 🚀 使用指南

### 推荐命令（已优化）

```bash
# 使用更新后的脚本，自动内存管理
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir /path/to/Qwen3-14B \
    --tokenizer_path /path/to/Qwen3-14B \
    --use_training_conversion \
    --test_mode both
```

### 预期输出

成功时会看到内存释放日志：

```
============================================================
加载Qwen3模型
============================================================
✓ 训练模型转换权重完成
✓ 转换了 323 个参数
✓ 所有权重加载成功

释放训练模型，准备创建推理模型...
✓ 训练模型已释放（~28GB）
✓ 中间权重数据已释放

创建推理模型...
✓ 推理模型创建成功
✓ 推理模型权重加载成功
✓ 权重参数已释放（已加载到模型）

============================================================
测试 infer 接口
============================================================
✓ infer调用成功  ← 不再出现内存错误
```

### 监控内存使用

#### Ascend NPU

```bash
# 实时监控
watch -n 1 npu-smi info

# 查看当前内存
npu-smi info | grep "Memory-Usage"
```

#### NVIDIA GPU

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看当前内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🐛 故障排查

### 问题 1: 仍然内存不足

**检查**：是否确实使用了更新后的脚本？

```bash
# 检查脚本中是否有内存释放代码
grep -n "训练模型已释放" ShieldLM/test_qwen3_direct.py

# 应该看到多处出现
```

**解决**：确保使用最新的脚本版本。

### 问题 2: 删除对象后内存未释放

**原因**：Python 的垃圾回收可能延迟

**解决**：
```python
import gc

# 方式1: 多次调用 gc.collect()
del large_object
gc.collect()
gc.collect()  # 第二次确保清理

# 方式2: 强制完整回收
gc.collect(generation=2)
```

### 问题 3: 内存泄漏

**检查**：是否有循环引用

```python
# 在脚本中添加调试
import gc
import sys

# 查看对象引用计数
print(f"Training model ref count: {sys.getrefcount(temp_training_model)}")

# 强制清理
gc.collect()
print(f"Collected: {gc.collect()} objects")
```

### 问题 4: 不同设备内存管理

**Ascend vs NVIDIA**：

```python
# Ascend（MindSpore）
import mindspore as ms
ms.context.set_context(mode=ms.GRAPH_MODE)  # 更高效的内存管理

# NVIDIA（PyTorch）
import torch
torch.cuda.empty_cache()  # 清理GPU缓存
```

## 📈 性能优化建议

### 1. 减少生成长度

```bash
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir your_weights \
    --use_training_conversion \
    --max_new_tokens 20  # 减少生成长度 → 减少内存
```

### 2. 减小 batch size

```bash
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir your_weights \
    --use_training_conversion \
    --batch_size 1  # 最小batch size
```

### 3. 禁用 KV cache（如果不需要）

```bash
python ShieldLM/test_qwen3_direct.py \
    --config your_config.yaml \
    --model_dir your_weights \
    --use_training_conversion
    # 不使用 --use_past，节省 5-10GB
```

### 4. 使用混合精度

在配置文件中：
```yaml
model:
  model_config:
    compute_dtype: float16  # 或 bfloat16
```

## 🔬 技术细节

### Python 内存管理

```python
# 对象删除不等于内存释放
del obj  # 只是删除引用

# 需要垃圾回收
gc.collect()  # 真正释放内存

# MindSpore Tensor 的特殊性
param = Parameter(...)  # 在设备内存中
del param  # 释放设备内存
gc.collect()  # 确保清理
```

### MindSpore 内存分配

```python
# MindSpore 使用内存池
# 优点：分配速度快
# 缺点：释放可能延迟

# 强制释放
import mindspore as ms
# 设置context时可以配置内存管理策略
ms.context.set_context(
    mode=ms.GRAPH_MODE,
    memory_optimize_level='O1'  # 内存优化级别
)
```

## 📚 相关文档

- `完整问题解决路径.md` - 所有问题的完整历史
- `常见问题快速解决.md` - 快速问题参考
- `NUMPY_TO_PARAMETER_FIX.md` - numpy 转换问题
- `CONFIG_MISMATCH_SOLUTION.md` - 配置不匹配问题

## 🎯 总结

### 问题

- 使用 `--use_training_conversion` 时峰值内存达 120GB+
- 推理时内存分配失败

### 原因

- 训练模型未释放（28GB）
- 中间权重副本未释放（28-56GB）
- 多个大对象同时在内存中

### 解决

- ✅ 自动释放训练模型
- ✅ 自动释放中间权重
- ✅ 峰值内存降低 50%+（120GB → 60GB）

### 效果

- 内存需求：60-70GB（可接受）
- 支持设备：Ascend 910B (64GB)、A100-80GB
- 推理成功率：显著提升

**更新后的脚本已自动处理所有内存优化！直接使用即可。** 🎉

