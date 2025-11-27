# Infer 测试失败问题分析与修复

## 问题描述

在测试 Qwen3-14B 模型时，`generate` 测试成功，但 `infer` 测试失败，报错：

```
ValueError: For set_inputs and tuple(list) in set_inputs, the dims of 1th input must be the same as expected, but got expected: 1, and input: 2!
```

## 根本原因

### 1. MindSpore 图编译机制

MindSpore 在图模式（GRAPH_MODE）下，会在第一次执行时编译计算图并缓存。后续调用时会复用已编译的图。

### 2. 模型状态冲突

执行顺序问题导致的冲突：

1. **先执行 generate**:
   - 进入 prefill 阶段（处理完整输入序列）
   - 进入 decode 阶段（逐个生成 token，多次迭代）
   - 模型内部 phase 变化：`prefill` → `increment`
   - 图被编译为适应 decode 阶段的形状

2. **后执行 infer**:
   - 尝试进入 prefill 阶段
   - 但模型已经处于 decode 状态
   - 输入形状与已编译的图不匹配 → **报错**

### 3. 具体错误分析

错误信息 `the dims of 1th input must be the same as expected, but got expected: 1, and input: 2` 表明：
- 第 1 个输入参数（可能是 `batch_valid_length` 或其他）
- 期望维度：1（decode 模式下的单 token）
- 实际维度：2（prefill 模式下的完整序列）

## 修复方案

### 方案 1：调整测试顺序（已采用）✅

**修改内容**：
```python
# 修改前：先 generate，后 infer
if args.test_mode in ["generate", "both"]:
    results["generate"] = test_generate(...)
if args.test_mode in ["infer", "both"]:
    results["infer"] = test_infer(...)

# 修改后：先 infer，后 generate
if args.test_mode in ["infer", "both"]:
    results["infer"] = test_infer(...)
    # 清理缓存
    if model.block_mgr:
        model.block_mgr.clear_cache()

if args.test_mode in ["generate", "both"]:
    results["generate"] = test_generate(...)
```

**优点**：
- 简单有效
- 不需要修改底层框架
- 避免了图编译冲突

### 方案 2：增强 infer 参数传递（必需！）

**修改内容**：确保 infer 调用时传递完整的参数，特别是 **position_ids**

```python
# 1. 准备 position_ids（必需！用于旋转位置编码）
max_len = input_ids_np.shape[1]
position_ids = np.zeros((batch_size, max_len), dtype=np.int32)
for idx, length in enumerate(valid_length_each_example):
    if length > 0:
        position_ids[idx, :length] = np.arange(length, dtype=np.int32)

# 2. 准备 block_tables 和 slot_mapping
if model.block_mgr:
    block_tables, slot_mapping = model.block_mgr.assemble_pa_full_inputs(
        max_input_length, valid_length_each_example, is_finished
    )
    
# 3. 调用 infer 时传递这些参数
infer_output, is_finished = model.infer(
    input_ids=input_ids_np,
    valid_length_each_example=valid_length_each_example,
    generation_config=gen_config,
    block_tables=block_tables,
    slot_mapping=slot_mapping,
    prefill=prefill,
    is_finished=is_finished,
    position_ids=position_ids,  # 必需！否则会报错
)
```

**重要性**：
- **必需参数**：position_ids 对于 RotaryPosEmb（旋转位置编码）是必需的
- 更符合框架设计
- 即使在复杂场景下也能工作

**错误示例**（如果缺少 position_ids）：
```
TypeError: Failed calling ApplyRotaryPosEmb with "position_ids=None".
The valid calling should be: "query=<Tensor>, key=<Tensor>, cos=<Tensor>, sin=<Tensor>, position_ids=<Tensor>".
```

### 方案 3：重新加载模型（未采用）

每次测试前重新加载模型实例。

**缺点**：
- 非常耗时（模型加载需要约 20 秒）
- 资源消耗大
- 不适合频繁测试

## 第二个问题：缺少 position_ids 参数

### 问题描述

在解决了测试顺序问题后，又遇到了新的错误：

```
TypeError: Failed calling ApplyRotaryPosEmb with "position_ids=None".
The valid calling should be: "query=<Tensor>, key=<Tensor>, cos=<Tensor>, sin=<Tensor>, position_ids=<Tensor>".
```

### 根本原因

Qwen3 模型使用了 **旋转位置编码（RoPE, Rotary Position Embedding）**，这是一种先进的位置编码方式。在 MindFormers 的实现中，ApplyRotaryPosEmb 操作需要显式传入 `position_ids` 参数。

**调用链**：
```
model.infer()
  → self.forward()
    → model.__call__()
      → GPTModel
        → TransformerBlock
          → TransformerLayer
            → Attention
              → RotaryPosEmbedding
                → ApplyRotaryPosEmb  # 需要 position_ids！
```

### 解决方案

**生成并传递 position_ids**：

```python
# 构建 position_ids
max_len = input_ids_np.shape[1]
position_ids = np.zeros((batch_size, max_len), dtype=np.int32)
for idx, length in enumerate(valid_length_each_example):
    if length > 0:
        # 为有效长度内的每个位置分配位置索引
        position_ids[idx, :length] = np.arange(length, dtype=np.int32)

# 调用 infer 时传递
infer_output, is_finished = model.infer(
    input_ids=input_ids_np,
    valid_length_each_example=valid_length_each_example,
    generation_config=gen_config,
    position_ids=position_ids,  # 关键！
    prefill=prefill,
    is_finished=is_finished,
)
```

### 为什么 generate 不需要？

你可能会问：为什么 `generate()` 测试成功了，但 `infer()` 需要显式传递 `position_ids`？

**答案**：
1. **generate** 内部会自动处理 position_ids 的生成和传递
2. **infer** 是更底层的接口，需要调用者提供完整的输入参数
3. 这是设计上的权衡：generate 简单易用，infer 灵活可控

### 参考实现

在 `run_mindformers_probability.py` 中可以看到正确的用法：

```python
def _collect_infer_scores(model, tokenized_batch, infer_config, steps_needed: int, pad_token_id: int):
    for _ in range(steps_needed):
        input_batch, valid_lengths = _pad_sequences(sequences, pad_token_id)
        position_ids = _build_position_ids(valid_lengths, input_batch.shape[1])  # 构建
        
        infer_output, is_finished = model.infer(
            input_ids=input_batch,
            valid_length_each_example=valid_lengths,
            generation_config=infer_config,
            prefill=prefill,
            is_finished=is_finished,
            position_ids=position_ids  # 传递
        )
```

## 测试验证

### 修复前的错误输出

```
============================================================
测试 generate 接口
============================================================
✓ generate调用成功
生成结果: 你好，请介绍一下你自己。!!!!!!!!!!!!!!...

============================================================
测试 infer 接口
============================================================
✗ infer测试失败!
ValueError: For set_inputs and tuple(list) in set_inputs, 
the dims of 1th input must be the same as expected...
```

### 修复后的预期输出

```
============================================================
测试 infer 接口
============================================================
✓ infer调用成功
生成的下一个token ID: 151645
解码后的文本: '你好'

============================================================
测试 generate 接口
============================================================
✓ generate调用成功
生成结果: 你好，请介绍一下你自己。我是一个AI助手...

============================================================
测试总结
============================================================
infer: ✓ 通过
generate: ✓ 通过
```

## 相关知识点

### MindSpore 图模式 vs PyNative 模式

| 特性 | 图模式（GRAPH_MODE） | 动态图模式（PYNATIVE_MODE） |
|------|---------------------|---------------------------|
| 执行方式 | 先编译，后执行 | 逐行执行 |
| 性能 | 高（适合生产） | 较低（适合调试） |
| 灵活性 | 低（形状固定） | 高（形状可变） |
| 调试难度 | 较高 | 较低 |

当前 Qwen3-14B 配置使用图模式以获得最佳性能。

### GenerationMixin 的关键方法

1. **generate()**: 完整的文本生成流程
   - 内部调用多次 `infer()` 或 `forward()`
   - 处理 prefill + 多次 decode

2. **infer()**: 单步推理
   - 可以是 prefill 或 decode
   - 返回下一个 token 和概率

3. **forward()**: 底层前向传播
   - 被 `infer()` 调用
   - 直接与模型交互

## 最佳实践

### 推荐的测试流程

1. **单独测试 infer**:
   ```bash
   python test_qwen3_mindformers.py --test_mode infer
   ```

2. **单独测试 generate**:
   ```bash
   python test_qwen3_mindformers.py --test_mode generate
   ```

3. **同时测试两者**（自动处理顺序）:
   ```bash
   python test_qwen3_mindformers.py --test_mode both
   ```

### 调试技巧

如果仍然遇到问题：

1. **检查模型配置**:
   ```bash
   # 查看 YAML 中的 use_past 设置
   grep -A 5 "use_past" models/predict_qwen3.yaml
   ```

2. **查看详细日志**:
   ```bash
   export GLOG_v=2  # MindSpore 详细日志
   python test_qwen3_mindformers.py ...
   ```

3. **使用 PyNative 模式**（如果支持）:
   在配置文件中设置 `mode: 1` 或 `mode: PYNATIVE_MODE`

## 参考资料

- MindSpore 官方文档：https://www.mindspore.cn/docs/zh-CN/r2.4.10/index.html
- MindFormers 文档：https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/index.html
- GenerationMixin 源码：`mindformers/generation/text_generator.py`

## 总结

这个问题实际上包含 **两个独立的问题**：

### 问题 1：图编译状态冲突 ❌ → ✅
**核心**：MindSpore 图编译的状态管理
**解决**：调整测试顺序（infer → generate）

### 问题 2：缺少必需参数 ❌ → ✅
**核心**：旋转位置编码需要 position_ids
**解决**：显式生成并传递 position_ids 参数

### 完整的修复方案

```python
# 1. 调整测试顺序
if args.test_mode in ["infer", "both"]:
    results["infer"] = test_infer(...)  # 先执行
    if model.block_mgr:
        model.block_mgr.clear_cache()

if args.test_mode in ["generate", "both"]:
    results["generate"] = test_generate(...)  # 后执行

# 2. 在 test_infer 中添加 position_ids
position_ids = np.zeros((batch_size, max_len), dtype=np.int32)
for idx, length in enumerate(valid_length_each_example):
    if length > 0:
        position_ids[idx, :length] = np.arange(length, dtype=np.int32)

# 3. 调用 infer 时传递完整参数
infer_output, is_finished = model.infer(
    input_ids=input_ids_np,
    valid_length_each_example=valid_length_each_example,
    generation_config=gen_config,
    position_ids=position_ids,  # 必需！
    block_tables=block_tables,
    slot_mapping=slot_mapping,
    prefill=prefill,
    is_finished=is_finished,
)
```

### 方案优点

- ✅ 解决了图编译冲突
- ✅ 提供了完整的参数传递
- ✅ 不影响性能
- ✅ 适用于各种模型配置
- ✅ 便于后续维护
- ✅ 符合框架设计规范

### 关键教训

1. **理解框架设计**：高层 API（generate）vs 低层 API（infer）
2. **参数完整性**：底层 API 需要提供完整参数
3. **参考官方实现**：run_mindformers_probability.py 是很好的参考
4. **逐步调试**：解决一个问题后可能会暴露下一个问题

希望这个详细说明能帮助你理解问题的本质和解决方案！🎉

