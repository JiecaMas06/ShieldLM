# Qwen3-14B MindFormers接口测试说明

## 概述

这个测试脚本用于测试Qwen3-14B模型在MindFormers框架下的`generate`和`infer`接口功能，方便调试和验证模型推理能力。

## 文件说明

- `test_qwen3_mindformers.py`: 主测试脚本
- `test_qwen3_mindformers.sh`: 便捷运行脚本
- `TEST_QWEN3_README.md`: 本说明文档

## 主要功能

### 1. infer 接口测试（优先执行）
测试模型的单步推理功能，适用于需要逐步控制生成过程的场景。

**测试内容:**
- 输入准备和有效长度计算
- prefill阶段推理
- 输出概率和下一个token的获取
- block_tables 和 slot_mapping 的准备

**注意**: infer 测试会优先执行，以避免与 generate 的图编译冲突。

### 2. generate 接口测试
测试模型的完整文本生成功能，类似于常规的推理调用。

**测试内容:**
- 输入提示词的tokenize
- GenerationConfig配置
- 调用`model.generate()`
- 输出解码和显示

**注意**: generate 会进行多轮迭代，会改变模型的内部状态。

## 使用方法

### 方式1: 使用YAML配置文件

```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode both \
    --prompt "你好，请介绍一下你自己。" \
    --max_new_tokens 50
```

### 方式2: 直接指定模型路径

```bash
python test_qwen3_mindformers.py \
    --model_path /path/to/qwen3-14b \
    --tokenizer_path /path/to/qwen3-14b \
    --test_mode both \
    --prompt "你好，请介绍一下你自己。" \
    --max_new_tokens 50
```

### 方式3: 使用shell脚本

```bash
# 先修改test_qwen3_mindformers.sh中的配置路径
./test_qwen3_mindformers.sh
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | None | MindFormers YAML配置文件路径 |
| `--model_path` | str | None | 模型目录路径（不使用YAML时） |
| `--tokenizer_path` | str | None | 分词器路径，默认使用模型路径 |
| `--test_mode` | str | both | 测试模式: generate/infer/both |
| `--prompt` | str | "你好，请介绍一下你自己。" | 测试提示词 |
| `--max_new_tokens` | int | 50 | 最大生成token数量 |
| `--trust_remote_code` | flag | False | 是否信任远程代码 |
| `--use_past` | flag | False | 是否使用增量推理(KV cache)，默认不使用 |
| `--batch_size` | int | 1 | 批次大小 |

## 测试模式

### 1. 仅测试generate
```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode generate \
    --prompt "你好"
```

### 2. 仅测试infer
```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode infer \
    --prompt "你好"
```

### 3. 同时测试两者（默认）
```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode both \
    --prompt "你好"
```

### 4. 启用增量推理（KV cache）
如果需要测试增量推理功能，添加`--use_past`参数：
```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode both \
    --prompt "你好" \
    --use_past
```

**注意**：默认不使用增量推理，以确保测试的稳定性和可调试性。

## 输出示例

### 成功输出示例

```
============================================================
Qwen3-14B MindFormers接口测试
============================================================
测试模式: both
最大生成token数: 50
使用增量推理: False
============================================================

============================================================
开始加载模型和分词器
============================================================

使用YAML配置文件: models/predict_qwen3.yaml
✓ MindSpore上下文构建完成
正在导入Qwen模型注册模块...
✓ 成功导入 mindformers.models.qwen2
✓ 成功导入 mindformers.models.qwen
✓ 成功导入 mindformers.models.qwen3

正在从配置文件加载模型...
✓ 模型加载成功
✓ 模型设置为评估模式

============================================================
测试 generate 接口
============================================================

输入提示词: 你好，请介绍一下你自己。

正在对提示词进行tokenize...
✓ tokenize完成，input_ids shape: (1, 15)

构建GenerationConfig...
✓ GenerationConfig配置完成

开始调用model.generate()...
✓ generate调用成功

------------------------------------------------------------
生成结果:
------------------------------------------------------------
你好！我是一个人工智能助手，专门设计来帮助回答问题...
------------------------------------------------------------

============================================================
测试 infer 接口
============================================================
...
✓ infer调用成功

============================================================
测试总结
============================================================
generate: ✓ 通过
infer: ✓ 通过
============================================================
```

### 错误输出示例

如果出现错误，脚本会详细显示：
- 错误类型
- 错误信息
- 完整的错误堆栈

这有助于快速定位和调试问题。

## 调试技巧

### 1. 检查模型加载
如果模型加载失败，检查：
- YAML配置文件路径是否正确
- 模型路径是否存在
- 设备ID设置是否正确（DEVICE_ID环境变量）

### 2. 检查分词器
如果分词器出错，尝试：
- 指定`--trust_remote_code`参数
- 确认tokenizer文件存在（tokenizer.json, tokenizer_config.json等）

### 3. 内存问题
如果遇到内存不足：
- 减少`--max_new_tokens`
- 设置`--batch_size 1`
- 检查是否正确配置了模型精度（bfloat16/float16）

### 4. 增量推理问题
默认情况下不使用增量推理。如果需要测试增量推理（KV cache）：
- 添加`--use_past`参数来启用
- 注意：增量推理可能需要额外的配置和内存

## 参考资料

- **MindFormers官方文档**: https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/index.html
- **GenerationMixin源码**: `mindformers源码/mindformers.generation.text_generator.py`
- **参考实现**: `run_mindformers_probability.py`

## 重要说明

### 测试顺序问题

**为什么先测试 infer，再测试 generate？**

在 MindSpore 的图模式下，模型在第一次调用时会进行图编译，并缓存编译结果。如果先运行 `generate`（多次迭代，包含 prefill 和 decode 阶段），再运行 `infer`（单次 prefill），会导致输入形状不匹配的错误：

```
ValueError: For set_inputs and tuple(list) in set_inputs, the dims of 1th input must be the same as expected
```

**解决方案**：脚本会自动按以下顺序执行测试：
1. 先执行 `infer` 测试（如果选择）
2. 清理 block table cache
3. 再执行 `generate` 测试（如果选择）

这样可以避免图编译冲突。

## 常见问题

### Q1: infer 测试失败怎么办？
A: 常见原因和解决方案：
- **图编译冲突**: 确保先测试 infer，再测试 generate（脚本已自动处理）
- **缺少依赖**: 确保已正确安装 MindFormers 1.3.2 和 MindSpore 2.4.10
- **use_past 设置**: 默认不使用增量推理（use_past=False），某些模型配置可能需要特殊处理

### Q2: 如何切换到其他Qwen模型？
A: 修改`--config`指向对应的YAML配置文件，或直接指定`--model_path`到其他模型目录。

### Q3: 如何测试英文提示词？
A: 使用`--prompt`参数指定英文文本即可，例如：
```bash
--prompt "Hello, please introduce yourself."
```

### Q4: 如何查看更多调试信息？
A: 脚本已经包含详细的日志输出。如需更多信息，可以修改MindSpore的日志级别。

### Q5: 支持批量测试吗？
A: 当前版本主要用于单个样本的接口测试。批量测试可以通过修改`--batch_size`参数，并在代码中传入多个提示词来实现。

## 贡献与反馈

如果在使用过程中发现问题或有改进建议，欢迎反馈。

