# Chat Template问题修复说明

## 问题描述

### 症状
- `generate` 只生成 "!" 号（token ID 0）
- `infer` 返回的概率全部为 0
- 使用 `run_mindformers.py` 加载同一模型可以正常输出

### 根本原因

**Qwen3是一个Chat模型，需要特定的对话格式才能正常工作。**

在原始实现中，我们直接将用户输入传给模型：
```python
# ❌ 错误的方式
prompt = "你好，请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors='np')
```

这导致模型无法理解输入，因为：
1. Qwen3训练时使用了特定的chat template格式
2. 模型期望输入包含角色标记（system, user, assistant）
3. 没有正确的格式，模型会输出异常token

### 正确的格式

Qwen模型需要的chat template格式：
```
<|im_start|>system
<|im_end|>
<|im_start|>user
你好，请介绍一下你自己。
<|im_end|>
<|im_start|>assistant

```

## 解决方案

### 方案1: 使用 `apply_chat_template`（推荐）

```python
# ✅ 推荐的方式
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors=None
)
```

### 方案2: 手动添加chat template

```python
# ✅ 备用方式
formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(formatted_prompt, return_tensors='np')
```

## 修复内容

已在以下函数中添加chat template处理：

### `test_generate` 函数
- 检查tokenizer是否有 `apply_chat_template` 方法
- 优先使用 `apply_chat_template`
- 如果不可用，回退到手动添加chat template
- 添加详细的日志输出，方便调试

### `test_infer` 函数
- 与 `test_generate` 相同的处理逻辑
- 确保infer测试也使用正确的输入格式

## 对比

### 之前的输入
```
Token IDs: [108386, 37945, 109432, 107828, 1773]
对应文本: "你好，请介绍一下你自己。"
```

### 修复后的输入（预期）
```
Token IDs: [<|im_start|>, system, \n, <|im_end|>, \n, <|im_start|>, user, \n, 你, 好, ，, 请, 介, 绍, 一, 下, 你, 自, 己, 。, \n, <|im_end|>, \n, <|im_start|>, assistant, \n]
```

## 为什么 `run_mindformers.py` 可以正常工作？

在 `run_mindformers.py` 中：
1. 有 `_add_model_base_prompt()` 函数专门处理chat template
2. `build_shieldlm_prompt()` 调用它来格式化输入
3. 因此模型可以正常理解输入并生成响应

```python
def _add_model_base_prompt(ipt, model_base):
    """Wrap prompt by model-base specific chat template minimal support."""
    if model_base in ('qwen', 'internlm'):
        return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{ipt}\n<|im_end|>\n<|im_start|>assistant\n"
    # ...
```

## 其他模型的Chat Template

不同模型使用不同的chat template：

### Baichuan
```
<reserved_106>{prompt}<reserved_107>
```

### InternLM
```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n
```

### ChatGLM
```
[gMASK]sop<|user|> \n {prompt}<|assistant|> \n
```

## 验证步骤

1. 运行修复后的测试脚本：
```bash
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode both \
    --prompt "你好，请介绍一下你自己。" \
    --max_new_tokens 50
```

2. 检查输出：
   - 查看日志中是否显示 "✓ 使用tokenizer.apply_chat_template处理完成" 或 "手动添加chat template"
   - 查看 `input_ids` 的长度是否显著增加（应该从5增加到20+）
   - 查看生成的文本是否为有意义的回复（而不是"!!!"）
   - 查看infer的概率是否不全为0

3. 预期结果：
   - `generate` 应该生成正常的中文回复
   - `infer` 的概率应该有正常分布（top token概率 > 0）
   - 不再出现token ID 0或全"!"的输出

## 重要提示

⚠️ **所有Chat模型都需要使用正确的chat template！**

如果你在使用其他chat模型时遇到类似问题：
1. 首先检查是否正确应用了chat template
2. 查看模型文档或 `tokenizer_config.json` 中的 `chat_template` 字段
3. 使用 `tokenizer.apply_chat_template()` 或参考模型的官方示例

## 参考

- Qwen3官方文档: https://github.com/QwenLM/Qwen
- MindFormers chat template处理: `run_mindformers.py` 中的 `_add_model_base_prompt()`
- HuggingFace chat template: https://huggingface.co/docs/transformers/chat_templating

