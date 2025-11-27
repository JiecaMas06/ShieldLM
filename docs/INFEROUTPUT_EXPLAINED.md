# InferOutput 对象详解

## 📦 InferOutput 的结构

根据你的观察，`infer_mcore()` 返回的 `InferOutput` 对象包含三个字段：

```python
InferOutput(
    target_list=[token_id],           # 生成的 token ID 列表
    probs=array([[...]], dtype=...),  # 词表概率分布
    logits=None or array([...])       # 原始 logits（可选）
)
```

### 1. target_list (生成的 token)

**类型**: `list[int]`

**含义**: 模型预测的下一个（或多个）token 的 ID

**示例**:
```python
target_list = [0]        # 预测的 token ID 是 0
target_list = [151645]   # 预测的 token ID 是 151645 (可能是 <|im_end|>)
```

**用途**:
```python
next_token_id = target_list[0]
next_token_text = tokenizer.decode([next_token_id])
```

### 2. probs (概率分布)

**类型**: `numpy.ndarray`

**形状**: `(batch_size, vocab_size)` 或 `(vocab_size,)`

**含义**: **整个词表**的概率分布，每个位置对应一个 token 的概率

**示例**:
```python
probs = array([[0.001, 0.002, ..., 0.003]], dtype=float32)
# 形状: (1, 151936) - Qwen3 的词表大小是 151936

# probs[0][0] = token 0 的概率
# probs[0][1] = token 1 的概率
# ...
# probs[0][151935] = token 151935 的概率
```

**特点**:
- ✅ 概率之和为 1.0（经过 softmax）
- ✅ 包含所有可能 token 的概率
- ✅ 可以用于分析模型的不确定性
- ✅ 可以用于采样策略（top-k, top-p 等）

**用途**:
```python
import numpy as np

# 找出概率最高的 token
top_token = np.argmax(probs[0])

# 找出概率最高的前 5 个 tokens
top_5_indices = np.argsort(probs[0])[-5:][::-1]
top_5_probs = probs[0][top_5_indices]

# 计算熵（不确定性）
entropy = -np.sum(probs[0] * np.log(probs[0] + 1e-10))

# 采样（根据概率分布随机选择）
sampled_token = np.random.choice(len(probs[0]), p=probs[0])
```

### 3. logits (原始输出)

**类型**: `numpy.ndarray` 或 `None`

**形状**: `(batch_size, vocab_size)` 或 `(vocab_size,)`

**含义**: 模型最后一层的原始输出（未经 softmax）

**示例**:
```python
logits = array([[2.5, -1.3, ..., 0.8]], dtype=float32)
# 或
logits = None  # 如果 output_logits=False
```

**关系**:
```python
# probs 是 logits 经过 softmax 得到的
probs = softmax(logits)

# 可以反向计算 logits（如果需要）
logits_approx = np.log(probs + 1e-10)
```

## 🔍 与 run_mindformers_probability.py 的对应关系

在 `run_mindformers_probability.py` 中：

```python
def _collect_infer_scores(model, tokenized_batch, infer_config, steps_needed, pad_token_id):
    score_history: List[np.ndarray] = []
    
    for _ in range(steps_needed):
        infer_output, is_finished = model.infer(...)
        
        # 获取 probs
        probs_tensor = infer_output["probs"]  # 或 infer_output.probs
        score_history.append(_ms_to_np(probs_tensor))
        
        # 获取下一个 token
        target_list = infer_output["target_list"]  # 或 infer_output.target_list
        for idx, token in enumerate(target_list):
            sequences[idx].append(int(token))
    
    return score_history
```

### get_probs 函数的实现

```python
def get_probs(scores: List[np.ndarray], idx: int, lang: str, model_base: str):
    # scores 是多步的 probs 列表
    # scores[0] = 第一步的 probs (1, vocab_size)
    # scores[1] = 第二步的 probs (1, vocab_size)
    # ...
    
    token_place, safe_token, unsafe_token, controversial_token = _select_token_info(lang, model_base)
    
    # 选择特定位置的 probs
    if token_place >= len(scores):
        token_place = len(scores) - 1
    
    score_np = _ms_to_np(scores[token_place])[idx].astype(np.float32)
    
    # 从完整的词表概率中提取特定 tokens
    valid_scores = np.array([
        score_np[safe_token],           # safe token 的概率
        score_np[unsafe_token],         # unsafe token 的概率
        score_np[controversial_token]   # controversial token 的概率
    ], dtype=np.float32)
    
    # 归一化为三分类概率
    max_valid = np.max(valid_scores)
    exp_scores = np.exp(valid_scores - max_valid)
    probs = exp_scores / np.sum(exp_scores)
    
    return {
        'safe': float(probs[0]),
        'unsafe': float(probs[1]),
        'controversial': float(probs[2])
    }
```

## 📊 实际使用示例

### 示例 1: 基本使用

```python
# 调用 infer_mcore
infer_output, is_finished = model.infer_mcore(...)

# 提取字段
target_list = infer_output.target_list  # [151645]
probs = infer_output.probs              # (1, 151936)
logits = infer_output.logits            # None 或 (1, 151936)

# 使用 token
next_token = target_list[0]
token_text = tokenizer.decode([next_token])

# 分析概率
token_prob = probs[0][next_token]
print(f"Token {next_token} 的概率: {token_prob:.6f}")
```

### 示例 2: Top-K 分析

```python
import numpy as np

probs_np = infer_output.probs[0]  # (vocab_size,)

# 找出概率最高的 10 个 tokens
top_k = 10
top_indices = np.argsort(probs_np)[-top_k:][::-1]
top_probs = probs_np[top_indices]

print("Top-10 tokens:")
for idx, prob in zip(top_indices, top_probs):
    token_text = tokenizer.decode([int(idx)])
    print(f"  {idx:6d} ({prob:8.6f}): '{token_text}'")
```

### 示例 3: 特定 Token 的概率

```python
# 查询特定 token 的概率（如 ShieldLM 的分类 tokens）
safe_token_id = 41479      # "safe" 的 token ID
unsafe_token_id = 86009    # "unsafe" 的 token ID

probs_np = infer_output.probs[0]

safe_prob = probs_np[safe_token_id]
unsafe_prob = probs_np[unsafe_token_id]

print(f"Safe 概率: {safe_prob:.6f}")
print(f"Unsafe 概率: {unsafe_prob:.6f}")
```

### 示例 4: 熵和不确定性

```python
import numpy as np

probs_np = infer_output.probs[0]

# 计算熵（信息熵）
entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
print(f"熵: {entropy:.4f}")

# 归一化熵（0-1 之间）
max_entropy = np.log(len(probs_np))
normalized_entropy = entropy / max_entropy
print(f"归一化熵: {normalized_entropy:.4f}")

# 熵越高，模型越不确定
if normalized_entropy > 0.5:
    print("模型不确定性较高")
else:
    print("模型比较确定")
```

## 🎯 关键要点

### 1. 词表大小

Qwen3-14B 的词表大小是 **151,936**，所以 `probs` 的形状是 `(1, 151936)` 或 `(151936,)`。

### 2. 概率分布的特点

- **所有概率之和为 1.0**
- 大多数 token 的概率非常小（接近 0）
- 只有少数 token 有较高的概率
- 预测的 token 通常是概率最高的那个（greedy 模式）

### 3. 与 generate 的关系

`generate()` 内部会多次调用 `infer()` 或 `infer_mcore()`：

```
generate() 循环:
  ├─ 第1次: infer_mcore() → probs[0], target[0]
  ├─ 第2次: infer_mcore() → probs[1], target[1]
  ├─ 第3次: infer_mcore() → probs[2], target[2]
  └─ ...
```

每次 infer 返回：
- 当前位置的完整词表概率分布
- 根据策略选择的 token ID

### 4. 采样策略的应用

不同的采样策略使用 `probs`：

| 策略 | 使用方式 |
|------|---------|
| Greedy | `argmax(probs)` |
| Top-K | 只保留概率最高的 K 个，重新归一化 |
| Top-P | 保留累积概率达到 P 的 tokens |
| Temperature | 调整 logits: `logits / temperature` |

## 📝 调试建议

在你的测试代码中，现在会显示：

```
解析infer输出...
  - infer_output 类型: InferOutput
  - 检测到 InferOutput 对象
  - 提取的 target_list: [151645]
  
✓ 解析后的 target_list: [151645]
  - target_list 类型: list

✓ probs（词表概率分布）:
  - shape: (1, 151936)
  - dtype: float32
  - Top 5 tokens:
    Token 151645: 0.850000 ('<|im_end|>')
    Token 108386: 0.080000 ('你')
    Token 151643: 0.030000 ('<|endoftext|>')
    ...
```

这样你就能清楚地看到模型的预测分布了！

## 🔗 相关文档

- `run_mindformers_probability.py` - 概率提取的完整实现
- `PROBLEM_5_FIX.md` - InferOutput 类型处理
- `test_qwen3_mindformers.py` - 测试代码

---

希望这个文档帮助你理解 InferOutput 的结构和用途！🎉

