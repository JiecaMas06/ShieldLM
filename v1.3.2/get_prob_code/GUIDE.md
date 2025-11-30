# First Token Logits Extractor 使用指南

本工具用于从 Baichuan2-13B 和 Qwen-14B 模型中提取首个生成token的logits概率分布。支持单条输入和JSONL批量处理。

## 环境要求

### 硬件要求
- Ascend 800T A2 或兼容的华为昇腾硬件
- 至少 60GB 显存（用于加载13B/14B模型）

### 软件依赖
- MindSpore >= 2.0
- MindFormers（已安装在当前仓库）
- Python >= 3.8
- sentencepiece（用于Baichuan2分词器）
- tiktoken（用于Qwen分词器）

### 安装依赖
```bash
pip install sentencepiece tiktoken
```

## 模型权重准备

### Baichuan2-13B
1. 下载MindSpore格式权重：[Baichuan2-13B-Chat.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2-13B-Chat.ckpt)
2. 下载分词器：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model)

### Qwen-14B
1. 下载MindSpore格式权重：[qwen_14b_base.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_14b_base.ckpt)
2. 下载分词器：[qwen.tiktoken](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken)

## 使用方法

### 单条输入

#### 基本用法
```bash
python get_prob/extract_first_token_logits.py \
    --model_type baichuan2-13b \
    --checkpoint /path/to/Baichuan2-13B-Chat.ckpt \
    --tokenizer /path/to/tokenizer.model \
    --input "你好，请介绍一下"
```

#### 获取Top-K结果
```bash
python get_prob/extract_first_token_logits.py \
    --model_type baichuan2-13b \
    --checkpoint /path/to/Baichuan2-13B-Chat.ckpt \
    --tokenizer /path/to/tokenizer.model \
    --input "你好，请介绍一下" \
    --top_k 10
```

#### 返回概率分布
```bash
python get_prob/extract_first_token_logits.py \
    --model_type qwen-14b \
    --checkpoint /path/to/qwen_14b_base.ckpt \
    --tokenizer /path/to/qwen.tiktoken \
    --input "人工智能的发展" \
    --return_probs \
    --top_k 5
```

### JSONL批量处理

#### 输入文件格式
JSONL文件每行一个JSON对象，支持以下格式：

**通用格式**（用于logits提取）：
```json
{"text": "输入文本1"}
{"text": "输入文本2"}
```

或：
```json
{"input": "输入文本1"}
{"input": "输入文本2"}
```

**安全评估格式**（用于safety_eval模式）：
```json
{"query": "用户问题1", "response": "模型回复1"}
{"query": "用户问题2", "response": "模型回复2"}
```

#### 批量处理命令
```bash
python get_prob/extract_first_token_logits.py \
    --model_type baichuan2-13b \
    --checkpoint /path/to/Baichuan2-13B-Chat.ckpt \
    --tokenizer /path/to/tokenizer.model \
    --input_file /path/to/input.jsonl \
    --output /path/to/output.jsonl \
    --top_k 10
```

#### 安全评估模式
```bash
python get_prob/extract_first_token_logits.py \
    --model_type qwen-14b \
    --checkpoint /path/to/qwen_14b_base.ckpt \
    --tokenizer /path/to/qwen.tiktoken \
    --input_file /path/to/safety_data.jsonl \
    --output /path/to/results.jsonl \
    --safety_eval \
    --lang zh
```

### Python API 使用

```python
from get_prob.extract_first_token_logits import FirstTokenLogitsExtractor

# 初始化提取器
extractor = FirstTokenLogitsExtractor(
    model_type="baichuan2-13b",
    checkpoint_path="/path/to/Baichuan2-13B-Chat.ckpt",
    tokenizer_path="/path/to/tokenizer.model",
    device_id=0,
    seq_length=4096
)

# 提取原始logits
logits = extractor.extract("你好，请介绍一下")
print(f"Logits shape: {logits.shape}")  # (vocab_size,)

# 提取概率分布
probs = extractor.extract("你好，请介绍一下", return_probs=True)
print(f"Probabilities sum: {probs.sum()}")  # 约等于1.0

# 获取Top-K结果
token_ids, values = extractor.extract("你好", top_k=10, return_probs=True)
for tid, val in zip(token_ids, values):
    token = extractor.tokenizer.decode([int(tid)])
    print(f"Token: {token}, Probability: {val:.4f}")

# 安全评估（需要先配置SAFETY_TOKENS）
probs = extractor.get_safety_probs("formatted_prompt", lang="zh")
print(f"Safe: {probs['safe']}, Unsafe: {probs['unsafe']}")
```

## 命令行参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_type` | str | 是 | - | 模型类型：`baichuan2-13b` 或 `qwen-14b` |
| `--checkpoint` | str | 是 | - | 模型权重文件路径（.ckpt格式） |
| `--tokenizer` | str | 是 | - | 分词器文件路径 |
| `--input` | str | 二选一 | - | 单条输入文本 |
| `--input_file` | str | 二选一 | - | 输入JSONL文件路径 |
| `--config` | str | 否 | None | 可选的配置文件路径（.yaml格式） |
| `--output` | str | 否 | None | 输出文件路径 |
| `--top_k` | int | 否 | None | 只返回Top-K个token的结果 |
| `--return_probs` | flag | 否 | False | 返回softmax后的概率分布 |
| `--device_id` | int | 否 | 0 | 设备ID |
| `--seq_length` | int | 否 | 4096 | 序列长度 |
| `--safety_eval` | flag | 否 | False | 启用安全评估模式 |
| `--lang` | str | 否 | zh | 安全评估语言（zh/en） |
| `--rules` | str | 否 | None | 安全评估规则文件路径 |

## 查找和验证Token ID的方法

在使用安全评估功能时，需要配置特定token（如"安全"、"不安全"、"有争议"）的ID。以下是在MindFormers中查找正确token ID的方法：

### 方法1：使用tokenizer.encode()

```python
import sys
sys.path.insert(0, '/path/to/mindformers')
sys.path.insert(0, '/path/to/mindformers/research/baichuan2')  # 或 qwen

# Baichuan2
from baichuan2_tokenizer import Baichuan2Tokenizer
tokenizer = Baichuan2Tokenizer(vocab_file='/path/to/tokenizer.model')

# 查找token ID
text = "安全"
token_ids = tokenizer.encode(text)
print(f"'{text}' -> Token IDs: {token_ids}")

# 验证：解码回文本
decoded = tokenizer.decode(token_ids)
print(f"Token IDs {token_ids} -> '{decoded}'")

# 查找多个关键词
keywords = ["安全", "不安全", "有争议", "safe", "unsafe", "controversial"]
for kw in keywords:
    ids = tokenizer.encode(kw)
    print(f"'{kw}' -> {ids}")
```

### 方法2：使用tokenizer的词汇表

```python
# 获取完整词汇表
vocab = tokenizer.get_vocab()

# 搜索包含特定字符的token
search_term = "安全"
matches = [(token, idx) for token, idx in vocab.items() if search_term in str(token)]
print(f"Tokens containing '{search_term}':")
for token, idx in matches[:20]:  # 显示前20个
    print(f"  {idx}: {token}")
```

### 方法3：验证token ID的正确性

```python
# 假设找到的token ID
safe_token_id = 12345  # 替换为实际值

# 验证1：解码单个token
decoded = tokenizer.decode([safe_token_id])
print(f"Token ID {safe_token_id} decodes to: '{decoded}'")

# 验证2：检查是否为单token
text = "安全"
encoded = tokenizer.encode(text)
if len(encoded) == 1:
    print(f"'{text}' is a single token with ID: {encoded[0]}")
else:
    print(f"'{text}' is split into multiple tokens: {encoded}")
    # 可能需要使用第一个token或寻找其他表示
```

### Qwen Tokenizer 示例

```python
sys.path.insert(0, '/path/to/mindformers/research/qwen')
from qwen_tokenizer import QwenTokenizer

tokenizer = QwenTokenizer(vocab_file='/path/to/qwen.tiktoken')

# Qwen使用tiktoken，查找方式类似
keywords_zh = ["安全", "不安全", "有争议"]
keywords_en = ["safe", "unsafe", "controversial"]

for kw in keywords_zh + keywords_en:
    ids = tokenizer.encode(kw)
    print(f"'{kw}' -> {ids}")
```

### 更新脚本中的Token ID

找到正确的token ID后，更新 `extract_first_token_logits.py` 中的 `SAFETY_TOKENS` 字典：

```python
SAFETY_TOKENS = {
    "qwen-14b": {
        "zh": {
            "token_place": 3,
            "safe": 12345,        # 替换为实际值
            "unsafe": 23456,      # 替换为实际值
            "controversial": 34567  # 替换为实际值
        },
        "en": {
            "token_place": 3,
            "safe": 6092,         # 替换为实际值
            "unsafe": 19860,      # 替换为实际值
            "controversial": 20129  # 替换为实际值
        }
    },
    "baichuan2-13b": {
        # 同样替换
    }
}
```

## 输出格式说明

### 单条输入 - 完整Logits
```
Extracted logits shape: (125696,)
Sum: -12345.67
Max: 15.234
Min: -25.678
```

### 单条输入 - Top-K
```
Top-10 tokens:
--------------------------------------------------
1. Token ID: 12345, Token: '自己', Logit: 15.234
2. Token ID: 23456, Token: '我', Logit: 14.567
...
```

### JSONL批量输出
```json
{"text": "输入文本", "top_tokens": [12345, 23456], "top_values": [0.234, 0.123]}
{"query": "问题", "response": "回复", "prob": {"safe": 0.8, "unsafe": 0.15, "controversial": 0.05}}
```

## 注意事项

1. **显存要求**：13B/14B模型需要较大显存，建议使用60GB以上显存的设备
2. **首次加载**：首次加载模型可能需要较长时间进行图编译
3. **use_past设置**：本工具禁用了KV cache（use_past=False），以确保单次前向传播的正确性
4. **Token ID配置**：使用安全评估功能前，必须先配置正确的token ID
5. **精度**：默认使用float16精度进行推理

