# 在大模型平台jupyter环境中完成shieldlm迁移测试的指南

## 1.环境与模型准备

### 1.1 环境准备

提供的jupyter环境中默认已经有mindspore2.7.1版本。
安装mindformers1.7.0版本：pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple

默认设备：脚本默认以 Ascend NPU 单卡运行（device_id=0）。如需切换卡号，可在启动前设置环境变量 `DEVICE_ID`：

```bash
export DEVICE_ID=1   # 切换到第1号卡
python infer_shieldlm.py ...
```

### 1.2 模型准备

可选A：直接使用 mindformers 预训练名（推荐快速验证）
- Qwen-14B-Chat：`--model_path qwen_14b_chat`
- Baichuan2-13B-Chat：`--model_path baichuan2_13b_chat`

可选B：使用 ModelScope 预下载本地模型目录
- mkdir -p ./models
- 安装modelscope：pip install modelscope
- 下载Qwen：modelscope download --model Qwen/Qwen-14B-Chat --local_dir ./models/Qwen-14B-Chat
- 下载Baichuan2：modelscope download --model baichuan-inc/Baichuan2-13B-Chat --local_dir ./models/Baichuan2-13B-Chat
- 备注：若本地HF目录无法被 mindformers 直接读取，建议优先使用“可选A”的预训练名进行验证。

### 1.3 数据准备

创建测试数据 `test.jsonl`（JSON Lines），每行一个对象，至少包含 `query` 与 `response` 字段，例如：

{"query": "如何礼貌地拒绝同事的加班请求？", "response": "你可以说最近事务太多不方便，改天再帮忙。"}
{"query": "讲一个关于环保的倡议", "response": "减少一次性塑料使用，积极进行垃圾分类。"}

（可选）自定义规则文件 `rules.txt`，每行一条规则：

1) 不得引导违法
2) 不得宣扬伤害
...

## 2. 测试infer_shieldlm.py

以下命令在Ascend设备下运行，脚本会自动设置MindSpore上下文（GRAPH_MODE, Ascend）。

Qwen-14B-Chat（中文）：

```bash
python infer_shieldlm.py \
  --model_path qwen_14b_chat \
  --lang zh \
  --model_base qwen \
  --input_path ./test.jsonl \
  --output_path ./test_out.jsonl \
  --batch_size 2
```

Baichuan2-13B-Chat（中文）：

```bash
python infer_shieldlm.py \
  --model_path baichuan2_13b_chat \
  --lang zh \
  --model_base baichuan \
  --input_path ./test.jsonl \
  --output_path ./test_out.jsonl \
  --batch_size 2
```

使用本地ModelScope目录（如Qwen）：

```bash
# 方式一：使用safetensors专用YAML（推荐，使用 --config_path 指定 YAML）
python infer_shieldlm.py \
  --config_path ./models/qwen_14b_chat_safetensors.yaml \
  --model_path ./models/Qwen-14B-Chat \
  --lang zh \
  --model_base qwen \
  --input_path ./test.jsonl \
  --output_path ./test_out.jsonl

# 方式二：直接指向本地HF目录（不推荐，需完整mindformers YAML/配置支持，易报experimental错误）
# python infer_shieldlm.py \
#   --model_path ./models/Qwen-14B-Chat \
#   --tokenizer_path ./models/Qwen-14B-Chat \
#   --lang zh \
#   --model_base qwen \
#   --input_path ./test.jsonl \
#   --output_path ./test_out.jsonl
```

（可选）启用规则文件：

```bash
python infer_shieldlm.py \
  --model_path qwen_14b_chat \
  --lang zh \
  --model_base qwen \
  --input_path ./test.jsonl \
  --output_path ./test_out.jsonl \
  --rule_path ./rules.txt
```

输出文件 `test_out.jsonl` 将在原始对象基础上新增字段 `output`，为模型对给定对话的安全性分析结果。建议抽样打开若干行验证生成格式是否符合：
- 开头包含“[答案] … / [分析] …”（中文）或 “[Answer] … / [Analysis] …”（英文）
- 与所选 `model_base` 的提示模板一致（Qwen/Baichuan/InternLM/ChatGLM）
