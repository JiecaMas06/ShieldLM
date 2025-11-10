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

仅使用“离线本地目录”方案（不依赖MindFormers内置配置与远端资源）：
- 准备目录：`./models/Qwen-14B-Chat`
- 安装并用 ModelScope 下载（或手动放入本地同等文件）：
  - `pip install modelscope`
  - `modelscope download --model Qwen/Qwen-14B-Chat --local_dir ./models/Qwen-14B-Chat`
- 目录内需包含：
  - safetensors 权重（例如多个 `.safetensors` 分片）
  - 本地分词器文件（至少其一，优先 `tokenizer.json`；若无，则可用 `tokenizer_config.json`、`spiece.model`/`sentencepiece.bpe.model`、`vocab.json`+`merges.txt`、或发行版提供的 `qwen.tiktoken` 等）

注意：本项目推理严格走“本地文件加载”，不会访问 `modelers.cn` 或其它远端地址。

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

使用本地ModelScope目录（Qwen-14B-Chat，中文，离线）：

```bash
python infer_shieldlm.py \
  --config_path ./models/qwen_14b_chat_safetensors.yaml \
  --model_path ./models/Qwen-14B-Chat \
  --tokenizer_path ./models/Qwen-14B-Chat/tokenizer_config.json \ 
  --lang zh \
  --model_base qwen \
  --input_path ./test.jsonl \
  --output_path ./test_out.jsonl
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
