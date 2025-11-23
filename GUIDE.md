# 在大模型平台jupyter环境中完成shieldlm迁移测试的指南

## 1.环境与模型准备

### 1.1 环境准备

登录https://xihe.mindspore.cn/training-projects ，打开**jupyter在线编程**，选择Ascend-snt9b，镜像选择python3.9-ms2.7.1-cann8.3.RC1，启动创建。

提供的jupyter环境中默认已经有mindspore2.7.1版本。

安装mindformers1.7.0版本：pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple

默认设备：脚本默认以 Ascend NPU 单卡运行（device_id=0）。

### 1.2 模型准备

仅使用“离线本地目录”方案（不依赖MindFormers内置配置与远端资源）：
- 准备目录：`./models/Qwen3-14B`
- 安装并用 ModelScope 下载（或手动放入本地同等文件）：
  - `pip install modelscope`
  - `modelscope download --model Qwen/Qwen3-14B --local_dir ./models/Qwen3-14B`
- 目录内需包含：
  - safetensors 权重（例如多个 `.safetensors` 分片）
  - 本地分词器文件（至少其一，优先 `tokenizer.json`；若无，则可用 `tokenizer_config.json`、`spiece.model`/`sentencepiece.bpe.model`、`vocab.json`+`merges.txt`、或发行版提供的 `qwen.tiktoken` 等）

注意：本项目推理严格走“本地文件加载”，不会访问 `modelers.cn` 或其它远端地址。

### 1.3 数据准备

（暂时使用test.jsonl，上传该文件即可）
创建测试数据 `test.jsonl`（JSON Lines），每行一个对象，至少包含 `query` 与 `response` 字段，例如：

{"query": "如何礼貌地拒绝同事的加班请求？", "response": "你可以说最近事务太多不方便，改天再帮忙。"}
{"query": "讲一个关于环保的倡议", "response": "减少一次性塑料使用，积极进行垃圾分类。"}

（可选）自定义规则文件 `rules.txt`，每行一条规则：

1) 不得引导违法
2) 不得宣扬伤害
...

## 2. 测试infer_shieldlm.py的同等功能（已测试完成）

以下命令在Ascend设备下运行，脚本会自动设置MindSpore上下文（GRAPH_MODE, Ascend）。
先上传run_mindformers.py文件到jupyter环境中。
使用本地ModelScope目录（Qwen-14B-Chat，中文，离线）：

以自带配置的Qwen3-14B模型为例：
**先更改hf文件夹中的generation_config.json文件，将max_new_tokens设置为1024。**
再运行以下命令，即可实现推理输出：
python run_mindformers.py --config predict_qwen3.yaml --run_mode predict --use_parallel False --input_jsonl ./test.jsonl --output_jsonl ./output.jsonl --lang zh  --predict_batch_size 2 --model_base qwen

建议不使用旧的Qwen-14B-Chat模型，其配置兼容性差。

使用Baichuan2-13B模型：（同样不建议使用Baichuan2模型，兼容性差，可以考虑更换为最新的glm4系列模型。）

## 3. 测试get_probability功能（正在测试实现方案）

`get_probability_ms.py` 用于计算模型对输入文本安全性评估的概率分布（safe/unsafe/controversial）。

### 3.1 基本用法

使用 Qwen3-14B 模型计算概率（中文），推荐显式指定与 `run_mindformers.py` 相同的 YAML：

```bash
python get_probability_ms.py \
  --model_path ./models/Qwen3-14B \
  --config_path ./predict_qwen3.yaml \
  --lang zh \
  --model_base qwen \
  --batch_size 1
```

脚本会对内置示例 `(query, response)` 做一次推理，并在终端打印模型生成结果和三类概率。

### 3.2 参数说明

- `--model_path`：模型权重目录（必需），用于兜底加载以及定位本地 tokenizer 资产
- `--tokenizer_path`：分词器路径（可选），默认会优先从 YAML 中的 `pretrained_model_dir` 自动推断，若无则回退到 `model_path`
- `--config_path`：MindFormers YAML 配置文件（强烈推荐），与 `run_mindformers.py` 使用的配置保持一致，用于通过 `MindFormerConfig + build_context + AutoModel.from_config` 加载模型
- `--lang`：语言，`zh` 或 `en`（必需）
- `--model_base`：模型类型，可选 `qwen`、`baichuan`、`internlm`、`chatglm`（必需），会决定 prompt 模板及标签 token 映射
- `--rule_path`：自定义规则文件（可选），每行一条规则，将注入到 ShieldLM 提示词中
- `--batch_size`：推理时的 batch 大小（可选，默认 1）

### 3.3 输出格式

当前脚本默认对内置示例 `(query, response)` 做一次推理，在终端打印两部分信息：

- 模型生成的完整输出文本
- 三类标签的概率分布，例如：

```text
Predict probability:  {'safe': 0.85, 'unsafe': 0.10, 'controversial': 0.05}
```

### 3.4 使用自定义规则

如需在评估时注入自定义规则，可通过 `--rule_path` 指定规则文件：

```bash
python get_probability_ms.py \
  --model_path ./models/Qwen3-14B \
  --config_path ./models/predict_qwen3.yaml \
  --lang zh \
  --model_base qwen \
  --rule_path ./rules.txt
```

