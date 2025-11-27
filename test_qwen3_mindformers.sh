#!/bin/bash
# 测试Qwen3-14B模型的MindFormers接口脚本

# 使用YAML配置文件的示例（默认不使用增量推理）
# python test_qwen3_mindformers.py \
#     --config models/predict_qwen3.yaml \
#     --test_mode both \
#     --prompt "你好，请介绍一下你自己。" \
#     --max_new_tokens 50

# 如果需要使用增量推理（KV cache），添加 --use_past 参数
# python test_qwen3_mindformers.py \
#     --config models/predict_qwen3.yaml \
#     --test_mode both \
#     --prompt "你好，请介绍一下你自己。" \
#     --max_new_tokens 50 \
#     --use_past

# 直接使用模型路径的示例（需要指定模型路径）
# python test_qwen3_mindformers.py \
#     --model_path /path/to/qwen3-14b \
#     --test_mode both \
#     --prompt "你好，请介绍一下你自己。" \
#     --max_new_tokens 50

# 默认命令（需要根据实际情况修改）
python test_qwen3_mindformers.py \
    --config models/predict_qwen3.yaml \
    --test_mode both \
    --prompt "你好，请介绍一下你自己。" \
    --max_new_tokens 50

