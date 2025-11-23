# 本项目目标为：将ShieldLM的代码从Pytorch支持迁移到Mindspore支持，在云端昇腾设备上完成复现

## 官方文档参考

Mindspore2.7.1 https://www.mindspore.cn/docs/zh-CN/r2.7.1/api_python/index.html
Mindspore Transformers1.7.0（mindformers）https://www.mindspore.cn/mindformers/docs/zh-CN/r1.7.0/index.html
可以通过fetch MCP访问官网文档进行查询；如果fetch失败，改用curl进行查看。

## 阶段目标实现

### 第一阶段：完成infer_shieldlm.py的迁移，并完成测试，确认模型基本生成功能。

[ ] 测试Qwen3-14B的推理功能

### 第二阶段：实现get_probability_ms的迁移

[ ] 测试Qwen3-14B的get_probability功能

### 第三阶段：实现训练过程的迁移

[ ] 测试Qwen3-14B的训练功能

### 第四阶段：执行训练测试
