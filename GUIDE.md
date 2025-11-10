# 在大模型平台jupyter环境中完成shieldlm迁移测试的指南

## 1.环境与模型准备

### 1.1 环境准备

提供的jupyter环境中默认已经有mindspore2.7.1版本。
安装mindformers1.7.0版本：pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.7.1/MindFormers/any/mindformers-1.7.0-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple

### 1.2 模型准备

安装modelscope：pip install modelscope
下载Qwen模型：modelscope download --model Qwen/Qwen-14B-Chat --local_dir ./models/Qwen-14B-Chat
下载Baichuan2模型：modelscope download --model baichuan-inc/Baichuan2-13B-Chat --local_dir ./models/Baichuan2-13B-Chat
暂时仅采用此两个模型进行测试。

### 1.3 数据准备

创建test.json文件。

