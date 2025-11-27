#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Qwen3-14B模型的MindFormers generate和infer接口
"""
import argparse
import importlib
import json
import os
from typing import Optional

import mindspore as ms
import numpy as np

from mindformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, MindFormerConfig
from mindformers.core.context import build_context
from mindformers.generation import GenerationConfig
from mindformers.tools.register import MindFormerRegister
from mindspore import Parameter


def parse_args():
    parser = argparse.ArgumentParser(description="测试Qwen3模型的generate和infer接口")
    parser.add_argument("--config", dest="config_path", type=str, default=None,
                        help="MindFormers YAML配置文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型目录路径（不使用YAML时的备用选项）")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器目录路径，默认使用模型路径")
    parser.add_argument("--test_mode", type=str, default="both",
                        choices=["generate", "infer", "both"],
                        help="测试模式：generate/infer/both")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。",
                        help="测试用的提示词")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大token数量")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="是否信任远程代码")
    parser.add_argument("--use_past", action="store_true", default=False,
                        help="是否使用增量推理（KV cache），默认不使用")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    return parser.parse_args()


def _import_model_register():
    """导入Qwen系列模型注册模块"""
    try:
        print("正在导入Qwen模型注册模块...")
        importlib.import_module("mindformers.models.qwen2")
        print("✓ 成功导入 mindformers.models.qwen2")
        
        importlib.import_module("mindformers.models.qwen")
        print("✓ 成功导入 mindformers.models.qwen")
        
        try:
            importlib.import_module("mindformers.models.qwen3")
            print("✓ 成功导入 mindformers.models.qwen3")
        except Exception as e:
            print(f"⚠ 未找到qwen3模块: {e}")
    except Exception as e:
        print(f"⚠ 模型注册导入失败: {e}")


def _load_local_tokenizer(path_like: str, trust_remote_code: bool = False):
    """加载本地分词器"""
    base = str(path_like)
    if os.path.isfile(base):
        base = os.path.dirname(base)
    if not os.path.isdir(base):
        raise RuntimeError(f"分词器路径不存在或不是目录: {base}")
    
    print(f"正在加载分词器: {base}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        print("✓ 使用MindFormers AutoTokenizer加载成功")
        return tokenizer
    except Exception as e:
        print(f"⚠ MindFormers AutoTokenizer加载失败: {e}")
        print("尝试使用HuggingFace AutoTokenizer...")
        from transformers import AutoTokenizer as HFAutoTokenizer
        tokenizer = HFAutoTokenizer.from_pretrained(
            base,
            padding_side="left",
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        print("✓ 使用HuggingFace AutoTokenizer加载成功")
        return tokenizer


def check_checkpoint_files(model_path):
    """
    检查checkpoint文件是否存在
    """
    print("\n" + "="*60)
    print("检查Checkpoint文件")
    print("="*60)
    
    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ 模型路径不存在: {model_path}")
        return False
    
    print(f"\n模型路径: {model_path}")
    
    # 检查各种可能的checkpoint文件
    checkpoint_patterns = [
        "*.safetensors",
        "*.ckpt",
        "*.bin",
        "*.pth",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    
    found_files = []
    total_size = 0
    
    for pattern in checkpoint_patterns:
        if '*' in pattern:
            # 使用glob查找
            import glob
            files = glob.glob(os.path.join(model_path, pattern))
            found_files.extend(files)
        else:
            # 直接检查
            file_path = os.path.join(model_path, pattern)
            if os.path.exists(file_path):
                found_files.append(file_path)
    
    if found_files:
        print(f"\n✓ 找到 {len(found_files)} 个checkpoint文件:")
        for f in found_files:
            size = os.path.getsize(f)
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  - {os.path.basename(f)}: {size_mb:.2f} MB")
        
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"\n总大小: {total_size_gb:.2f} GB")
        
        # 对于14B模型，期望的大小大约是 28GB (bf16)
        expected_size_gb = 28
        if total_size_gb < expected_size_gb * 0.5:
            print(f"\n⚠️ 警告：checkpoint文件大小 ({total_size_gb:.2f} GB) 远小于预期 (~{expected_size_gb} GB)")
            print("  这可能表明checkpoint文件不完整")
        elif total_size_gb < expected_size_gb * 0.9:
            print(f"\n⚠️ checkpoint文件大小 ({total_size_gb:.2f} GB) 略小于预期 (~{expected_size_gb} GB)")
        else:
            print(f"\n✓ checkpoint文件大小正常")
        
        return True
    else:
        print("\n❌ 未找到任何checkpoint文件！")
        print("  模型可能使用随机初始化的权重")
        return False


def verify_model_weights(model, tokenizer=None):
    """
    验证模型权重是否正确加载
    
    检查内容：
    1. 是否有参数全为0（表明未加载）
    2. 是否有NaN或Inf（表明加载错误）
    3. 关键层的统计信息
    4. embedding层的权重检查
    """
    print("\n" + "="*60)
    print("验证模型权重加载")
    print("="*60)
    
    all_params = []
    zero_params = []
    nan_inf_params = []
    suspicious_params = []
    
    # 收集所有参数
    print("\n正在检查模型参数...")
    for name, param in model.parameters_and_names():
        if isinstance(param, Parameter):
            try:
                # 转换为NumPy并确保是标准类型
                param_data = param.asnumpy()
                
                # 转换为float32以确保兼容性
                if param_data.dtype not in [np.float32, np.float64, np.int32, np.int64]:
                    param_data = param_data.astype(np.float32)
                
                all_params.append((name, param_data))
                
                # 检查全零参数
                if np.all(param_data == 0):
                    zero_params.append(name)
                
                # 检查NaN或Inf（仅对浮点类型）
                if param_data.dtype in [np.float32, np.float64]:
                    if np.any(np.isnan(param_data)) or np.any(np.isinf(param_data)):
                        nan_inf_params.append(name)
                
                # 检查可疑参数（标准差极小）
                if param_data.size > 1:
                    try:
                        # 确保是浮点类型再计算标准差
                        if param_data.dtype not in [np.float32, np.float64]:
                            param_float = param_data.astype(np.float32)
                        else:
                            param_float = param_data
                        
                        std = np.std(param_float)
                        if std < 1e-10 and not np.all(param_data == 0):
                            suspicious_params.append((name, std))
                    except Exception as e:
                        # 如果计算失败，跳过这个参数
                        pass
            except Exception as e:
                print(f"  ⚠ 无法检查参数 {name}: {e}")
                continue
    
    print(f"✓ 总共检查了 {len(all_params)} 个参数")
    
    # 报告问题
    print("\n" + "="*60)
    print("权重检查结果")
    print("="*60)
    
    if zero_params:
        print(f"\n⚠️ 发现 {len(zero_params)} 个全零参数（可能未正确加载）:")
        for name in zero_params[:10]:  # 只显示前10个
            print(f"  - {name}")
        if len(zero_params) > 10:
            print(f"  ... 还有 {len(zero_params) - 10} 个")
    else:
        print("\n✓ 没有发现全零参数")
    
    if nan_inf_params:
        print(f"\n❌ 发现 {len(nan_inf_params)} 个包含NaN/Inf的参数:")
        for name in nan_inf_params:
            print(f"  - {name}")
    else:
        print("✓ 没有发现NaN/Inf参数")
    
    if suspicious_params:
        print(f"\n⚠️ 发现 {len(suspicious_params)} 个可疑参数（标准差极小）:")
        for name, std in suspicious_params[:5]:
            print(f"  - {name}: std={std:.2e}")
        if len(suspicious_params) > 5:
            print(f"  ... 还有 {len(suspicious_params) - 5} 个")
    
    # 显示关键层的统计信息
    print("\n" + "="*60)
    print("关键层权重统计")
    print("="*60)
    
    key_layers = ['embedding', 'wte', 'lm_head', 'output', 'attention', 'q_proj', 'k_proj', 'v_proj']
    
    displayed_count = 0
    for name, param_data in all_params:
        # 检查是否是关键层
        is_key_layer = any(key in name.lower() for key in key_layers)
        
        if is_key_layer and displayed_count < 10:
            try:
                # 确保是浮点类型
                if param_data.dtype not in [np.float32, np.float64]:
                    param_float = param_data.astype(np.float32)
                else:
                    param_float = param_data
                
                print(f"\n参数: {name}")
                print(f"  - shape: {param_data.shape}")
                print(f"  - dtype: {param_data.dtype}")
                print(f"  - mean: {np.mean(param_float):.6f}")
                print(f"  - std: {np.std(param_float):.6f}")
                print(f"  - min: {np.min(param_float):.6f}")
                print(f"  - max: {np.max(param_float):.6f}")
                print(f"  - 非零元素比例: {np.count_nonzero(param_data) / param_data.size * 100:.2f}%")
                displayed_count += 1
            except Exception as e:
                print(f"\n参数: {name}")
                print(f"  - shape: {param_data.shape}")
                print(f"  - dtype: {param_data.dtype}")
                print(f"  ⚠ 无法计算统计信息: {e}")
                displayed_count += 1
    
    # 特别检查embedding层
    print("\n" + "="*60)
    print("Embedding层特别检查")
    print("="*60)
    
    embedding_found = False
    for name, param_data in all_params:
        if 'embedding' in name.lower() or 'wte' in name.lower():
            if not embedding_found:
                try:
                    print(f"\n找到embedding层: {name}")
                    print(f"  - shape: {param_data.shape}")
                    
                    # 确保是浮点类型
                    if param_data.dtype not in [np.float32, np.float64]:
                        param_float = param_data.astype(np.float32)
                    else:
                        param_float = param_data
                    
                    # 如果有tokenizer，测试几个常见token
                    if tokenizer is not None:
                        print("\n测试常见token的embedding:")
                        test_tokens = {
                            "你": None,
                            "好": None,
                            "hello": None,
                            "world": None,
                        }
                        
                        for text, _ in test_tokens.items():
                            try:
                                token_id = tokenizer.encode(text, add_special_tokens=False)
                                if isinstance(token_id, list) and len(token_id) > 0:
                                    token_id = token_id[0]
                                if token_id < param_float.shape[0]:
                                    embedding_vec = param_float[token_id]
                                    norm = np.linalg.norm(embedding_vec)
                                    mean_val = np.mean(embedding_vec)
                                    print(f"  - '{text}' (id={token_id}): norm={norm:.6f}, mean={mean_val:.6f}")
                            except Exception as e:
                                print(f"  - '{text}': 测试失败 ({e})")
                    
                    # 随机采样几个embedding检查
                    print("\n随机采样5个token的embedding:")
                    sample_indices = np.random.choice(param_float.shape[0], size=min(5, param_float.shape[0]), replace=False)
                    for idx in sample_indices:
                        vec = param_float[idx]
                        norm = np.linalg.norm(vec)
                        is_zero = np.all(vec == 0)
                        print(f"  - Token {idx}: norm={norm:.6f}, is_zero={is_zero}")
                    
                    embedding_found = True
                    break
                except Exception as e:
                    print(f"\n找到embedding层: {name}")
                    print(f"  ⚠ 无法完成embedding检查: {e}")
                    embedding_found = True
                    break
    
    if not embedding_found:
        print("\n⚠️ 未找到embedding层")
    
    # 总体评估
    print("\n" + "="*60)
    print("权重加载评估")
    print("="*60)
    
    has_issues = len(zero_params) > 0 or len(nan_inf_params) > 0
    
    if has_issues:
        print("\n❌ 权重加载存在问题！")
        if zero_params:
            print(f"  - {len(zero_params)} 个参数为全零")
        if nan_inf_params:
            print(f"  - {len(nan_inf_params)} 个参数包含NaN/Inf")
        print("\n可能的原因：")
        print("  1. checkpoint文件损坏或不完整")
        print("  2. 模型配置与checkpoint不匹配")
        print("  3. 加载过程中出现错误但被忽略")
        print("  4. safetensors文件读取失败")
        print("\n建议：")
        print("  - 检查checkpoint文件完整性")
        print("  - 查看加载日志中的警告/错误信息")
        print("  - 验证模型配置是否正确")
    else:
        print("\n✓ 权重加载看起来正常")
        print(f"  - 所有 {len(all_params)} 个参数都已初始化")
        print("  - 没有发现明显的加载问题")
        
        if suspicious_params:
            print(f"\n⚠️ 但有 {len(suspicious_params)} 个参数的标准差极小，请关注")
    
    print("="*60 + "\n")
    
    return not has_issues


def create_model_tokenizer(args):
    """创建模型和分词器"""
    print("\n" + "="*60)
    print("开始加载模型和分词器")
    print("="*60)
    
    cfg = None
    model = None
    tokenizer = None
    
    # 方式1: 使用YAML配置文件
    if args.config_path and (args.config_path.endswith(".yaml") or args.config_path.endswith(".yml")):
        print(f"\n使用YAML配置文件: {args.config_path}")
        cfg = MindFormerConfig(args.config_path, run_mode="predict")
        
        if args.trust_remote_code:
            cfg.trust_remote_code = True
        
        # 设置设备ID
        device_id_str = os.getenv("DEVICE_ID", None)
        if device_id_str is not None:
            try:
                cfg.context.device_id = int(device_id_str)
                print(f"设置设备ID: {cfg.context.device_id}")
            except Exception as e:
                print(f"⚠ 设置设备ID失败: {e}")
        
        # 构建上下文
        build_context(cfg)
        print("✓ MindSpore上下文构建完成")
        
        # 导入模型注册
        _import_model_register()
        
        # 检查checkpoint文件
        pretrained_dir = getattr(cfg, "pretrained_model_dir", None)
        if pretrained_dir is None and hasattr(cfg, "model") and hasattr(cfg.model, "model_config"):
            pretrained_dir = getattr(cfg.model.model_config, "pretrained_model_dir", None)
        if pretrained_dir:
            check_checkpoint_files(pretrained_dir)
        
        # 加载模型（重要：使用正确的MindFormers API加载权重）
        print("\n正在加载模型...")
        
        # 获取YAML文件所在的目录（包含config的目录）
        yaml_dir = os.path.dirname(args.config_path) if args.config_path else None
        
        # 尝试使用AutoModel.from_pretrained加载（指向YAML所在目录）
        if yaml_dir and os.path.exists(yaml_dir):
            print(f"  尝试使用AutoModel.from_pretrained加载")
            print(f"  YAML目录: {yaml_dir}")
            try:
                # 方式1：指向包含YAML的目录，MindFormers会自动读取YAML并加载权重
                model = AutoModel.from_pretrained(yaml_dir)
                print("✓ 模型及权重加载成功（AutoModel.from_pretrained）")
            except Exception as e:
                print(f"⚠ AutoModel.from_pretrained失败: {e}")
                
                # 方式2：尝试直接指向pretrained_model_dir
                if pretrained_dir and os.path.exists(pretrained_dir):
                    print(f"  尝试直接加载pretrained_dir: {pretrained_dir}")
                    try:
                        model = AutoModel.from_pretrained(pretrained_dir)
                        print("✓ 模型及权重加载成功（从pretrained_dir）")
                    except Exception as e2:
                        print(f"⚠ 从pretrained_dir加载也失败: {e2}")
                        print("  回退到from_config...")
                        model = AutoModel.from_config(args.config_path)
                        print("⚠ 使用from_config创建模型结构（权重未加载）")
                else:
                    print("  回退到from_config...")
                    model = AutoModel.from_config(args.config_path)
                    print("⚠ 使用from_config创建模型结构（权重未加载）")
        else:
            print(f"⚠ 未找到YAML目录或pretrained_dir")
            model = AutoModel.from_config(args.config_path)
            print("✓ 模型结构创建成功（无权重）")
        
        # 加载分词器
        print("\n正在加载分词器...")
        tokenizer_cfg = getattr(getattr(cfg, "processor", None), "tokenizer", None)
        if tokenizer_cfg:
            try:
                tokenizer = MindFormerRegister.get_instance_from_cfg(tokenizer_cfg, "tokenizer")
                print("✓ 从配置文件加载分词器成功")
            except Exception as e:
                print(f"⚠ 从配置文件加载分词器失败: {e}")
        
        # 备用分词器加载方式
        if tokenizer is None:
            tokenizer_path = args.tokenizer_path
            if tokenizer_path is None:
                pretrained_dir = getattr(cfg, "pretrained_model_dir", None)
                if pretrained_dir is None and hasattr(cfg, "model") and hasattr(cfg.model, "model_config"):
                    pretrained_dir = getattr(cfg.model.model_config, "pretrained_model_dir", None)
                tokenizer_path = pretrained_dir or args.model_path
            
            if tokenizer_path:
                tokenizer = _load_local_tokenizer(
                    tokenizer_path,
                    trust_remote_code=getattr(cfg, "trust_remote_code", args.trust_remote_code),
                )
    
    # 方式2: 直接使用模型路径
    else:
        print(f"\n使用模型路径: {args.model_path}")
        device_id = int(os.getenv("DEVICE_ID", "0"))
        print(f"设置设备ID: {device_id}")
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)
        
        _import_model_register()
        
        # 检查checkpoint文件
        if args.model_path:
            check_checkpoint_files(args.model_path)
        
        print("\n正在从路径加载模型...")
        try:
            # 使用AutoModel.from_pretrained（MindFormers的正确API）
            model = AutoModel.from_pretrained(args.model_path)
            print("✓ 模型及权重加载成功（AutoModel.from_pretrained）")
        except Exception as e:
            print(f"⚠ AutoModel.from_pretrained失败: {e}")
            print("  请确保模型路径包含YAML配置文件或是支持的模型名称")
            raise
        
        tokenizer_path = args.tokenizer_path or args.model_path
        tokenizer = _load_local_tokenizer(tokenizer_path, trust_remote_code=args.trust_remote_code)
    
    # 设置模型为评估模式
    model.set_train(False)
    print("✓ 模型设置为评估模式")
    
    # 确保分词器有必要的特殊token
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = "<|endoftext|>"
        print(f"设置eos_token: {tokenizer.eos_token}")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置pad_token: {tokenizer.pad_token}")
    
    print("\n" + "="*60)
    print("模型和分词器加载完成")
    print("="*60 + "\n")
    
    # 验证权重加载
    weights_ok = verify_model_weights(model, tokenizer)
    if not weights_ok:
        print("\n❌ 警告：权重加载验证未通过，模型可能无法正常工作！")
        print("建议：检查上面的详细信息，确认checkpoint文件是否正确\n")
    
    return model, tokenizer


def quick_forward_test(model, tokenizer):
    """
    快速前向传播测试，验证模型能否正常输出
    """
    print("\n" + "="*60)
    print("快速前向传播测试")
    print("="*60)
    
    try:
        # 创建一个简单的测试输入
        test_text = "Hello"
        print(f"\n测试输入: '{test_text}'")
        
        # Tokenize
        inputs = tokenizer(test_text, return_tensors='np', padding=True)
        input_ids = inputs['input_ids']
        print(f"✓ input_ids shape: {input_ids.shape}")
        
        # 尝试前向传播
        print("\n执行前向传播...")
        input_ids_tensor = ms.Tensor(input_ids, dtype=ms.int32)
        
        # 设置为评估模式
        model.set_train(False)
        
        # 执行前向传播
        outputs = model(input_ids_tensor)
        
        # 检查输出
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        if hasattr(logits, 'asnumpy'):
            logits_np = logits.asnumpy()
        else:
            logits_np = np.array(logits)
        
        # 确保是浮点类型
        if logits_np.dtype not in [np.float32, np.float64]:
            logits_np = logits_np.astype(np.float32)
        
        print(f"✓ 前向传播成功")
        print(f"  - 输出shape: {logits_np.shape}")
        print(f"  - 输出dtype: {logits_np.dtype}")
        print(f"  - 输出统计:")
        print(f"    - min: {np.min(logits_np):.6f}")
        print(f"    - max: {np.max(logits_np):.6f}")
        print(f"    - mean: {np.mean(logits_np):.6f}")
        print(f"    - std: {np.std(logits_np):.6f}")
        
        # 检查是否全为0
        if np.all(logits_np == 0):
            print("\n❌ 警告：输出全为0！这表明模型权重可能未正确加载")
            return False
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(logits_np)) or np.any(np.isinf(logits_np)):
            print("\n❌ 警告：输出包含NaN或Inf！")
            return False
        
        # 计算softmax并检查概率分布
        # 取最后一个位置的logits
        last_logits = logits_np[0, -1, :]
        
        # 确保是float32或float64
        if last_logits.dtype not in [np.float32, np.float64]:
            last_logits = last_logits.astype(np.float32)
        
        # 手动计算softmax (避免数值溢出)
        logits_max = np.max(last_logits)
        exp_logits = np.exp(last_logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_probs = probs[top_5_indices]
        
        print(f"\n  - Top 5 预测token:")
        for idx, prob in zip(top_5_indices, top_5_probs):
            try:
                token_text = tokenizer.decode([int(idx)], skip_special_tokens=False)
                print(f"    Token {idx}: {prob:.6f} ('{token_text}')")
            except:
                print(f"    Token {idx}: {prob:.6f}")
        
        if np.max(probs) < 0.01:
            print("\n⚠️ 警告：最高概率很小，可能存在问题")
        
        print("\n✓ 快速前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 快速前向传播测试失败!")
        print(f"错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_generate(model, tokenizer, prompt: str, args):
    """测试generate接口"""
    print("\n" + "="*60)
    print("测试 generate 接口")
    print("="*60)
    
    try:
        print(f"\n输入提示词(原始): {prompt}")
        
        # 应用chat template（对于chat模型非常重要）
        print("\n正在应用chat template...")
        if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
            # 使用标准的chat template格式
            messages = [{"role": "user", "content": prompt}]
            try:
                # 尝试使用apply_chat_template
                input_ids = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    return_tensors=None
                )
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist()
                input_ids = [input_ids]  # 添加batch维度
                print(f"✓ 使用tokenizer.apply_chat_template处理完成")
            except Exception as e:
                print(f"⚠ apply_chat_template失败: {e}")
                print("  回退到手动添加chat template...")
                # 手动添加Qwen chat template
                formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
                input_ids = inputs['input_ids'].tolist()
        else:
            print("⚠ tokenizer没有apply_chat_template方法，手动添加chat template...")
            # 手动添加Qwen chat template
            formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            print(f"  格式化后的prompt: {formatted_prompt[:100]}...")
            inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
            input_ids = inputs['input_ids'].tolist()
        
        print(f"✓ tokenize完成，input_ids shape: {np.array(input_ids).shape}")
        print(f"  input_ids: {input_ids[0][:10]}... (显示前10个)")
        print(f"  input_ids长度: {len(input_ids[0])}")
        
        # 构建生成配置
        print("\n构建GenerationConfig...")
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            use_past=args.use_past,
            return_dict_in_generate=False,
        )
        print(f"✓ GenerationConfig配置完成")
        print(f"  - max_new_tokens: {gen_config.max_new_tokens}")
        print(f"  - use_past: {gen_config.use_past}")
        print(f"  - eos_token_id: {gen_config.eos_token_id}")
        print(f"  - pad_token_id: {gen_config.pad_token_id}")
        
        # 调用generate
        print("\n开始调用model.generate()...")
        output_ids = model.generate(
            input_ids=input_ids,
            generation_config=gen_config
        )
        print("✓ generate调用成功")
        
        # 解码输出
        print("\n正在解码输出...")
        if isinstance(output_ids, list):
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        print("\n" + "-"*60)
        print("生成结果:")
        print("-"*60)
        print(output_text)
        print("-"*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ generate测试失败!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        print(traceback.format_exc())
        return False


def test_infer(model, tokenizer, prompt: str, args):
    """测试infer接口"""
    print("\n" + "="*60)
    print("测试 infer 接口")
    print("="*60)
    
    try:
        print(f"\n输入提示词(原始): {prompt}")
        
        # 应用chat template（对于chat模型非常重要）
        print("\n正在应用chat template...")
        if hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template):
            # 使用标准的chat template格式
            messages = [{"role": "user", "content": prompt}]
            try:
                # 尝试使用apply_chat_template
                input_ids = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    return_tensors=None
                )
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist()
                input_ids_np = np.array([input_ids], dtype=np.int32)  # 添加batch维度
                print(f"✓ 使用tokenizer.apply_chat_template处理完成")
            except Exception as e:
                print(f"⚠ apply_chat_template失败: {e}")
                print("  回退到手动添加chat template...")
                # 手动添加Qwen chat template
                formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
                input_ids_np = inputs['input_ids']
        else:
            print("⚠ tokenizer没有apply_chat_template方法，手动添加chat template...")
            # 手动添加Qwen chat template
            formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            print(f"  格式化后的prompt: {formatted_prompt[:100]}...")
            inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
            input_ids_np = inputs['input_ids']
        
        print(f"✓ tokenize完成，input_ids shape: {input_ids_np.shape}")
        print(f"  input_ids: {input_ids_np[0][:10].tolist()}... (显示前10个)")
        print(f"  input_ids长度: {input_ids_np.shape[1]}")
        
        # 计算有效长度
        batch_size = input_ids_np.shape[0]
        pad_token_id = int(tokenizer.pad_token_id)
        valid_length_each_example = []
        for i in range(batch_size):
            # 计算非padding的长度
            valid_indices = np.where(input_ids_np[i] != pad_token_id)[0]
            if len(valid_indices) > 0:
                valid_len = int(np.max(valid_indices)) + 1
            else:
                valid_len = input_ids_np.shape[1]
            valid_length_each_example.append(valid_len)
        valid_length_each_example = np.array(valid_length_each_example, dtype=np.int32)
        print(f"✓ 计算有效长度: {valid_length_each_example}")
        
        # 构建生成配置
        print("\n构建GenerationConfig...")
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            use_past=args.use_past,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,  # 启用 logits 输出，用于诊断
        )
        print(f"✓ GenerationConfig配置完成")
        print(f"  - use_past: {gen_config.use_past}")
        print(f"  - output_scores: {gen_config.output_scores}")
        print(f"  - output_logits: {gen_config.output_logits}")
        
        # 准备infer需要的参数
        is_finished = [False] * batch_size
        prefill = True
        
        # 检测模型架构类型
        from mindformers.core.context import is_legacy_model
        use_legacy = is_legacy_model()
        print(f"\n模型架构: {'Legacy' if use_legacy else 'MCore (Parallel Core)'}")
        
        # 准备 position_ids（必需参数，仅 Legacy 模型需要显式传递）
        print("\n准备 position_ids...")
        max_len = input_ids_np.shape[1]
        position_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        for idx, length in enumerate(valid_length_each_example):
            if length > 0:
                position_ids[idx, :length] = np.arange(length, dtype=np.int32)
        print(f"✓ position_ids shape: {position_ids.shape}")
        
        # 初始化 block_mgr（参考 generate() 的逻辑）
        print("\n初始化 block manager...")
        if not use_legacy:
            # MCore 模型需要初始化这些组件
            if hasattr(model, '_set_block_mgr'):
                try:
                    model._set_block_mgr(batch_size, model.config.seq_length)
                    print("✓ block_mgr 初始化成功")
                except Exception as e:
                    print(f"⚠ block_mgr 初始化失败: {e}")
            
            if hasattr(model, '_set_kv_cache'):
                try:
                    model._set_kv_cache()
                    print("✓ kv_cache 初始化成功")
                except Exception as e:
                    print(f"⚠ kv_cache 初始化失败: {e}")
            
            if hasattr(model, '_set_lower_triangle_mask'):
                try:
                    model._set_lower_triangle_mask()
                    print("✓ lower_triangle_mask 初始化成功")
                except Exception as e:
                    print(f"⚠ lower_triangle_mask 初始化失败: {e}")
            
            if hasattr(model, 'set_dynamic_inputs'):
                try:
                    model.set_dynamic_inputs()
                    print("✓ 动态输入设置成功")
                except Exception as e:
                    print(f"⚠ 动态输入设置失败: {e}")
        elif gen_config.use_past:
            # Legacy 模型如果使用 use_past 也需要初始化
            if hasattr(model, '_set_block_mgr'):
                try:
                    model._set_block_mgr(batch_size, model.config.seq_length)
                    print("✓ block_mgr 初始化成功")
                except Exception as e:
                    print(f"⚠ block_mgr 初始化失败: {e}")
            
            if hasattr(model, 'set_dynamic_inputs') and model.config.is_dynamic:
                try:
                    model.set_dynamic_inputs()
                    print("✓ 动态输入设置成功")
                except Exception as e:
                    print(f"⚠ 动态输入设置失败: {e}")
        
        # 准备 block_tables 和 slot_mapping
        print("\n准备 block_tables 和 slot_mapping...")
        block_tables = None
        slot_mapping = None
        if hasattr(model, 'block_mgr') and model.block_mgr:
            try:
                max_input_length = input_ids_np.shape[1]
                block_tables, slot_mapping = model.block_mgr.assemble_pa_full_inputs(
                    max_input_length,
                    valid_length_each_example,
                    is_finished
                )
                print(f"✓ block_tables shape: {block_tables.shape if hasattr(block_tables, 'shape') else 'N/A'}")
                print(f"✓ slot_mapping shape: {slot_mapping.shape if hasattr(slot_mapping, 'shape') else 'N/A'}")
            except Exception as e:
                print(f"⚠ 准备 block_tables/slot_mapping 失败: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print(f"⚠ block_mgr 不存在或未初始化")
        
        print(f"\n开始调用 infer...")
        print(f"  - prefill: {prefill}")
        print(f"  - is_finished: {is_finished}")
        print(f"  - use_past: {gen_config.use_past}")
        
        # 根据模型架构选择调用方式
        if use_legacy:
            # Legacy 模型：使用普通 infer 接口
            print("  - 使用 Legacy infer 接口")
            if block_tables is not None and slot_mapping is not None:
                infer_output, is_finished = model.infer(
                    input_ids=input_ids_np,
                    valid_length_each_example=valid_length_each_example,
                    generation_config=gen_config,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    prefill=prefill,
                    is_finished=is_finished,
                    position_ids=position_ids,
                )
            else:
                infer_output, is_finished = model.infer(
                    input_ids=input_ids_np,
                    valid_length_each_example=valid_length_each_example,
                    generation_config=gen_config,
                    prefill=prefill,
                    is_finished=is_finished,
                    position_ids=position_ids,
                )
        else:
            # MCore 模型：使用 infer_mcore 接口
            # position_ids 会在内部自动生成（prepare_inputs_for_generation_mcore）
            print("  - 使用 MCore infer_mcore 接口")
            if block_tables is None or slot_mapping is None:
                print("\n✗ 错误：block_tables 或 slot_mapping 未准备好")
                print("可能的原因：")
                print("  1. block_mgr 初始化失败")
                print("  2. 模型配置问题")
                print("  3. 序列长度超出限制")
                print("\n建议：")
                print("  - 检查模型配置文件中的 seq_length 和 num_blocks")
                print("  - 尝试减少输入序列长度")
                print("  - 查看上面的初始化日志获取更多信息")
                raise ValueError("MCore 模型需要 block_tables 和 slot_mapping，但准备失败")
            
            print(f"  - block_tables: {block_tables.shape if hasattr(block_tables, 'shape') else type(block_tables)}")
            print(f"  - slot_mapping: {slot_mapping.shape if hasattr(slot_mapping, 'shape') else type(slot_mapping)}")
            
            infer_output, is_finished = model.infer_mcore(
                input_ids=input_ids_np,
                valid_length_each_example=valid_length_each_example,
                generation_config=gen_config,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                prefill=prefill,
                is_finished=is_finished,
            )
        
        print("✓ infer调用成功")
        
        # 解析输出
        print("\n解析infer输出...")
        print(f"  - infer_output 类型: {type(infer_output).__name__}")
        print(f"  - infer_output 值: {infer_output}")
        
        # 处理不同类型的输出
        target_list = None
        probs = None
        logits = None
        
        if isinstance(infer_output, dict):
            # 字典格式（Legacy 或某些配置）
            print("  - 检测到字典格式")
            target_list = infer_output.get("target_list")
            probs = infer_output.get("probs")
            logits = infer_output.get("logits")
        elif hasattr(infer_output, 'target_list'):
            # InferOutput 对象（命名元组或类）
            print("  - 检测到 InferOutput 对象")
            target_list = infer_output.target_list
            probs = getattr(infer_output, 'probs', None)
            logits = getattr(infer_output, 'logits', None)
            print(f"  - 提取的 target_list: {target_list}")
            print(f"  - 提取的 target_list 类型: {type(target_list)}")
        elif isinstance(infer_output, (list, tuple)):
            # 直接返回列表或元组
            print("  - 检测到列表/元组格式")
            target_list = infer_output
        else:
            # 尝试直接使用
            print("  - 使用默认处理")
            target_list = infer_output
        
        print(f"\n✓ 解析后的 target_list: {target_list}")
        print(f"  - target_list 类型: {type(target_list).__name__}")
        
        if probs is not None:
            print(f"\n✓ probs（词表概率分布）:")
            print(f"  - shape: {probs.shape if hasattr(probs, 'shape') else 'N/A'}")
            print(f"  - dtype: {probs.dtype if hasattr(probs, 'dtype') else 'N/A'}")
            
            # 详细的统计诊断
            if hasattr(probs, 'shape') and len(probs.shape) >= 2:
                probs_np = probs[0] if len(probs.shape) == 2 else probs
                if hasattr(probs_np, 'asnumpy'):
                    probs_np = probs_np.asnumpy()
                
                # 统计信息
                print(f"\n  📊 probs 统计诊断:")
                print(f"    - min: {np.min(probs_np):.10f}")
                print(f"    - max: {np.max(probs_np):.10f}")
                print(f"    - mean: {np.mean(probs_np):.10f}")
                print(f"    - sum: {np.sum(probs_np):.10f} (应该接近 1.0)")
                print(f"    - 非零元素数: {np.count_nonzero(probs_np)}/{len(probs_np)}")
                print(f"    - >0.001 的元素数: {np.sum(probs_np > 0.001)}")
                print(f"    - >0.01 的元素数: {np.sum(probs_np > 0.01)}")
                
                # 显示概率最高的前5个token
                top_k_indices = np.argsort(probs_np)[-5:][::-1]
                top_k_probs = probs_np[top_k_indices]
                print(f"\n  - Top 5 tokens:")
                for idx, prob in zip(top_k_indices, top_k_probs):
                    try:
                        token_text = tokenizer.decode([int(idx)], skip_special_tokens=False)
                        print(f"    Token {idx}: {prob:.10f} ('{token_text}')")
                    except:
                        print(f"    Token {idx}: {prob:.10f}")
                
                # ⚠️ 如果所有概率都是 0，这是严重问题
                if np.max(probs_np) == 0:
                    print(f"\n  ⚠️⚠️⚠️ 警告：所有概率都是 0！这表明可能有问题：")
                    print(f"    1. 模型权重可能没有正确加载")
                    print(f"    2. 数值计算可能下溢到 0")
                    print(f"    3. 需要检查 logits（如果可用）")
        
        if logits is not None:
            print(f"\n✓ logits（原始输出）:")
            print(f"  - shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}")
            if hasattr(logits, 'shape'):
                logits_np = logits[0] if len(logits.shape) == 2 else logits
                if hasattr(logits_np, 'asnumpy'):
                    logits_np = logits_np.asnumpy()
                print(f"  - min: {np.min(logits_np):.6f}")
                print(f"  - max: {np.max(logits_np):.6f}")
                print(f"  - mean: {np.mean(logits_np):.6f}")
        else:
            print(f"\n⚠️ logits 为 None (output_logits=False)")
        
        # 解码下一个token
        if target_list is not None:
            # 确保 target_list 是可索引的
            if isinstance(target_list, (list, tuple)):
                next_token = target_list[0]
            else:
                next_token = target_list
            
            # 转换为整数
            if hasattr(next_token, 'item'):
                # Tensor 或 numpy 数组
                next_token = next_token.item()
            next_token = int(next_token)
            
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=False)
            print(f"\n生成的下一个token ID: {next_token}")
            print(f"解码后的文本: '{next_token_text}'")
        
        print(f"\nis_finished状态: {is_finished}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ infer测试失败!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        print(traceback.format_exc())
        return False


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Qwen3-14B MindFormers接口测试")
    print("="*60)
    print(f"测试模式: {args.test_mode}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"使用增量推理: {args.use_past}")
    print("="*60)
    
    # 加载模型和分词器
    model, tokenizer = create_model_tokenizer(args)
    
    # 快速前向传播测试
    forward_test_ok = quick_forward_test(model, tokenizer)
    if not forward_test_ok:
        print("\n❌ 快速前向传播测试未通过，建议先解决权重加载问题")
        print("是否继续运行完整测试？按Ctrl+C取消，或等待5秒自动继续...")
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n用户取消测试")
            return
    
    # 运行测试
    results = {}
    
    # 重要：先测试 infer，再测试 generate，避免图编译冲突
    if args.test_mode in ["infer", "both"]:
        results["infer"] = test_infer(model, tokenizer, args.prompt, args)
        # 清理 block table cache
        if hasattr(model, 'block_mgr') and model.block_mgr:
            try:
                model.block_mgr.clear_cache()
                print("\n✓ 已清理 block table cache")
            except Exception:
                pass
    
    if args.test_mode in ["generate", "both"]:
        results["generate"] = test_generate(model, tokenizer, args.prompt, args)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

