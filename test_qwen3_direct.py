#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用MindFormers Qwen3模型类测试Qwen3-14B
正确引入mindformers的Qwen3支持，加载权重和分词器，测试generate与infer
"""
import argparse
import json
import os
import sys
from typing import Optional

import mindspore as ms
import numpy as np

# 导入 mindformers（应该已经安装）
from mindformers import AutoTokenizer, MindFormerConfig
from mindformers.generation import GenerationConfig
from mindformers.core.context import build_context

# 验证 qwen3 模块是否可用
try:
    import mindformers.models.qwen3
    print("✓ 检测到 mindformers.models.qwen3 模块")
except ImportError as e:
    print(f"⚠ 无法导入 mindformers.models.qwen3: {e}")
    print("可能需要更新 mindformers 版本或从源码安装")


def parse_args():
    parser = argparse.ArgumentParser(description="直接测试Qwen3模型")
    parser.add_argument("--config", dest="config_path", type=str, required=True,
                        help="MindFormers YAML配置文件路径")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="模型权重目录路径（包含.safetensors或.ckpt文件）")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="分词器目录路径，默认使用模型路径")
    parser.add_argument("--test_mode", type=str, default="both",
                        choices=["generate", "infer", "both"],
                        help="测试模式：generate/infer/both")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。",
                        help="测试用的提示词")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="生成的最大token数量")
    parser.add_argument("--use_past", action="store_true", default=False,
                        help="是否使用增量推理（KV cache）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--use_training_conversion", action="store_true",
                        help="使用训练模型进行权重转换（当推理模型缺少convert_weight_dict时）")
    return parser.parse_args()


def manual_convert_weight_names(params_dict, model):
    """
    手动转换 HuggingFace 权重名称到 MindFormers 格式
    
    使用模型的 weight_mapping 进行名称转换
    注意：此方法仅做简单的名称映射，不进行 QKV/FFN 合并
    """
    if not hasattr(model, 'weight_mapping'):
        print("  ⚠ 模型没有 weight_mapping 属性，跳过转换")
        return params_dict
    
    weight_mapping = model.weight_mapping
    converted_dict = {}
    
    for old_name, param in params_dict.items():
        new_name = old_name
        
        # 应用 weight_mapping 转换
        for hf_pattern, mf_pattern in weight_mapping:
            if hf_pattern in new_name:
                new_name = new_name.replace(hf_pattern, mf_pattern)
        
        converted_dict[new_name] = param
    
    print(f"  转换了 {len(converted_dict)} 个参数")
    
    # 显示几个转换示例
    print("  转换示例:")
    sample_count = 0
    for old_name, param in params_dict.items():
        new_name = old_name
        for hf_pattern, mf_pattern in weight_mapping:
            if hf_pattern in new_name:
                new_name = new_name.replace(hf_pattern, mf_pattern)
        if new_name != old_name and sample_count < 3:
            print(f"    {old_name[:60]}... -> {new_name[:60]}...")
            sample_count += 1
    
    return converted_dict


def load_and_display_config_json(model_dir):
    """
    从模型目录加载并显示 config.json
    
    Args:
        model_dir: 模型目录
        
    Returns:
        dict: 配置字典，如果不存在返回 None
    """
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"  ⚠️ 未找到 config.json: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        print(f"  ✓ 找到 config.json")
        print(f"  关键配置项:")
        for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                    'num_key_value_heads', 'vocab_size', 'intermediate_size']:
            if key in config_dict:
                print(f"    - {key}: {config_dict[key]}")
        
        return config_dict
    except Exception as e:
        print(f"  ⚠️ 读取 config.json 失败: {e}")
        return None


def verify_weight_config_match(params_dict, model_config):
    """
    验证权重与模型配置是否匹配
    
    Args:
        params_dict: 加载的权重字典
        model_config: 模型配置
        
    Returns:
        bool: 是否匹配
        dict: 检测到的配置（如果可以从权重推断）
    """
    print("\n" + "="*60)
    print("验证权重与配置的匹配性")
    print("="*60)
    
    detected_config = {}
    issues = []
    
    # 检查 embedding 层
    embedding_keys = [
        'model.embed_tokens.weight',
        'embedding.word_embeddings.weight'
    ]
    
    embedding_weight = None
    for key in embedding_keys:
        if key in params_dict:
            embedding_weight = params_dict[key]
            break
    
    if embedding_weight is not None:
        if hasattr(embedding_weight, 'shape'):
            shape = embedding_weight.shape
        elif hasattr(embedding_weight, 'asnumpy'):
            shape = embedding_weight.asnumpy().shape
        else:
            shape = None
        
        if shape is not None:
            detected_vocab_size, detected_hidden_size = shape
            detected_config['vocab_size'] = detected_vocab_size
            detected_config['hidden_size'] = detected_hidden_size
            
            print(f"\n从权重检测到的配置:")
            print(f"  - vocab_size: {detected_vocab_size}")
            print(f"  - hidden_size: {detected_hidden_size}")
            
            print(f"\n当前模型配置:")
            print(f"  - vocab_size: {model_config.vocab_size}")
            print(f"  - hidden_size: {model_config.hidden_size}")
            
            # 检查是否匹配
            if detected_vocab_size != model_config.vocab_size:
                issues.append(f"vocab_size 不匹配：权重={detected_vocab_size}, 配置={model_config.vocab_size}")
            
            if detected_hidden_size != model_config.hidden_size:
                issues.append(f"hidden_size 不匹配：权重={detected_hidden_size}, 配置={model_config.hidden_size}")
    
    if issues:
        print("\n" + "⚠️ "*20)
        print("检测到配置不匹配问题:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print("⚠️ "*20)
        return False, detected_config
    else:
        print("\n✓ 权重与配置匹配")
        return True, detected_config


def check_checkpoint_files(model_path):
    """检查checkpoint文件是否存在"""
    print("\n" + "="*60)
    print("检查Checkpoint文件")
    print("="*60)
    
    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ 模型路径不存在: {model_path}")
        return False
    
    print(f"模型路径: {model_path}")
    
    # 检查各种可能的checkpoint文件
    import glob
    checkpoint_patterns = [
        "*.safetensors",
        "*.ckpt",
        "*.bin",
    ]
    
    found_files = []
    total_size = 0
    
    for pattern in checkpoint_patterns:
        files = glob.glob(os.path.join(model_path, pattern))
        found_files.extend(files)
    
    if found_files:
        print(f"\n✓ 找到 {len(found_files)} 个checkpoint文件:")
        for f in found_files:
            size = os.path.getsize(f)
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  - {os.path.basename(f)}: {size_mb:.2f} MB")
        
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"\n总大小: {total_size_gb:.2f} GB")
        return True
    else:
        print("\n❌ 未找到任何checkpoint文件！")
        return False


def load_tokenizer(tokenizer_path, trust_remote_code=True):
    """加载分词器"""
    print("\n" + "="*60)
    print("加载分词器")
    print("="*60)
    
    if not os.path.exists(tokenizer_path):
        raise RuntimeError(f"分词器路径不存在: {tokenizer_path}")
    
    print(f"分词器路径: {tokenizer_path}")
    
    try:
        # 首先尝试使用 MindFormers AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        print("✓ 使用MindFormers AutoTokenizer加载成功")
    except Exception as e:
        print(f"⚠ MindFormers AutoTokenizer加载失败: {e}")
        print("尝试使用HuggingFace AutoTokenizer...")
        from transformers import AutoTokenizer as HFAutoTokenizer
        tokenizer = HFAutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        print("✓ 使用HuggingFace AutoTokenizer加载成功")
    
    # 确保分词器有必要的特殊token
    if getattr(tokenizer, "eos_token", None) is None:
        tokenizer.eos_token = "<|endoftext|>"
        print(f"设置eos_token: {tokenizer.eos_token}")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置pad_token: {tokenizer.pad_token}")
    
    print(f"✓ 词汇表大小: {len(tokenizer)}")
    print(f"✓ eos_token_id: {tokenizer.eos_token_id}")
    print(f"✓ pad_token_id: {tokenizer.pad_token_id}")
    
    return tokenizer


def load_model_from_config(config_path, model_dir=None, use_training_for_conversion=False):
    """
    从配置文件加载Qwen3模型
    
    关键点：
    1. 设置RUN_MODE=predict环境变量以使用推理模型
    2. 直接导入Qwen3模型类
    3. 使用from_pretrained加载权重
    
    Args:
        config_path: 配置文件路径
        model_dir: 模型权重目录
        use_training_for_conversion: 是否使用训练模型进行权重转换（解决 InferenceQwen3ForCausalLM 缺少 convert_weight_dict 的问题）
    """
    print("\n" + "="*60)
    print("加载Qwen3模型")
    print("="*60)
    
    # 1. 设置环境变量为推理模式（除非需要用训练模式转换）
    if use_training_for_conversion:
        # 暂时不设置 RUN_MODE，让它创建训练模型用于转换
        print("✓ 使用训练模型模式进行权重转换")
    else:
        os.environ["RUN_MODE"] = "predict"
        print("✓ 设置 RUN_MODE=predict")
    
    # 2. 加载配置文件
    print(f"\n读取配置文件: {config_path}")
    cfg = MindFormerConfig(config_path, run_mode="predict")
    
    # 3. 设置设备ID
    device_id_str = os.getenv("DEVICE_ID", "0")
    try:
        cfg.context.device_id = int(device_id_str)
        print(f"✓ 设置设备ID: {cfg.context.device_id}")
    except Exception as e:
        print(f"⚠ 设置设备ID失败: {e}")
    
    # 4. 构建上下文
    print("\n构建MindSpore上下文...")
    build_context(cfg)
    print("✓ MindSpore上下文构建完成")
    
    # 5. 直接导入Qwen3模型
    print("\n导入Qwen3模型类...")
    try:
        from mindformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
        print("✓ 成功导入 Qwen3ForCausalLM 和 Qwen3Config")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确认mindformers路径正确，且包含qwen3模块")
        raise
    
    # 6. 检查checkpoint文件和配置
    if model_dir:
        check_checkpoint_files(model_dir)
        
        # 显示 config.json 内容
        print("\n" + "="*60)
        print("检查权重目录的 config.json")
        print("="*60)
        weight_config = load_and_display_config_json(model_dir)
    else:
        weight_config = None
    
    # 7. 创建模型配置
    print("\n" + "="*60)
    print("创建Qwen3Config")
    print("="*60)
    
    # 首先尝试从权重目录的 config.json 加载配置（最准确）
    qwen3_config = None
    if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"\n尝试从权重目录加载 config.json: {model_dir}")
        try:
            qwen3_config = Qwen3Config.from_pretrained(model_dir)
            print("✓ 从权重目录的 config.json 加载配置成功")
        except Exception as e:
            print(f"⚠ 从权重目录加载配置失败: {e}")
            print("  将使用 YAML 配置文件")
    
    # 如果没有从权重目录加载成功，使用 YAML 配置
    if qwen3_config is None:
        print("\n使用 YAML 配置文件创建配置...")
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'model_config'):
            model_config_dict = cfg.model.model_config.to_dict()
            qwen3_config = Qwen3Config(**model_config_dict)
            print("✓ 从YAML配置创建Qwen3Config")
        else:
            # 如果配置文件中没有明确的model_config，尝试从配置文件直接构建
            qwen3_config = Qwen3Config.from_pretrained(os.path.dirname(config_path))
            print("✓ 从目录加载Qwen3Config")
        
        # 如果有权重配置，警告可能的不匹配
        if weight_config:
            print("\n⚠️ 注意：YAML 配置可能与权重不匹配")
            print("建议：确保 YAML 中的配置与权重目录的 config.json 一致")
    
    print(f"\n模型配置:")
    print(f"  - hidden_size: {qwen3_config.hidden_size}")
    print(f"  - num_hidden_layers: {qwen3_config.num_hidden_layers}")
    print(f"  - num_attention_heads: {qwen3_config.num_attention_heads}")
    print(f"  - num_key_value_heads: {qwen3_config.num_key_value_heads}")
    print(f"  - vocab_size: {qwen3_config.vocab_size}")
    print(f"  - intermediate_size: {qwen3_config.intermediate_size}")
    
    # 8. 创建模型实例
    print("\n创建Qwen3模型实例...")
    if use_training_for_conversion:
        # 先创建训练模型用于权重转换
        os.environ.pop("RUN_MODE", None)  # 移除 RUN_MODE 以创建训练模型
        training_model = Qwen3ForCausalLM(config=qwen3_config)
        print("✓ 训练模型实例创建成功（用于权重转换）")
        print(f"  - 模型类型: {type(training_model).__name__}")
        
        # 稍后会用它来转换权重，然后创建推理模型
        model = None  # 暂时为None
        temp_training_model = training_model
    else:
        model = Qwen3ForCausalLM(config=qwen3_config)
        print("✓ 模型实例创建成功")
        print(f"  - 模型类型: {type(model).__name__}")
        temp_training_model = None
    
    # 9. 加载权重
    if model_dir and os.path.exists(model_dir):
        print(f"\n从目录加载权重: {model_dir}")
        try:
            # 查找权重文件
            import glob
            safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
            ckpt_files = glob.glob(os.path.join(model_dir, "*.ckpt"))
            
            if safetensors_files:
                # 加载 safetensors 文件
                print(f"找到 {len(safetensors_files)} 个 safetensors 文件")
                from mindspore import load_checkpoint as ms_load_checkpoint
                from mindspore import load_param_into_net
                
                # 如果有多个文件，需要合并加载
                params_dict = {}
                for sf_file in safetensors_files:
                    print(f"  加载: {os.path.basename(sf_file)}")
                    file_params = ms_load_checkpoint(sf_file, format='safetensors')
                    params_dict.update(file_params)
                
                # 验证权重与配置的匹配性（在转换前，使用原始权重名称）
                match_ok, detected_config = verify_weight_config_match(params_dict, qwen3_config)
                
                if not match_ok:
                    print("\n" + "="*60)
                    print("建议修复方案")
                    print("="*60)
                    print("1. 使用权重目录中的 config.json（推荐）")
                    print(f"   确保权重目录 {model_dir} 包含 config.json 文件")
                    print("   脚本会自动从该文件加载正确的配置")
                    print("\n2. 手动修改 YAML 配置文件")
                    if 'hidden_size' in detected_config:
                        print(f"   将 hidden_size 改为: {detected_config['hidden_size']}")
                    if 'vocab_size' in detected_config:
                        print(f"   将 vocab_size 改为: {detected_config['vocab_size']}")
                    print("\n3. 常见 Qwen3 模型配置:")
                    print("   Qwen3-0.5B:  hidden_size=896,  num_layers=24")
                    print("   Qwen3-1.8B:  hidden_size=2048, num_layers=28")
                    print("   Qwen3-4B:    hidden_size=2560, num_layers=40")
                    print("   Qwen3-7B:    hidden_size=3584, num_layers=28")
                    print("   Qwen3-14B:   hidden_size=5120, num_layers=48")
                    print("   Qwen3-32B:   hidden_size=5120, num_layers=64")
                    print("="*60)
                    
                    # 询问是否继续（通过检测到的配置创建新模型）
                    print("\n尝试使用检测到的配置重新创建模型...")
                    if detected_config:
                        # 更新配置
                        for key, value in detected_config.items():
                            setattr(qwen3_config, key, value)
                        print(f"✓ 配置已更新:")
                        for key, value in detected_config.items():
                            print(f"  - {key}: {value}")
                        
                        # 重新创建模型
                        print("\n重新创建模型实例...")
                        if use_training_for_conversion:
                            os.environ.pop("RUN_MODE", None)
                            training_model = Qwen3ForCausalLM(config=qwen3_config)
                            print("✓ 训练模型实例重新创建成功")
                            model = None
                            temp_training_model = training_model
                        else:
                            model = Qwen3ForCausalLM(config=qwen3_config)
                            print("✓ 模型实例重新创建成功")
                            temp_training_model = None
                
                # 检查是否需要转换权重名称（HuggingFace -> MindFormers）
                print("\n检查权重名称映射...")
                if any('model.embed_tokens' in k for k in params_dict.keys()):
                    print("  检测到 HuggingFace 权重格式，执行名称转换...")
                    
                    # 尝试多种转换方法
                    converted = False
                    
                    # 方法1: 使用训练模型的 convert_weight_dict（如果用户指定了 --use_training_conversion）
                    if temp_training_model is not None:
                        print("  使用训练模型的 convert_weight_dict 方法...")
                        try:
                            params_dict = temp_training_model.convert_weight_dict(params_dict)
                            print("  ✓ 训练模型转换权重完成")
                            
                            # 将 numpy 数组转换为 Parameter 对象
                            print("  转换 numpy 数组为 Parameter 对象...")
                            from mindspore import Parameter
                            converted_params = {}
                            for name, value in params_dict.items():
                                if isinstance(value, np.ndarray):
                                    # 转换为 Tensor 再转为 Parameter
                                    converted_params[name] = Parameter(
                                        ms.Tensor(value),
                                        name=name,
                                        requires_grad=False
                                    )
                                elif isinstance(value, Parameter):
                                    converted_params[name] = value
                                else:
                                    # 如果是 MindSpore Tensor，转为 Parameter
                                    converted_params[name] = Parameter(
                                        value,
                                        name=name,
                                        requires_grad=False
                                    )
                            params_dict = converted_params
                            print(f"  ✓ 转换了 {len(params_dict)} 个参数")
                            converted = True
                        except Exception as e:
                            print(f"  ⚠ 训练模型转换失败: {e}")
                            import traceback
                            print(traceback.format_exc())
                    
                    # 方法2: 如果当前模型有 convert_weight_dict 方法
                    if not converted and model is not None and hasattr(model, 'convert_weight_dict') and callable(model.convert_weight_dict):
                        try:
                            params_dict = model.convert_weight_dict(params_dict)
                            print("  ✓ 使用模型的 convert_weight_dict 方法转换完成")
                            converted = True
                        except Exception as e:
                            print(f"  ⚠ 模型的 convert_weight_dict 失败: {e}")
                    
                    # 方法3: 使用 weight_mapping 手动转换
                    if not converted:
                        print("  使用 weight_mapping 手动转换权重名称...")
                        model_for_mapping = model if model is not None else temp_training_model
                        params_dict = manual_convert_weight_names(params_dict, model_for_mapping)
                        print("  ✓ 手动权重名称转换完成")
                        converted = True
                
                # 加载权重到模型
                # 确保所有参数都是 Parameter 对象
                print("\n  确保参数格式正确...")
                final_params = {}
                for name, value in params_dict.items():
                    if isinstance(value, Parameter):
                        final_params[name] = value
                    elif isinstance(value, np.ndarray):
                        final_params[name] = Parameter(
                            ms.Tensor(value),
                            name=name,
                            requires_grad=False
                        )
                    elif hasattr(value, 'asnumpy'):  # MindSpore Tensor
                        final_params[name] = Parameter(
                            value,
                            name=name,
                            requires_grad=False
                        )
                    else:
                        # 尝试直接转换
                        final_params[name] = Parameter(
                            ms.Tensor(value),
                            name=name,
                            requires_grad=False
                        )
                print(f"  ✓ {len(final_params)} 个参数格式已确认")
                
                # 加载权重到模型
                target_model = model if model is not None else temp_training_model
                not_loaded = load_param_into_net(target_model, final_params)
                if not_loaded:
                    print(f"  ⚠ 以下参数未加载: {not_loaded}")
                else:
                    print("  ✓ 所有权重加载成功")
                print("✓ safetensors 权重加载完成")
                
                # 如果使用了训练模型转换，现在创建推理模型
                if temp_training_model is not None:
                    print("\n释放训练模型，准备创建推理模型...")
                    
                    # 重要：删除训练模型以释放内存（约 28GB）
                    del temp_training_model
                    temp_training_model = None
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    print("✓ 训练模型已释放（~28GB）")
                    
                    # 删除原始权重字典，只保留 final_params
                    if 'params_dict' in locals():
                        del params_dict
                    gc.collect()
                    print("✓ 中间权重数据已释放")
                    
                    print("\n创建推理模型...")
                    os.environ["RUN_MODE"] = "predict"
                    model = Qwen3ForCausalLM(config=qwen3_config)
                    print("✓ 推理模型创建成功")
                    print(f"  - 模型类型: {type(model).__name__}")
                    
                    # 将转换后的权重加载到推理模型
                    print("  将权重加载到推理模型...")
                    # 使用已经转换好的 final_params
                    not_loaded = load_param_into_net(model, final_params)
                    if not_loaded:
                        print(f"  ⚠ 以下参数未加载到推理模型: {not_loaded}")
                    else:
                        print("  ✓ 推理模型权重加载成功")
                    
                    # 权重加载后，也可以删除 final_params 释放内存（约 28GB）
                    del final_params
                    gc.collect()
                    print("✓ 权重参数已释放（已加载到模型）")
                
            elif ckpt_files:
                # 加载 MindSpore checkpoint 文件
                print(f"找到 {len(ckpt_files)} 个 ckpt 文件")
                from mindspore import load_checkpoint as ms_load_checkpoint
                from mindspore import load_param_into_net
                
                params_dict = {}
                for ckpt_file in ckpt_files:
                    print(f"  加载: {os.path.basename(ckpt_file)}")
                    file_params = ms_load_checkpoint(ckpt_file)
                    params_dict.update(file_params)
                
                # ckpt 文件加载的应该已经是 Parameter 对象，但还是检查一下
                print("\n  检查参数格式...")
                final_params_ckpt = {}
                for name, value in params_dict.items():
                    if isinstance(value, Parameter):
                        final_params_ckpt[name] = value
                    else:
                        print(f"  转换参数 {name} 为 Parameter")
                        if isinstance(value, np.ndarray):
                            final_params_ckpt[name] = Parameter(ms.Tensor(value), name=name, requires_grad=False)
                        else:
                            final_params_ckpt[name] = Parameter(value, name=name, requires_grad=False)
                print(f"  ✓ {len(final_params_ckpt)} 个参数格式已确认")
                
                target_model = model if model is not None else temp_training_model
                not_loaded = load_param_into_net(target_model, final_params_ckpt)
                if not_loaded:
                    print(f"  ⚠ 以下参数未加载: {not_loaded}")
                else:
                    print("  ✓ 所有权重加载成功")
                print("✓ ckpt 权重加载完成")
                
                # 如果使用了训练模型，现在创建推理模型
                if temp_training_model is not None:
                    print("\n释放训练模型...")
                    del temp_training_model
                    temp_training_model = None
                    import gc
                    gc.collect()
                    print("✓ 训练模型已释放")
                    
                    print("\n创建推理模型...")
                    os.environ["RUN_MODE"] = "predict"
                    model = Qwen3ForCausalLM(config=qwen3_config)
                    print("✓ 推理模型创建成功")
                    print(f"  - 模型类型: {type(model).__name__}")
                    
                    print("  将权重加载到推理模型...")
                    not_loaded = load_param_into_net(model, final_params_ckpt)
                    if not_loaded:
                        print(f"  ⚠ 以下参数未加载到推理模型: {not_loaded}")
                    else:
                        print("  ✓ 推理模型权重加载成功")
                    
                    # 释放权重参数
                    del final_params_ckpt
                    gc.collect()
                    print("✓ 权重参数已释放")
                
            else:
                print("⚠ 未找到权重文件（.safetensors 或 .ckpt）")
                print("模型将使用随机初始化的权重")
                
                # 如果使用了训练模型但没有权重，仍需创建推理模型
                if temp_training_model is not None:
                    print("\n释放训练模型...")
                    del temp_training_model
                    temp_training_model = None
                    import gc
                    gc.collect()
                    print("✓ 训练模型已释放")
                    
                    print("\n创建推理模型（无权重）...")
                    os.environ["RUN_MODE"] = "predict"
                    model = Qwen3ForCausalLM(config=qwen3_config)
                    print("✓ 推理模型创建成功（随机初始化）")
                
        except Exception as e:
            print(f"⚠ 权重加载失败: {e}")
            import traceback
            print(traceback.format_exc())
            print("模型将使用随机初始化的权重")
    else:
        print("\n⚠ 未指定model_dir，模型使用随机初始化的权重")
        
        # 如果使用了训练模型但没有指定权重目录，仍需创建推理模型
        if temp_training_model is not None:
            print("\n释放训练模型...")
            del temp_training_model
            temp_training_model = None
            import gc
            gc.collect()
            print("✓ 训练模型已释放")
            
            print("\n创建推理模型（无权重）...")
            os.environ["RUN_MODE"] = "predict"
            model = Qwen3ForCausalLM(config=qwen3_config)
            print("✓ 推理模型创建成功（随机初始化）")
    
    # 10. 确保 model 不是 None
    if model is None:
        raise RuntimeError("模型创建失败，model 为 None")
    
    # 11. 设置为评估模式
    model.set_train(False)
    print("✓ 模型设置为评估模式")
    
    print("\n" + "="*60)
    print("模型加载完成")
    print("="*60)
    
    return model, qwen3_config


def verify_model_weights(model, tokenizer=None):
    """验证模型权重是否正确加载"""
    print("\n" + "="*60)
    print("验证模型权重")
    print("="*60)
    
    all_params = []
    zero_params = []
    
    print("正在检查模型参数...")
    for name, param in model.parameters_and_names():
        if hasattr(param, 'asnumpy'):
            try:
                param_data = param.asnumpy()
                all_params.append((name, param_data))
                
                # 检查全零参数
                if np.all(param_data == 0):
                    zero_params.append(name)
            except Exception as e:
                print(f"  ⚠ 无法检查参数 {name}: {e}")
                continue
    
    print(f"✓ 总共检查了 {len(all_params)} 个参数")
    
    if zero_params:
        print(f"\n⚠️ 发现 {len(zero_params)} 个全零参数（可能未正确加载）:")
        for name in zero_params[:10]:
            print(f"  - {name}")
        if len(zero_params) > 10:
            print(f"  ... 还有 {len(zero_params) - 10} 个")
        return False
    else:
        print("\n✓ 没有发现全零参数")
        return True


def test_generate(model, tokenizer, prompt, args):
    """测试generate接口"""
    print("\n" + "="*60)
    print("测试 generate 接口")
    print("="*60)
    
    try:
        print(f"\n输入提示词: {prompt}")
        
        # 应用chat template
        print("\n应用chat template...")
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors=None
                )
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist()
                input_ids = [input_ids]
                print("✓ 使用apply_chat_template处理完成")
            except Exception as e:
                print(f"⚠ apply_chat_template失败: {e}")
                # 手动添加chat template
                formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
                input_ids = inputs['input_ids'].tolist()
        else:
            print("⚠ tokenizer没有apply_chat_template方法")
            formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
            input_ids = inputs['input_ids'].tolist()
        
        print(f"✓ 输入长度: {len(input_ids[0])}")
        
        # 构建GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=1,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            use_past=args.use_past,
        )
        print(f"✓ GenerationConfig配置完成")
        print(f"  - max_new_tokens: {gen_config.max_new_tokens}")
        print(f"  - use_past: {gen_config.use_past}")
        
        # 调用generate
        print("\n开始生成...")
        output_ids = model.generate(
            input_ids=input_ids,
            generation_config=gen_config
        )
        print("✓ generate调用成功")
        
        # 解码输出
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
        print(f"错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_infer(model, tokenizer, prompt, args):
    """测试infer接口"""
    print("\n" + "="*60)
    print("测试 infer 接口")
    print("="*60)
    
    try:
        print(f"\n输入提示词: {prompt}")
        
        # 应用chat template
        print("\n应用chat template...")
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors=None
                )
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist()
                input_ids_np = np.array([input_ids], dtype=np.int32)
                print("✓ 使用apply_chat_template处理完成")
            except Exception as e:
                print(f"⚠ apply_chat_template失败: {e}")
                formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
                input_ids_np = inputs['input_ids']
        else:
            print("⚠ tokenizer没有apply_chat_template方法")
            formatted_prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(formatted_prompt, return_tensors='np', padding=True)
            input_ids_np = inputs['input_ids']
        
        print(f"✓ 输入shape: {input_ids_np.shape}")
        
        # 计算有效长度
        batch_size = input_ids_np.shape[0]
        pad_token_id = int(tokenizer.pad_token_id)
        valid_length_each_example = []
        for i in range(batch_size):
            valid_indices = np.where(input_ids_np[i] != pad_token_id)[0]
            if len(valid_indices) > 0:
                valid_len = int(np.max(valid_indices)) + 1
            else:
                valid_len = input_ids_np.shape[1]
            valid_length_each_example.append(valid_len)
        valid_length_each_example = np.array(valid_length_each_example, dtype=np.int32)
        print(f"✓ 有效长度: {valid_length_each_example}")
        
        # 构建GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=1,
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            use_past=args.use_past,
            return_dict_in_generate=True,
        )
        print("✓ GenerationConfig配置完成")
        
        # 检查模型是否有infer方法
        if not hasattr(model, 'infer') and not hasattr(model, 'infer_mcore'):
            print("\n⚠️ 模型没有infer或infer_mcore方法")
            print("可能原因：")
            print("  1. 模型是训练模型而非推理模型")
            print("  2. 需要设置RUN_MODE=predict环境变量")
            return False
        
        # 准备infer参数
        is_finished = [False] * batch_size
        prefill = True
        
        # 检查是否是MCore模型
        from mindformers.core.context import is_legacy_model
        use_legacy = is_legacy_model()
        print(f"\n模型架构: {'Legacy' if use_legacy else 'MCore'}")
        
        # 初始化必要的组件
        if hasattr(model, '_set_block_mgr'):
            try:
                model._set_block_mgr(batch_size, model.config.seq_length)
                print("✓ block_mgr初始化成功")
            except Exception as e:
                print(f"⚠ block_mgr初始化失败: {e}")
        
        if hasattr(model, '_set_kv_cache'):
            try:
                model._set_kv_cache()
                print("✓ kv_cache初始化成功")
            except Exception as e:
                print(f"⚠ kv_cache初始化失败: {e}")
        
        # 准备block_tables和slot_mapping
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
                print(f"✓ block_tables和slot_mapping准备完成")
            except Exception as e:
                print(f"⚠ 准备block_tables/slot_mapping失败: {e}")
        
        # 调用infer
        print("\n开始调用infer...")
        if use_legacy and hasattr(model, 'infer'):
            # Legacy模型
            position_ids = np.zeros((batch_size, input_ids_np.shape[1]), dtype=np.int32)
            for idx, length in enumerate(valid_length_each_example):
                if length > 0:
                    position_ids[idx, :length] = np.arange(length, dtype=np.int32)
            
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
        elif hasattr(model, 'infer_mcore'):
            # MCore模型
            if block_tables is None or slot_mapping is None:
                print("\n✗ MCore模型需要block_tables和slot_mapping")
                return False
            
            infer_output, is_finished = model.infer_mcore(
                input_ids=input_ids_np,
                valid_length_each_example=valid_length_each_example,
                generation_config=gen_config,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                prefill=prefill,
                is_finished=is_finished,
            )
        else:
            print("\n✗ 未找到合适的infer方法")
            return False
        
        print("✓ infer调用成功")
        
        # 解析输出
        print("\n解析输出...")
        target_list = None
        if isinstance(infer_output, dict):
            target_list = infer_output.get("target_list")
        elif hasattr(infer_output, 'target_list'):
            target_list = infer_output.target_list
        else:
            target_list = infer_output
        
        if target_list is not None:
            if isinstance(target_list, (list, tuple)):
                next_token = target_list[0]
            else:
                next_token = target_list
            
            if hasattr(next_token, 'item'):
                next_token = next_token.item()
            next_token = int(next_token)
            
            next_token_text = tokenizer.decode([next_token], skip_special_tokens=False)
            print(f"\n生成的下一个token ID: {next_token}")
            print(f"解码后的文本: '{next_token_text}'")
        
        print(f"is_finished状态: {is_finished}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ infer测试失败!")
        print(f"错误: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Qwen3-14B 直接加载测试")
    print("="*60)
    print(f"配置文件: {args.config_path}")
    print(f"模型目录: {args.model_dir}")
    print(f"测试模式: {args.test_mode}")
    print("="*60)
    
    # 1. 加载模型
    model, model_config = load_model_from_config(
        args.config_path, 
        args.model_dir,
        use_training_for_conversion=args.use_training_conversion
    )
    
    # 2. 加载分词器
    tokenizer_path = args.tokenizer_path or args.model_dir
    if not tokenizer_path:
        # 尝试从配置文件推断
        cfg = MindFormerConfig(args.config_path)
        if hasattr(cfg, 'pretrained_model_dir'):
            tokenizer_path = cfg.pretrained_model_dir
        else:
            print("❌ 无法确定分词器路径，请使用--tokenizer_path指定")
            return
    
    tokenizer = load_tokenizer(tokenizer_path)
    
    # 3. 验证权重
    weights_ok = verify_model_weights(model, tokenizer)
    if not weights_ok:
        print("\n⚠️ 权重验证未通过，但继续测试...")
    
    # 4. 运行测试
    results = {}
    
    # 先测试infer再测试generate
    if args.test_mode in ["infer", "both"]:
        results["infer"] = test_infer(model, tokenizer, args.prompt, args)
    
    if args.test_mode in ["generate", "both"]:
        results["generate"] = test_generate(model, tokenizer, args.prompt, args)
    
    # 5. 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

