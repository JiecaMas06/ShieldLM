#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证已安装的 mindformers 是否包含 Qwen3 支持
"""
import sys


def verify_qwen3_support():
    """验证 Qwen3 支持"""
    print("="*60)
    print("验证 MindFormers Qwen3 支持")
    print("="*60)
    
    # 1. 检查 mindformers 是否已安装
    print("\n1. 检查 mindformers 安装...")
    try:
        import mindformers
        print(f"✓ mindformers 已安装")
        print(f"  版本: {getattr(mindformers, '__version__', '未知')}")
        print(f"  路径: {mindformers.__file__}")
    except ImportError as e:
        print(f"❌ mindformers 未安装: {e}")
        return False
    
    # 2. 检查 qwen3 模块
    print("\n2. 检查 qwen3 模块...")
    try:
        from mindformers.models import qwen3
        print(f"✓ qwen3 模块存在")
        print(f"  路径: {qwen3.__file__}")
    except ImportError as e:
        print(f"❌ qwen3 模块不存在: {e}")
        print("  可能原因:")
        print("  - mindformers 版本过旧")
        print("  - 需要从源码安装最新版本")
        return False
    
    # 3. 检查 Qwen3 相关类
    print("\n3. 检查 Qwen3 类...")
    try:
        from mindformers.models.qwen3 import (
            Qwen3Config,
            Qwen3ForCausalLM,
            InferenceQwen3ForCausalLM,
            TrainingQwen3ForCausalLM,
            Qwen3PreTrainedModel
        )
        print("✓ 所有 Qwen3 类都可用:")
        print("  - Qwen3Config")
        print("  - Qwen3ForCausalLM")
        print("  - InferenceQwen3ForCausalLM")
        print("  - TrainingQwen3ForCausalLM")
        print("  - Qwen3PreTrainedModel")
    except ImportError as e:
        print(f"❌ 部分 Qwen3 类不可用: {e}")
        return False
    
    # 4. 检查模型注册
    print("\n4. 检查模型是否已注册到 MindFormer...")
    try:
        from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
        
        # 检查配置是否注册
        config_registered = MindFormerRegister.is_exist(
            module_type=MindFormerModuleType.CONFIG,
            class_name='Qwen3Config'
        )
        print(f"  Qwen3Config 注册状态: {'✓ 已注册' if config_registered else '❌ 未注册'}")
        
        # 检查模型是否注册
        model_registered = MindFormerRegister.is_exist(
            module_type=MindFormerModuleType.MODELS,
            class_name='Qwen3ForCausalLM'
        )
        print(f"  Qwen3ForCausalLM 注册状态: {'✓ 已注册' if model_registered else '❌ 未注册'}")
        
    except Exception as e:
        print(f"  ⚠ 检查注册状态时出错: {e}")
    
    # 5. 测试创建配置
    print("\n5. 测试创建 Qwen3Config...")
    try:
        config = Qwen3Config(
            vocab_size=151936,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
        )
        print("✓ Qwen3Config 创建成功")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - num_hidden_layers: {config.num_hidden_layers}")
        print(f"  - vocab_size: {config.vocab_size}")
    except Exception as e:
        print(f"❌ Qwen3Config 创建失败: {e}")
        return False
    
    # 6. 测试创建模型（不加载权重）
    print("\n6. 测试创建 Qwen3 模型（不加载权重）...")
    try:
        import os
        os.environ["RUN_MODE"] = "predict"  # 设置为推理模式
        
        model = Qwen3ForCausalLM(config=config)
        print("✓ Qwen3ForCausalLM 创建成功")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - 实际使用的类: {model.__class__.__name__}")
        
        # 检查模型是否有必要的方法
        has_generate = hasattr(model, 'generate')
        has_infer = hasattr(model, 'infer') or hasattr(model, 'infer_mcore')
        print(f"  - 支持 generate: {'✓' if has_generate else '❌'}")
        print(f"  - 支持 infer: {'✓' if has_infer else '❌'}")
        
    except Exception as e:
        print(f"❌ Qwen3ForCausalLM 创建失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    print("\n" + "="*60)
    print("✓ Qwen3 支持验证通过！")
    print("="*60)
    print("\n您可以直接使用以下方式导入 Qwen3:")
    print("  from mindformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM")
    print("\n或者使用 AutoModel:")
    print("  from mindformers import AutoModel, AutoConfig")
    print("  config = AutoConfig.from_pretrained('qwen3')")
    print("  model = AutoModel.from_pretrained('qwen3')")
    
    return True


if __name__ == "__main__":
    success = verify_qwen3_support()
    sys.exit(0 if success else 1)

